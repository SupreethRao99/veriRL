"""Shared runtime utilities for the Modal and HF Jobs training adapters.

Both adapters need to: wait for the env server, optionally spin up a vLLM
subprocess, and resolve resume checkpoints. This module centralises that logic
so neither adapter file duplicates it.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import requests


def wait_for_env_server(env_url: str, retries: int = 30, delay: int = 2) -> None:
    """Poll the VeriRL environment server until its /health endpoint responds.

    Args:
        env_url: Base URL of the VeriRL environment server.
        retries: Maximum number of poll attempts before raising.
        delay: Seconds to wait between each attempt.

    Raises:
        RuntimeError: If the server does not respond within ``retries * delay`` seconds.
    """
    print(f"[VeriRL] Waiting for env server at {env_url} ...")
    for _ in range(retries):
        try:
            if requests.get(f"{env_url}/health", timeout=5).status_code == 200:
                print("[VeriRL] Env server ready.")
                return
        except Exception:
            pass
        time.sleep(delay)
    raise RuntimeError(
        f"VeriRL env server at {env_url} not reachable after {retries * delay}s"
    )


def set_single_node_dist_env() -> None:
    """Set PyTorch distributed env vars for single-node, single-process training.

    Must be called before any CUDA context is opened. Configures RANK,
    LOCAL_RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT, and
    PYTORCH_CUDA_ALLOC_CONF for GRPOTrainer's internal process group.
    """
    os.environ.update({
        "RANK": "0",
        "LOCAL_RANK": "0",
        "WORLD_SIZE": "1",
        "MASTER_ADDR": "localhost",
        "MASTER_PORT": "12355",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })


def latest_checkpoint(root: str | Path) -> str | None:
    """Return the path of the highest-numbered ``checkpoint-N`` directory, or None.

    Args:
        root: Directory to search for ``checkpoint-N`` subdirectories.

    Returns:
        Absolute path string to the latest checkpoint, or ``None`` if none exist.
    """
    root = Path(root)
    checkpoints: list[tuple[int, Path]] = []
    for candidate in root.glob("checkpoint-*"):
        if not candidate.is_dir():
            continue
        try:
            step = int(candidate.name.rsplit("-", 1)[1])
        except (IndexError, ValueError):
            continue
        checkpoints.append((step, candidate))
    if not checkpoints:
        return None
    return str(max(checkpoints, key=lambda item: item[0])[1])


def start_vllm_server(
    vllm_model: str,
    max_model_len: int,
    port: int = 8001,
    log_path: str = "/tmp/vllm_server.log",
) -> subprocess.Popen:
    """Launch a ``trl vllm-serve`` subprocess on GPU 1 and wait until it is healthy.

    Strips PyTorch distributed env vars from the subprocess environment so
    vLLM's own ``dist.init_process_group`` does not conflict with the training
    TCPStore running at MASTER_PORT.

    Args:
        vllm_model: HuggingFace model ID or local path for vLLM to serve.
        max_model_len: Maximum token sequence length for the KV cache.
        port: HTTP port the vLLM server listens on.
        log_path: File path for combined vLLM stdout/stderr.

    Returns:
        The running ``subprocess.Popen`` handle for the vLLM server.

    Raises:
        RuntimeError: If the process exits early or fails to start within 360 s.
    """
    trl_bin = str(Path(sys.executable).parent / "trl")
    trl_ver = subprocess.run(
        [sys.executable, "-c", "import trl; print(trl.__version__)"],
        capture_output=True,
        text=True,
    )
    print(f"[VeriRL] Starting vLLM server on GPU 1, port {port} ...")
    print(f"[VeriRL] trl binary: {trl_bin}  version: {trl_ver.stdout.strip()}")

    _DIST_KEYS = {
        "RANK", "LOCAL_RANK", "WORLD_SIZE",
        "MASTER_ADDR", "MASTER_PORT",
        "TORCHELASTIC_RESTART_COUNT", "TORCHELASTIC_MAX_RESTARTS",
    }
    vllm_env = {k: v for k, v in os.environ.items() if k not in _DIST_KEYS}
    vllm_env.update({"CUDA_VISIBLE_DEVICES": "1", "PYTHONUNBUFFERED": "1"})

    vllm_log = open(log_path, "w")
    proc = subprocess.Popen(
        [
            trl_bin, "vllm-serve",
            "--model", vllm_model,
            "--port", str(port),
            "--gpu-memory-utilization", "0.9",
            "--max-model-len", str(max_model_len),
        ],
        env=vllm_env,
        stdout=vllm_log,
        stderr=subprocess.STDOUT,
    )

    for i in range(180):  # up to 360 s — first run downloads the model
        if proc.poll() is not None:
            vllm_log.flush()
            tail = open(log_path).read()[-3000:]
            raise RuntimeError(
                f"vLLM server exited early (code {proc.returncode}):\n{tail}"
            )
        try:
            if requests.get(f"http://localhost:{port}/health", timeout=2).status_code == 200:
                print("[VeriRL] vLLM server ready.")
                return proc
        except Exception:
            pass
        if i % 30 == 29:
            vllm_log.flush()
            print(f"[VeriRL] vLLM still starting ({(i + 1) * 2}s) ...")
        time.sleep(2)

    proc.kill()
    tail = open(log_path).read()[-3000:]
    raise RuntimeError(f"vLLM server failed to start within 360s. Log:\n{tail}")


def build_vllm_kwargs(
    gpu_count: int,
    vllm_model: str,
    max_model_len: int,
    vllm_port: int = 8001,
) -> dict:
    """Build the vLLM configuration kwargs dict for GRPOConfig.

    Chooses *server mode* when two or more GPUs are available (vLLM on GPU 1,
    training on GPU 0) and *colocate mode* otherwise. In colocate mode the
    context window is capped at 8192 to avoid OOM on a single card.

    Args:
        gpu_count: Number of available CUDA devices (``torch.cuda.device_count()``).
        vllm_model: HuggingFace model ID served by vLLM (unused in colocate mode).
        max_model_len: Maximum sequence length from the training config.
        vllm_port: Port the vLLM server listens on (server mode only).

    Returns:
        Dict ready to unpack as ``GRPOConfig(**vllm_kwargs)``.
    """
    if gpu_count >= 2:
        return {
            "use_vllm": True,
            "vllm_mode": "server",
            "vllm_server_host": "localhost",
            "vllm_server_port": vllm_port,
            "vllm_gpu_memory_utilization": 0.9,
            "vllm_max_model_length": max_model_len,
        }
    return {
        "use_vllm": True,
        "vllm_mode": "colocate",
        "vllm_gpu_memory_utilization": 0.5,
        "vllm_max_model_length": min(max_model_len, 8192),
    }


def resolve_resume_checkpoint(
    output_dir: str | Path,
    hub_repo_id: str,
    hf_token: str,
) -> str | None:
    """Resolve the VERIRL_RESUME_FROM_CHECKPOINT env var to a local checkpoint path.

    Resolution order:
      1. Env var unset → return ``None`` (fresh start).
      2. Env var is an explicit path (not ``'latest'``) → return it directly.
      3. Search ``output_dir`` for the highest-numbered checkpoint.
      4. Download from ``hub_repo_id`` and search the downloaded snapshot.

    Args:
        output_dir: Local directory where checkpoints are written.
        hub_repo_id: HuggingFace Hub repo to download from as a fallback.
        hf_token: HuggingFace token for authenticated Hub downloads.

    Returns:
        Absolute path to the checkpoint directory, or ``None`` for a fresh start.

    Raises:
        RuntimeError: If the env var is ``'latest'`` but no checkpoint is found.
    """
    from huggingface_hub import snapshot_download

    requested = os.environ.get("VERIRL_RESUME_FROM_CHECKPOINT", "").strip()
    if not requested:
        return None

    if requested not in {"latest", "last-checkpoint"}:
        print(f"[VeriRL] Resuming GRPO from explicit checkpoint: {requested}")
        return requested

    local_latest = latest_checkpoint(output_dir)
    if local_latest:
        print(f"[VeriRL] Resuming GRPO from local checkpoint: {local_latest}")
        return local_latest

    resume_dir = Path(output_dir) / "hub_resume"
    print(f"[VeriRL] Downloading checkpoints from {hub_repo_id} ...")
    snapshot_download(
        repo_id=hub_repo_id,
        token=hf_token,
        local_dir=resume_dir,
        allow_patterns=["last-checkpoint/**", "checkpoint-*/**"],
    )

    last_checkpoint = resume_dir / "last-checkpoint"
    if last_checkpoint.is_dir():
        print(f"[VeriRL] Resuming GRPO from Hub checkpoint: {last_checkpoint}")
        return str(last_checkpoint)

    hub_latest = latest_checkpoint(resume_dir)
    if hub_latest:
        print(f"[VeriRL] Resuming GRPO from Hub checkpoint: {hub_latest}")
        return hub_latest

    raise RuntimeError(
        f"VERIRL_RESUME_FROM_CHECKPOINT={requested!r}, but no checkpoint was found "
        f"locally in {output_dir} or on Hub at {hub_repo_id}"
    )
