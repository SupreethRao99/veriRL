# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch>=2.3.1",
#   "transformers>=4.57.0",
#   "trl>=1.0.0",
#   "peft>=0.12.0",
#   "accelerate>=0.33.0",
#   "bitsandbytes>=0.43.3",
#   "wandb>=0.17.0",
#   "huggingface_hub>=0.24.0",
#   "datasets>=3.5.1",
#   "numpy",
#   "omegaconf>=2.3.0",
#   "liger-kernel",
#   "jmespath>=1.0.0",
#   "requests",
#   "trackio",
#   "vllm>=0.11.0,<=0.18.0; sys_platform == 'linux'",
# ]
# ///
"""
RLVR GRPO training script for HF Jobs.

GPU strategy (detected at runtime from torch.cuda.device_count()):
  1 GPU  → colocate mode: vLLM and training share the GPU
  2 GPUs → server mode:   vLLM on GPU 1, training on GPU 0

Environment variables (injected via `hf jobs uv run --secrets / --env`):
  VERIRL_ENV_URL  (required) URL of the running VeriRL env server (e.g. your HF Space)
  HF_TOKEN        (required) HuggingFace write token
  WANDB_API_KEY   (optional) Weights & Biases key
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap: clone repo → identical to Modal's add_local_dir
# ---------------------------------------------------------------------------

_REPO = "https://github.com/SupreethRao99/veriRL.git"
_WORKDIR = "/tmp/verirl"
_GIT_REF = os.environ.get("VERIRL_GIT_REF", "feat/working-grpo")

if not os.path.exists(_WORKDIR):
    subprocess.run(
        ["git", "clone", "--depth=1", "--branch", _GIT_REF, _REPO, _WORKDIR],
        check=True,
    )

sys.path.insert(0, _WORKDIR)
os.chdir(_WORKDIR)

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

import requests  # noqa: E402
import torch  # noqa: E402

from training.config import TrainConfig  # noqa: E402
from training.trainer import build_grpo_config, run_training, setup_auth  # noqa: E402

_OUTPUT_DIR = f"{_WORKDIR}/checkpoints"
os.makedirs(_OUTPUT_DIR, exist_ok=True)


def _latest_checkpoint(root: str | Path) -> str | None:
    root = Path(root)
    checkpoints = []
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


def _resolve_resume_checkpoint(config: TrainConfig, hf_token: str) -> str | None:
    requested = os.environ.get("VERIRL_RESUME_FROM_CHECKPOINT", "").strip()
    if not requested:
        return None

    if requested not in {"latest", "last-checkpoint"}:
        print(f"[VeriRL] Resuming GRPO from explicit checkpoint: {requested}")
        return requested

    local_latest = _latest_checkpoint(_OUTPUT_DIR)
    if local_latest:
        print(f"[VeriRL] Resuming GRPO from local checkpoint: {local_latest}")
        return local_latest

    from huggingface_hub import snapshot_download  # noqa: E402

    resume_dir = Path(_OUTPUT_DIR) / "hub_resume"
    print(f"[VeriRL] Downloading checkpoints from {config.hf_output_repo} ...")
    snapshot_download(
        repo_id=config.hf_output_repo,
        token=hf_token,
        local_dir=resume_dir,
        allow_patterns=["last-checkpoint/**", "checkpoint-*/**"],
    )

    last_checkpoint = resume_dir / "last-checkpoint"
    if last_checkpoint.is_dir():
        print(f"[VeriRL] Resuming GRPO from Hub checkpoint: {last_checkpoint}")
        return str(last_checkpoint)

    hub_latest = _latest_checkpoint(resume_dir)
    if hub_latest:
        print(f"[VeriRL] Resuming GRPO from Hub checkpoint: {hub_latest}")
        return hub_latest

    raise RuntimeError(
        f"VERIRL_RESUME_FROM_CHECKPOINT={requested!r}, but no checkpoint was found"
    )

os.environ.update(
    {
        "RANK": "0",
        "LOCAL_RANK": "0",
        "WORLD_SIZE": "1",
        "MASTER_ADDR": "localhost",
        "MASTER_PORT": "12355",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }
)

# Wait for env server before touching the GPU (same as modal_infra.py)
env_url = os.environ["VERIRL_ENV_URL"]
print(f"[VeriRL] Waiting for env server at {env_url} ...")
for _ in range(30):
    try:
        if requests.get(f"{env_url}/health", timeout=5).status_code == 200:
            print("[VeriRL] Env server ready.")
            break
    except Exception:
        pass
    time.sleep(2)
else:
    raise RuntimeError(f"VeriRL env server at {env_url} not reachable after 60s")

gpu_count = torch.cuda.device_count()
print(
    f"[VeriRL] GPU count: {gpu_count} → mode: {'server' if gpu_count >= 2 else 'colocate'}"
)

hf_token, wandb_key = setup_auth()
config = TrainConfig.from_yaml(env_url=env_url)

vllm_proc = None

if gpu_count >= 2:
    # ── Server mode ──────────────────────────────────────────────────────────
    # Start vLLM on GPU 1 BEFORE opening a CUDA context on this process,
    # then restrict this process to GPU 0. Mirrors modal_infra.py server mode.
    vllm_port = 8001
    print(f"[VeriRL] Starting vLLM server on GPU 1, port {vllm_port} ...")

    trl_bin = str(__import__("pathlib").Path(sys.executable).parent / "trl")
    trl_ver = subprocess.run(
        [sys.executable, "-c", "import trl; print(trl.__version__)"],
        capture_output=True,
        text=True,
    )
    print(f"[VeriRL] trl binary: {trl_bin}  version: {trl_ver.stdout.strip()}")

    _DIST_KEYS = {
        "RANK", "LOCAL_RANK", "WORLD_SIZE",
        "MASTER_ADDR", "MASTER_PORT",
        "TORCHELASTIC_RESTART_COUNT", "TORCHELASTIC_MAX_RESTARTS",
    }
    vllm_env = {k: v for k, v in os.environ.items() if k not in _DIST_KEYS}
    vllm_env.update({"CUDA_VISIBLE_DEVICES": "1", "PYTHONUNBUFFERED": "1"})

    vllm_log = open("/tmp/vllm_server.log", "w")
    vllm_proc = subprocess.Popen(
        [
            trl_bin, "vllm-serve",
            "--model", config.vllm_base_model,
            "--port", str(vllm_port),
            "--gpu-memory-utilization", "0.9",
            "--max-model-len", str(config.max_model_length),
        ],
        env=vllm_env,
        stdout=vllm_log,
        stderr=subprocess.STDOUT,
    )
    for i in range(180):  # 360s — first run downloads model
        if vllm_proc.poll() is not None:
            vllm_log.flush()
            tail = open("/tmp/vllm_server.log").read()[-3000:]
            raise RuntimeError(
                f"vLLM server exited early (code {vllm_proc.returncode}):\n{tail}"
            )
        try:
            if (
                requests.get(f"http://localhost:{vllm_port}/health", timeout=2).status_code
                == 200
            ):
                print("[VeriRL] vLLM server ready.")
                break
        except Exception:
            pass
        if i % 30 == 29:
            vllm_log.flush()
            print(f"[VeriRL] vLLM still starting ({(i + 1) * 2}s) ...")
        time.sleep(2)
    else:
        vllm_proc.kill()
        tail = open("/tmp/vllm_server.log").read()[-3000:]
        raise RuntimeError(f"vLLM server failed to start within 360s. Log:\n{tail}")

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    vllm_kwargs = dict(
        use_vllm=True,
        vllm_mode="server",
        vllm_server_host="localhost",
        vllm_server_port=vllm_port,
        vllm_gpu_memory_utilization=0.9,
        vllm_max_model_length=config.max_model_length,
    )
else:
    # ── Colocate mode ────────────────────────────────────────────────────────
    vllm_kwargs = dict(
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.5,
        vllm_max_model_length=min(config.max_model_length, 8192),
    )

print(f"[VeriRL] base_model  = {config.base_model}")
print(f"[VeriRL] env_url     = {config.env_url}")
print(f"[VeriRL] output_repo = {config.hf_output_repo}")

try:
    resume_from_checkpoint = _resolve_resume_checkpoint(config, hf_token)
    grpo_config = build_grpo_config(
        config,
        hf_token,
        wandb_key,
        output_dir=_OUTPUT_DIR,
        **vllm_kwargs,
    )
    result = run_training(
        config,
        grpo_config,
        hf_token,
        f"{_OUTPUT_DIR}/final",
        resume_from_checkpoint=resume_from_checkpoint,
    )
    print(f"[hf_jobs/grpo] Done: {result}")
finally:
    if vllm_proc is not None:
        vllm_proc.terminate()
