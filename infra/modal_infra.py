"""VeriRL Modal infrastructure — SFT warm-start and RLVR training.

All Modal-specific code lives here. Training logic in ``training/`` has zero
Modal imports and runs identically locally or on Modal.

The VeriRL environment server runs as a separate Modal deployment
(``modal_env.py``). Set ``VERIRL_ENV_URL`` in the ``"verirl-training"`` secret
to point at that deployment before running ``train``.

GPU strategy (``config.yaml`` → ``modal.gpu_count``):
  1 GPU  → colocate mode: vLLM and training share the GPU
            (``max_model_length`` capped at 8192 to avoid OOM)
  2 GPUs → server mode: vLLM on GPU 1 (full 22 GB KV cache),
            training on GPU 0

Usage
-----
  modal run modal_infra.py::sft    # SFT warm-start on PyraNet-Verilog (H100, ~8h)
  modal run modal_infra.py::train  # RLVR GRPO from SFT checkpoint (2×L4, ~4h)
"""

from __future__ import annotations

import os
from pathlib import Path

import modal
from omegaconf import OmegaConf


# ---------------------------------------------------------------------------
# Config — read at import time so GPU decorator arguments are correct.
# ---------------------------------------------------------------------------

CONTAINER_ROOT = "/root/verirl"
CHECKPOINTS_DIR = f"{CONTAINER_ROOT}/checkpoints"
HF_CACHE_DIR = "/root/.cache/huggingface"

_cfg_candidates = [
    Path(CONTAINER_ROOT) / "config.yaml",
    Path(__file__).parent.parent / "config.yaml",  # infra/../config.yaml
]
_cfg_path = next(p for p in _cfg_candidates if p.exists())
_modal_cfg = OmegaConf.load(_cfg_path).modal
_GPU_COUNT = int(_modal_cfg.get("gpu_count", 1))
_GPU_SPEC = (
    f"{_modal_cfg.gpu_type}:{_GPU_COUNT}"
    if _GPU_COUNT > 1
    else str(_modal_cfg.gpu_type)
)


# ---------------------------------------------------------------------------
# Container images
# ---------------------------------------------------------------------------

_LOCAL_DIR_KWARGS = dict(
    remote_path=CONTAINER_ROOT,
    copy=True,
    ignore=[".git", "__pycache__", ".venv", ".pytest_cache", "*.pyc", "*.egg-info"],
)

# SFT image: Unsloth-based, TRL 0.x — incompatible with the GRPO vLLM image.
sft_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync(extras=["sft"])
    .env({"TOKENIZERS_PARALLELISM": "false", "PYTHONPATH": CONTAINER_ROOT})
    .add_local_dir(".", **_LOCAL_DIR_KWARGS)
    .run_commands(f"uv pip install --no-deps -e {CONTAINER_ROOT}")
)

# RLVR image: TRL 1.x + vLLM — separate from SFT because Unsloth pins TRL 0.x.
rlvr_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync(extras=["grpo"])
    .env({"TOKENIZERS_PARALLELISM": "false", "PYTHONPATH": CONTAINER_ROOT})
    .add_local_dir(".", **_LOCAL_DIR_KWARGS)
    .run_commands(f"uv pip install --no-deps -e {CONTAINER_ROOT}")
)


# ---------------------------------------------------------------------------
# Shared volumes and secrets
# ---------------------------------------------------------------------------

checkpoints_vol = modal.Volume.from_name("verirl-rlvr-checkpoints", create_if_missing=True)
hf_cache_vol    = modal.Volume.from_name("huggingface-cache", create_if_missing=True)

verirl_secrets = modal.Secret.from_name("verirl-training")
hf_secret      = modal.Secret.from_name("huggingface-secret")
wandb_secret   = modal.Secret.from_name("wandb-secret")

_SECRETS = [verirl_secrets, hf_secret, wandb_secret]
_VOLUMES = {CHECKPOINTS_DIR: checkpoints_vol, HF_CACHE_DIR: hf_cache_vol}

app = modal.App("verirl-rlvr")


# ---------------------------------------------------------------------------
# Phase 1 — SFT warm-start
# ---------------------------------------------------------------------------


@app.function(
    image=sft_image,
    gpu="H100",
    timeout=8 * 3600,
    secrets=_SECRETS,
    volumes=_VOLUMES,
    memory=65536,
)
def sft() -> dict:
    """SFT warm-start: fine-tune Qwen3-4B-Thinking on PyraNet-Verilog (692K samples).

    Returns:
        A dict with ``status``, ``output_repo``, and ``merged_repo`` keys.
    """
    from training.config import SFTConfig
    from training.sft import run_sft
    from training.trainer import setup_auth

    hf_token, wandb_key = setup_auth()
    config = SFTConfig.from_yaml()
    return run_sft(config, hf_token, wandb_key, output_dir=f"{CHECKPOINTS_DIR}/sft")


# ---------------------------------------------------------------------------
# Phase 2 — RLVR training (GRPO from SFT checkpoint)
# ---------------------------------------------------------------------------


@app.function(
    image=rlvr_image,
    gpu=_GPU_SPEC,
    timeout=8 * 3600,
    secrets=_SECRETS,
    volumes=_VOLUMES,
    memory=65536,
)
def train() -> dict:
    """RLVR: agentic GRPO with vLLM, starting from the SFT checkpoint.

    Automatically selects vLLM server mode (2 GPUs) or colocate mode (1 GPU)
    based on ``config.yaml → modal.gpu_count``.

    Returns:
        A dict with ``status`` and ``output_repo`` keys.
    """
    import torch

    from training.config import TrainConfig
    from training.runtime import (
        build_vllm_kwargs,
        set_single_node_dist_env,
        start_vllm_server,
        wait_for_env_server,
    )
    from training.trainer import build_grpo_config, run_training, setup_auth

    set_single_node_dist_env()

    env_url = os.environ["VERIRL_ENV_URL"]
    wait_for_env_server(env_url)

    gpu_count = int(
        OmegaConf.load(Path(CONTAINER_ROOT) / "config.yaml").modal.get("gpu_count", 1)
    )
    print(f"[VeriRL] GPU count: {gpu_count} → mode: {'server' if gpu_count >= 2 else 'colocate'}")

    hf_token, wandb_key = setup_auth()
    config = TrainConfig.from_yaml(env_url=env_url)

    vllm_proc = None
    try:
        if gpu_count >= 2:
            vllm_proc = start_vllm_server(config.vllm_base_model, config.max_model_length)
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"

        vllm_kwargs = build_vllm_kwargs(
            gpu_count, config.vllm_base_model, config.max_model_length
        )
        grpo_config = build_grpo_config(
            config, hf_token, wandb_key, output_dir=CHECKPOINTS_DIR, **vllm_kwargs
        )
        return run_training(config, grpo_config, hf_token, f"{CHECKPOINTS_DIR}/final")
    finally:
        if vllm_proc is not None:
            vllm_proc.terminate()


# ---------------------------------------------------------------------------
# SFT model sanity check
# ---------------------------------------------------------------------------


@app.function(
    image=sft_image,
    gpu="L4",
    timeout=600,
    secrets=_SECRETS,
    volumes=_VOLUMES,
)
def test_sft(task: str = "relu_clip") -> str:
    """Run ``test_sft.py`` against the pushed SFT checkpoint on an L4 GPU.

    Args:
        task: VeriRL task ID to test against (default: ``"relu_clip"``).

    Returns:
        Combined stdout from the test script.

    Raises:
        RuntimeError: If the script exits with a non-zero return code.
    """
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, f"{CONTAINER_ROOT}/training/test_sft.py", "--task", task],
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    if result.stderr:
        print("[stderr]", result.stderr[-3000:])
    if result.returncode != 0:
        raise RuntimeError(f"test_sft.py exited with code {result.returncode}")
    return result.stdout


# ---------------------------------------------------------------------------
# Local entry point
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def main():
    """Default entry point: run SFT warm-start when invoked via ``modal run modal_infra.py``."""
    sft.remote()
