# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch>=2.3.1",
#   "transformers>=5.2.0",
#   "trl>=1.0.0",
#   "peft>=0.12.0",
#   "accelerate>=0.33.0",
#   "bitsandbytes>=0.43.3",
#   "wandb>=0.17.0",
#   "huggingface_hub>=0.24.0",
#   "openenv-core[core]>=0.2.2",
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
"""RLVR GRPO training entry point for HuggingFace Jobs.

GPU strategy (detected at runtime from ``torch.cuda.device_count()``):
  1 GPU  → colocate mode: vLLM and training share the GPU
  2 GPUs → server mode: vLLM on GPU 1, training on GPU 0

Environment variables (injected via ``hf jobs uv run --secrets / --env``):
  VERIRL_ENV_URL              (required) URL of the running VeriRL env server
  HF_TOKEN                    (required) HuggingFace write token
  WANDB_API_KEY               (optional) Weights & Biases key
  VERIRL_RESUME_FROM_CHECKPOINT (optional) ``'latest'`` or explicit checkpoint path
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap: clone the repo so all training.* imports resolve correctly.
# Mirrors Modal's add_local_dir — HF Jobs does not pre-install the project.
# ---------------------------------------------------------------------------

_REPO = "https://github.com/SupreethRao99/veriRL.git"
_WORKDIR = "/tmp/verirl_env"
_GIT_REF = os.environ.get("VERIRL_GIT_REF", "feat/working-grpo")
_OUTPUT_DIR = f"{_WORKDIR}/checkpoints"

if not os.path.exists(_WORKDIR):
    subprocess.run(
        ["git", "clone", "--depth=1", "--branch", _GIT_REF, _REPO, _WORKDIR],
        check=True,
    )

sys.path.insert(0, _WORKDIR)
sys.path.insert(0, str(Path(_WORKDIR).parent))
os.chdir(_WORKDIR)
os.makedirs(_OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Training — all imports after the bootstrap so the repo is on sys.path.
# noqa: E402 comments suppress import-order linting for the block below.
# ---------------------------------------------------------------------------

import torch  # noqa: E402

from training.config import TrainConfig  # noqa: E402
from training.runtime import (  # noqa: E402
    build_vllm_kwargs,
    resolve_resume_checkpoint,
    set_single_node_dist_env,
    start_vllm_server,
    wait_for_env_server,
)
from training.trainer import build_grpo_config, run_training, setup_auth  # noqa: E402

set_single_node_dist_env()

env_url = os.environ["VERIRL_ENV_URL"]
wait_for_env_server(env_url)

gpu_count = torch.cuda.device_count()
print(f"[VeriRL] GPU count: {gpu_count} → mode: {'server' if gpu_count >= 2 else 'colocate'}")

hf_token, wandb_key = setup_auth()
config = TrainConfig.from_yaml(env_url=env_url)

vllm_proc = None
try:
    if gpu_count >= 2:
        vllm_proc = start_vllm_server(config.vllm_base_model, config.max_model_length)
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    vllm_kwargs = build_vllm_kwargs(gpu_count, config.vllm_base_model, config.max_model_length)
    resume = resolve_resume_checkpoint(_OUTPUT_DIR, config.hf_output_repo, hf_token)
    grpo_config = build_grpo_config(
        config, hf_token, wandb_key, output_dir=_OUTPUT_DIR, **vllm_kwargs
    )
    result = run_training(
        config,
        grpo_config,
        hf_token,
        f"{_OUTPUT_DIR}/final",
        resume_from_checkpoint=resume,
    )
    print(f"[hf_jobs/grpo] Done: {result}")
finally:
    if vllm_proc is not None:
        vllm_proc.terminate()
