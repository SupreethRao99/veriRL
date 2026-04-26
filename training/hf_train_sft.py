# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch>=2.3.1",
#   "transformers>=4.57.0",
#   "trl>=0.14.0,<1.0.0",
#   "peft>=0.12.0",
#   "accelerate>=0.33.0",
#   "bitsandbytes>=0.43.3",
#   "wandb>=0.17.0",
#   "huggingface_hub>=0.24.0",
#   "datasets>=3.5.1",
#   "numpy",
#   "omegaconf>=2.3.0",
#   "trackio",
#   "unsloth==2026.4.8; sys_platform == 'linux'",
# ]
# ///
"""SFT warm-start training entry point for HuggingFace Jobs.

Environment variables (injected via ``hf jobs uv run --secrets``):
  HF_TOKEN      (required) HuggingFace write token
  WANDB_API_KEY (optional) Weights & Biases key
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
_OUTPUT_DIR = f"{_WORKDIR}/checkpoints/sft"

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
# Training — all imports after bootstrap so the repo is on sys.path.
# noqa: E402 comments suppress import-order linting for the block below.
# ---------------------------------------------------------------------------

from training.config import SFTConfig  # noqa: E402
from training.sft import run_sft  # noqa: E402
from training.trainer import setup_auth  # noqa: E402

hf_token, wandb_key = setup_auth()
config = SFTConfig.from_yaml()
result = run_sft(config, hf_token, wandb_key, output_dir=_OUTPUT_DIR)
print(f"[hf_jobs/sft] Done: {result}")
