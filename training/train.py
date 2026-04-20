"""
VeriRL RLVR Training — Modal entry point
=========================================
Fine-tunes Qwen2.5-Coder-3B-Instruct on all 10 VeriRL tasks using GRPO.

Architecture
------------
- Algorithm:   GRPO (TRL GRPOTrainer + environment_factory)
- LoRA:        QLoRA rank-64, NF4 4-bit quantisation
- Inference:   optional vLLM colocate mode for faster generation
- Checkpoints: Modal Volume + HF Hub push

Setup
-----
0. Install training deps locally (macOS-safe — vllm/trl[vllm] skipped on non-Linux):
       uv sync --extra training

1. Deploy the VeriRL server (or use a hosted HF Space):
       docker run -p 8000:8000 ghcr.io/SupreethRao99/veriRL:latest

2. Create the Modal secret:
       modal secret create verirl-training \\
           HF_TOKEN=hf_xxx \\
           WANDB_API_KEY=wandb_xxx \\
           VERIRL_ENV_URL=https://your-space.hf.space

3. Run:
       modal run training/train.py              # standard QLoRA
       modal run training/train.py::train_vllm  # + vLLM colocate
       modal run training/train.py::smoke_test  # connectivity check (no GPU)
"""

from __future__ import annotations

import os
import textwrap
from pathlib import Path

import modal

from training.config import TrainConfig
from training.trainer import build_grpo_config, run_training, setup_auth

# ---------------------------------------------------------------------------
# Modal app, image, volumes, secrets
# ---------------------------------------------------------------------------

app = modal.App("verirl-rlvr")

# Build the training image.
# - uv_pip_install installs the full dep set (always Linux in the container).
# - add_local_python_source copies training/ into the image so `import training`
#   resolves inside the container.
training_image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "torch==2.3.1",
        "trl[vllm]==0.28.0",
        "vllm==0.12.0",
        "transformers==4.57",
        "peft>=0.12.0",
        "accelerate>=0.33.0",
        "bitsandbytes>=0.43.3",
        "omegaconf>=2.3.0",
        "openenv-core[core]>=0.2.2",
        "python-dotenv>=1.0.0",
        "wandb>=0.17.0",
        "huggingface_hub>=0.24.0",
        "datasets>=3.5.1",
        "numpy",
    )
    # Install the full repo as a package so Modal gets the current branch's
    # code (10 tasks, multi-file, formal) rather than the stale PyPI release.
    .add_local_python_source(".")
    .add_local_python_source("training")
    .env({"TOKENIZERS_PARALLELISM": "false"})
)

verirl_secrets = modal.Secret.from_name("verirl-training")

CHECKPOINTS_DIR = Path("/checkpoints")
checkpoints_volume = modal.Volume.from_name(
    "verirl-rlvr-checkpoints", create_if_missing=True
)

# Shared decorator kwargs to avoid duplication across training functions
_GPU_KWARGS = dict(
    image=training_image,
    gpu=modal.gpu.A100(count=1, size="80GB"),
    timeout=4 * 3600,
    secrets=[verirl_secrets],
    volumes={str(CHECKPOINTS_DIR): checkpoints_volume},
    memory=65536,
)

# ---------------------------------------------------------------------------
# Training functions
# ---------------------------------------------------------------------------


@app.function(**_GPU_KWARGS)
def train() -> dict:
    """Standard QLoRA GRPO training on a single A100-80GB."""
    hf_token, wandb_key = setup_auth()
    config = TrainConfig.from_yaml(
        env_url=os.environ.get("VERIRL_ENV_URL", "http://localhost:8000")
    )
    grpo_config = build_grpo_config(
        config, hf_token, wandb_key, output_dir=str(CHECKPOINTS_DIR)
    )
    return run_training(config, grpo_config, hf_token, str(CHECKPOINTS_DIR / "final"))


@app.function(**_GPU_KWARGS)
def train_vllm() -> dict:
    """
    QLoRA GRPO training with vLLM colocate mode for faster generation.

    vLLM runs inside the trainer process and shares GPU memory with the model,
    improving generation throughput at the cost of higher peak memory usage.
    """
    # Required environment variables for vLLM colocate mode
    os.environ.update({
        "RANK":        "0",
        "LOCAL_RANK":  "0",
        "WORLD_SIZE":  "1",
        "MASTER_ADDR": "localhost",
        "MASTER_PORT": "12355",
    })

    hf_token, wandb_key = setup_auth()
    config = TrainConfig.from_yaml(
        env_url=os.environ.get("VERIRL_ENV_URL", "http://localhost:8000")
    )
    grpo_config = build_grpo_config(
        config, hf_token, wandb_key,
        output_dir=str(CHECKPOINTS_DIR),
        use_vllm=True,
        vllm_mode="colocate",
    )
    return run_training(config, grpo_config, hf_token, str(CHECKPOINTS_DIR / "final"))


# ---------------------------------------------------------------------------
# Smoke test — verifies connectivity and one manual episode, no GPU needed
# ---------------------------------------------------------------------------


@app.function(
    image=training_image,
    timeout=300,
    secrets=[verirl_secrets],
)
def smoke_test() -> dict:
    """Validate env connectivity and a complete write→compile→sim→submit episode."""
    from verirl_env import VerirlAction, verirl_env  # type: ignore

    env_url = os.environ.get("VERIRL_ENV_URL", "http://localhost:8000")
    print(f"[smoke_test] Connecting to VeriRL at {env_url}")

    simple_verilog = textwrap.dedent("""
        module relu_clip #(parameter IN_W=8, parameter OUT_W=4) (
            input  wire signed [IN_W-1:0]  in_val,
            output wire        [OUT_W-1:0] out_val,
            output wire                    saturated
        );
            localparam integer MAX_OUT = (1 << OUT_W) - 1;
            wire neg = in_val[IN_W-1];
            wire [IN_W-1:0] relu_out = neg ? {IN_W{1'b0}} : in_val;
            wire pos_clip = (relu_out > MAX_OUT[IN_W-1:0]);
            assign out_val   = pos_clip ? MAX_OUT[OUT_W-1:0] : relu_out[OUT_W-1:0];
            assign saturated = neg | pos_clip;
        endmodule
    """).strip()

    env = verirl_env(base_url=env_url)
    score = 0.0
    try:
        result = env.reset(task_id="relu_clip")
        print(f"[smoke_test] task_spec length: {len(result.observation.task_spec)}")

        env.step(VerirlAction(
            action_type="write_file", filename="design.v", verilog_src=simple_verilog
        ))
        result = env.step(VerirlAction(action_type="run_compile"))
        print(f"[smoke_test] compile_ok={result.observation.compile_ok}")

        result = env.step(VerirlAction(action_type="run_sim"))
        obs = result.observation
        print(f"[smoke_test] sim: {obs.tests_passed}/{obs.tests_total} tests passed")

        result = env.step(VerirlAction(action_type="submit"))
        score = float(result.observation.final_score or 0.0)
        print(f"[smoke_test] final_score={score:.3f}")
    finally:
        env.close()

    print(f"[smoke_test] PASSED — relu_clip score={score:.3f}")
    return {"status": "ok", "relu_clip_score": score}


# ---------------------------------------------------------------------------
# Local entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="VeriRL RLVR training")
    parser.add_argument("--smoke", action="store_true", help="Run smoke test (no GPU)")
    parser.add_argument("--vllm",  action="store_true", help="Use vLLM colocate mode")
    args = parser.parse_args()

    with app.run():
        if args.smoke:
            print(smoke_test.remote())
        elif args.vllm:
            print(train_vllm.remote())
        else:
            print(train.remote())
