"""
Local training runner — no Modal dependency.

For Modal-based training (remote GPU, volumes, secrets) use modal_infra.py instead.

Usage
-----
  python training/train.py                # standard GRPO
  python training/train.py --vllm        # + vLLM colocate mode
  python training/train.py --smoke       # connectivity smoke test

Environment
-----------
  VERIRL_ENV_URL    URL of the running VeriRL environment server
  HF_TOKEN          HuggingFace token (required for model download + hub push)
  WANDB_API_KEY     (optional) Weights & Biases key
"""

from __future__ import annotations

import argparse
import os
import textwrap

from training.config import TrainConfig
from training.trainer import build_grpo_config, run_training, setup_auth


def _smoke_test() -> dict:
    from verirl_env import VerirlAction, verirl_env  # type: ignore

    env_url = os.environ.get("VERIRL_ENV_URL", "http://localhost:8000")
    print(f"[smoke_test] Connecting to {env_url}")

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
        env.step(VerirlAction(action_type="write_file", filename="design.v", verilog_src=simple_verilog))
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VeriRL local training runner")
    parser.add_argument("--smoke", action="store_true", help="Run connectivity smoke test")
    parser.add_argument("--vllm",  action="store_true", help="Enable vLLM colocate mode")
    parser.add_argument("--output-dir", default="./checkpoints", help="Checkpoint output directory")
    args = parser.parse_args()

    if args.smoke:
        print(_smoke_test())
    else:
        hf_token, wandb_key = setup_auth()
        config = TrainConfig.from_yaml(
            env_url=os.environ.get("VERIRL_ENV_URL", "http://localhost:8000")
        )

        extra = {}
        if args.vllm:
            os.environ.update({
                "RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": "1",
                "MASTER_ADDR": "localhost", "MASTER_PORT": "12355",
            })
            extra = {"use_vllm": True, "vllm_mode": "colocate"}

        grpo_config = build_grpo_config(
            config, hf_token, wandb_key,
            output_dir=args.output_dir,
            **extra,
        )
        result = run_training(config, grpo_config, hf_token, f"{args.output_dir}/final")
        print(result)
