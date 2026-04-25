"""
VeriRL environment server — Modal CPU deployment.

Hosts the OpenEnv VeriRL environment as a persistent ASGI web service.
No GPU required — iverilog/yosys/graphviz run on CPU.

Usage
-----
  modal deploy modal_env.py         # Deploy; prints stable HTTPS URL
  modal run modal_env.py::smoke_test  # Connectivity + episode check (no GPU)

After deploying, add the printed URL to your Modal secret "verirl-training"
as VERIRL_ENV_URL so the training app can reach it.
"""

import modal

CONTAINER_ROOT = "/root/verirl"

_LOCAL_DIR_KWARGS = dict(
    remote_path=CONTAINER_ROOT,
    copy=True,
    ignore=[".git", "__pycache__", ".venv", ".pytest_cache", "*.pyc", "*.egg-info"],
)

env_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("iverilog", "yosys", "graphviz")
    .uv_sync()
    .env({"PYTHONPATH": CONTAINER_ROOT})
    .add_local_dir(".", **_LOCAL_DIR_KWARGS)
    .run_commands(f"uv pip install --no-deps -e {CONTAINER_ROOT}")
)

app = modal.App("verirl-env")

verirl_secrets = modal.Secret.from_name("verirl-training")


@app.cls(
    image=env_image,
    cpu=4.0,
    memory=8192,
    min_containers=1,
    secrets=[verirl_secrets],
)
@modal.concurrent(max_inputs=64)
class VerirlEnvServer:
    @modal.asgi_app()
    def serve(self):
        from server.app import app as fastapi_app
        return fastapi_app


# ---------------------------------------------------------------------------
# Smoke test — validates a full write→compile→sim→submit episode
# ---------------------------------------------------------------------------

@app.function(
    image=env_image,
    timeout=300,
    secrets=[verirl_secrets],
)
def smoke_test() -> dict:
    """Validate env connectivity and a complete episode against the deployed server."""
    import os
    import textwrap
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
