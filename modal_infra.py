"""
VeriRL Modal infrastructure — SFT warm-start and RLVR training.

All Modal-specific code lives here. Training logic in training/ has zero Modal
imports and runs identically locally or on Modal.

The VeriRL environment server runs as a separate Modal deployment (modal_env.py).
Set VERIRL_ENV_URL in the "verirl-training" secret to point at that deployment.

GPU strategy (config.yaml modal.gpu_count):
  1 GPU  → colocate mode: vLLM and training share the GPU (max_model_length capped at 8192)
  2 GPUs → server mode:   vLLM on GPU 1 (full 22 GB KV cache), training on GPU 0

Usage
-----
  modal run modal_infra.py::sft    # SFT warm-start on PyraNet-Verilog
  modal run modal_infra.py::train  # RLVR (GRPO + vLLM) from SFT checkpoint
"""

from pathlib import Path

import modal
from omegaconf import OmegaConf

# ---------------------------------------------------------------------------
# Container layout (defined first — used by import-time config read below)
# ---------------------------------------------------------------------------

CONTAINER_ROOT = "/root/verirl"

# ---------------------------------------------------------------------------
# Read Modal config at import time so the function decorator is correct.
# Use CONTAINER_ROOT when running inside Modal; fall back to __file__ locally.
# ---------------------------------------------------------------------------

_cfg_candidates = [
    Path(CONTAINER_ROOT) / "config.yaml",
    Path(__file__).parent / "config.yaml",
]
_cfg_path = next(p for p in _cfg_candidates if p.exists())
_modal_cfg = OmegaConf.load(_cfg_path).modal
_GPU_COUNT = int(_modal_cfg.get("gpu_count", 1))
_GPU_SPEC = (
    f"{_modal_cfg.gpu_type}:{_GPU_COUNT}"
    if _GPU_COUNT > 1
    else str(_modal_cfg.gpu_type)
)
CHECKPOINTS_DIR = f"{CONTAINER_ROOT}/checkpoints"
HF_CACHE_DIR = "/root/.cache/huggingface"

_LOCAL_DIR_KWARGS = dict(
    remote_path=CONTAINER_ROOT,
    copy=True,
    ignore=[".git", "__pycache__", ".venv", ".pytest_cache", "*.pyc", "*.egg-info"],
)

# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------

# SFT image: Unsloth-based, TRL 0.x — no vLLM
sft_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync(extras=["sft"])
    .env(
        {
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONPATH": CONTAINER_ROOT,
        }
    )
    .add_local_dir(".", **_LOCAL_DIR_KWARGS)
    .run_commands(f"uv pip install --no-deps -e {CONTAINER_ROOT}")
)

# RLVR image: TRL 1.x + vLLM — separate from SFT because Unsloth pins TRL 0.x
rlvr_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync(extras=["grpo"])
    .env(
        {
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONPATH": CONTAINER_ROOT,
        }
    )
    .add_local_dir(".", **_LOCAL_DIR_KWARGS)
    .run_commands(f"uv pip install --no-deps -e {CONTAINER_ROOT}")
)

# ---------------------------------------------------------------------------
# Volumes & secrets
# ---------------------------------------------------------------------------

checkpoints_vol = modal.Volume.from_name(
    "verirl-rlvr-checkpoints", create_if_missing=True
)
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)

verirl_secrets = modal.Secret.from_name("verirl-training")
hf_secret = modal.Secret.from_name("huggingface-secret")
wandb_secret = modal.Secret.from_name("wandb-secret")

_SECRETS = [verirl_secrets, hf_secret, wandb_secret]
_VOLUMES = {CHECKPOINTS_DIR: checkpoints_vol, HF_CACHE_DIR: hf_cache_vol}

app = modal.App("verirl-rlvr")

# ---------------------------------------------------------------------------
# SFT warm-start
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
    """SFT warm-start: fine-tune Qwen3-4B-Thinking on PyraNet-Verilog (692K Verilog samples)."""
    from training.config import SFTConfig
    from training.sft import run_sft
    from training.trainer import setup_auth

    hf_token, wandb_key = setup_auth()
    config = SFTConfig.from_yaml()
    return run_sft(config, hf_token, wandb_key, output_dir=f"{CHECKPOINTS_DIR}/sft")


# ---------------------------------------------------------------------------
# RLVR training (starts from SFT checkpoint)
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
    """RLVR: QLoRA GRPO with vLLM.

    GPU strategy (config.yaml modal.gpu_count):
      1 GPU  → colocate mode, max_model_length capped at 8192
      2 GPUs → server mode, vLLM on GPU 1 with full 22 GB, training on GPU 0
    """
    import os
    import subprocess
    import sys
    import time
    from pathlib import Path

    import requests
    from omegaconf import OmegaConf

    from training.config import TrainConfig
    from training.trainer import build_grpo_config, run_training, setup_auth

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

    # Warm up the env server before touching the GPU
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

    gpu_count = int(
        OmegaConf.load(Path(CONTAINER_ROOT) / "config.yaml").modal.get("gpu_count", 1)
    )
    print(
        f"[VeriRL] GPU count: {gpu_count} → mode: {'server' if gpu_count >= 2 else 'colocate'}"
    )

    hf_token, wandb_key = setup_auth()
    config = TrainConfig.from_yaml(env_url=env_url)

    vllm_proc = None

    if gpu_count >= 2:
        # ── Server mode ──────────────────────────────────────────────────────
        # Start vLLM server on GPU 1 BEFORE initialising CUDA on this process,
        # then restrict this process to GPU 0 via CUDA_VISIBLE_DEVICES.
        vllm_port = 8001
        print(f"[VeriRL] Starting vLLM server on GPU 1, port {vllm_port} ...")

        # trl CLI lives alongside the venv Python — derive path rather than relying on PATH.
        trl_bin = str(Path(sys.executable).parent / "trl")
        trl_ver = subprocess.run(
            [sys.executable, "-c", "import trl; print(trl.__version__)"],
            capture_output=True,
            text=True,
        )
        print(f"[VeriRL] trl binary: {trl_bin}  version: {trl_ver.stdout.strip()}")

        # Strip torch-distributed env vars so vLLM's own dist.init_process_group
        # doesn't collide with the training TCPStore at MASTER_PORT.
        _DIST_KEYS = {
            "RANK",
            "LOCAL_RANK",
            "WORLD_SIZE",
            "MASTER_ADDR",
            "MASTER_PORT",
            "TORCHELASTIC_RESTART_COUNT",
            "TORCHELASTIC_MAX_RESTARTS",
        }
        vllm_env = {k: v for k, v in os.environ.items() if k not in _DIST_KEYS}
        vllm_env.update(
            {
                "CUDA_VISIBLE_DEVICES": "1",  # vLLM sees only GPU 1; training process gets GPU 0
                "PYTHONUNBUFFERED": "1",
            }
        )

        vllm_log = open("/tmp/vllm_server.log", "w")
        vllm_proc = subprocess.Popen(
            [
                trl_bin,
                "vllm-serve",
                "--model",
                config.vllm_base_model,
                "--port",
                str(vllm_port),
                "--gpu-memory-utilization",
                "0.9",
                "--max-model-len",
                str(config.max_model_length),
            ],
            env=vllm_env,
            stdout=vllm_log,
            stderr=subprocess.STDOUT,
        )
        for i in range(180):  # 360s — first run downloads ~8 GB model
            if vllm_proc.poll() is not None:
                vllm_log.flush()
                tail = open("/tmp/vllm_server.log").read()[-3000:]
                raise RuntimeError(
                    f"vLLM server exited early (code {vllm_proc.returncode}):\n{tail}"
                )
            try:
                if (
                    requests.get(
                        f"http://localhost:{vllm_port}/health", timeout=2
                    ).status_code
                    == 200
                ):
                    print("[VeriRL] vLLM server ready.")
                    break
            except Exception:
                pass
            if i % 30 == 29:
                vllm_log.flush()
                print(f"[VeriRL] vLLM still starting ({(i + 1) * 2}s)... last log:")
                print(open("/tmp/vllm_server.log").read()[-500:])
            time.sleep(2)
        else:
            vllm_proc.kill()
            tail = open("/tmp/vllm_server.log").read()[-3000:]
            raise RuntimeError(f"vLLM server failed to start within 360s. Log:\n{tail}")

        # Restrict this process to GPU 0 before any CUDA context is opened.
        # vLLM owns GPU 1; both processes must see disjoint device sets or
        # NCCL communicator init will conflict (TRL enforces this from v0.19.2+).
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
        # ── Colocate mode ────────────────────────────────────────────────────
        # vLLM and training share the single L4. Cap context to avoid OOM.
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
        grpo_config = build_grpo_config(
            config,
            hf_token,
            wandb_key,
            output_dir=CHECKPOINTS_DIR,
            **vllm_kwargs,
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
    """Run test_sft.py against the pushed SFT checkpoint on a L4 GPU."""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, f"{CONTAINER_ROOT}/test_sft.py", "--task", task],
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
# Local entry point — `modal run modal_infra.py` kicks off SFT first.
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def main():
    sft.remote()
