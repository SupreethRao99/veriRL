"""
VeriRL HF Jobs — submit SFT warm-start and RLVR (GRPO) training to Hugging Face Jobs.

Analogous to modal_infra.py: all training logic lives in training/ and is
backend-agnostic. This file is HF Jobs-specific glue only.

The VeriRL environment server must be reachable from HF Jobs compute.
Point VERIRL_ENV_URL at your deployed HF Space before running `train`.

GPU strategy (set via --flavor):
  sft:   a10g-large  (1×A10G, 24 GB)      — Unsloth SFT
  train: a10g-largex2 (2×A10G, 48 GB)    — vLLM server mode (GPU 1) + training (GPU 0)
         or a100-large (1×A100, 80 GB)    — colocate mode on a single big card

Usage
-----
  python hf_jobs.py sft                        # SFT warm-start
  python hf_jobs.py train                      # RLVR GRPO (requires VERIRL_ENV_URL)
  python hf_jobs.py ps                         # list running jobs
  python hf_jobs.py logs <job-id>              # tail job logs
  python hf_jobs.py --dry-run sft              # print hf CLI command without submitting
  python hf_jobs.py train --flavor a100-large  # override hardware

Prerequisites
-------------
  pip install huggingface_hub hf_xet            # or: uv add huggingface_hub
  huggingface-cli login                          # authenticate
  # Register secrets on HF Hub (one-time):
  #   HF_TOKEN      — write-access token
  #   WANDB_API_KEY — W&B key (optional but recommended)
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_GITHUB_ORG = "SupreethRao99"
_GITHUB_REPO = "veriRL"
_BRANCH_DEFAULT = os.environ.get("VERIRL_GIT_REF", "feat/working-grpo")

_SFT_FLAVOR_DEFAULT = "a10g-large"
_GRPO_FLAVOR_DEFAULT = "a10g-largex2"
_TIMEOUT_DEFAULT = "8h"

# Local .env values are forwarded via --env-file when present. If .env is not
# present, fall back to HF Jobs' token forwarding for HF_TOKEN.
_SECRETS = ["HF_TOKEN"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _raw_url(git_ref: str, path: str) -> str:
    return f"https://raw.githubusercontent.com/{_GITHUB_ORG}/{_GITHUB_REPO}/{git_ref}/{path}"


def _run_hf(*args: str, dry_run: bool = False) -> int:
    cmd = ["hf", "jobs", *args]
    if dry_run:
        print("[dry-run]", " ".join(cmd))
        return 0
    return subprocess.run(cmd).returncode


def _submit(
    script_url: str,
    flavor: str,
    timeout: str,
    env: dict[str, str] | None = None,
    dry_run: bool = False,
) -> int:
    args = ["uv", "run", "--flavor", flavor, "--timeout", timeout]
    has_env_file = os.path.exists(".env")
    if has_env_file:
        args += ["--env-file", ".env"]
    else:
        for secret in _SECRETS:
            args += ["--secrets", secret]
    for k, v in (env or {}).items():
        args += ["--env", f"{k}={v}"]
    args.append(script_url)

    print(f"[hf_jobs] script  : {script_url}")
    print(f"[hf_jobs] flavor  : {flavor}   timeout: {timeout}")
    if has_env_file:
        print("[hf_jobs] env-file: .env")
    else:
        print("[hf_jobs] secret  : HF_TOKEN=<provided by hf CLI>")
    if env:
        for k, v in env.items():
            print(f"[hf_jobs] env     : {k}={v}")
    return _run_hf(*args, dry_run=dry_run)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_sft(flavor: str, timeout: str, git_ref: str, dry_run: bool) -> int:
    return _submit(
        script_url=_raw_url(git_ref, "training/hf_train_sft.py"),
        flavor=flavor,
        timeout=timeout,
        env={"VERIRL_GIT_REF": git_ref},
        dry_run=dry_run,
    )


def cmd_train(
    flavor: str,
    timeout: str,
    git_ref: str,
    resume_from_checkpoint: str | None,
    dry_run: bool,
) -> int:
    env_url = os.environ.get("VERIRL_ENV_URL", "").strip()
    if not env_url:
        print(
            "ERROR: VERIRL_ENV_URL is not set.\n"
            "Set it to your deployed VeriRL env server, e.g.:\n"
            "  export VERIRL_ENV_URL=https://<username>-verirl-env.hf.space",
            file=sys.stderr,
        )
        return 1
    env = {"VERIRL_ENV_URL": env_url, "VERIRL_GIT_REF": git_ref}
    if resume_from_checkpoint:
        env["VERIRL_RESUME_FROM_CHECKPOINT"] = resume_from_checkpoint
    return _submit(
        script_url=_raw_url(git_ref, "training/hf_train_grpo.py"),
        flavor=flavor,
        timeout=timeout,
        env=env,
        dry_run=dry_run,
    )


def cmd_ps(dry_run: bool) -> int:
    return _run_hf("ps", dry_run=dry_run)


def cmd_logs(job_id: str, dry_run: bool) -> int:
    return _run_hf("logs", job_id, dry_run=dry_run)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Submit VeriRL training jobs to Hugging Face Jobs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the hf CLI command without submitting",
    )
    sub = p.add_subparsers(dest="command", required=True)

    sft_p = sub.add_parser("sft", help="SFT warm-start on PyraNet-Verilog")
    sft_p.add_argument("--flavor", default=_SFT_FLAVOR_DEFAULT, help="HF Jobs hardware flavor")
    sft_p.add_argument("--timeout", default=_TIMEOUT_DEFAULT)
    sft_p.add_argument("--git-ref", default=_BRANCH_DEFAULT, help="Git ref to run on HF Jobs")

    train_p = sub.add_parser(
        "train", help="RLVR GRPO training (requires VERIRL_ENV_URL env var)"
    )
    train_p.add_argument("--flavor", default=_GRPO_FLAVOR_DEFAULT, help="HF Jobs hardware flavor")
    train_p.add_argument("--timeout", default=_TIMEOUT_DEFAULT)
    train_p.add_argument("--git-ref", default=_BRANCH_DEFAULT, help="Git ref to run on HF Jobs")
    train_p.add_argument(
        "--resume-from-checkpoint",
        default=None,
        help="Checkpoint path, `latest`, or Hub subfolder to pass to GRPO resume",
    )

    sub.add_parser("ps", help="List running HF Jobs")

    logs_p = sub.add_parser("logs", help="Tail logs for a job")
    logs_p.add_argument("job_id", help="Job ID from `hf_jobs.py ps`")

    return p


if __name__ == "__main__":
    args = _parser().parse_args()

    if args.command == "sft":
        sys.exit(cmd_sft(args.flavor, args.timeout, args.git_ref, args.dry_run))
    elif args.command == "train":
        sys.exit(cmd_train(args.flavor, args.timeout, args.git_ref, args.resume_from_checkpoint, args.dry_run))
    elif args.command == "ps":
        sys.exit(cmd_ps(args.dry_run))
    elif args.command == "logs":
        sys.exit(cmd_logs(args.job_id, args.dry_run))
