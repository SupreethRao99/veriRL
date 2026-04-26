"""VeriRL HF Jobs — submit SFT and RLVR (GRPO) training to HuggingFace Jobs.

Analogous to ``modal_infra.py``: all training logic lives in ``training/`` and
is backend-agnostic. This file contains only HF Jobs-specific glue.

The VeriRL environment server must be reachable from HF Jobs compute.
Point ``VERIRL_ENV_URL`` at your deployed HF Space before running ``train``.

GPU flavors (set via ``--flavor``):
  sft:   ``h200``   (1×H200, 141 GB)   — Unsloth SFT
  train: ``a10g-largex2`` (2×A10G, 48 GB)   — vLLM server mode
         ``h200-large``   (1×H200, 141 GB)   — colocate mode on a single card

Usage
-----
  python hf_jobs.py sft                        # SFT warm-start
  python hf_jobs.py train                      # RLVR GRPO (requires VERIRL_ENV_URL)
  python hf_jobs.py ps                         # list running jobs
  python hf_jobs.py logs <job-id>              # tail job logs
  python hf_jobs.py --dry-run sft              # print hf CLI command without submitting
  python hf_jobs.py train --flavor h200        # override hardware

Prerequisites
-------------
  pip install huggingface_hub hf_xet
  huggingface-cli login
  # Register secrets on HF Hub (one-time):
  #   HF_TOKEN      — write-access token
  #   WANDB_API_KEY — W&B key (optional but recommended)
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


_GITHUB_ORG = "SupreethRao99"
_GITHUB_REPO = "veriRL"
_BRANCH_DEFAULT = os.environ.get("VERIRL_GIT_REF", "feat/working-grpo")

_SFT_FLAVOR_DEFAULT = "a10g-large"
_GRPO_FLAVOR_DEFAULT = "a10g-largex2"
_TIMEOUT_DEFAULT = "8h"


def _raw_url(git_ref: str, path: str) -> str:
    """Build a raw.githubusercontent.com URL for a file at the given git ref.

    Args:
        git_ref: Branch name or commit SHA.
        path: Repo-relative file path (e.g. ``"training/hf_train_sft.py"``).

    Returns:
        Full raw GitHub URL string.
    """
    return f"https://raw.githubusercontent.com/{_GITHUB_ORG}/{_GITHUB_REPO}/{git_ref}/{path}"


def _hf_token_secret_arg() -> str:
    """Read the HF token from env or the cached HF credentials file.

    Returns:
        A ``"HF_TOKEN=<value>"`` string suitable for passing to ``--secrets``.
    """
    token = os.environ.get("HF_TOKEN")
    if not token:
        cache = Path.home() / ".cache" / "huggingface" / "token"
        if cache.exists():
            token = cache.read_text().strip()
    if not token:
        raise RuntimeError(
            "HF_TOKEN not set and ~/.cache/huggingface/token not found. "
            "Run: huggingface-cli login"
        )
    return f"HF_TOKEN={token}"


def _run_hf(*args: str, dry_run: bool = False) -> int:
    """Run an ``hf jobs`` subcommand, optionally in dry-run mode.

    Args:
        *args: Arguments to append after ``hf jobs``.
        dry_run: If ``True``, print the command instead of executing it.

    Returns:
        The subprocess return code (always 0 in dry-run mode).
    """
    cmd = ["hf", "jobs", *args]
    if dry_run:
        redacted = [
            "HF_TOKEN=<redacted>" if arg.startswith("HF_TOKEN=") else arg
            for arg in cmd
        ]
        print("[dry-run]", " ".join(redacted))
        return 0
    return subprocess.run(cmd).returncode


def _submit(
    script_url: str,
    flavor: str,
    timeout: str,
    env: dict[str, str] | None = None,
    dry_run: bool = False,
) -> int:
    """Build and execute an ``hf jobs uv run`` submission command.

    Args:
        script_url: Raw GitHub URL of the script to run on HF Jobs.
        flavor: HF Jobs hardware flavor (e.g. ``"a10g-largex2"``).
        timeout: Job timeout string (e.g. ``"8h"``).
        env: Extra environment variables to pass via ``--env K=V``.
        dry_run: If ``True``, print the command without submitting.

    Returns:
        The ``hf`` CLI return code.
    """
    args = ["uv", "run", "--flavor", flavor, "--timeout", timeout]
    if os.path.exists(".env.secrets"):
        args += ["--secrets-file", ".env.secrets"]
    if os.path.exists(".env"):
        args += ["--env-file", ".env"]
    for k, v in (env or {}).items():
        args += ["--env", f"{k}={v}"]
    args.append(script_url)

    print(f"[hf_jobs] script  : {script_url}")
    print(f"[hf_jobs] flavor  : {flavor}   timeout: {timeout}")
    if os.path.exists(".env.secrets"):
        print("[hf_jobs] secrets-file: .env.secrets")
    if os.path.exists(".env"):
        print("[hf_jobs] env-file: .env")
    for k, v in (env or {}).items():
        print(f"[hf_jobs] env     : {k}={v}")
    return _run_hf(*args, dry_run=dry_run)


def cmd_sft(flavor: str, timeout: str, git_ref: str, dry_run: bool) -> int:
    """Submit the SFT warm-start job to HF Jobs.

    Args:
        flavor: HF Jobs hardware flavor.
        timeout: Job timeout string.
        git_ref: Git branch or commit SHA to run.
        dry_run: If ``True``, print without submitting.

    Returns:
        The ``hf`` CLI return code.
    """
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
    """Submit the RLVR GRPO training job to HF Jobs.

    Requires ``VERIRL_ENV_URL`` to be set in the local environment.

    Args:
        flavor: HF Jobs hardware flavor.
        timeout: Job timeout string.
        git_ref: Git branch or commit SHA to run.
        resume_from_checkpoint: Checkpoint path or ``'latest'`` to resume from,
            or ``None`` to start fresh.
        dry_run: If ``True``, print without submitting.

    Returns:
        The ``hf`` CLI return code, or 1 if ``VERIRL_ENV_URL`` is not set.
    """
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


def cmd_eval(
    flavor: str,
    timeout: str,
    git_ref: str,
    grpo_model: str | None,
    n_runs: int,
    dry_run: bool,
) -> int:
    """Submit the model comparison evaluation job to HF Jobs (1×A10G).

    Runs Base → SFT → GRPO inference on easy tasks and prints a markdown
    score table. Requires ``VERIRL_ENV_URL`` in ``.env`` and HF_TOKEN in
    ``.env.secrets``.

    Args:
        flavor: HF Jobs hardware flavor (default: ``a10g-large``).
        timeout: Job timeout string (e.g. ``"4h"``).
        git_ref: Git branch or commit SHA to run.
        grpo_model: HF repo ID of the GRPO checkpoint, or ``None`` to use default.
        n_runs: Number of episodes per (model, task) pair.
        dry_run: If ``True``, print the hf CLI command without submitting.

    Returns:
        The ``hf`` CLI return code, or 1 if ``VERIRL_ENV_URL`` is not set.
    """
    from dotenv import load_dotenv
    load_dotenv()

    env_url = os.environ.get("VERIRL_ENV_URL", "").strip()
    if not env_url:
        print(
            "ERROR: VERIRL_ENV_URL is not set. Add it to .env or export it.",
            file=sys.stderr,
        )
        return 1

    env: dict[str, str] = {
        "VERIRL_ENV_URL": env_url,
        "VERIRL_GIT_REF": git_ref,
        "EVAL_N_RUNS": str(n_runs),
    }
    if grpo_model:
        env["VERIRL_GRPO_MODEL"] = grpo_model

    return _submit(
        script_url=_raw_url(git_ref, "training/hf_eval_models.py"),
        flavor=flavor,
        timeout=timeout,
        env=env,
        dry_run=dry_run,
    )


def cmd_ps(dry_run: bool) -> int:
    """List all running HF Jobs.

    Args:
        dry_run: If ``True``, print the command without executing.

    Returns:
        The ``hf`` CLI return code.
    """
    return _run_hf("ps", dry_run=dry_run)


def cmd_logs(job_id: str, dry_run: bool) -> int:
    """Tail the logs for a specific HF Job.

    Args:
        job_id: Job ID from ``hf_jobs.py ps``.
        dry_run: If ``True``, print the command without executing.

    Returns:
        The ``hf`` CLI return code.
    """
    return _run_hf("logs", job_id, dry_run=dry_run)


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the ``hf_jobs.py`` CLI."""
    p = argparse.ArgumentParser(
        description="Submit VeriRL training jobs to HuggingFace Jobs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--dry-run", action="store_true", help="Print the hf CLI command without submitting")
    sub = p.add_subparsers(dest="command", required=True)

    sft_p = sub.add_parser("sft", help="SFT warm-start on PyraNet-Verilog")
    sft_p.add_argument("--flavor", default=_SFT_FLAVOR_DEFAULT, help="HF Jobs hardware flavor")
    sft_p.add_argument("--timeout", default=_TIMEOUT_DEFAULT)
    sft_p.add_argument("--git-ref", default=_BRANCH_DEFAULT, help="Git ref to run on HF Jobs")

    train_p = sub.add_parser("train", help="RLVR GRPO training (requires VERIRL_ENV_URL)")
    train_p.add_argument("--flavor", default=_GRPO_FLAVOR_DEFAULT, help="HF Jobs hardware flavor")
    train_p.add_argument("--timeout", default=_TIMEOUT_DEFAULT)
    train_p.add_argument("--git-ref", default=_BRANCH_DEFAULT, help="Git ref to run on HF Jobs")
    train_p.add_argument(
        "--resume-from-checkpoint",
        default=None,
        help="Checkpoint path, 'latest', or Hub subfolder to resume GRPO from",
    )

    sub.add_parser("ps", help="List running HF Jobs")

    logs_p = sub.add_parser("logs", help="Tail logs for a job")
    logs_p.add_argument("job_id", help="Job ID from `hf_jobs.py ps`")

    eval_p = sub.add_parser("eval", help="Base vs SFT vs GRPO comparison on easy tasks (1×L4)")
    eval_p.add_argument("--flavor", default="a10g-large", help="HF Jobs hardware flavor")
    eval_p.add_argument("--timeout", default="4h")
    eval_p.add_argument("--git-ref", default=_BRANCH_DEFAULT)
    eval_p.add_argument("--grpo-model", default=None,
                        help="HF repo ID of the GRPO checkpoint (overrides default)")
    eval_p.add_argument("--n-runs", type=int, default=3,
                        help="Episodes per (model, task) pair (default: 3)")

    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()

    if args.command == "sft":
        sys.exit(cmd_sft(args.flavor, args.timeout, args.git_ref, args.dry_run))
    elif args.command == "train":
        sys.exit(cmd_train(
            args.flavor, args.timeout, args.git_ref,
            args.resume_from_checkpoint, args.dry_run,
        ))
    elif args.command == "eval":
        sys.exit(cmd_eval(
            args.flavor, args.timeout, args.git_ref,
            args.grpo_model, args.n_runs, args.dry_run,
        ))
    elif args.command == "ps":
        sys.exit(cmd_ps(args.dry_run))
    elif args.command == "logs":
        sys.exit(cmd_logs(args.job_id, args.dry_run))
