# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch>=2.3.1",
#   "transformers>=4.45.0",
#   "peft>=0.12.0",
#   "accelerate>=0.33.0",
#   "bitsandbytes>=0.43.3",
#   "openai>=1.0.0",
#   "openenv-core[core]>=0.2.2",
#   "huggingface_hub>=0.24.0",
#   "requests",
#   "python-dotenv",
#   "vllm>=0.11.0,<=0.18.0; sys_platform == 'linux'",
# ]
# ///
"""Model comparison evaluation — Base vs SFT vs GRPO on easy VeriRL tasks.

Runs on HuggingFace Jobs (1×A10G, 24 GB VRAM / 46 GB RAM). For each model:
  - Base: loaded as bitsandbytes-4bit via vLLM (--load-format bitsandbytes)
  - SFT:  loaded as bf16 via vLLM (standard)
  - GRPO: LoRA adapter merged into SFT-merged base on CPU, saved to /tmp,
          then loaded as bf16 via vLLM

N_RUNS episodes are run per (model, task). Final scores are printed as a
markdown table to stdout; all progress goes to stderr.

Required env vars:
  VERIRL_ENV_URL   URL of the deployed VeriRL environment server
  HF_TOKEN         HuggingFace token with read access to private repos

Optional:
  EVAL_N_RUNS      Episodes per (model, task) — default 3
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from statistics import mean
from typing import Optional

# ---------------------------------------------------------------------------
# Bootstrap: clone repo so openenv-verirl_env resolves correctly.
# ---------------------------------------------------------------------------

_REPO = "https://github.com/SupreethRao99/veriRL.git"
_WORKDIR = "/tmp/verirl_env"
_GIT_REF = os.environ.get("VERIRL_GIT_REF", "feat/working-grpo")

if not os.path.exists(_WORKDIR):
    subprocess.run(
        ["git", "clone", "--depth=1", "--branch", _GIT_REF, _REPO, _WORKDIR],
        check=True,
    )

sys.path.insert(0, _WORKDIR)
os.chdir(_WORKDIR)

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import requests  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from openai import OpenAI  # noqa: E402

from verirl_env import VerirlAction, verirl_env  # noqa: E402

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

VLLM_PORT = 8001
VLLM_BASE_URL = f"http://localhost:{VLLM_PORT}/v1"

HF_TOKEN = os.environ.get("HF_TOKEN", "")
ENV_URL = os.environ["VERIRL_ENV_URL"]
N_RUNS = int(os.environ.get("EVAL_N_RUNS", "3"))
GRPO_MERGED_DIR = "/tmp/verirl_grpo_merged"

# Model definitions. Each entry:
#   (display_label, hf_model_id, served_name, extra_vllm_flags)
# served_name is what the OpenAI client sends as model= in chat requests.
# extra_vllm_flags are appended to the vllm serve command for that model only.
_MODELS_RAW: list[tuple[str, str, list[str]]] = [
    (
        "Base (Qwen3-4B-Thinking)",
        "unsloth/Qwen3-4B-Thinking-2507-unsloth-bnb-4bit",
        ["--quantization", "bitsandbytes", "--load-format", "bitsandbytes"],
    ),
    (
        "+ SFT",
        "Supreeth/verirl-sft-qwen3-4b-thinking-merged",
        [],
    ),
    (
        "+ GRPO",
        GRPO_MERGED_DIR,  # populated after merge step below
        [],
    ),
]

EASY_TASKS = ["mac_unit", "relu_clip", "barrel_shifter"]
TASK_BUDGETS_SEC = {"mac_unit": 240, "relu_clip": 180, "barrel_shifter": 180}

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an expert RTL hardware designer. Implement the given Verilog specification correctly.

    REQUIRED WORKFLOW — follow this sequence every episode:
      1. write_file   — write a complete, synthesizable Verilog module.
      2. run_compile  — check for syntax errors; fix with write_file if needed.
      3. run_sim      — run the testbench; fix failures with write_file.
      4. submit       — only after attempting compile and sim.

    NEVER submit without first running run_compile and run_sim.

    Available actions — respond with exactly one JSON object, no markdown:
      {"action_type": "write_file", "filename": "design.v", "verilog_src": "<full module>", "message": "..."}
      {"action_type": "run_compile", "message": "checking syntax"}
      {"action_type": "run_sim",     "message": "running testbench"}
      {"action_type": "run_synth",   "message": "checking area"}
      {"action_type": "run_formal",  "message": "checking formal properties"}
      {"action_type": "submit",      "message": "final submission"}

    Design rules:
    - No `initial` blocks in the design module (testbench only).
    - Use always @(posedge clk) for sequential logic.
    - Fully combinational modules: use assign or always @(*).
    - Pay close attention to pipeline depth and timing requirements.
    """
).strip()


# ---------------------------------------------------------------------------
# GRPO merge step
# ---------------------------------------------------------------------------


def merge_grpo_adapter() -> None:
    """Merge the GRPO LoRA adapter into the SFT-merged base and save to disk.

    Runs on CPU so the GPU is free for vLLM afterwards. The merged weights
    are saved to GRPO_MERGED_DIR in bf16 and reused across restarts.
    """
    if Path(GRPO_MERGED_DIR).exists():
        print(f"[eval] GRPO merged model already at {GRPO_MERGED_DIR}, skipping merge.",
              file=sys.stderr, flush=True)
        return

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    sft_base = "Supreeth/verirl-sft-qwen3-4b-thinking-merged"
    adapter  = "Supreeth/verirl-rlvr-qwen3-4b-thinking"

    print(f"[eval] Merging GRPO adapter {adapter} into {sft_base} ...", file=sys.stderr, flush=True)
    print("[eval] Loading base on CPU (bf16) ...", file=sys.stderr, flush=True)

    hf_kwargs = {"token": HF_TOKEN} if HF_TOKEN else {}

    base_model = AutoModelForCausalLM.from_pretrained(
        sft_base,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        **hf_kwargs,
    )
    tokenizer = AutoTokenizer.from_pretrained(sft_base, **hf_kwargs)

    print("[eval] Applying LoRA adapter ...", file=sys.stderr, flush=True)
    peft_model = PeftModel.from_pretrained(base_model, adapter, **hf_kwargs)

    print("[eval] Merging and unloading ...", file=sys.stderr, flush=True)
    merged = peft_model.merge_and_unload()

    print(f"[eval] Saving merged model to {GRPO_MERGED_DIR} ...", file=sys.stderr, flush=True)
    merged.save_pretrained(GRPO_MERGED_DIR, safe_serialization=True)
    tokenizer.save_pretrained(GRPO_MERGED_DIR)

    # Free RAM before vLLM starts
    del merged, peft_model, base_model
    import gc
    gc.collect()
    print("[eval] GRPO merge complete.", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# vLLM server management
# ---------------------------------------------------------------------------


def _wait_for_vllm(timeout: int = 180) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"http://localhost:{VLLM_PORT}/health", timeout=3)
            if r.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(4)
    raise RuntimeError(f"vLLM server did not become healthy within {timeout}s")


def start_vllm(model_id: str, extra_flags: list[str]) -> tuple[subprocess.Popen, str]:
    """Start a vLLM OpenAI-compatible server.

    Returns (process, served_model_name). served_model_name is the identifier
    the OpenAI client must pass as model= in chat requests.
    """
    # vLLM uses the last path component as the served model name when loading
    # from a local directory, otherwise the full repo ID.
    served_name = Path(model_id).name if Path(model_id).exists() else model_id

    print(f"[eval] Starting vLLM: {model_id}", file=sys.stderr, flush=True)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0"}
    if HF_TOKEN:
        env["HF_TOKEN"] = HF_TOKEN
        env["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_id,
        "--served-model-name", served_name,
        "--port", str(VLLM_PORT),
        "--dtype", "bfloat16",
        "--max-model-len", "4096",
        "--gpu-memory-utilization", "0.88",
        "--trust-remote-code",
        "--enable-prefix-caching",
        *extra_flags,
    ]

    proc = subprocess.Popen(cmd, env=env, stdout=sys.stderr, stderr=sys.stderr)
    _wait_for_vllm()
    print(f"[eval] vLLM ready — served as '{served_name}'", file=sys.stderr, flush=True)
    return proc, served_name


def stop_vllm(proc: subprocess.Popen) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
    time.sleep(5)  # let the port release


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------


def _fmt_obs(obs) -> str:
    parts = []
    if obs.task_spec:
        parts.append(f"TASK SPECIFICATION:\n{obs.task_spec}")
    if obs.tool_stdout:
        parts.append(f"TOOL OUTPUT:\n{obs.tool_stdout}")
    if obs.tool_stderr:
        parts.append(f"ERRORS:\n{obs.tool_stderr}")
    parts.append(
        f"Status: compile={'OK' if obs.compile_ok else 'FAIL'} | "
        f"tests={obs.tests_passed}/{obs.tests_total} | "
        f"turn={obs.turn_number} | remaining={obs.turns_remaining}"
    )
    return "\n\n".join(parts)


def _parse_action(text: str) -> VerirlAction:
    t = text.strip()
    for fence in ("```json", "```"):
        if fence in t:
            t = t.split(fence)[1].split("```")[0].strip()
            break
    s, e = t.find("{"), t.rfind("}") + 1
    if s >= 0 and e > s:
        try:
            data = json.loads(t[s:e])
            valid = VerirlAction.model_fields
            return VerirlAction(**{k: v for k, v in data.items() if k in valid})
        except Exception:
            pass
    return VerirlAction(action_type="submit", message="parse error")


def _clamp(v: float) -> float:
    return round(min(max(float(v), 0.01), 0.99), 4)


async def run_episode(task_id: str, served_model: str, llm: OpenAI) -> float:
    """Run one episode. Returns final_score ∈ [0.01, 0.99]."""
    budget = TASK_BUDGETS_SEC[task_id]
    t0 = time.time()
    final_score = 0.01
    obs = None

    env = verirl_env(base_url=ENV_URL)
    try:
        result = await env.reset(task_id=task_id)
        obs = result.observation
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _fmt_obs(obs)},
        ]

        for step in range(1, 50):
            if time.time() - t0 > budget:
                action = VerirlAction(action_type="submit", message="time budget exceeded")
            else:
                try:
                    resp = llm.chat.completions.create(
                        model=served_model,
                        messages=messages,
                        max_tokens=512,
                        # Disable Qwen3 thinking mode — we need clean JSON output.
                        # Without this, <think> blocks break JSON extraction and long
                        # reasoning causes WebSocket keepalive timeouts on the HF Space.
                        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                    )
                    text = resp.choices[0].message.content or ""
                    action = _parse_action(text)
                    messages.append({"role": "assistant", "content": text})
                except Exception as exc:
                    print(f"[eval]     LLM error step {step}: {exc}", file=sys.stderr)
                    action = VerirlAction(action_type="submit", message="llm error")

            try:
                result = await env.step(action)
            except Exception as exc:
                print(f"[eval]     env.step error: {exc}", file=sys.stderr)
                break

            obs = result.observation
            messages.append({"role": "user", "content": _fmt_obs(obs)})
            print(
                f"[eval]     step={step} {action.action_type} "
                f"compile={'OK' if obs.compile_ok else 'FAIL'} "
                f"sim={obs.tests_passed}/{obs.tests_total}",
                file=sys.stderr, flush=True,
            )

            if result.done:
                final_score = _clamp(obs.final_score or 0.01)
                break

    finally:
        if final_score == 0.01 and obs is not None and obs.current_verilog:
            try:
                r = await env.step(VerirlAction(action_type="submit", message="safety submit"))
                final_score = _clamp(r.observation.final_score or 0.01)
            except Exception:
                pass
        await env.close()

    return final_score


async def evaluate_model(label: str, served_model: str) -> dict[str, list[float]]:
    """Run N_RUNS episodes per easy task. Returns task_id → [scores]."""
    llm = OpenAI(base_url=VLLM_BASE_URL, api_key="EMPTY")
    scores: dict[str, list[float]] = {t: [] for t in EASY_TASKS}

    for task_id in EASY_TASKS:
        for run in range(N_RUNS):
            print(
                f"[eval] [{label}] {task_id} run {run + 1}/{N_RUNS}",
                file=sys.stderr, flush=True,
            )
            score = await run_episode(task_id, served_model, llm)
            scores[task_id].append(score)
            print(
                f"[eval] [{label}] {task_id} run {run + 1} → {score:.4f}",
                file=sys.stderr, flush=True,
            )

    return scores


# ---------------------------------------------------------------------------
# Markdown table
# ---------------------------------------------------------------------------


def print_table(all_scores: dict[str, dict[str, list[float]]]) -> None:
    labels = [label for label, _, _ in _MODELS_RAW]
    task_labels = {
        "mac_unit":       "MAC unit (easy)",
        "relu_clip":      "ReLU-Clip (easy)",
        "barrel_shifter": "Barrel Shifter (easy)",
    }

    print("| Task | " + " | ".join(labels) + " |")
    print("|------|" + "|".join(["------"] * len(labels)) + "|")

    all_means: dict[str, list[float]] = {lb: [] for lb in labels}

    for task_id, task_label in task_labels.items():
        row = f"| {task_label} |"
        for lb in labels:
            s = all_scores.get(lb, {}).get(task_id, [])
            m = mean(s) if s else 0.0
            all_means[lb].append(m)
            row += f" {m:.3f} |"
        print(row)

    mean_row = "| **Mean** |"
    for lb in labels:
        m = mean(all_means[lb]) if all_means[lb] else 0.0
        mean_row += f" **{m:.3f}** |"
    print(mean_row)

    print()
    print(f"*Scores: weighted EDA-tool score ∈ [0.01, 0.99]. Each cell = mean of {N_RUNS} runs.*")

    print("\n<details>")
    print("<summary>Raw per-run scores</summary>\n")
    print("```json")
    print(json.dumps(
        {lb: {t: s for t, s in ts.items()} for lb, ts in all_scores.items()},
        indent=2,
    ))
    print("```\n</details>")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    print(
        f"[eval] VeriRL model comparison | "
        f"{len(_MODELS_RAW)} models × {len(EASY_TASKS)} tasks × {N_RUNS} runs",
        file=sys.stderr, flush=True,
    )
    print(f"[eval] ENV_URL = {ENV_URL}", file=sys.stderr, flush=True)

    # Merge GRPO adapter before any vLLM process starts (needs free RAM)
    merge_grpo_adapter()

    all_scores: dict[str, dict[str, list[float]]] = {}

    for label, model_id, extra_flags in _MODELS_RAW:
        print(f"\n[eval] ══════ {label} ({model_id}) ══════", file=sys.stderr, flush=True)
        proc, served_name = start_vllm(model_id, extra_flags)
        try:
            all_scores[label] = await evaluate_model(label, served_name)
        finally:
            stop_vllm(proc)

    print("\n## VeriRL Evaluation Results\n")
    print_table(all_scores)


if __name__ == "__main__":
    asyncio.run(main())
