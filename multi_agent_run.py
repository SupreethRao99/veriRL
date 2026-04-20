"""
Multi-Agent Evolutionary Inference Demo — VeriRL
=================================================
Demonstrates the ShinkaEvolve-inspired evolutionary loop at inference time:

  1. Designer LLM runs N independent episodes on the same task.
     VeriRL scores each design with real EDA tools.

  2. Verifier LLM receives the top design and its EDA score breakdown.
     It identifies likely edge cases and undetected weaknesses.

  3. Mutator / Evolution step: top-K designs + Verifier findings are
     assembled into an evolution prompt. Designer LLM writes an evolved
     design that synthesises the best elements of each attempt.

  4. VeriRL scores the evolved design. We report: avg_baseline → evolved.

This pipeline shows judges what the model will learn to do INTERNALLY
after evolutionary GRPO training — in a single forward pass.

Usage
-----
    python multi_agent_run.py --task relu_clip --candidates 3 --model gpt-4o

Required env vars (same as inference.py):
    OPENAI_API_KEY  or  HF_TOKEN
    API_BASE_URL    (default: https://router.huggingface.co/v1)
    ENV_BASE_URL    (default: http://localhost:8000)

STDOUT format complies with CLAUDE.md:
    [START] task=<task> env=verirl-multiagent model=<model>
    [STEP]  step=<n> action=<action> reward=<r> done=<bool> error=<msg|null>
    [END]   success=<bool> steps=<n> score=<score> rewards=<r1,...>
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import time
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

from verirl_env import VerirlAction, verirl_env

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_KEY      = os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:8000")
BENCHMARK    = "verirl-multiagent"

TASK_BUDGETS: dict[str, int] = {
    "mac_unit": 240, "relu_clip": 180, "barrel_shifter": 180,
    "axi_fifo": 360, "register_file": 300, "ring_buffer": 360,
    "dot_product": 300, "fir_filter": 360,
    "systolic_array": 480, "fp16_adder": 600,
}

# ---------------------------------------------------------------------------
# System prompts for each agent role
# ---------------------------------------------------------------------------

DESIGNER_SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert RTL hardware designer. Implement the given Verilog specification.

    Workflow — always follow this sequence:
      1. write_file   — write the complete, synthesizable Verilog module(s)
      2. run_compile  — check syntax; fix errors and rewrite if needed
      3. run_sim      — run testbench; fix FAIL lines; rewrite + recompile
      4. submit       — when tests pass or turns are nearly exhausted

    Rules:
    - No `initial` blocks in the design (testbench only)
    - Sequential: always @(posedge clk)
    - Combinational: assign / always @(*)
    - Multi-module: separate write_file calls per file

    Respond with exactly ONE JSON action per turn, no markdown:
      {"action_type": "write_file", "filename": "design.v", "verilog_src": "..."}
      {"action_type": "run_compile"}
      {"action_type": "run_sim"}
      {"action_type": "run_synth"}
      {"action_type": "run_formal"}
      {"action_type": "submit"}

    NEVER submit without first running run_compile and run_sim.
""").strip()

EVOLUTION_DESIGNER_SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert RTL hardware designer performing an evolutionary design improvement.

    You will be shown K previous Verilog implementations of the same hardware task,
    each with its EDA evaluation scores (compile, sim, timing, area, formal).

    Your task: synthesise an EVOLVED design that:
      1. Combines the best structural elements from each attempt
      2. Fixes the specific weaknesses revealed by the EDA scores
      3. Outperforms all previous attempts

    Then use the standard workflow:
      write_file → run_compile → run_sim → submit

    Respond with exactly ONE JSON action per turn.
    NEVER submit without running run_compile and run_sim first.
""").strip()

VERIFIER_SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert RTL verification engineer. You will receive:
      - A hardware task specification
      - A Verilog design that partially passes the testbench
      - Its EDA score breakdown (compile, sim, timing, area)

    Your job: identify 2-3 specific edge cases or design weaknesses NOT caught by
    the existing testbench. Focus on:
      - Boundary conditions (e.g. max value, zero, negative)
      - Timing edge cases (e.g. pipeline flush, back-to-back inputs)
      - Corner cases in the spec (e.g. overflow, carry, NaN handling)

    Also write a minimal Verilog testbench snippet that would expose ONE of these
    weaknesses. Put the testbench in a ```verilog ... ``` block.

    Be concise: 3-5 sentences of analysis + 1 testbench snippet.
""").strip()

# ---------------------------------------------------------------------------
# Logging helpers (compliant with CLAUDE.md STDOUT format)
# ---------------------------------------------------------------------------


def _log_start(task: str, model: str) -> None:
    print(f"[START] task={task} env={BENCHMARK} model={model}", flush=True)


def _log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error or 'null'}",
        flush=True,
    )


def _log_end(success: bool, steps: int, rewards: list[float], score: float) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


def _safe_score(raw) -> float:
    return round(min(max(float(raw), 0.01), 0.99), 2)


def _sanitize_error(s: str, max_len: int = 80) -> Optional[str]:
    if not s:
        return None
    return " ".join(s.split())[:max_len] or None


def _parse_action(text: str) -> tuple[VerirlAction, Optional[str]]:
    """Extract a JSON action from LLM response."""
    t = text.strip()
    for fence in ("```json", "```"):
        if fence in t:
            t = t.split(fence, 1)[1].split("```", 1)[0].strip()
            break
    start, end = t.find("{"), t.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            data = json.loads(t[start:end])
            valid = VerirlAction.model_fields
            return VerirlAction(**{k: v for k, v in data.items() if k in valid}), None
        except Exception as exc:
            return VerirlAction(action_type="submit", message="parse error"), _sanitize_error(str(exc), 60)
    return VerirlAction(action_type="submit", message="no json"), "parse_error: no JSON found"


def _action_label(action: VerirlAction) -> str:
    if action.action_type == "write_file" and action.verilog_src:
        return f"write_file({len(action.verilog_src)}chars)"
    return action.action_type


def _format_obs(obs) -> str:
    parts = []
    if obs.task_spec:
        parts.append(f"TASK SPECIFICATION:\n{obs.task_spec}")
    if obs.tool_stdout:
        parts.append(f"TOOL OUTPUT:\n{obs.tool_stdout}")
    if obs.tool_stderr:
        parts.append(f"ERRORS:\n{obs.tool_stderr}")
    if getattr(obs, "current_files", None):
        summary = ", ".join(f"{n}({len(s)}chars)" for n, s in sorted(obs.current_files.items()))
        parts.append(f"Files on disk: {summary}")
    parts.append(
        f"Status: compile={'OK' if obs.compile_ok else 'FAIL'} | "
        f"tests={obs.tests_passed}/{obs.tests_total} | "
        f"turn={obs.turn_number} | remaining={obs.turns_remaining}"
    )
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Designer episode
# ---------------------------------------------------------------------------


def run_designer_episode(
    client: OpenAI,
    model: str,
    task_id: str,
    system_prompt: str,
    extra_context: str = "",
    global_step_offset: int = 0,
    verbose: bool = True,
) -> dict:
    """
    Run one complete Designer episode. Returns:
        {task_id, code, score, score_breakdown, steps, rewards, done}
    """
    budget = TASK_BUDGETS.get(task_id, 300)
    start = time.time()
    env = verirl_env(base_url=ENV_BASE_URL)
    rewards: list[float] = []
    step = global_step_offset
    last_verilog = ""
    final_score = 0.01
    score_breakdown: dict = {}
    done = False

    try:
        result = env.reset(task_id=task_id)
        obs = result.observation

        messages: list[dict] = [{"role": "system", "content": system_prompt}]
        if extra_context:
            messages.append({"role": "user", "content": extra_context})
        messages.append({"role": "user", "content": _format_obs(obs)})

        for _ in range(100):
            if time.time() - start > budget:
                action = VerirlAction(action_type="submit", message="time budget exceeded")
            else:
                try:
                    resp = client.chat.completions.create(
                        model=model, messages=messages, temperature=0.7
                    )
                    text = resp.choices[0].message.content or ""
                    action, err = _parse_action(text)
                    messages.append({"role": "assistant", "content": text})
                except Exception as exc:
                    action = VerirlAction(action_type="submit", message="llm error")
                    err = _sanitize_error(str(exc))

            # Track last written Verilog for the evolution buffer
            if action.action_type == "write_file" and action.verilog_src:
                last_verilog = action.verilog_src

            result = env.step(action)
            obs = result.observation
            reward = _safe_score(getattr(result, "reward", obs.reward if hasattr(obs, "reward") else 0.01))
            rewards.append(reward)
            step += 1
            done = obs.done

            if verbose:
                err_str = _sanitize_error(obs.tool_stderr or "") if not obs.compile_ok else None
                _log_step(step, _action_label(action), reward, done, err_str)

            messages.append({"role": "user", "content": _format_obs(obs)})

            if done:
                final_score = _safe_score(obs.final_score or 0.01)
                score_breakdown = obs.score_breakdown or {}
                break

    finally:
        env.close()

    return {
        "task_id":        task_id,
        "code":           last_verilog,
        "score":          final_score,
        "score_breakdown": score_breakdown,
        "steps":          step - global_step_offset,
        "rewards":        rewards,
        "done":           done,
    }


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------


def run_verifier(
    client: OpenAI,
    model: str,
    task_spec: str,
    top_design: dict,
) -> str:
    """
    Verifier LLM: analyse the top design and identify edge cases.
    Optionally runs the extracted adversarial testbench against the design
    via subprocess iverilog (best-effort; skipped if iverilog not available).

    Returns: a bug_report string to be included in the evolution prompt.
    """
    score = top_design.get("score", 0.0)
    breakdown = top_design.get("score_breakdown") or {}
    bd_str = ", ".join(f"{k}={v:.2f}" for k, v in sorted(breakdown.items())) or "N/A"
    code = top_design.get("code", "")

    user_msg = (
        f"TASK SPECIFICATION:\n{task_spec}\n\n"
        f"DESIGN TO VERIFY (score={score:.3f}, {bd_str}):\n"
        f"```verilog\n{code}\n```"
    )

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": VERIFIER_SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            temperature=0.3,
            max_tokens=600,
        )
        verifier_report = resp.choices[0].message.content or ""
    except Exception as exc:
        return f"Verifier unavailable: {exc}"

    # Best-effort: extract testbench and run via iverilog
    adversarial_result = _try_run_adversarial_tb(code, verifier_report)
    if adversarial_result:
        verifier_report += f"\n\n[Adversarial testbench result]: {adversarial_result}"

    return verifier_report


def _try_run_adversarial_tb(design_code: str, verifier_text: str) -> Optional[str]:
    """Extract a Verilog testbench from verifier output and run it via iverilog."""
    import re
    match = re.search(r"```(?:verilog|systemverilog)?\s*(.*?)```", verifier_text, re.DOTALL | re.IGNORECASE)
    if not match:
        return None
    tb_code = match.group(1).strip()
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            design_path = os.path.join(tmpdir, "design.v")
            tb_path     = os.path.join(tmpdir, "adversarial_tb.v")
            out_path    = os.path.join(tmpdir, "sim.vvp")
            with open(design_path, "w") as f:
                f.write(design_code)
            with open(tb_path, "w") as f:
                f.write(tb_code)
            compile_result = subprocess.run(
                ["iverilog", "-o", out_path, design_path, tb_path],
                capture_output=True, text=True, timeout=15,
            )
            if compile_result.returncode != 0:
                return f"COMPILE ERROR: {' '.join(compile_result.stderr.split()[:20])}"
            sim_result = subprocess.run(
                ["vvp", out_path],
                capture_output=True, text=True, timeout=15,
            )
            output = sim_result.stdout + sim_result.stderr
            # Truncate to 200 chars for STDOUT compliance
            return output.replace("\n", " ").strip()[:200] or "Simulation completed (no output)"
    except FileNotFoundError:
        return None  # iverilog not installed — skip silently
    except Exception as exc:
        return f"adversarial_error: {str(exc)[:80]}"


# ---------------------------------------------------------------------------
# Evolution prompt builder
# ---------------------------------------------------------------------------


def _build_evolution_context(
    top_k_results: list[dict],
    verifier_report: str,
    task_spec: str,
) -> str:
    """Compose the user-facing evolution context block."""
    designs_text = ""
    for i, r in enumerate(top_k_results, 1):
        score = r.get("score", 0.0)
        bd = r.get("score_breakdown") or {}
        bd_str = ", ".join(f"{k}={v:.2f}" for k, v in sorted(bd.items())) or "N/A"
        code = r.get("code", "")
        designs_text += (
            f"\n--- Attempt {i} (score={score:.3f} | {bd_str}) ---\n"
            f"```verilog\n{code}\n```\n"
        )

    # Identify weakest dimension across all attempts
    all_bd: dict[str, list[float]] = {}
    for r in top_k_results:
        for k, v in (r.get("score_breakdown") or {}).items():
            all_bd.setdefault(k, []).append(v)
    if all_bd:
        weakest = min(all_bd, key=lambda k: sum(all_bd[k]) / len(all_bd[k]))
        focus = f"Prioritise improving the '{weakest}' dimension (lowest average score)."
    else:
        focus = "Focus on correctness and synthesis quality."

    return (
        f"TASK SPECIFICATION:\n{task_spec}\n\n"
        f"PREVIOUS DESIGN ATTEMPTS:\n{designs_text}\n"
        f"VERIFIER FINDINGS:\n{verifier_report}\n\n"
        f"{focus}\n\n"
        "Synthesise an EVOLVED Verilog design that incorporates the best structural "
        "elements of each attempt and fixes the weaknesses identified above."
    )


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------


def run_evolutionary_pipeline(
    task_id: str,
    model: str,
    n_candidates: int = 3,
    evolution_top_k: int = 2,
    verbose: bool = True,
) -> dict:
    """
    Full Designer → Verifier → Mutator → Evolution pipeline for one task.

    Returns summary dict with baseline scores, evolved score, and improvement.
    """
    client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

    _log_start(task=task_id, model=model)

    all_rewards: list[float] = []
    global_step = 0

    # ── Step 1: N independent Designer episodes ──────────────────────────────
    print(f"# Phase 1 — Running {n_candidates} independent design attempts", file=sys.stderr)
    candidates: list[dict] = []
    for i in range(n_candidates):
        print(f"# Candidate {i + 1}/{n_candidates} ...", file=sys.stderr)
        result = run_designer_episode(
            client=client,
            model=model,
            task_id=task_id,
            system_prompt=DESIGNER_SYSTEM_PROMPT,
            global_step_offset=global_step,
            verbose=verbose,
        )
        candidates.append(result)
        all_rewards.extend(result["rewards"])
        global_step += result["steps"]

    candidates.sort(key=lambda r: -r["score"])
    baseline_scores = [c["score"] for c in candidates]
    avg_baseline = sum(baseline_scores) / len(baseline_scores)
    top_score = baseline_scores[0]
    print(
        f"# Phase 1 done — scores: {[f'{s:.3f}' for s in baseline_scores]} "
        f"avg={avg_baseline:.3f} top={top_score:.3f}",
        file=sys.stderr,
    )

    # ── Step 2: Verifier analyses top design ─────────────────────────────────
    print("# Phase 2 — Verifier analysing top design ...", file=sys.stderr)
    # Retrieve task spec from the first candidate's reset (we just need the spec text)
    env_tmp = verirl_env(base_url=ENV_BASE_URL)
    try:
        spec_result = env_tmp.reset(task_id=task_id)
        task_spec = spec_result.observation.task_spec or f"Implement the {task_id} module."
    finally:
        env_tmp.close()

    verifier_report = run_verifier(
        client=client,
        model=model,
        task_spec=task_spec,
        top_design=candidates[0],
    )
    print(f"# Verifier report ({len(verifier_report)} chars)", file=sys.stderr)

    # ── Step 3: Evolution episode ─────────────────────────────────────────────
    print("# Phase 3 — Evolution episode (synthesising improved design) ...", file=sys.stderr)
    top_k = candidates[:evolution_top_k]
    evolution_context = _build_evolution_context(top_k, verifier_report, task_spec)

    evolved = run_designer_episode(
        client=client,
        model=model,
        task_id=task_id,
        system_prompt=EVOLUTION_DESIGNER_SYSTEM_PROMPT,
        extra_context=evolution_context,
        global_step_offset=global_step,
        verbose=verbose,
    )
    all_rewards.extend(evolved["rewards"])
    global_step += evolved["steps"]
    evolved_score = evolved["score"]

    improvement = evolved_score - avg_baseline
    success = evolved_score > avg_baseline

    print(
        f"# Evolution complete — avg_baseline={avg_baseline:.3f} "
        f"evolved={evolved_score:.3f} improvement={improvement:+.3f}",
        file=sys.stderr,
    )

    _log_end(
        success=success,
        steps=global_step,
        rewards=all_rewards,
        score=evolved_score,
    )

    return {
        "task_id":        task_id,
        "baseline_scores": baseline_scores,
        "avg_baseline":   avg_baseline,
        "evolved_score":  evolved_score,
        "improvement":    improvement,
        "success":        success,
        "total_steps":    global_step,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="VeriRL Multi-Agent Evolutionary Inference Demo"
    )
    parser.add_argument(
        "--task", required=True,
        choices=list(TASK_BUDGETS.keys()),
        help="Task ID to run",
    )
    parser.add_argument(
        "--candidates", type=int, default=3,
        help="Number of independent Designer episodes (default: 3)",
    )
    parser.add_argument(
        "--top-k", type=int, default=2,
        help="Top-K candidates used for evolution prompt (default: 2)",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("MODEL_NAME", "gpt-4o-mini"),
        help="LLM model name (default: gpt-4o-mini or $MODEL_NAME)",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-step [STEP] lines (only print [START] and [END])",
    )
    args = parser.parse_args()

    summary = run_evolutionary_pipeline(
        task_id=args.task,
        model=args.model,
        n_candidates=args.candidates,
        evolution_top_k=args.top_k,
        verbose=not args.quiet,
    )

    print(
        f"\n# Summary: {summary['task_id']} | "
        f"baseline_avg={summary['avg_baseline']:.3f} | "
        f"evolved={summary['evolved_score']:.3f} | "
        f"improvement={summary['improvement']:+.3f}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
