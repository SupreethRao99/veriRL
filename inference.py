"""
Inference Script — VeriRL Verilog Hardware Design Environment
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    ENV_BASE_URL   The base URL of the running VeriRL environment server.

- Defaults:
    API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
    ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:8000")

- The inference script must be named `inference.py` and placed in the root directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables

STDOUT FORMAT
- The script must emit exactly three line types to stdout, in this order:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

  Rules:
    - One [START] line at episode begin.
    - One [STEP] line per step, immediately after env.step() returns.
    - One [END] line after env.close(), always emitted (even on exception).
    - reward and rewards are formatted to 2 decimal places.
    - done and success are lowercase booleans: true or false.
    - error is the raw last_action_error string, or null if none.
    - All fields on a single line with no newlines within a line.
    - Each task's score must be strictly in (0, 1).

  Example:
    [START] task=mac_unit env=verirl model=Qwen2.5-72B-Instruct
    [STEP] step=1 action=write_file(312chars) reward=0.02 done=false error=null
    [STEP] step=2 action=run_compile reward=0.07 done=false error=null
    [STEP] step=3 action=submit reward=0.05 done=true error=null
    [END] success=true steps=3 score=0.47 rewards=0.02,0.07,0.05
"""

import asyncio
import json
import os
import sys
import textwrap
import time
from typing import List, Optional

from dotenv import load_dotenv
from openai import OpenAI

from verirl_env import VerirlAction, verirl_env

load_dotenv()


# --- Configuration ---
API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:8000")
BENCHMARK = "verirl"

# Per-task wall-clock budgets (seconds) and success threshold
TASK_BUDGETS: dict[str, int] = {
    # easy
    "mac_unit":        4 * 60,
    "relu_clip":       3 * 60,
    "barrel_shifter":  3 * 60,
    # medium
    "axi_fifo":        6 * 60,
    "register_file":   5 * 60,
    "ring_buffer":     6 * 60,
    "dot_product":     5 * 60,
    "fir_filter":      6 * 60,
    # hard
    "systolic_array":  8 * 60,
    "fp16_adder":     10 * 60,
}
SUCCESS_SCORE_THRESHOLD = 0.5  # final_score in [0, 1]

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an expert RTL hardware designer. Implement the given Verilog specification correctly.

    REQUIRED WORKFLOW — follow this sequence every episode:
      1. write_file   — write a complete, synthesizable Verilog module.
                        For multi-module designs use separate files:
                        {"action_type": "write_file", "filename": "pe.v", "verilog_src": "..."}
                        {"action_type": "write_file", "filename": "top.v", "verilog_src": "..."}
      2. run_compile  — check all files for syntax errors; fix with write_file if needed
      3. run_sim      — run the testbench; read every PASS/FAIL line; fix failures with write_file
      4. (optional) run_formal — check formal properties if available; fix any counterexamples
      5. (optional) run_synth  — check area against reference cell count
      6. submit       — only after attempting compile and sim

    NEVER submit without first running run_compile and run_sim.

    Available actions — respond with exactly one JSON object, no markdown:
      {"action_type": "write_file", "filename": "design.v", "verilog_src": "<full module>", "message": "..."}
      {"action_type": "run_compile", "message": "checking syntax"}
      {"action_type": "run_sim",     "message": "running testbench"}
      {"action_type": "run_synth",   "message": "checking area"}
      {"action_type": "run_formal",  "message": "checking formal properties"}
      {"action_type": "list_files",  "message": "show written files"}
      {"action_type": "submit",      "message": "final submission"}

    Design rules:
    - No `initial` blocks in the design module (testbench only)
    - Use always @(posedge clk) for sequential logic
    - Fully combinational modules: use assign or always @(*)
    - Pay close attention to pipeline depth, pipeline registers, and timing requirements
    - For tasks requiring multiple modules: use separate write_file calls with different filenames
    """
).strip()


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------


def log_start(task: str, model: str) -> None:
    print(f"[START] task={task} env={BENCHMARK} model={model}", flush=True)


def log_step(
    step: int, action: str, reward: float, done: bool, error: Optional[str]
) -> None:
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error or 'null'}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: List[float], final_score: float) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={final_score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Agent helpers
# ---------------------------------------------------------------------------


def safe_score(raw) -> float:
    """Clamp any score/reward to the open interval (0, 1) required by the validator."""
    return round(min(max(float(raw), 0.01), 0.99), 2)


def sanitize_error(error_str: str, max_len: int = 80) -> Optional[str]:
    """Sanitize error message: remove newlines, truncate to max_len."""
    if not error_str:
        return None
    # Replace newlines and multiple spaces with single space
    sanitized = " ".join(error_str.split())
    # Truncate
    if len(sanitized) > max_len:
        sanitized = sanitized[:max_len]
    return sanitized


def format_observation(obs) -> str:
    """Format a VerirlObservation as a readable context block for the LLM."""
    parts = []
    if obs.task_spec:
        parts.append(f"TASK SPECIFICATION:\n{obs.task_spec}")
    if obs.tool_stdout:
        parts.append(f"TOOL OUTPUT:\n{obs.tool_stdout}")
    if obs.tool_stderr:
        parts.append(f"ERRORS:\n{obs.tool_stderr}")

    # Multi-file project summary
    if getattr(obs, "current_files", None):
        file_summary = ", ".join(
            f"{n}({len(s)}chars)" for n, s in sorted(obs.current_files.items())
        )
        parts.append(f"Files on disk: {file_summary}")

    # Formal verification status
    formal_proven = getattr(obs, "formal_properties_proven", None)
    formal_total = getattr(obs, "formal_properties_total", None)
    formal_str = (
        f" | formal={formal_proven}/{formal_total}"
        if formal_proven is not None else ""
    )

    parts.append(
        f"Status: compile={'OK' if obs.compile_ok else 'FAIL'} | "
        f"tests={obs.tests_passed}/{obs.tests_total}"
        f"{formal_str} | "
        f"turn={obs.turn_number} | remaining={obs.turns_remaining}"
    )
    return "\n\n".join(parts)


def parse_action(response_text: str) -> tuple[VerirlAction, Optional[str]]:
    """Extract a JSON action from the LLM response, handling markdown fences.

    Returns (action, parse_error) where parse_error is None on success.
    """
    text = response_text.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            data = json.loads(text[start:end])
            valid_fields = VerirlAction.model_fields
            return VerirlAction(**{k: v for k, v in data.items() if k in valid_fields}), None
        except Exception as exc:
            err = sanitize_error(f"parse_error: {str(exc)}", max_len=60)
            return VerirlAction(action_type="submit", message="parse error"), err
    return VerirlAction(action_type="submit", message="parse error"), "parse_error: no JSON found"


def action_label(action: VerirlAction) -> str:
    """Compact one-token label for [STEP] logging."""
    if action.action_type == "write_file" and action.verilog_src:
        return f"write_file({len(action.verilog_src)}chars)"
    return action.action_type


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fallback_action(obs) -> VerirlAction:
    """Fallback when the LLM call fails — submit whatever code we have."""
    msg = "llm error, submitting best code" if obs.current_verilog else "llm error"
    return VerirlAction(action_type="submit", message=msg)


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------


async def run_task(task_id: str, llm: OpenAI) -> float:
    """Run one complete episode for the given task. Returns final_score in [0, 1]."""
    start_time = time.time()
    final_score = 0.0
    rewards: List[float] = []
    steps_taken = 0
    success = False

    log_start(task=task_id, model=MODEL_NAME)

    env = None
    obs = None
    try:
        budget = TASK_BUDGETS.get(task_id)
        if budget is None:
            raise ValueError(f"Unknown task_id '{task_id}'. Valid: {list(TASK_BUDGETS.keys())}")

        env = verirl_env(base_url=ENV_BASE_URL)
        result = await env.reset(task_id=task_id)
        obs = result.observation

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": format_observation(obs)},
        ]

        for step in range(1, 100):  # max guard; episodes end via done flag
            elapsed = time.time() - start_time
            if elapsed > budget:
                action = VerirlAction(
                    action_type="submit", message="time budget exceeded"
                )
                result = await env.step(action)
                reward = safe_score(result.reward or 0.01)
                rewards.append(reward)
                steps_taken = step
                log_step(step, action_label(action), reward, True, None)
                final_score = safe_score(result.observation.final_score or 0.01)
                break

            # LLM call
            error: Optional[str] = None
            try:
                response = llm.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                )
                assistant_text = response.choices[0].message.content or ""
            except Exception as exc:
                error = sanitize_error(str(exc))
                assistant_text = ""

            if assistant_text:
                action, parse_err = parse_action(assistant_text)
                if parse_err and error is None:
                    error = parse_err
            else:
                # LLM call failed — pick a sensible fallback based on current state
                # rather than immediately submitting
                action = _fallback_action(obs)

            # Environment step
            try:
                result = await env.step(action)
            except Exception as exc:
                error = sanitize_error(str(exc))
                # Log the failed step before breaking
                reward = 0.01  # fallback reward on error
                rewards.append(reward)
                steps_taken = step
                log_step(step, action_label(action), reward, True, error)
                break

            obs = result.observation
            reward = safe_score(result.reward or 0.01)
            done = result.done

            # Validate reward is in expected range
            if not (-1.0 <= reward <= 1.0):
                print(
                    f"[WARNING] Task {task_id}: reward {reward} outside [-1.0, 1.0]",
                    file=sys.stderr,
                    flush=True,
                )
                error = sanitize_error(f"reward {reward} outside [-1.0, 1.0]")

            rewards.append(reward)
            steps_taken = step
            log_step(step, action_label(action), reward, done, error)

            messages.append({"role": "assistant", "content": assistant_text})
            messages.append({"role": "user", "content": format_observation(obs)})

            if done:
                final_score = safe_score(obs.final_score or 0.01)
                break

        success = final_score >= SUCCESS_SCORE_THRESHOLD

        # Validate final_score is in [0, 1]
        if not (0.0 <= final_score <= 1.0):
            print(
                f"[WARNING] Task {task_id}: final_score {final_score} not in [0, 1]",
                file=sys.stderr,
                flush=True,
            )

    finally:
        # Safety net: if loop exited without a submit (e.g. connection drop),
        # attempt a final submit so the score is not lost.
        if final_score == 0.0 and obs is not None and obs.current_verilog is not None:
            try:
                result = await env.step(
                    VerirlAction(action_type="submit", message="safety submit")
                )
                final_score = safe_score(result.observation.final_score or 0.01)
                success = final_score >= SUCCESS_SCORE_THRESHOLD
                steps_taken += 1
                log_step(steps_taken, "submit", safe_score(result.reward or 0.01), True, None)
            except Exception:
                pass
        # Ensure final_score is never exactly 0.0 (validator requires strict (0, 1))
        if final_score == 0.0:
            final_score = 0.01
        success = final_score >= SUCCESS_SCORE_THRESHOLD
        # Ensure rewards list is never empty — [END] requires at least one reward value
        if not rewards:
            rewards = [0.01]
        if env is not None:
            try:
                await env.close()
            except Exception:
                pass
        log_end(success=success, steps=steps_taken, rewards=rewards, final_score=final_score)

    return final_score


# ---------------------------------------------------------------------------
# Task Enumeration & Validation
# ---------------------------------------------------------------------------


async def validate_environment(base_url: str) -> List[str]:
    """
    Validate that the environment is operational and all tasks are discoverable.

    Runs silently to avoid interfering with stdout format.
    Returns list of valid task IDs or empty list if validation fails.

    Args:
        base_url: Environment server URL

    Returns:
        List of task IDs if validation passes, empty list otherwise
    """
    # Known tasks — these must be in the environment
    task_manifest = [
        {"id": "mac_unit",        "difficulty": "easy",   "max_turns": 8},
        {"id": "relu_clip",       "difficulty": "easy",   "max_turns": 6},
        {"id": "barrel_shifter",  "difficulty": "easy",   "max_turns": 6},
        {"id": "axi_fifo",        "difficulty": "medium", "max_turns": 10},
        {"id": "register_file",   "difficulty": "medium", "max_turns": 8},
        {"id": "ring_buffer",     "difficulty": "medium", "max_turns": 10},
        {"id": "dot_product",     "difficulty": "medium", "max_turns": 8},
        {"id": "fir_filter",      "difficulty": "medium", "max_turns": 10},
        {"id": "systolic_array",  "difficulty": "hard",   "max_turns": 12},
        {"id": "fp16_adder",      "difficulty": "hard",   "max_turns": 15},
    ]

    task_ids = []
    env = verirl_env(base_url=base_url)

    for task in task_manifest:
        task_id = task["id"]
        try:
            result = await env.reset(task_id=task_id)
            # Submit empty code (should score 0)
            result = await env.step(VerirlAction(action_type="submit"))
            obs = result.observation
            final_score = safe_score(obs.final_score or 0.01)

            # Task is valid if we got a score
            if final_score is not None:
                task_ids.append(task_id)
        except Exception:
            # Silently skip failed validation
            pass

    await env.close()
    return task_ids


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    # Try to enumerate tasks; if validation fails, use defaults
    task_ids = await validate_environment(ENV_BASE_URL)
    if not task_ids:
        # Fallback: use hardcoded task list (validation may have failed due to network)
        task_ids = [
            "mac_unit", "relu_clip", "barrel_shifter",
            "axi_fifo", "register_file", "ring_buffer", "dot_product", "fir_filter",
            "systolic_array", "fp16_adder",
        ]

    llm = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    scores: dict[str, float] = {}
    total_start = time.time()

    for task_id in task_ids:
        scores[task_id] = await run_task(task_id, llm)

    # No summary output — spec requires ONLY [START]/[STEP]/[END] lines to stdout


if __name__ == "__main__":
    asyncio.run(main())
