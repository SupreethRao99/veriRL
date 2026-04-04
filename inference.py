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
    [END]   success=<true|false> steps=<n> rewards=<r1,r2,...,rn>

  Rules:
    - One [START] line at episode begin.
    - One [STEP] line per step, immediately after env.step() returns.
    - One [END] line after env.close(), always emitted (even on exception).
    - reward and rewards are formatted to 2 decimal places.
    - done and success are lowercase booleans: true or false.
    - error is the raw last_action_error string, or null if none.
    - All fields on a single line with no newlines within a line.

  Example:
    [START] task=mac_unit env=verirl model=Qwen2.5-72B-Instruct
    [STEP] step=1 action=write_file(312chars) reward=0.02 done=false error=null
    [STEP] step=2 action=run_compile reward=0.07 done=false error=null
    [STEP] step=3 action=submit reward=0.05 done=true error=null
    [END] success=true steps=3 rewards=0.02,0.07,0.05
"""

import asyncio
import json
import os
import textwrap
import time
from typing import List, Optional

from dotenv import load_dotenv
from openai import OpenAI

from verirl_env import VerirlAction, verirl_env

load_dotenv()


# --- Configuration ---
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:8000")
BENCHMARK = "verirl"

# Per-task wall-clock budgets (seconds) and success threshold
TASK_BUDGETS: dict[str, int] = {
    "mac_unit": 4 * 60,
    "axi_fifo": 6 * 60,
    "systolic_array": 8 * 60,
}
SUCCESS_SCORE_THRESHOLD = 0.5  # final_score in [0, 1]

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an expert RTL hardware designer. Implement the given Verilog specification correctly.

    REQUIRED WORKFLOW — follow this sequence every episode:
      1. write_file   — write a complete, synthesizable Verilog module (no `initial` blocks in design)
      2. run_compile  — check for syntax errors; if errors appear, write_file again with fixes
      3. run_sim      — run the testbench; read every PASS/FAIL line; fix failures with write_file
      4. (repeat 2-3 until all tests pass or turns are low)
      5. submit       — only submit after attempting compile and sim

    NEVER submit without first running run_compile and run_sim — you will score 0 otherwise.

    Available actions — respond with exactly one JSON object, no markdown:
      {"action_type": "write_file", "verilog_src": "<full module here>", "message": "reasoning"}
      {"action_type": "run_compile", "message": "checking syntax"}
      {"action_type": "run_sim",     "message": "running testbench"}
      {"action_type": "run_synth",   "message": "checking area"}
      {"action_type": "submit",      "message": "final submission"}

    Design rules:
    - No `initial` blocks in the design module (testbench only)
    - Use always @(posedge clk) for sequential logic
    - Pay close attention to pipeline depth, pipeline registers, and timing requirements
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


def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Agent helpers
# ---------------------------------------------------------------------------


def format_observation(obs) -> str:
    """Format a VerirlObservation as a readable context block for the LLM."""
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
            return VerirlAction(action_type="submit", message="parse error"), f"parse_error: {str(exc)[:60]}"
    return VerirlAction(action_type="submit", message="parse error"), "parse_error: no JSON found in response"


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
    budget = TASK_BUDGETS[task_id]
    start_time = time.time()
    final_score = 0.0
    rewards: List[float] = []
    steps_taken = 0
    success = False

    log_start(task=task_id, model=MODEL_NAME)

    env = verirl_env(base_url=ENV_BASE_URL)
    obs = None
    try:
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
                reward = result.reward or 0.0
                rewards.append(reward)
                steps_taken = step
                log_step(step, action_label(action), reward, True, None)
                final_score = result.observation.final_score or 0.0
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
                error = str(exc)[:80]
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
                error = str(exc)[:80]
                break

            obs = result.observation
            reward = result.reward or 0.0
            done = result.done

            rewards.append(reward)
            steps_taken = step
            log_step(step, action_label(action), reward, done, error)

            messages.append({"role": "assistant", "content": assistant_text})
            messages.append({"role": "user", "content": format_observation(obs)})

            if done:
                final_score = obs.final_score or 0.0
                break

        success = final_score >= SUCCESS_SCORE_THRESHOLD

    finally:
        # Safety net: if loop exited without a submit (e.g. connection drop),
        # attempt a final submit so the score is not lost.
        if final_score == 0.0 and obs is not None and obs.current_verilog is not None:
            try:
                result = await env.step(
                    VerirlAction(action_type="submit", message="safety submit")
                )
                final_score = result.observation.final_score or 0.0
                success = final_score >= SUCCESS_SCORE_THRESHOLD
                steps_taken += 1
                log_step(steps_taken, "submit", result.reward or 0.0, True, None)
            except Exception:
                pass
        try:
            await env.close()
        except Exception:
            pass
        log_end(success=success, steps=steps_taken, rewards=rewards)

    return final_score


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    llm = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    tasks = ["mac_unit", "axi_fifo", "systolic_array"]
    scores: dict[str, float] = {}
    total_start = time.time()

    for task_id in tasks:
        scores[task_id] = await run_task(task_id, llm)

    total_elapsed = time.time() - total_start
    mean_score = sum(scores.values()) / len(scores)

    print("\n# Summary", flush=True)
    for task_id, score in scores.items():
        print(f"#   {task_id:20s}: {score:.3f}", flush=True)
    print(f"#   {'mean':20s}: {mean_score:.3f}", flush=True)
    print(f"#   total_time: {total_elapsed:.0f}s", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
