"""Task curriculum: difficulty buckets, sampling, system prompts, and evolution helpers."""

from __future__ import annotations

import random
import textwrap
from typing import Any

from training.config import TrainConfig

TASKS_BY_DIFFICULTY: dict[str, list[str]] = {
    "easy":   ["mac_unit", "relu_clip", "barrel_shifter"],
    "medium": ["axi_fifo", "register_file", "ring_buffer", "dot_product", "fir_filter"],
    "hard":   ["systolic_array", "fp16_adder"],
}

ALL_TASKS: list[str] = [t for tasks in TASKS_BY_DIFFICULTY.values() for t in tasks]

SYSTEM_PROMPT: str = textwrap.dedent("""
    You are an expert RTL hardware designer. Your goal is to implement a correct,
    synthesizable Verilog module that passes all testbench checks.

    Use the available tools in this order:
      1. write_file   — write a complete Verilog module (repeat for each file)
      2. run_compile  — check syntax; fix errors and rewrite if needed
      3. run_sim      — run testbench; fix FAIL lines, then rewrite + recompile
      4. submit       — when all tests pass, or when turns are nearly exhausted

    Rules:
    - No `initial` blocks in the design (testbench only)
    - Sequential logic: always @(posedge clk)
    - Combinational logic: assign / always @(*)
    - Multi-file designs: call write_file once per file with a distinct filename
""").strip()


def sample_task(config: TrainConfig, rng: random.Random = random) -> str:
    """Sample a task ID weighted by difficulty bucket."""
    difficulties = list(config.task_difficulty_weights.keys())
    weights = [config.task_difficulty_weights[d] for d in difficulties]
    chosen = rng.choices(difficulties, weights=weights, k=1)[0]
    return rng.choice(TASKS_BY_DIFFICULTY[chosen])


# ---------------------------------------------------------------------------
# Evolutionary GRPO — prompt templates
# ---------------------------------------------------------------------------

EVOLUTION_SYSTEM_PROMPT: str = textwrap.dedent("""
    You are an expert RTL hardware designer reviewing multiple previous design attempts.
    You will be shown K Verilog implementations of the same hardware task, each with its
    EDA evaluation scores (compile, simulation, timing, area, formal verification).

    Your task: synthesize an EVOLVED design that:
      1. Combines the best structural elements from each attempt
      2. Fixes the specific weaknesses revealed by the lowest-scoring dimensions
      3. Produces a superior implementation that outperforms all previous attempts

    Score dimensions:
      compile  — syntax correctness (iverilog)
      sim      — testbench pass rate (iverilog + vvp)
      timing   — cycle-accurate timing constraints
      area     — synthesis cell count vs reference (yosys)
      formal   — formal property verification (SymbiYosys)

    Use the same tool workflow as before:
      write_file → run_compile → run_sim → submit
    Do not skip compile or sim steps.
""").strip()


def build_evolution_prompt(
    top_k_results: list[dict[str, Any]],
    task_spec: str,
) -> list[dict[str, str]]:
    """Build a chat-format evolution prompt from top-K previous design results.

    Args:
        top_k_results: List of dicts with keys: code (str), score (float),
                       score_breakdown (dict[str, float] | None).
        task_spec:     Full task specification markdown.

    Returns:
        List of chat message dicts (system + user) ready for the LLM.
    """
    designs_text = ""
    for i, result in enumerate(top_k_results, 1):
        code = result.get("code", "")
        score = result.get("score", 0.0)
        breakdown = result.get("score_breakdown") or {}
        if breakdown:
            bd_str = ", ".join(f"{k}={v:.2f}" for k, v in sorted(breakdown.items()))
        else:
            bd_str = "breakdown unavailable"
        designs_text += (
            f"\n--- Design {i} (overall={score:.3f} | {bd_str}) ---\n"
            f"```verilog\n{code}\n```\n"
        )

    # Identify the weakest dimension to focus the evolution on
    all_breakdowns = [r.get("score_breakdown") or {} for r in top_k_results]
    merged: dict[str, list[float]] = {}
    for bd in all_breakdowns:
        for k, v in bd.items():
            merged.setdefault(k, []).append(v)
    if merged:
        weakest = min(merged, key=lambda k: sum(merged[k]) / len(merged[k]))
        focus_hint = f"Focus your evolution on improving the '{weakest}' dimension, which scored lowest on average."
    else:
        focus_hint = "Focus on correctness and synthesis quality."

    user_content = (
        f"TASK SPECIFICATION:\n{task_spec}\n\n"
        f"PREVIOUS DESIGN ATTEMPTS:\n{designs_text}\n"
        f"{focus_hint}\n\n"
        "Now write an EVOLVED Verilog design that synthesizes the best elements of "
        "each attempt and fixes the identified weaknesses."
    )

    return [
        {"role": "system", "content": EVOLUTION_SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]
