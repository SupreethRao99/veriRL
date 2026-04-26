"""Task curriculum: difficulty buckets, sampling weights, and the system prompt."""

from __future__ import annotations

import random
import textwrap

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

    Workflow (follow in order):
      1. write_file    — implement the module
      2. run_compile   — fix any syntax errors, then rewrite + recompile until clean
      3. run_sim       — fix failing tests, rewrite + recompile until all pass
      4. run_synth     — verify synthesizability (optional but recommended)
      5. run_formal    — optional; use for edge-case confidence on complex designs
      6. submit        — YOU MUST ALWAYS CALL THIS. Call once tests pass, or when
                         you are running low on turns. This is mandatory — the
                         episode is not scored until you call submit.

    Rules:
    - No `initial` blocks in the design (testbench only)
    - Sequential logic: always @(posedge clk)
    - Combinational logic: assign / always @(*)
    - Multi-file designs: call write_file once per file with a distinct filename
    - Always read the full tool output before deciding your next action
    - IMPORTANT: You MUST call submit() at the end of every episode, no exceptions.
""").strip()


def sample_task(config: TrainConfig, rng: random.Random = random) -> str:
    """Sample a task ID weighted by difficulty bucket.

    Chooses a difficulty bucket according to ``config.task_difficulty_weights``,
    then uniformly samples a task from that bucket.

    Args:
        config: Training config containing the ``task_difficulty_weights`` dict.
        rng: Random source (defaults to the module-level ``random`` instance).

    Returns:
        A task ID string such as ``"mac_unit"`` or ``"fp16_adder"``.
    """
    difficulties = list(config.task_difficulty_weights.keys())
    weights = [config.task_difficulty_weights[d] for d in difficulties]
    chosen = rng.choices(difficulties, weights=weights, k=1)[0]
    return rng.choice(TASKS_BY_DIFFICULTY[chosen])
