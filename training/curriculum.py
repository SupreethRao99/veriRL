"""Task curriculum: difficulty buckets, sampling, and system prompt."""

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
