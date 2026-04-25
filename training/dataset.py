"""Dataset builder for GRPO training."""

from __future__ import annotations

import random
from pathlib import Path

from datasets import Dataset

from training.config import TrainConfig
from training.curriculum import ALL_TASKS, SYSTEM_PROMPT, TASKS_BY_DIFFICULTY

# Map task_id → subdirectory name under problems/
_TASK_DIRS: dict[str, str] = {
    "mac_unit":       "task1_mac",
    "axi_fifo":       "task2_axi_fifo",
    "systolic_array": "task3_systolic",
    "relu_clip":      "task4_relu_clip",
    "barrel_shifter": "task5_barrel_shifter",
    "register_file":  "task6_register_file",
    "ring_buffer":    "task7_ring_buffer",
    "dot_product":    "task8_dot_product",
    "fir_filter":     "task9_fir_filter",
    "fp16_adder":     "task10_fp16_adder",
}


def _load_specs_from_disk() -> dict[str, str]:
    """Read spec.md for every task directly from the problems/ directory."""
    candidates = [
        Path(__file__).parent.parent / "problems",
        Path(__file__).parent.parent.parent / "problems",
    ]
    try:
        import verirl_env as _ve  # type: ignore
        candidates.insert(0, Path(_ve.__file__).parent / "problems")
    except ImportError:
        pass

    problems_dir: Path | None = next((p for p in candidates if p.is_dir()), None)
    if problems_dir is None:
        return {}

    specs: dict[str, str] = {}
    for task_id, subdir in _TASK_DIRS.items():
        spec_file = problems_dir / subdir / "spec.md"
        if spec_file.exists():
            specs[task_id] = spec_file.read_text()
    return specs


def build_dataset(config: TrainConfig, n_samples: int = 400) -> Dataset:
    """
    Build a HuggingFace Dataset of (prompt, task_id) pairs for GRPO.

    Records are ordered easy → medium → hard so TRL trains on simpler tasks
    first. GRPOTrainer requires a standard Dataset, so the rows are laid out in
    curriculum order even though the trainer may still sample from it.

    With generation_batch_size=4 and num_generations=2, TRL consumes roughly
    n_steps * (generation_batch_size / num_generations) / steps_per_generation
    unique prompts. For max_steps=200 that is ~100 unique prompts, so
    n_samples=400 gives ~2 passes through the curriculum.

    The task_id column is forwarded to VerirlToolEnv.reset() so every sample
    targets a specific task.
    """
    rng = random.Random(42)

    specs = _load_specs_from_disk()
    for task_id in ALL_TASKS:
        if task_id not in specs:
            specs[task_id] = f"Implement the {task_id} Verilog module."

    weights = config.task_difficulty_weights
    total_weight = sum(weights.values())

    records = []
    # Build phases in order: easy first, medium next, hard last.
    for difficulty in ["easy", "medium", "hard"]:
        w = weights.get(difficulty, 0.0)
        n = round(n_samples * w / total_weight)
        allowed_tasks = set(config.task_ids) if config.task_ids else None
        tasks = TASKS_BY_DIFFICULTY[difficulty]
        if allowed_tasks is not None:
            tasks = [task for task in tasks if task in allowed_tasks]
        if not tasks:
            continue
        for _ in range(n):
            task_id = rng.choice(tasks)
            records.append({
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": f"TASK SPECIFICATION:\n{specs[task_id]}"},
                ],
                "task_id": task_id,
            })

    # Trim or pad to exactly n_samples (pad with easy tasks).
    easy_tasks = config.task_ids or TASKS_BY_DIFFICULTY["easy"]
    while len(records) < n_samples:
        task_id = rng.choice(easy_tasks)
        records.insert(0, {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": f"TASK SPECIFICATION:\n{specs[task_id]}"},
            ],
            "task_id": task_id,
        })
    records = records[:n_samples]

    return Dataset.from_list(records)
