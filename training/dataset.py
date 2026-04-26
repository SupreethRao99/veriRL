"""Dataset builder for GRPO curriculum training."""

from __future__ import annotations

import random
from pathlib import Path

from datasets import Dataset

from training.config import TrainConfig
from training.curriculum import ALL_TASKS, SYSTEM_PROMPT, TASKS_BY_DIFFICULTY


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
    """Read ``spec.md`` for every task from the ``problems/`` directory.

    Tries several candidate locations (installed package path, repository root,
    parent of repository root) so the function works both when the package is
    installed as a wheel and when run directly from the cloned repo.

    Returns:
        A dict mapping ``task_id`` → spec Markdown string. Tasks whose spec
        file is not found are omitted; callers should supply fallback text.
    """
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
    """Build a HuggingFace ``Dataset`` of ``(prompt, task_id)`` pairs for GRPO.

    Records are ordered easy → medium → hard so TRL trains on simpler tasks
    first. The ordering is soft — GRPOTrainer may still sample non-sequentially
    — but it biases the early training steps toward tractable tasks.

    With ``num_generations=2`` and ``per_device_train_batch_size=4``, TRL
    consumes roughly ``max_steps * batch / num_generations`` unique prompts.
    For ``max_steps=200`` that is ~400 unique prompts, so the default
    ``n_samples=400`` gives one pass through the curriculum.

    The ``task_id`` column is forwarded to ``VerirlToolEnv.reset()`` so every
    sample targets a specific task rather than sampling randomly at runtime.

    Args:
        config: Training config supplying ``task_difficulty_weights`` and
            ``task_ids`` (the allowed task subset).
        n_samples: Total number of dataset rows to produce.

    Returns:
        A HuggingFace ``Dataset`` with ``prompt`` (list of chat messages) and
        ``task_id`` (str) columns.
    """
    rng = random.Random(42)

    specs = _load_specs_from_disk()
    for task_id in ALL_TASKS:
        if task_id not in specs:
            specs[task_id] = f"Implement the {task_id} Verilog module."

    weights = config.task_difficulty_weights
    total_weight = sum(weights.values())

    records = []
    for difficulty in ["easy", "medium", "hard"]:
        w = weights.get(difficulty, 0.0)
        n = round(n_samples * w / total_weight)
        allowed_tasks = set(config.task_ids) if config.task_ids else None
        tasks = TASKS_BY_DIFFICULTY[difficulty]
        if allowed_tasks is not None:
            tasks = [t for t in tasks if t in allowed_tasks]
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
