"""Dataset builder for GRPO training."""

from __future__ import annotations

import random
from pathlib import Path

from datasets import Dataset

from training.config import TrainConfig
from training.curriculum import ALL_TASKS, SYSTEM_PROMPT, sample_task

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
    """
    Read spec.md for every task directly from the problems/ directory.

    Works whether we're running from the repo root or from an installed
    package (problems/ is bundled as package data under verirl_env/).
    """
    # Try repo-root layout first, then installed-package layout.
    candidates = [
        Path(__file__).parent.parent / "problems",  # repo root
        Path(__file__).parent.parent.parent / "problems",  # installed as verirl_env.training
    ]
    # Also try via the installed package's __file__
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


def build_dataset(config: TrainConfig, n_samples: int = 2000) -> Dataset:
    """
    Build a HuggingFace Dataset of (prompt, task_id) pairs for GRPO.

    The task_id column is forwarded to VerirlToolEnv.reset() as a keyword
    argument so every training sample targets a specific task.

    Specs are read from disk (problems/*/spec.md). If a spec file is missing
    for a task, we fall back to a one-line placeholder.
    """
    rng = random.Random(42)

    specs = _load_specs_from_disk()

    # Fill in any missing tasks with a short fallback
    for task_id in ALL_TASKS:
        if task_id not in specs:
            specs[task_id] = f"Implement the {task_id} Verilog module."

    records = []
    for _ in range(n_samples):
        task_id = sample_task(config, rng)
        records.append({
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": f"TASK SPECIFICATION:\n{specs[task_id]}"},
            ],
            "task_id": task_id,
        })
    return Dataset.from_list(records)
