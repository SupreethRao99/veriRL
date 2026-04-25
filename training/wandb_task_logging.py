"""Per-task reward logging for GRPO runs.

The reward function sees task-level episode scores, while the trainer owns the
global step. Buffering here lets a callback flush reward means against the
trainer step so W&B plots reward-vs-step per task.
"""

from __future__ import annotations

import collections
import os
import threading
from collections.abc import Mapping


_LOCK = threading.Lock()
_PENDING: dict[str, list[float]] = collections.defaultdict(list)
_METRICS_DEFINED = False


def record_task_reward(task_id: str | None, reward: float) -> None:
    """Buffer one completed episode reward for later trainer-step logging."""
    task = (task_id or "unknown").strip() or "unknown"
    with _LOCK:
        _PENDING[task].append(float(reward))


def _is_wandb_ready() -> bool:
    if os.environ.get("WANDB_DISABLED", "").lower() in {"1", "true", "yes"}:
        return False
    try:
        import wandb
    except Exception:
        return False
    return wandb.run is not None


def flush_task_rewards(step: int, *, is_world_process_zero: bool = True) -> Mapping[str, float]:
    """Log buffered per-task reward means to W&B at the provided trainer step."""
    if not is_world_process_zero:
        return {}

    with _LOCK:
        if not _PENDING:
            return {}
        pending = {task: values[:] for task, values in _PENDING.items()}
        _PENDING.clear()

    if not _is_wandb_ready():
        return {}

    import wandb

    global _METRICS_DEFINED
    if not _METRICS_DEFINED:
        wandb.define_metric("verirl/global_step")
        wandb.define_metric("reward_by_task/*", step_metric="verirl/global_step")
        wandb.define_metric("reward_count_by_task/*", step_metric="verirl/global_step")
        _METRICS_DEFINED = True

    metrics: dict[str, float] = {"verirl/global_step": float(step)}
    for task, values in sorted(pending.items()):
        if not values:
            continue
        metrics[f"reward_by_task/{task}"] = sum(values) / len(values)
        metrics[f"reward_count_by_task/{task}"] = float(len(values))

    if len(metrics) > 1:
        # Do not pass W&B's internal `step`; TRL also logs to W&B and may have
        # advanced it already. `verirl/global_step` is the x-axis for these
        # task reward plots.
        wandb.log(metrics)
    return metrics


def clear_task_rewards() -> None:
    """Clear pending rewards, useful before starting or resuming a run."""
    with _LOCK:
        _PENDING.clear()
