"""Per-task reward logging for GRPO runs.

The reward functions see task-level episode scores, while the trainer owns the
global step counter. This module buffers rewards by task ID and flushes mean
values to W&B aligned to the trainer step via a ``TrainerCallback``, producing
one ``reward_by_task/<task>`` metric per task per gradient step.
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
    """Buffer one completed episode reward for later trainer-step logging.

    Thread-safe; may be called concurrently from multiple reward functions.

    Args:
        task_id: The task that just finished (e.g. ``"mac_unit"``).
            ``None`` or empty string is normalised to ``"unknown"``.
        reward: Final score for the episode, in [0.01, 0.99].
    """
    task = (task_id or "unknown").strip() or "unknown"
    with _LOCK:
        _PENDING[task].append(float(reward))


def _is_wandb_ready() -> bool:
    """Return ``True`` if W&B is installed, not disabled, and has an active run."""
    if os.environ.get("WANDB_DISABLED", "").lower() in {"1", "true", "yes"}:
        return False
    try:
        import wandb
    except Exception:
        return False
    return wandb.run is not None


def flush_task_rewards(
    step: int, *, is_world_process_zero: bool = True
) -> Mapping[str, float]:
    """Log buffered per-task reward means to W&B at the given trainer step.

    Called by ``WandbTaskRewardCallback.on_step_end`` after every gradient
    update. No-ops silently when W&B is not initialised or the buffer is empty.

    Args:
        step: Trainer global step to use as the x-axis value.
        is_world_process_zero: When ``False`` (non-primary rank in multi-GPU),
            this function is a no-op to avoid duplicate W&B writes.

    Returns:
        The metrics dict that was logged, or an empty dict if nothing was logged.
    """
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
        # Do not pass W&B's internal ``step`` argument — TRL may have already
        # advanced it. ``verirl/global_step`` is the x-axis for task reward plots.
        wandb.log(metrics)
    return metrics


def clear_task_rewards() -> None:
    """Clear all pending reward buffers.

    Call before starting or resuming a training run to prevent stale rewards
    from a previous run contaminating the first flush.
    """
    with _LOCK:
        _PENDING.clear()
