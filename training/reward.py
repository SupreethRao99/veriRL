"""Reward function and evolution buffer for TRL's GRPOTrainer."""

from __future__ import annotations

import threading
from typing import Any

# ---------------------------------------------------------------------------
# Evolution buffer — collects top-K designs per task across training steps.
# reward_func populates this; trainer.py reads it to build evolution prompts.
# ---------------------------------------------------------------------------

_BUFFER_MAX_PER_TASK = 5  # keep at most this many designs per task

_buffer_lock = threading.Lock()
# task_id → list of (code, score, score_breakdown), sorted by score descending
_evolution_buffer: dict[str, list[tuple[str, float, dict[str, Any]]]] = {}


def _update_buffer(
    task_id: str,
    code: str,
    score: float,
    score_threshold: float = 0.40,
) -> None:
    """Thread-safe insert of a completed design into the evolution buffer."""
    if not task_id or not code or score < score_threshold:
        return
    with _buffer_lock:
        bucket = _evolution_buffer.setdefault(task_id, [])
        bucket.append((code, score, {}))
        # Keep sorted descending by score, prune to max size
        bucket.sort(key=lambda x: -x[1])
        del bucket[_BUFFER_MAX_PER_TASK:]


def get_evolution_buffer() -> dict[str, list[tuple[str, float, dict[str, Any]]]]:
    """Return a snapshot of the current evolution buffer (deep-copied for safety)."""
    with _buffer_lock:
        return {tid: list(entries) for tid, entries in _evolution_buffer.items()}


def clear_evolution_buffer() -> None:
    """Clear the evolution buffer (called between training phases if needed)."""
    with _buffer_lock:
        _evolution_buffer.clear()


# ---------------------------------------------------------------------------
# GRPO reward function
# ---------------------------------------------------------------------------


def reward_func(environments: list, **kwargs) -> list[float]:
    """
    Read final_score from each VerirlToolEnv after episode completion.

    TRL calls this after every rollout batch. Each element of `environments`
    is a VerirlToolEnv instance whose self.reward was set when submit() was
    called (or left at 0.01 if the episode ended without a submit).

    Side-effect: high-scoring completions are added to _evolution_buffer so
    that the evolution training phase can build synthesis prompts from them.
    """
    rewards: list[float] = []
    for env in environments:
        score = env.reward
        _update_buffer(
            task_id=getattr(env, "task_id", ""),
            code=getattr(env, "last_verilog_src", ""),
            score=score,
        )
        rewards.append(score)
    return rewards
