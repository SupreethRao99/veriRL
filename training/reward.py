"""Reward function for environment_factory-mode GRPOTrainer.

When environment_factory is active, TRL calls reward_func(environments, ...)
where each element is a VerirlToolEnv instance. The episode score is already
stored on env.reward by submit() or the auto-grade path in _step().
"""

from __future__ import annotations


def reward_func(environments, **kwargs) -> list[float]:
    rewards = []
    for env in environments:
        # Pop the oldest completed episode's reward. If the queue is empty the
        # episode ended without submit/auto-grade (e.g. TRL turn limit hit before
        # any tool call), so reward is 0.
        r = env._reward_queue.popleft() if env._reward_queue else 0.0
        print(f"[reward] task={env.task_id} score={r:.3f} queue_remaining={len(env._reward_queue)} env_id={id(env)}")
        rewards.append(r)
    return rewards
