"""Reward function for TRL's GRPOTrainer."""

from __future__ import annotations


def reward_func(environments: list, **kwargs) -> list[float]:
    """
    Read final_score from each VerirlToolEnv after episode completion.

    TRL calls this after every rollout batch. Each element of `environments`
    is a VerirlToolEnv instance whose self.reward was set when submit() was
    called (or left at 0.01 if the episode ended without a submit).
    """
    return [env.reward for env in environments]
