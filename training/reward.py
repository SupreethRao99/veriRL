"""Reward functions for environment_factory-mode GRPOTrainer.

When environment_factory is active, TRL calls each reward function with
VerirlToolEnv instances. The terminal reward components are already queued by
submit() or the auto-grade path in _step().
"""

from __future__ import annotations

from training.wandb_task_logging import record_task_reward


_ZERO_COMPONENTS = {"tool": 0.0, "compile": 0.0, "sim": 0.0, "final": 0.0}


def _components_for_env(env) -> dict[str, float]:
    cached = getattr(env, "_current_reward_components", None)
    if cached is not None:
        return cached

    if getattr(env, "_reward_component_queue", None):
        components = env._reward_component_queue.popleft()
    else:
        components = dict(_ZERO_COMPONENTS)
    env._current_reward_components = components
    return components


def _component_reward(environments, key: str, *, clear: bool = False) -> list[float]:
    rewards = []
    for env in environments:
        components = _components_for_env(env)
        r = float(components.get(key, 0.0))
        if key == "final":
            record_task_reward(env.task_id, r)
        queue_remaining = len(getattr(env, "_reward_component_queue", []))
        print(f"[reward/{key}] task={env.task_id} score={r:.3f} queue_remaining={queue_remaining} env_id={id(env)}")
        if clear and hasattr(env, "_current_reward_components"):
            delattr(env, "_current_reward_components")
        rewards.append(r)
    return rewards


def tool_use_reward(environments, **kwargs) -> list[float]:
    return _component_reward(environments, "tool")


def compile_reward(environments, **kwargs) -> list[float]:
    return _component_reward(environments, "compile")


def sim_reward(environments, **kwargs) -> list[float]:
    return _component_reward(environments, "sim")


def final_score_reward(environments, **kwargs) -> list[float]:
    return _component_reward(environments, "final", clear=True)
