"""Reward functions for environment_factory-mode GRPOTrainer.

When ``environment_factory`` is active, TRL calls each reward function with
a list of ``VerirlToolEnv`` instances after the rollout phase. The terminal
reward components (tool, compile, sim, final) are already queued by
``submit()`` or the auto-grade path in ``_step()``.

The four functions correspond to the four entries in ``config.reward_weights``:
  [tool_use_reward, compile_reward, sim_reward, final_score_reward]
"""

from __future__ import annotations

from training.wandb_task_logging import record_task_components

# Mirrors config.yaml training.grpo.reward_weights — [tool, compile, sim, final]
_REWARD_WEIGHTS = (0.05, 0.25, 0.40, 0.30)


_ZERO_COMPONENTS: dict[str, float] = {
    "tool": 0.0,
    "compile": 0.0,
    "sim": 0.0,
    "final": 0.0,
}


def _components_for_env(env) -> dict[str, float]:
    """Return the cached reward components for an environment, popping from the queue if needed.

    Components are cached on ``env._current_reward_components`` so all four
    reward functions read the same episode's data. The cache is cleared by
    ``final_score_reward`` (the last function called) via the ``clear`` flag
    in ``_component_reward``.

    Args:
        env: A ``VerirlToolEnv`` instance.

    Returns:
        Dict with ``"tool"``, ``"compile"``, ``"sim"``, ``"final"`` values.
    """
    cached = getattr(env, "_current_reward_components", None)
    if cached is not None:
        return cached
    if getattr(env, "_reward_component_queue", None):
        components = env._reward_component_queue.popleft()
    elif hasattr(env, "partial_reward_components"):
        # Episode did not terminate before TRL sampled rewards — use latest obs
        # state rather than returning all-zeros, which kills the learning signal.
        components = env.partial_reward_components()
        print(
            f"[reward/partial] task={getattr(env, 'task_id', '?')} "
            f"tool_calls={getattr(env, '_tool_calls', 0)} "
            f"compile={getattr(env, '_compile_ok', False)} "
            f"sim={getattr(env, '_tests_passed', 0)}/{getattr(env, '_tests_total', 0)} "
            f"components={components}"
        )
    else:
        components = dict(_ZERO_COMPONENTS)
    env._current_reward_components = components
    return components


def _component_reward(environments, key: str, *, clear: bool = False) -> list[float]:
    """Extract one reward component from each environment and log it.

    Args:
        environments: List of ``VerirlToolEnv`` instances from TRL.
        key: Which component to extract (``"tool"``, ``"compile"``, ``"sim"``,
            or ``"final"``).
        clear: If ``True``, delete the cached components after reading so the
            next episode starts fresh. Should only be ``True`` for the last
            reward function (``final_score_reward``).

    Returns:
        List of floats, one per environment, in the same order as ``environments``.
    """
    rewards = []
    for env in environments:
        components = _components_for_env(env)
        r = float(components.get(key, 0.0))
        queue_remaining = len(getattr(env, "_reward_component_queue", []))
        print(
            f"[reward/{key}] task={env.task_id} score={r:.3f} "
            f"queue_remaining={queue_remaining} env_id={id(env)}"
        )
        if clear:
            # Last reward function — log all components + weighted composite to
            # W&B, then reset the cache for the next episode.
            w_tool, w_compile, w_sim, w_final = _REWARD_WEIGHTS
            composite = (
                w_tool    * float(components.get("tool",    0.0))
                + w_compile * float(components.get("compile", 0.0))
                + w_sim     * float(components.get("sim",     0.0))
                + w_final   * float(components.get("final",   0.0))
            )
            record_task_components(env.task_id, components, composite)
            if hasattr(env, "_current_reward_components"):
                delattr(env, "_current_reward_components")
        rewards.append(r)
    return rewards


def tool_use_reward(environments, **kwargs) -> list[float]:
    """Reward for calling ``submit`` (1.0) vs. partial tool use (0–0.5).

    Incentivises the agent to complete the loop rather than stopping early.
    """
    return _component_reward(environments, "tool")


def compile_reward(environments, **kwargs) -> list[float]:
    """Reward for producing a design that compiles cleanly (1.0) or not (0.0)."""
    return _component_reward(environments, "compile")


def sim_reward(environments, **kwargs) -> list[float]:
    """Reward proportional to the fraction of simulation test cases that passed."""
    return _component_reward(environments, "sim")


def final_score_reward(environments, **kwargs) -> list[float]:
    """Weighted EDA-tool score from ``submit`` in [0.01, 0.99].

    This is the last reward function called per step and clears the cached
    components so the next episode starts with a clean slate.
    """
    return _component_reward(environments, "final", clear=True)
