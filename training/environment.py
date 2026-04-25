"""VeriRL tool environment for TRL's ``environment_factory`` API.

Each ``VerirlToolEnv`` instance owns a dedicated asyncio event loop running in
a background daemon thread. All WebSocket client calls (reset, step) are
dispatched to that loop via ``run_coroutine_threadsafe``, ensuring the
connection is always driven by the same loop regardless of which thread TRL
calls from.

TRL reuses environment instances across multiple rollouts within a gradient
step and calls reward functions once at the end of the step. Each completed
episode's shaped reward components are pushed onto ``_reward_component_queue``;
the reward functions in ``reward.py`` pop from the front of that queue.
"""

from __future__ import annotations

import asyncio
import collections
import random
import threading

from verirl_env import VerirlAction, verirl_env  # type: ignore

from training.curriculum import ALL_TASKS


def _format_obs(obs) -> str:
    """Convert a ``VerirlObservation`` to a human-readable string for the model.

    Args:
        obs: A ``VerirlObservation`` returned by ``reset()`` or ``step()``.

    Returns:
        A newline-separated string of observation sections (task spec, tool
        output, file summary, status line, and optional episode-done message).
    """
    parts = []
    if obs.task_spec:
        parts.append(f"TASK:\n{obs.task_spec}")
    if obs.tool_stdout:
        parts.append(f"OUTPUT:\n{obs.tool_stdout}")
    if obs.tool_stderr:
        parts.append(f"ERRORS:\n{obs.tool_stderr}")
    if getattr(obs, "current_files", None):
        summary = ", ".join(
            f"{n}({len(s)}B)" for n, s in sorted(obs.current_files.items())
        )
        parts.append(f"Files: {summary}")
    parts.append(
        f"compile={'OK' if obs.compile_ok else 'FAIL'} "
        f"tests={obs.tests_passed}/{obs.tests_total} "
        f"turn={obs.turn_number}/{obs.turn_number + obs.turns_remaining}"
    )
    if obs.done:
        score = _score(obs)
        parts.append(f"EPISODE DONE — final_score={score:.3f}")
    return "\n\n".join(parts)


def _score(obs) -> float:
    """Extract the final score from an observation, treating missing values as 0.

    Args:
        obs: A ``VerirlObservation`` (terminal or non-terminal).

    Returns:
        Float score in [0.0, 1.0]; 0.0 when ``final_score`` is absent or None.
    """
    s = getattr(obs, "final_score", None)
    return float(s) if s is not None else 0.0


def _reward_components(obs, *, submitted: bool, tool_calls: int) -> dict[str, float]:
    """Compute shaped reward components from a terminal observation.

    The four components map directly to the four reward functions registered
    with GRPOTrainer:

    - ``tool``:    1.0 if the agent called ``submit``; otherwise a partial
                   credit proportional to how many tool calls were made.
    - ``compile``: 1.0 if the current design compiles cleanly, else 0.0.
    - ``sim``:     Fraction of simulation test cases that passed.
    - ``final``:   The weighted EDA-tool score from ``submit`` in [0.01, 0.99].

    Args:
        obs: Terminal ``VerirlObservation`` (``obs.done == True``).
        submitted: Whether the last action was ``submit``.
        tool_calls: Total number of tool calls made in this episode.

    Returns:
        A dict with keys ``"tool"``, ``"compile"``, ``"sim"``, ``"final"``.
    """
    tests_total = int(getattr(obs, "tests_total", 0) or 0)
    tests_passed = int(getattr(obs, "tests_passed", 0) or 0)
    sim = tests_passed / tests_total if tests_total > 0 else 0.0
    final = max(0.01, min(0.99, _score(obs)))
    tool = 1.0 if submitted else min(0.5, tool_calls / 6.0)
    return {
        "tool":    float(tool),
        "compile": 1.0 if bool(getattr(obs, "compile_ok", False)) else 0.0,
        "sim":     float(max(0.0, min(1.0, sim))),
        "final":   float(final),
    }


def make_env_class(env_url: str):
    """Return a ``VerirlToolEnv`` class with ``env_url`` baked into its closure.

    This factory is used as the ``environment_factory`` argument to
    ``GRPOTrainer``. TRL calls the returned class with no arguments to
    instantiate new environments.

    Args:
        env_url: Base URL of the running VeriRL environment server.

    Returns:
        The ``VerirlToolEnv`` class, ready to pass to ``GRPOTrainer``.
    """

    class VerirlToolEnv:
        """Single VeriRL episode environment for TRL's agentic GRPO loop.

        Each instance manages one concurrent WebSocket connection to the VeriRL
        server. The connection is reset at the start of each episode via
        ``reset()``. Tool methods (``write_file``, ``run_compile``, etc.) map
        directly to VeriRL action types and return the formatted observation
        string consumed by the LLM at the next turn.
        """

        def __init__(self) -> None:
            """Initialise the environment and start its dedicated asyncio event loop."""
            self._loop = asyncio.new_event_loop()
            self._loop_thread = threading.Thread(
                target=self._loop.run_forever,
                daemon=True,
                name=f"verirl-env-{id(self)}",
            )
            self._loop_thread.start()

            self._client = verirl_env(base_url=env_url)
            self._reward_queue: collections.deque = collections.deque()
            self._reward_component_queue: collections.deque = collections.deque()
            self.reward: float = 0.0
            self.done: bool = False
            self.task_id: str = ""
            self.last_verilog_src: str = ""
            self._tool_calls: int = 0
            # Latest observation state — updated after every step for partial rewards
            self._compile_ok: bool = False
            self._tests_passed: int = 0
            self._tests_total: int = 0

        def _enqueue_rewards(self, components: dict[str, float]) -> float:
            """Push terminal reward components onto the queue and mark the episode done.

            Args:
                components: Dict with ``"tool"``, ``"compile"``, ``"sim"``, and
                    ``"final"`` reward values for this episode.

            Returns:
                The ``"final"`` reward value.
            """
            r = float(components["final"])
            self._reward_component_queue.append(components)
            self._reward_queue.append(r)
            self.reward = r
            self.done = True
            return r

        def _run(self, coro):
            """Submit a coroutine to this environment's dedicated event loop and block.

            Args:
                coro: An awaitable to execute on ``self._loop``.

            Returns:
                The coroutine's return value.
            """
            return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

        def _step(self, action: VerirlAction) -> str:
            """Execute an action and enqueue rewards if the server auto-grades.

            Called by all public tool methods. Increments the tool-call counter,
            executes the action over WebSocket, and enqueues reward components
            when the server signals ``obs.done``.

            Args:
                action: The VeriRL action to execute.

            Returns:
                Formatted observation string for the model.
            """
            self._tool_calls += 1
            result = self._run(self._client.step(action))
            obs = result.observation
            # Always track latest state so partial_reward_components() is accurate
            self._compile_ok = bool(getattr(obs, "compile_ok", False))
            self._tests_passed = int(getattr(obs, "tests_passed", 0) or 0)
            self._tests_total = int(getattr(obs, "tests_total", 0) or 0)
            if obs.done and not self.done:
                components = _reward_components(
                    obs,
                    submitted=action.action_type == "submit",
                    tool_calls=self._tool_calls,
                )
                r = self._enqueue_rewards(components)
                print(
                    f"[auto-grade] env_id={id(self)} task={self.task_id} "
                    f"raw={obs.final_score} r={r:.3f} components={components} "
                    f"action={action.action_type}"
                )
            return _format_obs(obs)

        def reset(self, task_id: str | None = None, **kwargs) -> str:
            """Reset the environment and start a new episode.

            Closes any existing WebSocket connection, reconnects to the server,
            and calls ``reset`` for the given task.

            Args:
                task_id: Task to reset to. If ``None``, a random task is chosen
                    from ``ALL_TASKS``.
                **kwargs: Ignored; present for TRL compatibility.

            Returns:
                Formatted observation string for the initial task specification.
            """
            try:
                self._run(self._client.close())
            except Exception:
                pass
            self._client = verirl_env(base_url=env_url)
            self.reward = 0.0
            self.done = False
            self.task_id = task_id or random.choice(ALL_TASKS)
            self.last_verilog_src = ""
            self._tool_calls = 0
            self._compile_ok = False
            self._tests_passed = 0
            self._tests_total = 0
            result = self._run(self._client.reset(task_id=self.task_id))
            return _format_obs(result.observation)

        def partial_reward_components(self) -> dict[str, float]:
            """Shaped reward from the latest obs state for non-terminal episodes.

            Called by ``reward.py`` when no completed-episode entry is in the
            queue. Gives the model a learning signal even when it ran out of
            turns without calling submit, or when TRL samples rewards mid-episode.

            Returns:
                Dict with ``"tool"``, ``"compile"``, ``"sim"``, ``"final"`` keys.
            """
            sim = (
                self._tests_passed / self._tests_total
                if self._tests_total > 0
                else 0.0
            )
            # No submit → tool credit capped at 0.5; scale by how many tools used
            tool = min(0.5, self._tool_calls / 6.0)
            return {
                "tool":    float(tool),
                "compile": 1.0 if self._compile_ok else 0.0,
                "sim":     float(max(0.0, min(1.0, sim))),
                "final":   0.01,  # minimum — no submit grade yet
            }

        def write_file(self, filename: str, verilog_src: str) -> str:
            """Write a Verilog source file to the design workspace.

            Args:
                filename: Target filename, e.g. ``'design.v'`` or ``'submodule.v'``.
                verilog_src: Complete, synthesizable Verilog module source.

            Returns:
                Updated workspace status observation.
            """
            print(f"[env] task={self.task_id} tool=write_file filename={filename} src_len={len(verilog_src)}")
            self.last_verilog_src = verilog_src
            return self._step(VerirlAction(
                action_type="write_file",
                filename=filename,
                verilog_src=verilog_src,
            ))

        def run_compile(self) -> str:
            """Compile the workspace to check for syntax errors.

            Returns:
                Compiler stdout/stderr. Read all errors before attempting a fix.
            """
            print(f"[env] task={self.task_id} tool=run_compile")
            return self._step(VerirlAction(action_type="run_compile"))

        def run_sim(self) -> str:
            """Run testbench simulation against the compiled design.

            Returns:
                Simulation results showing PASS/FAIL for each test case.
            """
            print(f"[env] task={self.task_id} tool=run_sim")
            return self._step(VerirlAction(action_type="run_sim"))

        def run_synth(self) -> str:
            """Run logic synthesis to verify the design is synthesizable.

            Returns:
                Synthesis report with resource usage, warnings, and errors.
            """
            print(f"[env] task={self.task_id} tool=run_synth")
            return self._step(VerirlAction(action_type="run_synth"))

        def run_formal(self) -> str:
            """Run formal verification on the design via SymbiYosys.

            Returns:
                Formal verification results (PASS / bounded / counterexample found).
            """
            print(f"[env] task={self.task_id} tool=run_formal")
            return self._step(VerirlAction(action_type="run_formal"))

        def submit(self) -> str:
            """Submit the design for final grading. Must be called at the end of every episode.

            If the episode is already complete (e.g. the server auto-graded on
            a prior ``run_sim`` call), returns immediately without a second
            server round-trip.

            Returns:
                Final score and a summary of test results, or a no-op message
                if the episode was already scored.
            """
            print(f"[env] task={self.task_id} tool=submit env_id={id(self)} done={self.done}")
            if self.done:
                print(
                    f"[submit-noop] env_id={id(self)} task={self.task_id} "
                    f"r={self.reward:.3f} queue_size={len(self._reward_component_queue)}"
                )
                return f"Episode already complete. Score: {self.reward:.3f}"

            try:
                self._tool_calls += 1
                result = self._run(self._client.step(VerirlAction(action_type="submit")))
            except Exception as exc:
                print(f"[submit-error] env_id={id(self)} task={self.task_id} error={exc}")
                components = {"tool": 1.0, "compile": 0.0, "sim": 0.0, "final": 0.01}
                self._enqueue_rewards(components)
                return f"Submit failed: {exc}"

            obs = result.observation
            components = _reward_components(obs, submitted=True, tool_calls=self._tool_calls)
            r = self._enqueue_rewards(components)
            print(
                f"[submit-ok] env_id={id(self)} task={self.task_id} "
                f"raw={obs.final_score} r={r:.3f} components={components} "
                f"queue_size={len(self._reward_component_queue)}"
            )
            return _format_obs(obs)

    return VerirlToolEnv
