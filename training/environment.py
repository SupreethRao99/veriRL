"""VeriRL tool environment for TRL's environment_factory API."""

from __future__ import annotations

import asyncio
import collections
import random
import threading

from verirl_env import VerirlAction, verirl_env  # type: ignore

from training.curriculum import ALL_TASKS


def _format_obs(obs) -> str:
    """Convert a VerirlObservation to a readable string for the model."""
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
    """Extract final score from observation, treating None/missing as 0."""
    s = getattr(obs, "final_score", None)
    return float(s) if s is not None else 0.0


def make_env_class(env_url: str):
    """
    Return a VerirlToolEnv class with env_url baked into its closure.

    Each instance owns a dedicated asyncio event loop running in a background
    daemon thread. All client calls (reset, step) are dispatched to that loop
    via run_coroutine_threadsafe, ensuring the WebSocket connection is always
    driven by the same loop regardless of which thread TRL calls from.

    TRL reuses env instances across multiple rollouts within a gradient step and
    calls reward_func once at the end of the step. Each completed episode's reward
    is pushed onto _reward_queue; reward_func pops from the front.
    """

    class VerirlToolEnv:
        def __init__(self) -> None:
            # Dedicated event loop for this env instance — prevents
            # "Future attached to a different loop" errors that occur when
            # TRL's agentic loop calls tool methods from threads that each
            # have their own (or no) event loop.
            self._loop = asyncio.new_event_loop()
            self._loop_thread = threading.Thread(
                target=self._loop.run_forever, daemon=True, name=f"verirl-env-{id(self)}"
            )
            self._loop_thread.start()

            self._client = verirl_env(base_url=env_url)
            self._reward_queue: collections.deque = collections.deque()
            self.reward: float = 0.0
            self.done: bool = False
            self.task_id: str = ""
            self.last_verilog_src: str = ""

        def _run(self, coro):
            """Submit coroutine to this env's dedicated loop and block until done."""
            return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

        def _step(self, action: VerirlAction) -> str:
            """Execute an action; enqueue reward if the server auto-grades."""
            result = self._run(self._client.step(action))
            obs = result.observation
            if obs.done and not self.done:
                r = max(0.01, min(0.99, _score(obs)))
                self._reward_queue.append(r)
                self.reward = r
                self.done = True
                print(f"[auto-grade] env_id={id(self)} task={self.task_id} raw={obs.final_score} r={r:.3f} action={action.action_type}")
            return _format_obs(obs)

        def reset(self, task_id: str | None = None, **kwargs) -> str:
            """Reset the environment for a new episode."""
            try:
                self._run(self._client.close())
            except Exception:
                pass
            self._client = verirl_env(base_url=env_url)
            self.reward = 0.0
            self.done = False
            self.task_id = task_id or random.choice(ALL_TASKS)
            self.last_verilog_src = ""
            result = self._run(self._client.reset(task_id=self.task_id))
            return _format_obs(result.observation)

        def write_file(self, filename: str, verilog_src: str) -> str:
            """Write a Verilog source file to the design workspace.

            Args:
                filename: Target filename, e.g. 'design.v' or 'submodule.v'
                verilog_src: The complete, synthesizable Verilog module source

            Returns:
                Updated workspace status after the write.
            """
            print(f"[env] task={self.task_id} tool=write_file filename={filename} src_len={len(verilog_src)}")
            self.last_verilog_src = verilog_src
            return self._step(
                VerirlAction(
                    action_type="write_file",
                    filename=filename,
                    verilog_src=verilog_src,
                )
            )

        def run_compile(self) -> str:
            """Compile the design to check for syntax errors.

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
            """Run formal verification on the design.

            Returns:
                Formal verification results (PASS / bounded / counterexample found).
            """
            print(f"[env] task={self.task_id} tool=run_formal")
            return self._step(VerirlAction(action_type="run_formal"))

        def submit(self) -> str:
            """Submit the design for final grading. Call when tests pass or turns are low.

            Returns:
                Final score and a summary of test results.
            """
            print(f"[env] task={self.task_id} tool=submit env_id={id(self)} done={self.done}")
            if self.done:
                # Episode already scored (e.g. server auto-graded on run_sim).
                # Return a no-op so TRL doesn't enter exception-handling mode.
                print(f"[submit-noop] env_id={id(self)} task={self.task_id} r={self.reward:.3f} queue_size={len(self._reward_queue)}")
                return f"Episode already complete. Score: {self.reward:.3f}"
            try:
                result = self._run(self._client.step(VerirlAction(action_type="submit")))
            except Exception as exc:
                print(f"[submit-error] env_id={id(self)} task={self.task_id} error={exc}")
                self._reward_queue.append(0.01)
                self.reward = 0.01
                self.done = True
                return f"Submit failed: {exc}"
            obs = result.observation
            r = max(0.01, min(0.99, _score(obs)))
            self._reward_queue.append(r)
            self.reward = r
            self.done = True
            print(f"[submit-ok] env_id={id(self)} task={self.task_id} raw={obs.final_score} r={r:.3f} queue_size={len(self._reward_queue)}")
            return _format_obs(obs)

    return VerirlToolEnv
