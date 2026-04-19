"""VeriRL tool environment for TRL's environment_factory API."""

from __future__ import annotations

import random

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
        score = getattr(obs, "final_score", None) or 0.01
        parts.append(f"EPISODE DONE — final_score={score:.3f}")
    return "\n\n".join(parts)


def make_env_class(env_url: str):
    """
    Return a VerirlToolEnv class with env_url baked into its closure.

    TRL's environment_factory expects a no-arg constructor, so the URL cannot
    be passed at instantiation time — it is captured here instead.

    TRL auto-discovers every public method (other than reset) as a callable
    tool, building a JSON schema from the typed signatures and docstrings.
    After each episode, reward_func reads self.reward from the instance.
    """

    class VerirlToolEnv:
        def __init__(self) -> None:
            self._client = verirl_env(base_url=env_url)
            self.reward: float = 0.0
            self.done: bool = False
            # Tracked for the evolution buffer in reward_func
            self.task_id: str = ""
            self.last_verilog_src: str = ""

        def _step(self, action: VerirlAction) -> str:
            """Execute an action, update done/reward if the episode auto-terminates."""
            result = self._client.step(action)
            obs = result.observation
            # The server auto-grades when max_turns is exhausted — detect that
            # here so reward is always set, even if the model never calls submit.
            if obs.done and not self.done:
                self.reward = max(0.01, min(0.99, float(obs.final_score or 0.01)))
                self.done = True
            return _format_obs(obs)

        def reset(self, task_id: str | None = None, **kwargs) -> str:
            """Reset the environment for a new episode."""
            try:
                self._client.close()
            except Exception:
                pass
            self._client = verirl_env(base_url=env_url)
            self.reward = 0.0
            self.done = False
            self.task_id = task_id or random.choice(ALL_TASKS)
            self.last_verilog_src = ""
            result = self._client.reset(task_id=self.task_id)
            return _format_obs(result.observation)

        def write_file(self, filename: str, verilog_src: str) -> str:
            """Write a Verilog source file to the design workspace.

            Args:
                filename: Target filename, e.g. 'design.v' or 'submodule.v'
                verilog_src: The complete, synthesizable Verilog module source

            Returns:
                Updated workspace status after the write.
            """
            self.last_verilog_src = verilog_src  # track for evolution buffer
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
            return self._step(VerirlAction(action_type="run_compile"))

        def run_sim(self) -> str:
            """Run testbench simulation against the compiled design.

            Returns:
                Simulation results showing PASS/FAIL for each test case.
            """
            return self._step(VerirlAction(action_type="run_sim"))

        def run_synth(self) -> str:
            """Run logic synthesis to verify the design is synthesizable.

            Returns:
                Synthesis report with resource usage, warnings, and errors.
            """
            return self._step(VerirlAction(action_type="run_synth"))

        def run_formal(self) -> str:
            """Run formal verification on the design.

            Returns:
                Formal verification results (PASS / bounded / counterexample found).
            """
            return self._step(VerirlAction(action_type="run_formal"))

        def submit(self) -> str:
            """Submit the design for final grading. Call when tests pass or turns are low.

            Returns:
                Final score and a summary of test results.
            """
            if self.done:
                raise ValueError("Design already submitted for this episode.")
            result = self._client.step(VerirlAction(action_type="submit"))
            obs = result.observation
            self.reward = max(0.01, min(0.99, float(obs.final_score or 0.01)))
            self.done = True
            return _format_obs(obs)

    return VerirlToolEnv
