# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
VeriRL Environment Implementation.

Server-side environment for evaluating Verilog RTL submissions against
real EDA tools (iverilog + yosys). Agents write hardware descriptions and
receive compilation feedback, simulation results, synthesis area, and final
weighted scores.
"""

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
from uuid import uuid4

# Support both in-repo and standalone imports
try:
    # In-repo imports (when running from OpenEnv repository)
    from openenv.core.env_server.interfaces import Environment

    from ..models import VerirlAction, VerirlObservation, VerirlState
    from .evaluator import VerilogEvaluator
except ImportError:
    from models import VerirlAction, VerirlObservation, VerirlState

    # Standalone imports (when environment is standalone with openenv from pip)
    from openenv.core.env_server.interfaces import Environment
    from server.evaluator import VerilogEvaluator


@dataclass
class Task:
    """A Verilog design task."""

    id: str
    name: str
    difficulty: str
    max_turns: int
    description: str
    spec: str
    testbench_path: str
    reference_cells: int


# Task metadata — maps task_id to config for _load_tasks
_TASK_CONFIGS = [
    {
        "id": "mac_unit",
        "name": "Pipelined MAC Unit",
        "difficulty": "easy",
        "max_turns": 8,
        "reference_cells": 120,
        "description": (
            "Design a 2-stage pipelined multiply-accumulate unit for signed 8-bit integers "
            "with a 32-bit accumulator."
        ),
        "dir": "task1_mac",
    },
    {
        "id": "axi_fifo",
        "name": "AXI-Stream FIFO",
        "difficulty": "medium",
        "max_turns": 10,
        "reference_cells": 200,
        "description": (
            "Implement a 4-entry AXI-Stream compliant FIFO with correct valid/ready "
            "handshake and backpressure."
        ),
        "dir": "task2_axi_fifo",
    },
    {
        "id": "systolic_array",
        "name": "4x4 Systolic Array",
        "difficulty": "hard",
        "max_turns": 12,
        "reference_cells": 600,
        "description": (
            "Design a weight-stationary 4x4 systolic array for INT8 matrix multiply, "
            "producing all 16 outputs within 7 clock cycles."
        ),
        "dir": "task3_systolic",
    },
]


class VerirlEnvironment(Environment):
    """
    Verilog hardware design environment.

    Agents write Verilog RTL for AI-accelerator primitives and are evaluated
    by real EDA simulation tools (iverilog + yosys). Supports concurrent
    sessions — each instance has fully isolated episode state.

    Available tasks:
    - mac_unit (easy): 2-stage pipelined MAC unit, 8 turns
    - axi_fifo (medium): AXI-Stream FIFO, 10 turns
    - systolic_array (hard): 4x4 weight-stationary systolic array, 12 turns

    Available actions:
    - write_file: Submit Verilog source code
    - run_compile: Check syntax via iverilog
    - run_sim: Functional simulation against testbench
    - run_synth: Synthesis via yosys (area estimation)
    - submit: Finalize episode and compute final score

    Example:
        >>> env = VerirlEnvironment()
        >>> obs = env.reset(task_id="mac_unit")
        >>> print(obs.task_spec)
        >>>
        >>> obs = env.step(VerirlAction(action_type="write_file", verilog_src=...))
        >>> obs = env.step(VerirlAction(action_type="run_compile"))
        >>> obs = env.step(VerirlAction(action_type="submit"))
        >>> print(f"Score: {obs.final_score}")
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(
        self,
        problems_dir: Optional[str] = None,
        max_turns: Optional[int] = None,
    ):
        """
        Initialize the VeriRL environment.

        Args:
            problems_dir: Path to problems directory, or None for auto-detection
            max_turns: Override max turns per episode (None = use task default)
        """
        self.problems_dir = (
            Path(problems_dir) if problems_dir else self._default_problems_dir()
        )
        self._max_turns_override = max_turns
        self.evaluator = VerilogEvaluator()
        self.tasks: Dict[str, Task] = self._load_tasks()

        # Episode state — reset on each call to reset()
        self._state = VerirlState()
        self._current_task: Optional[Task] = None
        self._current_verilog: Optional[str] = None
        self._compile_ok: bool = False
        self._tests_passed: int = 0
        self._tests_total: int = 0
        self._prev_sim_ratio: float = 0.0
        self._total_reward: float = 0.0
        self._turn_number: int = 0
        self._max_turns: int = 10

    def _default_problems_dir(self) -> Path:
        """Find the problems directory relative to this package."""
        env_dir = os.environ.get("VERIRL_PROBLEMS_DIR")
        if env_dir:
            p = Path(env_dir)
            if p.exists():
                return p

        pkg_problems = Path(__file__).parent.parent / "problems"
        if pkg_problems.exists():
            return pkg_problems

        raise FileNotFoundError(
            "No problems directory found. Set VERIRL_PROBLEMS_DIR or "
            "ensure 'problems/' exists in the package directory."
        )

    def _load_tasks(self) -> Dict[str, Task]:
        """Load all tasks from the problems directory."""
        tasks: Dict[str, Task] = {}
        for config in _TASK_CONFIGS:
            task_dir = self.problems_dir / config["dir"]
            if not task_dir.exists():
                continue
            spec_path = task_dir / "spec.md"
            testbench_path = task_dir / "testbench.v"
            if not testbench_path.exists():
                continue
            tasks[config["id"]] = Task(
                id=config["id"],
                name=config["name"],
                difficulty=config["difficulty"],
                max_turns=config["max_turns"],
                description=config["description"],
                spec=spec_path.read_text() if spec_path.exists() else "",
                testbench_path=str(testbench_path),
                reference_cells=config["reference_cells"],
            )
        return tasks

    def reset(self, task_id: Optional[str] = None, **kwargs) -> VerirlObservation:
        """
        Reset the environment and start a new episode.

        Args:
            task_id: Task to run ("mac_unit", "axi_fifo", "systolic_array"),
                     or None for random selection.

        Returns:
            Initial VerirlObservation with the full task specification.
        """
        if task_id is None:
            task_id = random.choice(list(self.tasks.keys()))
        if task_id not in self.tasks:
            raise ValueError(
                f"Unknown task_id '{task_id}'. Valid: {list(self.tasks.keys())}"
            )

        self._current_task = self.tasks[task_id]
        self._max_turns = self._max_turns_override or self._current_task.max_turns
        self._current_verilog = None
        self._compile_ok = False
        self._tests_passed = 0
        self._tests_total = 0
        self._prev_sim_ratio = 0.0
        self._total_reward = 0.0
        self._turn_number = 0

        self._state = VerirlState(
            episode_id=str(uuid4()),
            step_count=0,
            task_id=task_id,
            compile_ok=False,
            tests_passed=0,
            tests_total=0,
            total_reward=0.0,
            turns_remaining=self._max_turns,
            episode_done=False,
        )

        return VerirlObservation(
            task_spec=self._current_task.spec,
            tool_stdout="",
            tool_stderr="",
            compile_ok=False,
            tests_passed=0,
            tests_total=0,
            turn_number=0,
            turns_remaining=self._max_turns,
            current_verilog=None,
            done=False,
            reward=0.0,
        )

    def step(self, action: VerirlAction, **kwargs) -> VerirlObservation:
        """
        Execute one agent action and return the resulting observation.

        Args:
            action: VerirlAction with action_type and optional verilog_src.

        Returns:
            VerirlObservation with tool feedback and updated episode state.
        """
        if self._current_task is None:
            raise RuntimeError("Must call reset() before step().")
        if self._state.episode_done:
            raise RuntimeError(
                "Episode is done. Call reset() to start a new episode."
            )

        tool_stdout = ""
        tool_stderr = ""

        if action.action_type == "write_file":
            if not action.verilog_src:
                tool_stderr = "ERROR: write_file requires the verilog_src field."
            else:
                self._current_verilog = action.verilog_src
                # Reset compilation and simulation state — new code must be re-verified
                self._compile_ok = False
                self._tests_passed = 0
                self._tests_total = 0
                self._prev_sim_ratio = 0.0
                tool_stdout = "File written. Use run_compile to check syntax."

        elif action.action_type == "run_compile":
            if not self._current_verilog:
                tool_stderr = "ERROR: No file written yet. Use write_file first."
            else:
                result = self.evaluator.compile(self._current_verilog)
                self._compile_ok = result.success
                tool_stdout = result.stdout
                tool_stderr = result.stderr
                if result.success:
                    tool_stdout += "\nCompilation successful."

        elif action.action_type == "run_sim":
            if not self._current_verilog:
                tool_stderr = "ERROR: No file written yet. Use write_file first."
            elif not self._compile_ok:
                tool_stderr = (
                    "ERROR: Code does not compile. Fix errors before simulating."
                )
            else:
                result = self.evaluator.simulate(
                    self._current_verilog, self._current_task.testbench_path
                )
                self._tests_passed = result.tests_passed
                self._tests_total = result.tests_total
                tool_stdout = result.stdout
                tool_stderr = result.stderr

        elif action.action_type == "run_synth":
            if not self._current_verilog:
                tool_stderr = "ERROR: No file written yet. Use write_file first."
            elif not self._compile_ok:
                tool_stderr = (
                    "ERROR: Code does not compile. Fix errors before synthesizing."
                )
            else:
                result = self.evaluator.synthesize(self._current_verilog)
                ref = self._current_task.reference_cells
                tool_stdout = (
                    f"Cell count: {result.cell_count} "
                    f"(reference target: {ref} cells — smaller is better up to the reference)\n"
                    + result.stdout
                )
                tool_stderr = result.stderr

        elif action.action_type == "submit":
            return self._handle_submit()

        else:
            tool_stderr = (
                f"ERROR: Unknown action_type '{action.action_type}'. "
                "Valid: write_file, run_compile, run_sim, run_synth, submit"
            )

        # Advance turn counter
        self._turn_number += 1
        self._state.step_count = self._turn_number

        # Check if episode expired
        done = self._turn_number >= self._max_turns
        final_score = None
        score_breakdown = None

        if done:
            eval_result = self.evaluator.grade(
                verilog_src=self._current_verilog or "",
                task_id=self._current_task.id,
                testbench_path=self._current_task.testbench_path,
                reference_cells=self._current_task.reference_cells,
            )
            final_score = eval_result.final_score
            score_breakdown = eval_result.score_breakdown

        reward = self._calculate_step_reward()
        self._total_reward += reward
        self._state.episode_done = done
        self._sync_state(final_score)

        return VerirlObservation(
            task_spec="",
            tool_stdout=tool_stdout,
            tool_stderr=tool_stderr,
            compile_ok=self._compile_ok,
            tests_passed=self._tests_passed,
            tests_total=self._tests_total,
            turn_number=self._turn_number,
            turns_remaining=max(0, self._max_turns - self._turn_number),
            current_verilog=self._current_verilog,
            final_score=final_score,
            score_breakdown=score_breakdown,
            done=done,
            reward=reward,
        )

    def _handle_submit(self) -> VerirlObservation:
        """Grade current submission and end the episode."""
        eval_result = self.evaluator.grade(
            verilog_src=self._current_verilog or "",
            task_id=self._current_task.id,
            testbench_path=self._current_task.testbench_path,
            reference_cells=self._current_task.reference_cells,
        )
        self._turn_number += 1
        self._state.step_count = self._turn_number
        reward = self._calculate_step_reward()
        self._total_reward += reward
        self._state.episode_done = True
        self._sync_state(eval_result.final_score)

        return VerirlObservation(
            task_spec="",
            tool_stdout=f"Final score: {eval_result.final_score:.3f}\n"
            + eval_result.to_agent_feedback(),
            tool_stderr="",
            compile_ok=self._compile_ok,
            tests_passed=self._tests_passed,
            tests_total=self._tests_total,
            turn_number=self._turn_number,
            turns_remaining=0,
            current_verilog=self._current_verilog,
            final_score=eval_result.final_score,
            score_breakdown=eval_result.score_breakdown,
            done=True,
            reward=reward,
        )

    def _calculate_step_reward(self) -> float:
        """
        Per-step reward signal.

        Rewards progress (having code, compiling, passing tests) and
        applies a small time penalty to encourage efficiency.

        Components:
        - +0.02 for having any Verilog on file
        - +0.05 for a clean compile
        - +0.10 × (tests_passed / tests_total) for absolute test ratio
        - +0.15 × improvement in test ratio vs previous sim run (delta bonus)
        - -min(0.01 × turn_number, 0.05) time penalty (capped at 0.05)
        """
        reward = 0.0
        if self._current_verilog:
            reward += 0.02
        if self._compile_ok:
            reward += 0.05
        if self._tests_total > 0:
            current_ratio = self._tests_passed / self._tests_total
            reward += 0.10 * current_ratio
            # Delta bonus: reward improvement over the last sim run
            delta = current_ratio - self._prev_sim_ratio
            if delta > 0:
                reward += 0.15 * delta
            self._prev_sim_ratio = current_ratio
        # Time penalty capped at 0.05 so later turns aren't over-penalised
        reward -= min(0.01 * self._turn_number, 0.05)
        return max(reward, -0.05)

    def _sync_state(self, final_score: Optional[float]) -> None:
        """Update the public state object from internal episode variables."""
        self._state.compile_ok = self._compile_ok
        self._state.tests_passed = self._tests_passed
        self._state.tests_total = self._tests_total
        self._state.total_reward = self._total_reward
        self._state.turns_remaining = max(0, self._max_turns - self._turn_number)
        self._state.final_score = final_score

    @property
    def state(self) -> VerirlState:
        """Get current environment state."""
        return self._state

    @property
    def done(self) -> bool:
        """Check if episode is done."""
        return self._state.episode_done

    @property
    def reward(self) -> float:
        """Reward is returned per-step; this property satisfies the interface."""
        return 0.0

    def list_tasks(self) -> List[str]:
        """List all available task IDs."""
        return list(self.tasks.keys())

    @property
    def num_tasks(self) -> int:
        """Number of available tasks."""
        return len(self.tasks)
