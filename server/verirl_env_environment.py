# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
VeriRL Environment Implementation.

Server-side environment for evaluating Verilog RTL submissions against
real EDA tools (iverilog + yosys + SymbiYosys). Agents write hardware
descriptions and receive compilation feedback, simulation results,
synthesis area, formal verification results, and final weighted scores.

Multi-file support: agents may write multiple named source files;
all files are compiled and simulated together.
"""

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
from uuid import uuid4

try:
    from openenv.core.env_server.interfaces import Environment
    from ..models import VerirlAction, VerirlObservation, VerirlState
    from .evaluator import VerilogEvaluator
except ImportError:
    from models import VerirlAction, VerirlObservation, VerirlState
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
    properties_path: Optional[str] = None  # SymbiYosys formal properties


# ---------------------------------------------------------------------------
# Task metadata — maps to problems/<dir>/
# ---------------------------------------------------------------------------

_TASK_CONFIGS = [
    # ── Original 3 tasks ──────────────────────────────────────────────────
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
    # ── 7 new tasks ───────────────────────────────────────────────────────
    {
        "id": "relu_clip",
        "name": "Parameterized ReLU-Clip Unit",
        "difficulty": "easy",
        "max_turns": 6,
        "reference_cells": 15,
        "description": (
            "Design a fully combinational ReLU activation followed by saturating cast "
            "to a narrower unsigned integer — the activation primitive in every quantized "
            "neural-network inference pipeline."
        ),
        "dir": "task4_relu_clip",
        "has_formal": True,
    },
    {
        "id": "barrel_shifter",
        "name": "Parameterized Barrel Shifter",
        "difficulty": "easy",
        "max_turns": 6,
        "reference_cells": 30,
        "description": (
            "Design a fully combinational barrel shifter supporting left shift, "
            "logical right shift, and arithmetic right shift for any WIDTH."
        ),
        "dir": "task5_barrel_shifter",
    },
    {
        "id": "register_file",
        "name": "Dual-Read / Single-Write Register File",
        "difficulty": "medium",
        "max_turns": 8,
        "reference_cells": 150,
        "description": (
            "Implement a RISC-style register file with two asynchronous read ports, "
            "one synchronous write port, and register 0 hardwired to zero."
        ),
        "dir": "task6_register_file",
        "has_formal": True,
    },
    {
        "id": "ring_buffer",
        "name": "Parameterized Ring Buffer",
        "difficulty": "medium",
        "max_turns": 10,
        "reference_cells": 80,
        "description": (
            "Implement a power-of-2-depth ring buffer (circular FIFO) with push/pop "
            "interface, full/empty flags, and item count — the standard data structure "
            "for KV-cache management in inference accelerators."
        ),
        "dir": "task7_ring_buffer",
    },
    {
        "id": "dot_product",
        "name": "Pipelined 4-Element INT8 Dot Product",
        "difficulty": "medium",
        "max_turns": 8,
        "reference_cells": 100,
        "description": (
            "Design a 2-stage pipelined unit computing the dot product of two 4-element "
            "signed INT8 vectors — the innermost operation in attention score and "
            "fully-connected layer computation."
        ),
        "dir": "task8_dot_product",
    },
    {
        "id": "fir_filter",
        "name": "3-Tap FIR Filter",
        "difficulty": "medium",
        "max_turns": 10,
        "reference_cells": 80,
        "description": (
            "Implement a direct-form I 3-tap FIR filter for signed 8-bit samples with "
            "programmable coefficients — y[n] = h0·x[n] + h1·x[n-1] + h2·x[n-2]."
        ),
        "dir": "task9_fir_filter",
    },
    {
        "id": "fp16_adder",
        "name": "IEEE 754 FP16 Adder",
        "difficulty": "hard",
        "max_turns": 15,
        "reference_cells": 250,
        "description": (
            "Implement a combinational IEEE 754 half-precision (FP16) floating-point "
            "adder handling normal numbers, zero, infinity, and NaN — the arithmetic "
            "primitive of every GPU tensor core."
        ),
        "dir": "task10_fp16_adder",
        "has_formal": True,
    },
]


class VerirlEnvironment(Environment):
    """
    Verilog hardware design environment.

    Agents write synthesizable Verilog RTL for AI-accelerator primitives and are
    evaluated by real EDA tools (iverilog + yosys, optionally SymbiYosys).
    Supports multi-file projects and concurrent sessions.

    Available tasks (10):
      easy:        mac_unit, relu_clip, barrel_shifter
      medium:      axi_fifo, register_file, ring_buffer, dot_product, fir_filter
      hard:        systolic_array, fp16_adder

    Available actions:
      write_file   — write (or overwrite) a named Verilog source file
      run_compile  — check syntax of all files via iverilog
      run_sim      — functional simulation against testbench
      run_synth    — synthesis via yosys (area estimation)
      run_formal   — formal verification via SymbiYosys (if available)
      list_files   — show all files currently on disk
      submit       — finalize episode and compute final score
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(
        self,
        problems_dir: Optional[str] = None,
        max_turns: Optional[int] = None,
    ):
        self.problems_dir = (
            Path(problems_dir) if problems_dir else self._default_problems_dir()
        )
        self._max_turns_override = max_turns
        self.evaluator = VerilogEvaluator()
        self.tasks: Dict[str, Task] = self._load_tasks()

        # Episode state — reset on each reset()
        self._state = VerirlState()
        self._current_task: Optional[Task] = None
        self._files: Dict[str, str] = {}          # filename → source
        self._compile_ok: bool = False
        self._tests_passed: int = 0
        self._tests_total: int = 0
        self._prev_sim_ratio: float = 0.0
        self._formal_proven: int = 0
        self._formal_total: int = 0
        self._total_reward: float = 0.0
        self._turn_number: int = 0
        self._max_turns: int = 10

    def _default_problems_dir(self) -> Path:
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
        tasks: Dict[str, Task] = {}
        for config in _TASK_CONFIGS:
            task_dir = self.problems_dir / config["dir"]
            if not task_dir.exists():
                continue
            testbench_path = task_dir / "testbench.v"
            if not testbench_path.exists():
                continue
            spec_path = task_dir / "spec.md"
            props_path = task_dir / "properties.sv"
            tasks[config["id"]] = Task(
                id=config["id"],
                name=config["name"],
                difficulty=config["difficulty"],
                max_turns=config["max_turns"],
                description=config["description"],
                spec=spec_path.read_text() if spec_path.exists() else "",
                testbench_path=str(testbench_path),
                reference_cells=config["reference_cells"],
                properties_path=str(props_path) if props_path.exists() else None,
            )
        return tasks

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    def reset(self, task_id: Optional[str] = None, **kwargs) -> VerirlObservation:
        """Reset the environment and start a new episode."""
        if task_id is None:
            task_id = random.choice(list(self.tasks.keys()))
        if task_id not in self.tasks:
            raise ValueError(
                f"Unknown task_id '{task_id}'. Valid: {list(self.tasks.keys())}"
            )

        self._current_task = self.tasks[task_id]
        self._max_turns = self._max_turns_override or self._current_task.max_turns
        self._files = {}
        self._compile_ok = False
        self._tests_passed = 0
        self._tests_total = 0
        self._prev_sim_ratio = 0.0
        self._formal_proven = 0
        self._formal_total = 0
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
            current_files={},
            done=False,
            reward=0.01,
        )

    def step(self, action: VerirlAction, **kwargs) -> VerirlObservation:
        """Execute one agent action and return the resulting observation."""
        if self._current_task is None:
            raise RuntimeError("Must call reset() before step().")
        if self._state.episode_done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        tool_stdout = ""
        tool_stderr = ""

        if action.action_type == "write_file":
            if not action.verilog_src:
                tool_stderr = "ERROR: write_file requires the verilog_src field."
            else:
                fname = (action.filename or "design.v").strip()
                # Sanitize: strip any path components
                fname = Path(fname).name
                if not fname.endswith(".v") and not fname.endswith(".sv"):
                    fname = fname + ".v"
                self._files[fname] = action.verilog_src
                # Reset compile/sim state — new code must be re-verified
                self._compile_ok = False
                self._tests_passed = 0
                self._tests_total = 0
                self._prev_sim_ratio = 0.0
                n_files = len(self._files)
                file_list = ", ".join(sorted(self._files.keys()))
                tool_stdout = (
                    f"Wrote '{fname}'. Project now has {n_files} file(s): {file_list}. "
                    f"Use run_compile to check syntax."
                )

        elif action.action_type == "list_files":
            if not self._files:
                tool_stdout = "No files written yet."
            else:
                lines = [f"  {n}: {len(s)} chars" for n, s in sorted(self._files.items())]
                tool_stdout = "Current files:\n" + "\n".join(lines)

        elif action.action_type == "run_compile":
            if not self._files:
                tool_stderr = "ERROR: No files written yet. Use write_file first."
            else:
                result = self.evaluator.compile(self._files)
                self._compile_ok = result.success
                tool_stdout = result.stdout
                tool_stderr = result.stderr
                if result.success:
                    tool_stdout += "\nCompilation successful."

        elif action.action_type == "run_sim":
            if not self._files:
                tool_stderr = "ERROR: No files written yet. Use write_file first."
            elif not self._compile_ok:
                tool_stderr = "ERROR: Code does not compile. Fix errors before simulating."
            else:
                result = self.evaluator.simulate(
                    self._files, self._current_task.testbench_path
                )
                self._tests_passed = result.tests_passed
                self._tests_total = result.tests_total
                tool_stdout = result.stdout
                tool_stderr = result.stderr

        elif action.action_type == "run_synth":
            if not self._files:
                tool_stderr = "ERROR: No files written yet. Use write_file first."
            elif not self._compile_ok:
                tool_stderr = "ERROR: Code does not compile. Fix errors before synthesizing."
            else:
                result = self.evaluator.synthesize(self._files)
                ref = self._current_task.reference_cells
                tool_stdout = (
                    f"Cell count: {result.cell_count} "
                    f"(reference target: {ref} cells — smaller is better up to the reference)\n"
                    + result.stdout
                )
                tool_stderr = result.stderr

        elif action.action_type == "run_formal":
            if not self._files:
                tool_stderr = "ERROR: No files written yet. Use write_file first."
            elif not self._compile_ok:
                tool_stderr = "ERROR: Code does not compile. Fix errors before formal verification."
            elif not self._current_task.properties_path:
                tool_stderr = (
                    f"INFO: No formal properties defined for task '{self._current_task.id}'. "
                    "run_formal is not available for this task."
                )
            else:
                result = self.evaluator.formal_verify(
                    self._files, self._current_task.properties_path
                )
                self._formal_proven = result.properties_proven
                self._formal_total = result.properties_total
                if result.success:
                    tool_stdout = (
                        f"Formal verification PASSED: "
                        f"{result.properties_proven}/{result.properties_total} properties proven.\n"
                        + result.stdout
                    )
                else:
                    tool_stdout = (
                        f"Formal verification FAILED: "
                        f"{result.properties_proven}/{result.properties_total} properties proven.\n"
                        + result.stdout
                    )
                    if result.counterexample:
                        tool_stderr = f"Counterexample found:\n{result.counterexample}"
                    else:
                        tool_stderr = result.stderr

        elif action.action_type == "submit":
            return self._handle_submit()

        else:
            tool_stderr = (
                f"ERROR: Unknown action_type '{action.action_type}'. "
                "Valid: write_file, run_compile, run_sim, run_synth, run_formal, list_files, submit"
            )

        # Advance turn counter
        self._turn_number += 1
        self._state.step_count = self._turn_number
        done = self._turn_number >= self._max_turns

        final_score = None
        score_breakdown = None
        if done:
            eval_result = self.evaluator.grade(
                source=self._files or "",
                task_id=self._current_task.id,
                testbench_path=self._current_task.testbench_path,
                reference_cells=self._current_task.reference_cells,
                properties_path=self._current_task.properties_path,
            )
            final_score = eval_result.final_score
            score_breakdown = eval_result.score_breakdown

        reward = self._calculate_step_reward()
        self._total_reward += reward
        self._state.episode_done = done
        self._sync_state(final_score)

        # current_verilog: primary file for backward compatibility
        primary = self._files.get("design.v") or (
            next(iter(self._files.values())) if self._files else None
        )

        return VerirlObservation(
            task_spec="",
            tool_stdout=tool_stdout,
            tool_stderr=tool_stderr,
            compile_ok=self._compile_ok,
            tests_passed=self._tests_passed,
            tests_total=self._tests_total,
            turn_number=self._turn_number,
            turns_remaining=max(0, self._max_turns - self._turn_number),
            current_verilog=primary,
            current_files=dict(self._files),
            formal_properties_proven=self._formal_proven if self._formal_total else None,
            formal_properties_total=self._formal_total if self._formal_total else None,
            final_score=final_score,
            score_breakdown=score_breakdown,
            done=done,
            reward=reward,
        )

    def _handle_submit(self) -> VerirlObservation:
        """Grade current submission and end the episode."""
        eval_result = self.evaluator.grade(
            source=self._files or "",
            task_id=self._current_task.id,
            testbench_path=self._current_task.testbench_path,
            reference_cells=self._current_task.reference_cells,
            properties_path=self._current_task.properties_path,
        )
        self._turn_number += 1
        self._state.step_count = self._turn_number
        reward = self._calculate_step_reward()
        self._total_reward += reward
        self._state.episode_done = True
        self._sync_state(eval_result.final_score)

        primary = self._files.get("design.v") or (
            next(iter(self._files.values())) if self._files else None
        )
        formal = eval_result.formal

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
            current_verilog=primary,
            current_files=dict(self._files),
            formal_properties_proven=formal.properties_proven if formal else None,
            formal_properties_total=formal.properties_total if formal else None,
            final_score=eval_result.final_score,
            score_breakdown=eval_result.score_breakdown,
            done=True,
            reward=reward,
        )

    def _calculate_step_reward(self) -> float:
        """
        Per-step dense reward signal.

        Components:
        - +0.02  for having any Verilog on file
        - +0.05  for a clean compile
        - +0.10 × (tests_passed / tests_total) for absolute test ratio
        - +0.15 × improvement in test ratio vs previous sim run (delta bonus)
        - +0.05 × (formal_proven / formal_total) if formal was run
        - -min(0.01 × turn_number, 0.05) time penalty (capped)
        """
        reward = 0.0
        if self._files:
            reward += 0.02
        if self._compile_ok:
            reward += 0.05
        if self._tests_total > 0:
            current_ratio = self._tests_passed / self._tests_total
            reward += 0.10 * current_ratio
            delta = current_ratio - self._prev_sim_ratio
            if delta > 0:
                reward += 0.15 * delta
            self._prev_sim_ratio = current_ratio
        if self._formal_total > 0:
            reward += 0.05 * (self._formal_proven / self._formal_total)
        reward -= min(0.01 * self._turn_number, 0.05)
        return max(0.01, min(0.99, reward))

    def _sync_state(self, final_score: Optional[float]) -> None:
        self._state.compile_ok = self._compile_ok
        self._state.tests_passed = self._tests_passed
        self._state.tests_total = self._tests_total
        self._state.total_reward = self._total_reward
        self._state.turns_remaining = max(0, self._max_turns - self._turn_number)
        self._state.final_score = final_score

    @property
    def state(self) -> VerirlState:
        return self._state

    @property
    def done(self) -> bool:
        return self._state.episode_done

    @property
    def reward(self) -> float:
        return 0.0

    def list_tasks(self) -> List[str]:
        return list(self.tasks.keys())

    def list_tasks_by_difficulty(self, difficulty: str) -> List[str]:
        """Return task IDs filtered by difficulty ('easy', 'medium', 'hard')."""
        return [
            tid for tid, t in self.tasks.items()
            if t.difficulty == difficulty
        ]

    @property
    def num_tasks(self) -> int:
        return len(self.tasks)
