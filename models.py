# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the VeriRL Verilog hardware design environment.

Agents write and verify Verilog RTL for AI-accelerator hardware primitives,
receiving feedback from real EDA tools (iverilog, yosys).
"""

from typing import Dict, Optional

from pydantic import Field

try:
    from openenv.core.env_server.types import Action, Observation, State
except ImportError:
    from openenv.core.env_server.types import Action, Observation, State


class VerirlAction(Action):
    """Action for the VeriRL environment — Verilog code submission and EDA tool invocations."""

    action_type: str = Field(
        ...,
        description=(
            "One of: write_file, run_compile, run_sim, run_synth, "
            "run_formal, list_files, submit"
        ),
    )
    verilog_src: Optional[str] = Field(
        default=None, description="Verilog source code (required for write_file)"
    )
    filename: Optional[str] = Field(
        default=None,
        description=(
            "Target filename for write_file (e.g. 'pe.v', 'top.v'). "
            "Defaults to 'design.v' if omitted."
        ),
    )
    message: Optional[str] = Field(
        default=None, description="Agent reasoning note (logged but not graded)"
    )


class VerirlObservation(Observation):
    """Observation from the VeriRL environment — EDA tool feedback and episode status."""

    task_spec: str = Field(default="", description="Full task specification (markdown)")
    tool_stdout: str = Field(default="", description="EDA tool stdout output")
    tool_stderr: str = Field(default="", description="EDA tool stderr / error messages")
    compile_ok: bool = Field(default=False, description="Whether current code compiles")
    tests_passed: int = Field(default=0, ge=0, description="Simulation tests passed")
    tests_total: int = Field(default=0, ge=0, description="Total simulation tests")
    turn_number: int = Field(default=0, ge=0, description="Current turn number")
    turns_remaining: int = Field(default=0, ge=0, description="Turns remaining in episode")
    current_verilog: Optional[str] = Field(
        default=None,
        description="Primary Verilog source (design.v); use current_files for multi-file projects",
    )
    current_files: Optional[Dict[str, str]] = Field(
        default=None,
        description="All files currently on disk: {filename: source_code}",
    )
    formal_properties_proven: Optional[int] = Field(
        default=None, description="Number of formal properties proven (run_formal)"
    )
    formal_properties_total: Optional[int] = Field(
        default=None, description="Total formal properties checked"
    )
    final_score: Optional[float] = Field(
        default=None, description="Final score in [0.01, 0.99] (set on submit or episode expiry)"
    )
    score_breakdown: Optional[Dict[str, float]] = Field(
        default=None, description="Per-dimension scores: compile, sim, timing, area, formal"
    )


class VerirlState(State):
    """State for the VeriRL environment."""

    task_id: Optional[str] = Field(default=None, description="Current task ID")
    compile_ok: bool = Field(default=False, description="Whether code currently compiles")
    tests_passed: int = Field(default=0, ge=0, description="Simulation tests passed")
    tests_total: int = Field(default=0, ge=0, description="Total simulation tests")
    total_reward: float = Field(default=0.0, description="Cumulative reward this episode")
    turns_remaining: int = Field(default=0, ge=0, description="Turns remaining")
    episode_done: bool = Field(default=False, description="Whether episode is over")
    final_score: Optional[float] = Field(
        default=None, description="Final score in [0.01, 0.99] if episode is done"
    )
