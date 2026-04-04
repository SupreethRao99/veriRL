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
        description="One of: write_file, run_compile, run_sim, run_synth, submit",
    )
    verilog_src: Optional[str] = Field(
        default=None, description="Verilog source code (required for write_file)"
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
        default=None, description="Current Verilog source on file"
    )
    final_score: Optional[float] = Field(
        default=None, description="Final score in [0, 1] (set on submit or episode expiry)"
    )
    score_breakdown: Optional[Dict[str, float]] = Field(
        default=None, description="Per-dimension scores: compile, sim, timing, area"
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
        default=None, description="Final score if episode is done"
    )
