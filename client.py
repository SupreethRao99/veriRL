# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
VeriRL Environment Client.

This module provides the client for connecting to a VeriRL Environment server
via WebSocket for persistent multi-step episodes.

Example:
    >>> # Connect to a running server
    >>> with verirl_env(base_url="http://localhost:8000") as client:
    ...     result = client.reset(task_id="mac_unit")
    ...     print(result.observation.task_spec)
    ...
    ...     result = client.step(VerirlAction(
    ...         action_type="write_file",
    ...         verilog_src="module mac_unit(...); endmodule"
    ...     ))
    ...     result = client.step(VerirlAction(action_type="run_compile"))
    ...     result = client.step(VerirlAction(action_type="submit"))
    ...     print(f"Score: {result.observation.final_score}")

Example with Docker:
    >>> client = verirl_env.from_docker_image("verirl_env-env:latest")
    >>> try:
    ...     result = client.reset(task_id="axi_fifo")
    ...     result = client.step(VerirlAction(action_type="write_file", verilog_src=...))
    ...     result = client.step(VerirlAction(action_type="submit"))
    ... finally:
    ...     client.close()
"""

from typing import Any, Dict

# Support both in-repo and standalone imports
try:
    # In-repo imports (when running from OpenEnv repository)
    from openenv.core.client_types import StepResult
    from openenv.core.env_client import EnvClient

    from .models import VerirlAction, VerirlObservation, VerirlState
except ImportError:
    from models import VerirlAction, VerirlObservation, VerirlState

    # Standalone imports (when environment is standalone with openenv from pip)
    from openenv.core.client_types import StepResult
    from openenv.core.env_client import EnvClient


class verirl_env(EnvClient[VerirlAction, VerirlObservation, VerirlState]):
    """
    Client for the VeriRL Verilog hardware design environment.

    Maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step episodes. Each instance has its own
    dedicated environment session on the server.

    Agents submit Verilog code and EDA tool commands, receiving feedback
    including compilation status, simulation results, synthesis area, and
    final scores.

    Example:
        >>> with verirl_env(base_url="http://localhost:8000") as client:
        ...     result = client.reset(task_id="mac_unit")
        ...     result = client.step(VerirlAction(
        ...         action_type="write_file",
        ...         verilog_src="module mac_unit(...); endmodule"
        ...     ))
        ...     result = client.step(VerirlAction(action_type="run_sim"))
        ...     result = client.step(VerirlAction(action_type="submit"))
        ...     print(f"Score: {result.observation.final_score}")
    """

    def _step_payload(self, action: VerirlAction) -> Dict[str, Any]:
        """Convert VerirlAction to JSON payload for step request."""
        payload: Dict[str, Any] = {"action_type": action.action_type}
        if action.verilog_src is not None:
            payload["verilog_src"] = action.verilog_src
        if action.filename is not None:
            payload["filename"] = action.filename
        if action.message is not None:
            payload["message"] = action.message
        return payload

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[VerirlObservation]:
        """Parse server response into StepResult[VerirlObservation]."""
        obs_data = payload.get("observation", {})
        observation = VerirlObservation(
            task_spec=obs_data.get("task_spec", ""),
            tool_stdout=obs_data.get("tool_stdout", ""),
            tool_stderr=obs_data.get("tool_stderr", ""),
            compile_ok=obs_data.get("compile_ok", False),
            tests_passed=obs_data.get("tests_passed", 0),
            tests_total=obs_data.get("tests_total", 0),
            turn_number=obs_data.get("turn_number", 0),
            turns_remaining=obs_data.get("turns_remaining", 0),
            current_verilog=obs_data.get("current_verilog"),
            current_files=obs_data.get("current_files"),
            final_score=obs_data.get("final_score"),
            score_breakdown=obs_data.get("score_breakdown"),
            formal_properties_proven=obs_data.get("formal_properties_proven"),
            formal_properties_total=obs_data.get("formal_properties_total"),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> VerirlState:
        """Parse server response into VerirlState."""
        return VerirlState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            task_id=payload.get("task_id"),
            compile_ok=payload.get("compile_ok", False),
            tests_passed=payload.get("tests_passed", 0),
            tests_total=payload.get("tests_total", 0),
            total_reward=payload.get("total_reward", 0.0),
            turns_remaining=payload.get("turns_remaining", 0),
            episode_done=payload.get("episode_done", False),
            final_score=payload.get("final_score"),
        )
