"""Tests for Pydantic models."""

import pytest
from verirl_env.models import VerirlAction, VerirlObservation, VerirlState


class TestVerirlAction:
    """Test VerirlAction model."""

    def test_write_file_action(self):
        action = VerirlAction(action_type="write_file", verilog_src="module foo; endmodule")
        assert action.action_type == "write_file"
        assert action.verilog_src == "module foo; endmodule"
        assert action.message is None

    def test_write_file_with_message(self):
        action = VerirlAction(
            action_type="write_file",
            verilog_src="module bar; endmodule",
            message="Initial implementation"
        )
        assert action.message == "Initial implementation"

    def test_run_compile_action(self):
        action = VerirlAction(action_type="run_compile", message="Checking syntax")
        assert action.action_type == "run_compile"
        assert action.verilog_src is None
        assert action.message == "Checking syntax"

    def test_run_sim_action(self):
        action = VerirlAction(action_type="run_sim")
        assert action.action_type == "run_sim"

    def test_run_synth_action(self):
        action = VerirlAction(action_type="run_synth")
        assert action.action_type == "run_synth"

    def test_submit_action(self):
        action = VerirlAction(action_type="submit", message="Done")
        assert action.action_type == "submit"
        assert action.message == "Done"


class TestVerirlObservation:
    """Test VerirlObservation model."""

    def test_initial_observation(self):
        obs = VerirlObservation(
            task_spec="Design a MAC unit",
            tool_stdout="",
            tool_stderr="",
            compile_ok=False,
            tests_passed=0,
            tests_total=0,
            turn_number=0,
            turns_remaining=8,
            done=False,
            reward=0.0
        )
        assert obs.task_spec == "Design a MAC unit"
        assert obs.compile_ok is False
        assert obs.turn_number == 0
        assert obs.turns_remaining == 8
        assert obs.final_score is None

    def test_observation_with_results(self):
        obs = VerirlObservation(
            task_spec="",
            tool_stdout="Compilation successful",
            tool_stderr="",
            compile_ok=True,
            tests_passed=5,
            tests_total=7,
            turn_number=2,
            turns_remaining=6,
            current_verilog="module foo; endmodule",
            final_score=0.75,
            score_breakdown={"compile": 1.0, "sim": 0.714},
            done=False,
            reward=0.12
        )
        assert obs.compile_ok is True
        assert obs.tests_passed == 5
        assert obs.final_score == 0.75
        assert obs.score_breakdown["sim"] == 0.714


class TestVerirlState:
    """Test VerirlState model."""

    def test_default_state(self):
        state = VerirlState()
        assert state.episode_done is False
        assert state.compile_ok is False
        assert state.tests_passed == 0
        assert state.total_reward == 0.0

    def test_state_with_values(self):
        state = VerirlState(
            episode_id="ep123",
            task_id="mac_unit",
            compile_ok=True,
            tests_passed=3,
            tests_total=5,
            total_reward=0.25,
            episode_done=False,
            final_score=0.5
        )
        assert state.episode_id == "ep123"
        assert state.task_id == "mac_unit"
        assert state.compile_ok is True
        assert state.final_score == 0.5
