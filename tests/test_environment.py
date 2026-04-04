"""Tests for VerirlEnvironment."""

import pytest
from verirl_env.models import VerirlAction
from verirl_env.server.verirl_env_environment import VerirlEnvironment


class TestEnvironmentReset:
    """Test environment reset."""

    def test_reset_mac_unit(self, environment):
        obs = environment.reset(task_id="mac_unit")
        assert obs.task_spec != ""
        assert "mac" in obs.task_spec.lower()
        assert obs.compile_ok is False
        assert obs.tests_passed == 0
        assert obs.turn_number == 0
        assert obs.turns_remaining == 8
        assert obs.done is False

    def test_reset_axi_fifo(self, environment):
        obs = environment.reset(task_id="axi_fifo")
        assert obs.task_spec != ""
        assert "fifo" in obs.task_spec.lower()
        assert obs.turns_remaining == 10

    def test_reset_systolic_array(self, environment):
        obs = environment.reset(task_id="systolic_array")
        assert obs.task_spec != ""
        assert "systolic" in obs.task_spec.lower()
        assert obs.turns_remaining == 12

    def test_reset_random_task(self, environment):
        obs = environment.reset(task_id=None)
        assert obs.task_spec != ""
        assert obs.turns_remaining > 0


class TestEnvironmentStep:
    """Test environment steps."""

    def test_write_file_step(self, environment):
        environment.reset(task_id="mac_unit")
        obs = environment.step(VerirlAction(
            action_type="write_file",
            verilog_src="module mac_unit (input a); endmodule"
        ))
        assert obs.current_verilog is not None
        assert obs.turn_number == 1
        assert obs.turns_remaining == 7

    def test_write_empty_file(self, environment):
        environment.reset(task_id="mac_unit")
        obs = environment.step(VerirlAction(
            action_type="write_file",
            verilog_src=""
        ))
        assert "ERROR" in obs.tool_stderr

    def test_run_compile_no_file(self, environment):
        environment.reset(task_id="mac_unit")
        obs = environment.step(VerirlAction(action_type="run_compile"))
        assert "ERROR" in obs.tool_stderr

    def test_run_compile_after_write(self, environment, mac_reference_verilog, requires_iverilog):
        environment.reset(task_id="mac_unit")
        environment.step(VerirlAction(
            action_type="write_file",
            verilog_src=mac_reference_verilog
        ))
        obs = environment.step(VerirlAction(action_type="run_compile"))
        assert obs.compile_ok is True
        assert "successful" in obs.tool_stdout.lower()

    def test_run_sim_without_compile(self, environment, mac_reference_verilog):
        environment.reset(task_id="mac_unit")
        environment.step(VerirlAction(
            action_type="write_file",
            verilog_src=mac_reference_verilog
        ))
        obs = environment.step(VerirlAction(action_type="run_sim"))
        assert "ERROR" in obs.tool_stderr

    def test_run_sim_after_compile(self, environment, mac_reference_verilog, requires_eda_tools):
        environment.reset(task_id="mac_unit")
        environment.step(VerirlAction(
            action_type="write_file",
            verilog_src=mac_reference_verilog
        ))
        environment.step(VerirlAction(action_type="run_compile"))
        obs = environment.step(VerirlAction(action_type="run_sim"))
        assert obs.tests_total > 0
        assert obs.tests_passed > 0


class TestEnvironmentReward:
    """Test reward calculation."""

    def test_reward_write_file(self, environment, mac_reference_verilog):
        environment.reset(task_id="mac_unit")
        obs = environment.step(VerirlAction(
            action_type="write_file",
            verilog_src=mac_reference_verilog
        ))
        assert obs.reward > 0  # should get positive reward for writing

    def test_reward_compile(self, environment, mac_reference_verilog, requires_iverilog):
        environment.reset(task_id="mac_unit")
        environment.step(VerirlAction(
            action_type="write_file",
            verilog_src=mac_reference_verilog
        ))
        obs = environment.step(VerirlAction(action_type="run_compile"))
        assert obs.reward > 0.02  # compile adds to reward

    def test_reward_sim_passing(self, environment, mac_reference_verilog, requires_eda_tools):
        environment.reset(task_id="mac_unit")
        environment.step(VerirlAction(
            action_type="write_file",
            verilog_src=mac_reference_verilog
        ))
        environment.step(VerirlAction(action_type="run_compile"))
        obs = environment.step(VerirlAction(action_type="run_sim"))
        assert obs.reward > 0.0  # Should have some reward from sim/tests


class TestEnvironmentDone:
    """Test episode termination."""

    def test_submit_ends_episode(self, environment, mac_reference_verilog):
        environment.reset(task_id="mac_unit")
        environment.step(VerirlAction(
            action_type="write_file",
            verilog_src=mac_reference_verilog
        ))
        environment.step(VerirlAction(action_type="run_compile"))
        obs = environment.step(VerirlAction(action_type="submit"))
        assert obs.done is True
        assert obs.final_score is not None

    def test_max_turns_expires(self, environment, mac_reference_verilog):
        env = VerirlEnvironment(max_turns=2)
        env.reset(task_id="mac_unit")
        env.step(VerirlAction(action_type="write_file", verilog_src=mac_reference_verilog))
        obs = env.step(VerirlAction(action_type="run_compile"))
        assert obs.done is True  # episode expires at max_turns
        assert obs.final_score is not None


class TestTaskSelection:
    """Test task metadata and selection."""

    def test_list_tasks(self, environment):
        tasks = environment.list_tasks()
        assert "mac_unit" in tasks
        assert "axi_fifo" in tasks
        assert "systolic_array" in tasks

    def test_num_tasks(self, environment):
        assert environment.num_tasks == 3

    def test_invalid_task_id(self, environment):
        with pytest.raises(ValueError):
            environment.reset(task_id="invalid_task")
