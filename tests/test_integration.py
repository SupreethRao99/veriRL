"""Integration tests for complete workflows."""

import pytest
from verirl_env.models import VerirlAction


class TestMACWorkflow:
    """Test complete MAC unit workflow."""

    def test_mac_full_episode(self, environment, mac_reference_verilog, requires_eda_tools):
        # Reset
        obs = environment.reset(task_id="mac_unit")
        assert obs.turn_number == 0

        # Write code
        obs = environment.step(VerirlAction(
            action_type="write_file",
            verilog_src=mac_reference_verilog,
            message="Implementing pipelined MAC"
        ))
        assert obs.current_verilog == mac_reference_verilog
        assert obs.turn_number == 1

        # Compile
        obs = environment.step(VerirlAction(action_type="run_compile"))
        assert obs.compile_ok is True
        assert obs.turn_number == 2

        # Simulate
        obs = environment.step(VerirlAction(action_type="run_sim"))
        assert obs.tests_total > 0
        assert obs.turn_number == 3

        # Submit
        obs = environment.step(VerirlAction(action_type="submit"))
        assert obs.done is True
        assert obs.final_score > 0.0  # Should have some score
        assert obs.score_breakdown is not None


class TestAXIWorkflow:
    """Test complete AXI FIFO workflow."""

    def test_axi_full_episode(self, environment, axi_reference_verilog, requires_eda_tools):
        obs = environment.reset(task_id="axi_fifo")

        obs = environment.step(VerirlAction(
            action_type="write_file",
            verilog_src=axi_reference_verilog
        ))
        assert obs.turn_number == 1

        obs = environment.step(VerirlAction(action_type="run_compile"))
        assert obs.compile_ok is True

        obs = environment.step(VerirlAction(action_type="run_sim"))
        assert obs.tests_total > 0

        obs = environment.step(VerirlAction(action_type="submit"))
        assert obs.done is True
        assert obs.final_score > 0.0  # Should have some score


class TestSystolicWorkflow:
    """Test complete systolic array workflow."""

    def test_systolic_full_episode(self, environment, systolic_reference_verilog, requires_eda_tools):
        obs = environment.reset(task_id="systolic_array")

        obs = environment.step(VerirlAction(
            action_type="write_file",
            verilog_src=systolic_reference_verilog
        ))

        obs = environment.step(VerirlAction(action_type="run_compile"))
        assert obs.compile_ok is True

        obs = environment.step(VerirlAction(action_type="run_sim"))

        obs = environment.step(VerirlAction(action_type="submit"))
        assert obs.done is True


class TestErrorRecovery:
    """Test recovery from errors."""

    def test_broken_compile_then_fix(self, environment, mac_reference_verilog, requires_iverilog):
        environment.reset(task_id="mac_unit")

        # Submit broken code
        obs = environment.step(VerirlAction(
            action_type="write_file",
            verilog_src="module broken (input a, output b);"  # missing endmodule
        ))

        obs = environment.step(VerirlAction(action_type="run_compile"))
        assert obs.compile_ok is False

        # Fix and resubmit
        obs = environment.step(VerirlAction(
            action_type="write_file",
            verilog_src=mac_reference_verilog
        ))

        obs = environment.step(VerirlAction(action_type="run_compile"))
        assert obs.compile_ok is True
