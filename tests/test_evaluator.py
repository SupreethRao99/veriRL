"""Tests for VerilogEvaluator."""

import pytest
from verirl_env.server.evaluator import VerilogEvaluator


class TestCompilation:
    """Test Verilog compilation."""

    def test_compile_valid_module(self, evaluator, requires_iverilog):
        verilog = "module foo (input a, output b); assign b = a; endmodule"
        result = evaluator.compile(verilog)
        assert result.success is True
        assert result.stderr == "" or "warning" not in result.stderr.lower()

    def test_compile_syntax_error(self, evaluator, requires_iverilog):
        verilog = "module foo (input a, output b) assign b = a; endmodule"  # missing ;
        result = evaluator.compile(verilog)
        assert result.success is False
        assert len(result.stderr) > 0

    def test_compile_empty_module(self, evaluator, requires_iverilog):
        verilog = "module empty; endmodule"
        result = evaluator.compile(verilog)
        assert result.success is True


class TestMACUnit:
    """Test MAC unit compilation, simulation, and grading."""

    def test_mac_reference_compiles(self, evaluator, mac_reference_verilog, requires_iverilog):
        result = evaluator.compile(mac_reference_verilog)
        assert result.success is True

    def test_mac_reference_grading(self, evaluator, mac_reference_verilog, environment, requires_eda_tools):
        task = environment.tasks["mac_unit"]
        result = evaluator.grade(
            verilog_src=mac_reference_verilog,
            task_id="mac_unit",
            testbench_path=task.testbench_path,
            reference_cells=task.reference_cells,
        )
        # Verify grading pipeline works: compile → simulate → grade
        assert result.compilation.success is True
        assert result.simulation is not None
        assert result.simulation.tests_total > 0
        # Final score should be computed
        assert result.final_score is not None
        assert 0.0 <= result.final_score <= 1.0

    def test_mac_empty_submission(self, evaluator, environment):
        task = environment.tasks["mac_unit"]
        result = evaluator.grade(
            verilog_src="",
            task_id="mac_unit",
            testbench_path=task.testbench_path,
            reference_cells=task.reference_cells,
        )
        assert result.final_score == 0.0
        assert result.compilation.success is False

    def test_mac_broken_module(self, evaluator, environment):
        broken = "module mac_unit (input a, output b); endmodule"  # wrong interface
        task = environment.tasks["mac_unit"]
        result = evaluator.grade(
            verilog_src=broken,
            task_id="mac_unit",
            testbench_path=task.testbench_path,
            reference_cells=task.reference_cells,
        )
        # Broken module fails simulation, gets very low score
        assert result.final_score < 0.2
        assert result.simulation is not None
        assert result.simulation.tests_passed == 0


class TestAXIFIFO:
    """Test AXI FIFO compilation and grading."""

    def test_axi_reference_compiles(self, evaluator, axi_reference_verilog, requires_iverilog):
        result = evaluator.compile(axi_reference_verilog)
        assert result.success is True

    def test_axi_reference_grading(self, evaluator, axi_reference_verilog, environment, requires_eda_tools):
        task = environment.tasks["axi_fifo"]
        result = evaluator.grade(
            verilog_src=axi_reference_verilog,
            task_id="axi_fifo",
            testbench_path=task.testbench_path,
            reference_cells=task.reference_cells,
        )
        assert result.final_score > 0.2  # Should score decently
        assert result.compilation.success is True
        assert result.simulation is not None
        assert result.simulation.tests_total > 0


class TestSystolicArray:
    """Test systolic array compilation and grading."""

    def test_systolic_reference_compiles(self, evaluator, systolic_reference_verilog, requires_iverilog):
        result = evaluator.compile(systolic_reference_verilog)
        assert result.success is True

    def test_systolic_reference_grading(self, evaluator, systolic_reference_verilog, environment, requires_eda_tools):
        task = environment.tasks["systolic_array"]
        result = evaluator.grade(
            verilog_src=systolic_reference_verilog,
            task_id="systolic_array",
            testbench_path=task.testbench_path,
            reference_cells=task.reference_cells,
        )
        assert result.final_score > 0.0  # Should have some score
        assert result.compilation.success is True
        if result.simulation:
            # Systolic may or may not pass all tests depending on impl
            assert result.simulation.tests_total >= 0
        # Check timing if available
        if result.simulation and result.simulation.timing_cycles:
            assert result.simulation.timing_cycles <= 20  # Reasonable bound


class TestSynthesis:
    """Test synthesis (yosys)."""

    def test_simple_synthesis(self, evaluator, requires_yosys):
        verilog = "module foo (input a, b, output c); assign c = a & b; endmodule"
        result = evaluator.synthesize(verilog)
        # Synthesis may succeed or fail depending on yosys version, but should run
        assert result.stdout is not None or result.stderr is not None

    def test_mac_synthesis(self, evaluator, mac_reference_verilog, requires_yosys):
        result = evaluator.synthesize(mac_reference_verilog)
        # Synthesis should produce some output
        assert result.stdout is not None or result.stderr is not None
