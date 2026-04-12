"""Tests for new tasks 4–10 (relu_clip through fp16_adder)."""

import pytest
from verirl_env.models import VerirlAction
from verirl_env.server.verirl_env_environment import VerirlEnvironment


# ---------------------------------------------------------------------------
# Reset / metadata tests — no EDA tools needed
# ---------------------------------------------------------------------------


class TestNewTaskResets:
    """Verify reset observation fields for every new task."""

    @pytest.mark.parametrize("task_id,keyword,max_turns", [
        ("relu_clip",      "relu",      6),
        ("barrel_shifter", "shift",     6),
        ("register_file",  "register",  8),
        ("ring_buffer",    "ring",     10),
        ("dot_product",    "dot",       8),
        ("fir_filter",     "fir",      10),
        ("fp16_adder",     "fp16",     15),
    ])
    def test_reset_fields(self, environment, task_id, keyword, max_turns):
        obs = environment.reset(task_id=task_id)
        assert obs.task_spec != "", f"{task_id}: task_spec must not be empty"
        assert keyword in obs.task_spec.lower(), (
            f"{task_id}: expected '{keyword}' in task_spec"
        )
        assert obs.turns_remaining == max_turns, (
            f"{task_id}: expected {max_turns} turns, got {obs.turns_remaining}"
        )
        assert obs.compile_ok is False
        assert obs.tests_passed == 0
        assert obs.turn_number == 0
        assert obs.done is False

    def test_all_ten_tasks_load(self, environment):
        """Sanity check: all 10 tasks are present after loading."""
        assert environment.num_tasks == 10
        tasks = environment.list_tasks()
        for tid in [
            "mac_unit", "axi_fifo", "systolic_array",
            "relu_clip", "barrel_shifter", "register_file",
            "ring_buffer", "dot_product", "fir_filter", "fp16_adder",
        ]:
            assert tid in tasks, f"Task '{tid}' missing from environment"

    def test_list_tasks_by_difficulty(self, environment):
        easy   = environment.list_tasks_by_difficulty("easy")
        medium = environment.list_tasks_by_difficulty("medium")
        hard   = environment.list_tasks_by_difficulty("hard")

        assert set(easy)   == {"mac_unit", "relu_clip", "barrel_shifter"}
        assert set(medium) == {"axi_fifo", "register_file", "ring_buffer",
                               "dot_product", "fir_filter"}
        assert set(hard)   == {"systolic_array", "fp16_adder"}


# ---------------------------------------------------------------------------
# Multi-file write support
# ---------------------------------------------------------------------------


class TestMultiFileWrite:
    """Verify that multi-file projects compile correctly."""

    def test_write_two_files(self, environment, requires_iverilog):
        """Two files written separately should compile together."""
        top_v = """
module top (input wire clk, input wire signed [7:0] a, b,
            output wire [3:0] out);
    relu_clip #(.IN_W(8), .OUT_W(4)) u0 (.in_val(a), .out_val(out), .saturated());
endmodule
"""
        sub_v = """
module relu_clip #(parameter IN_W=8, parameter OUT_W=4) (
    input  wire signed [IN_W-1:0]  in_val,
    output wire        [OUT_W-1:0] out_val,
    output wire                    saturated
);
    localparam integer MAX_OUT = (1 << OUT_W) - 1;
    wire neg = in_val[IN_W-1];
    wire [IN_W-1:0] relu_out = neg ? {IN_W{1'b0}} : in_val;
    wire pos_clip = (relu_out > MAX_OUT[IN_W-1:0]);
    assign out_val   = pos_clip ? MAX_OUT[OUT_W-1:0] : relu_out[OUT_W-1:0];
    assign saturated = neg | pos_clip;
endmodule
"""
        environment.reset(task_id="relu_clip")
        environment.step(VerirlAction(action_type="write_file",
                                      filename="relu_clip.v",
                                      verilog_src=sub_v))
        environment.step(VerirlAction(action_type="write_file",
                                      filename="top.v",
                                      verilog_src=top_v))
        obs = environment.step(VerirlAction(action_type="run_compile"))
        # Both files should be tracked
        assert "relu_clip.v" in obs.current_files
        assert "top.v" in obs.current_files

    def test_list_files_action(self, environment):
        """list_files action reports files with sizes."""
        environment.reset(task_id="barrel_shifter")
        environment.step(VerirlAction(action_type="write_file",
                                      filename="design.v",
                                      verilog_src="module barrel_shifter; endmodule"))
        obs = environment.step(VerirlAction(action_type="list_files"))
        assert "design.v" in obs.tool_stdout
        assert "chars" in obs.tool_stdout


# ---------------------------------------------------------------------------
# Compile tests — iverilog required
# ---------------------------------------------------------------------------


class TestNewTasksCompile:
    """Reference implementations must compile without errors."""

    def test_relu_clip_compiles(self, environment, relu_clip_reference_verilog,
                                requires_iverilog):
        environment.reset(task_id="relu_clip")
        environment.step(VerirlAction(action_type="write_file",
                                      verilog_src=relu_clip_reference_verilog))
        obs = environment.step(VerirlAction(action_type="run_compile"))
        assert obs.compile_ok is True

    def test_barrel_shifter_compiles(self, environment,
                                     barrel_shifter_reference_verilog,
                                     requires_iverilog):
        environment.reset(task_id="barrel_shifter")
        environment.step(VerirlAction(action_type="write_file",
                                      verilog_src=barrel_shifter_reference_verilog))
        obs = environment.step(VerirlAction(action_type="run_compile"))
        assert obs.compile_ok is True

    def test_register_file_compiles(self, environment,
                                    register_file_reference_verilog,
                                    requires_iverilog):
        environment.reset(task_id="register_file")
        environment.step(VerirlAction(action_type="write_file",
                                      verilog_src=register_file_reference_verilog))
        obs = environment.step(VerirlAction(action_type="run_compile"))
        assert obs.compile_ok is True

    def test_ring_buffer_compiles(self, environment, ring_buffer_reference_verilog,
                                  requires_iverilog):
        environment.reset(task_id="ring_buffer")
        environment.step(VerirlAction(action_type="write_file",
                                      verilog_src=ring_buffer_reference_verilog))
        obs = environment.step(VerirlAction(action_type="run_compile"))
        assert obs.compile_ok is True

    def test_dot_product_compiles(self, environment, dot_product_reference_verilog,
                                  requires_iverilog):
        environment.reset(task_id="dot_product")
        environment.step(VerirlAction(action_type="write_file",
                                      verilog_src=dot_product_reference_verilog))
        obs = environment.step(VerirlAction(action_type="run_compile"))
        assert obs.compile_ok is True

    def test_fir_filter_compiles(self, environment, fir_filter_reference_verilog,
                                 requires_iverilog):
        environment.reset(task_id="fir_filter")
        environment.step(VerirlAction(action_type="write_file",
                                      verilog_src=fir_filter_reference_verilog))
        obs = environment.step(VerirlAction(action_type="run_compile"))
        assert obs.compile_ok is True

    def test_fp16_adder_compiles(self, environment, fp16_adder_reference_verilog,
                                 requires_iverilog):
        environment.reset(task_id="fp16_adder")
        environment.step(VerirlAction(action_type="write_file",
                                      verilog_src=fp16_adder_reference_verilog))
        obs = environment.step(VerirlAction(action_type="run_compile"))
        assert obs.compile_ok is True


# ---------------------------------------------------------------------------
# Simulation tests — iverilog + vvp required
# ---------------------------------------------------------------------------


class TestNewTasksSimulation:
    """Reference implementations must pass the majority of testbench checks."""

    def _run_sim(self, environment, task_id, verilog_src):
        environment.reset(task_id=task_id)
        environment.step(VerirlAction(action_type="write_file",
                                      verilog_src=verilog_src))
        environment.step(VerirlAction(action_type="run_compile"))
        return environment.step(VerirlAction(action_type="run_sim"))

    def test_relu_clip_sim(self, environment, relu_clip_reference_verilog,
                           requires_eda_tools):
        obs = self._run_sim(environment, "relu_clip", relu_clip_reference_verilog)
        assert obs.tests_total > 0, "Testbench produced no PASS/FAIL lines"
        assert obs.tests_passed == obs.tests_total, (
            f"relu_clip: {obs.tests_passed}/{obs.tests_total} tests passed"
        )

    def test_barrel_shifter_sim(self, environment, barrel_shifter_reference_verilog,
                                requires_eda_tools):
        obs = self._run_sim(environment, "barrel_shifter",
                            barrel_shifter_reference_verilog)
        assert obs.tests_total > 0
        assert obs.tests_passed == obs.tests_total, (
            f"barrel_shifter: {obs.tests_passed}/{obs.tests_total}"
        )

    def test_register_file_sim(self, environment, register_file_reference_verilog,
                               requires_eda_tools):
        obs = self._run_sim(environment, "register_file",
                            register_file_reference_verilog)
        assert obs.tests_total > 0
        assert obs.tests_passed == obs.tests_total

    def test_ring_buffer_sim(self, environment, ring_buffer_reference_verilog,
                             requires_eda_tools):
        obs = self._run_sim(environment, "ring_buffer", ring_buffer_reference_verilog)
        assert obs.tests_total > 0
        assert obs.tests_passed == obs.tests_total

    def test_dot_product_sim(self, environment, dot_product_reference_verilog,
                             requires_eda_tools):
        obs = self._run_sim(environment, "dot_product", dot_product_reference_verilog)
        assert obs.tests_total > 0
        assert obs.tests_passed == obs.tests_total

    def test_fir_filter_sim(self, environment, fir_filter_reference_verilog,
                            requires_eda_tools):
        obs = self._run_sim(environment, "fir_filter", fir_filter_reference_verilog)
        assert obs.tests_total > 0
        assert obs.tests_passed == obs.tests_total

    def test_fp16_adder_sim(self, environment, fp16_adder_reference_verilog,
                            requires_eda_tools):
        obs = self._run_sim(environment, "fp16_adder", fp16_adder_reference_verilog)
        assert obs.tests_total > 0
        # FP16 adder is hard — require at least 75% pass rate from reference
        pass_rate = obs.tests_passed / obs.tests_total
        assert pass_rate >= 0.75, (
            f"fp16_adder reference: only {obs.tests_passed}/{obs.tests_total} pass"
        )


# ---------------------------------------------------------------------------
# Grading tests — full pipeline with EDA tools
# ---------------------------------------------------------------------------


class TestNewTasksGrading:
    """Reference submissions should achieve high final scores."""

    def _grade(self, environment, task_id, verilog_src):
        environment.reset(task_id=task_id)
        environment.step(VerirlAction(action_type="write_file",
                                      verilog_src=verilog_src))
        environment.step(VerirlAction(action_type="run_compile"))
        environment.step(VerirlAction(action_type="run_sim"))
        obs = environment.step(VerirlAction(action_type="submit"))
        return obs

    def test_relu_clip_reference_scores_high(self, environment,
                                             relu_clip_reference_verilog,
                                             requires_eda_tools):
        obs = self._grade(environment, "relu_clip", relu_clip_reference_verilog)
        assert obs.final_score is not None
        assert obs.final_score >= 0.70, (
            f"relu_clip reference scored {obs.final_score:.3f}, expected >= 0.70"
        )

    def test_barrel_shifter_reference_scores_high(self, environment,
                                                  barrel_shifter_reference_verilog,
                                                  requires_eda_tools):
        obs = self._grade(environment, "barrel_shifter",
                          barrel_shifter_reference_verilog)
        assert obs.final_score is not None
        assert obs.final_score >= 0.70

    def test_register_file_reference_scores_high(self, environment,
                                                 register_file_reference_verilog,
                                                 requires_eda_tools):
        obs = self._grade(environment, "register_file",
                          register_file_reference_verilog)
        assert obs.final_score is not None
        assert obs.final_score >= 0.65

    def test_ring_buffer_reference_scores_high(self, environment,
                                               ring_buffer_reference_verilog,
                                               requires_eda_tools):
        obs = self._grade(environment, "ring_buffer", ring_buffer_reference_verilog)
        assert obs.final_score is not None
        assert obs.final_score >= 0.65

    def test_dot_product_reference_scores_high(self, environment,
                                               dot_product_reference_verilog,
                                               requires_eda_tools):
        obs = self._grade(environment, "dot_product", dot_product_reference_verilog)
        assert obs.final_score is not None
        assert obs.final_score >= 0.65

    def test_fir_filter_reference_scores_high(self, environment,
                                              fir_filter_reference_verilog,
                                              requires_eda_tools):
        obs = self._grade(environment, "fir_filter", fir_filter_reference_verilog)
        assert obs.final_score is not None
        assert obs.final_score >= 0.65

    def test_empty_submission_scores_minimum(self, environment, requires_eda_tools):
        """An empty submission should always return the minimum score."""
        for task_id in ("relu_clip", "barrel_shifter", "ring_buffer"):
            environment.reset(task_id=task_id)
            obs = environment.step(VerirlAction(action_type="submit"))
            assert obs.final_score == 0.01, (
                f"{task_id}: empty submission scored {obs.final_score}, expected 0.01"
            )

    def test_score_breakdown_present(self, environment, relu_clip_reference_verilog,
                                     requires_eda_tools):
        """score_breakdown must contain expected keys for relu_clip."""
        obs = self._grade(environment, "relu_clip", relu_clip_reference_verilog)
        assert obs.score_breakdown is not None
        assert "compile" in obs.score_breakdown
        assert "sim" in obs.score_breakdown

    def test_score_always_in_valid_range(self, environment,
                                         relu_clip_reference_verilog,
                                         requires_eda_tools):
        """final_score must always be in [0.01, 0.99]."""
        obs = self._grade(environment, "relu_clip", relu_clip_reference_verilog)
        assert 0.01 <= obs.final_score <= 0.99

    def test_run_formal_no_properties_task(self, environment,
                                           barrel_shifter_reference_verilog,
                                           requires_iverilog):
        """run_formal on a task with no properties file should return INFO, not crash."""
        environment.reset(task_id="barrel_shifter")
        environment.step(VerirlAction(action_type="write_file",
                                      verilog_src=barrel_shifter_reference_verilog))
        environment.step(VerirlAction(action_type="run_compile"))
        obs = environment.step(VerirlAction(action_type="run_formal"))
        assert "INFO" in obs.tool_stderr or "not available" in obs.tool_stderr.lower()
