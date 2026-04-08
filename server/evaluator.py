# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
VeriRL EDA Evaluator.

Wraps iverilog, vvp, and yosys to evaluate agent-submitted Verilog code.
Provides compilation checking, functional simulation, synthesis area estimation,
and weighted per-task scoring.

All feedback is curated to be actionable for LLM agents.
"""

import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Dict, Optional

TOOL_TIMEOUT = 30  # seconds — prevents infinite loops in agent-submitted code
SYNTH_TIMEOUT = 60  # synthesis is slower than compilation


@dataclass
class CompilationResult:
    success: bool
    stdout: str
    stderr: str


@dataclass
class SimulationResult:
    success: bool
    stdout: str
    stderr: str
    tests_passed: int
    tests_total: int
    timing_cycles: Optional[int] = None


@dataclass
class SynthesisResult:
    success: bool
    stdout: str
    stderr: str
    cell_count: int = 0


@dataclass
class EvalResult:
    compilation: CompilationResult
    simulation: Optional[SimulationResult] = None
    synthesis: Optional[SynthesisResult] = None
    final_score: Optional[float] = None
    score_breakdown: Optional[Dict[str, float]] = None

    def to_agent_feedback(self) -> str:
        """Format evaluation results as human-readable feedback for the agent."""
        parts = []
        if not self.compilation.success:
            parts.append(f"Compilation FAILED:\n{self.compilation.stderr}")
            return "\n".join(parts)
        parts.append("Compilation: OK")
        if self.simulation:
            if self.simulation.tests_total > 0:
                parts.append(
                    f"Simulation: {self.simulation.tests_passed}/{self.simulation.tests_total} "
                    f"tests passed"
                )
            if self.simulation.timing_cycles is not None:
                parts.append(f"Timing: {self.simulation.timing_cycles} cycles")
            if self.simulation.stdout:
                parts.append(f"Sim output:\n{self.simulation.stdout}")
            if self.simulation.stderr:
                parts.append(f"Sim errors:\n{self.simulation.stderr}")
        if self.synthesis:
            parts.append(f"Synthesis: {self.synthesis.cell_count} cells")
        if self.final_score is not None:
            parts.append(f"Score: {self.final_score:.3f}")
            if self.score_breakdown:
                breakdown_str = ", ".join(
                    f"{k}={v:.2f}" for k, v in self.score_breakdown.items()
                )
                parts.append(f"Breakdown: {breakdown_str}")
        return "\n".join(parts)


# Per-task grading weights
TASK_WEIGHTS: Dict[str, Dict[str, float]] = {
    "mac_unit": {
        "compile": 0.10,
        "sim": 0.60,
        "timing": 0.20,
        "area": 0.10,
    },
    "axi_fifo": {
        "compile": 0.10,
        "sim": 0.70,
        "area": 0.20,
    },
    "systolic_array": {
        "compile": 0.05,
        "sim": 0.50,
        "timing": 0.30,
        "area": 0.15,
    },
}

SYSTOLIC_TIMING_LIMIT = 10  # cycles — allows 7 + 3-cycle graceful degradation


class VerilogEvaluator:
    """
    Evaluates agent-submitted Verilog code using real EDA tools.

    Provides:
    - Compilation checking via iverilog
    - Functional simulation via iverilog + vvp against task testbenches
    - Synthesis via yosys for area estimation (cell count)
    - Weighted per-task grading across compile / sim / timing / area dimensions

    Example:
        >>> evaluator = VerilogEvaluator()
        >>> result = evaluator.grade(
        ...     verilog_src=my_verilog,
        ...     task_id="mac_unit",
        ...     testbench_path="/path/to/testbench.v",
        ...     reference_cells=120,
        ... )
        >>> print(f"Score: {result.final_score}")
        >>> print(result.to_agent_feedback())
    """

    def __init__(self, timeout: int = TOOL_TIMEOUT):
        self.timeout = timeout

    def compile(self, verilog_src: str) -> CompilationResult:
        """Syntax-check Verilog using iverilog. No binary output — checks errors only."""
        with tempfile.NamedTemporaryFile(suffix=".v", mode="w", delete=False) as f:
            f.write(verilog_src)
            tmpv = f.name
        try:
            result = subprocess.run(
                ["iverilog", "-o", "/dev/null", "-Wall", tmpv],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            return CompilationResult(
                success=(result.returncode == 0),
                stdout=result.stdout,
                stderr=result.stderr,
            )
        except subprocess.TimeoutExpired:
            return CompilationResult(
                success=False,
                stdout="",
                stderr="ERROR: compilation timed out after 30s",
            )
        except FileNotFoundError:
            return CompilationResult(
                success=False,
                stdout="",
                stderr="ERROR: iverilog not found — is it installed?",
            )
        finally:
            os.unlink(tmpv)

    def simulate(self, verilog_src: str, testbench_path: str) -> SimulationResult:
        """
        Compile agent code together with the task testbench and run with vvp.

        Testbenches use $display("PASS: ...") and $display("FAIL: ...") conventions.
        Counts PASS/FAIL lines to compute test score. Extracts "CYCLES: N" for timing.
        """
        with tempfile.NamedTemporaryFile(suffix=".v", mode="w", delete=False) as f:
            f.write(verilog_src)
            agent_file = f.name
        sim_binary = tempfile.mktemp(suffix=".vvp")
        try:
            compile_result = subprocess.run(
                ["iverilog", "-o", sim_binary, agent_file, testbench_path],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            if compile_result.returncode != 0:
                return SimulationResult(
                    success=False,
                    stdout=compile_result.stdout,
                    stderr=compile_result.stderr,
                    tests_passed=0,
                    tests_total=0,
                )
            sim_result = subprocess.run(
                ["vvp", sim_binary],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            stdout = sim_result.stdout
            passes = len(re.findall(r"^PASS:", stdout, re.MULTILINE))
            fails = len(re.findall(r"^FAIL:", stdout, re.MULTILINE))
            timing_match = re.search(r"CYCLES:\s*(\d+)", stdout)
            return SimulationResult(
                success=(sim_result.returncode == 0),
                stdout=stdout,
                stderr=sim_result.stderr,
                tests_passed=passes,
                tests_total=passes + fails,
                timing_cycles=int(timing_match.group(1)) if timing_match else None,
            )
        except subprocess.TimeoutExpired:
            return SimulationResult(
                success=False,
                stdout="",
                stderr="ERROR: simulation timed out — possible infinite loop in submitted code",
                tests_passed=0,
                tests_total=0,
            )
        finally:
            os.unlink(agent_file)
            if os.path.exists(sim_binary):
                os.unlink(sim_binary)

    def synthesize(self, verilog_src: str) -> SynthesisResult:
        """
        Run Yosys synthesis on agent's Verilog.

        Uses 'synth -flatten; stat' only — no optimization passes — so cell count
        is fully deterministic (same source always produces the same count).
        """
        with tempfile.NamedTemporaryFile(suffix=".v", mode="w", delete=False) as f:
            f.write(verilog_src)
            tmpv = f.name
        synth_script = f"read_verilog {tmpv}; synth -flatten; stat"
        try:
            result = subprocess.run(
                ["yosys", "-p", synth_script],
                capture_output=True,
                text=True,
                timeout=SYNTH_TIMEOUT,
            )
            combined = result.stdout + result.stderr
            cell_match = re.search(r"Number of cells:\s*(\d+)", combined)
            cell_count = int(cell_match.group(1)) if cell_match else 0
            return SynthesisResult(
                success=(result.returncode == 0 and cell_count > 0),
                stdout=result.stdout,
                stderr=result.stderr,
                cell_count=cell_count,
            )
        except subprocess.TimeoutExpired:
            return SynthesisResult(
                success=False,
                stdout="",
                stderr="ERROR: synthesis timed out",
                cell_count=0,
            )
        finally:
            os.unlink(tmpv)

    def grade(
        self,
        verilog_src: str,
        task_id: str,
        testbench_path: str,
        reference_cells: int,
    ) -> EvalResult:
        """
        Full grading pipeline: compile → simulate → synthesize → weighted score.

        Args:
            verilog_src: Agent-submitted Verilog source code
            task_id: Task identifier (mac_unit, axi_fifo, systolic_array)
            testbench_path: Absolute path to the task testbench file
            reference_cells: Reference synthesis cell count for area scoring

        Returns:
            EvalResult with final_score and all breakdown values strictly in (0, 1)
        """
        weights = TASK_WEIGHTS.get(task_id, {"compile": 1.0})
        breakdown: Dict[str, float] = {k: 0.0 for k in weights}

        if not verilog_src.strip():
            return EvalResult(
                compilation=CompilationResult(
                    success=False, stdout="", stderr="Empty submission"
                ),
                final_score=0.01,
                score_breakdown=breakdown,
            )

        # Step 1: Compilation
        comp = self.compile(verilog_src)
        if not comp.success:
            return EvalResult(
                compilation=comp,
                final_score=0.01,
                score_breakdown=breakdown,
            )
        breakdown["compile"] = 0.99  # Clamp to [0.01, 0.99]

        # Step 2: Simulation
        sim = self.simulate(verilog_src, testbench_path)
        if sim.tests_total > 0:
            ratio = sim.tests_passed / sim.tests_total
            breakdown["sim"] = max(0.01, min(ratio, 0.99))

        # Step 3: Synthesis — always run if code compiles; area/timing use results below
        synth = self.synthesize(verilog_src)

        # Step 4: Timing scoring (task-specific)
        if "timing" in breakdown:
            if task_id == "mac_unit" and synth is not None:
                # Verify 2-stage pipeline via DFF count.
                # After yosys tech mapping, DFFs appear as $_SDFF_* or $_SDFFE_* cells.
                # Count both pre-mapped ($dff) and post-mapped ($_SDFF, $_SDFFE) variants.
                dff_count = 0
                # Pre-mapped: $dff cells (appear in verbose log)
                dff_match = re.search(r"\$dff\s+(\d+)", synth.stdout)
                if dff_match:
                    dff_count = int(dff_match.group(1))
                # Post-mapped: count $_SDFF cells in stat output
                sdff_match = re.findall(r"\$_SDFF", synth.stdout)
                if sdff_match:
                    dff_count = len(sdff_match)

                if dff_count >= 2:
                    breakdown["timing"] = 0.99
                elif synth.cell_count >= 10:
                    breakdown["timing"] = 0.50
            elif task_id == "systolic_array":
                if sim.timing_cycles is not None:
                    cycles = sim.timing_cycles
                    if cycles <= SYSTOLIC_TIMING_LIMIT:
                        breakdown["timing"] = 0.99
                    elif cycles <= SYSTOLIC_TIMING_LIMIT + 3:
                        timing_score = 1.0 - (cycles - SYSTOLIC_TIMING_LIMIT) * 0.2
                        breakdown["timing"] = max(0.01, min(timing_score, 0.99))
                elif breakdown["sim"] > 0:
                    breakdown["timing"] = 0.20

        # Step 5: Area scoring — gated on functional correctness (sim > 0)
        # Prevents a trivially small but wrong module from scoring high on area.
        if (
            "area" in breakdown
            and synth is not None
            and synth.success
            and synth.cell_count > 0
            and reference_cells > 0
            and breakdown["sim"] > 0.0
        ):
            ratio = reference_cells / synth.cell_count
            breakdown["area"] = max(0.01, min(ratio, 0.99))

        # Clamp all breakdown values to (0, 1) — single point of score validation
        breakdown = {k: max(0.01, min(0.99, v)) for k, v in breakdown.items()}

        raw_score = round(sum(weights[k] * breakdown[k] for k in weights), 4)
        # Validator requires scores strictly in (0, 1) — clamp to [0.01, 0.99]
        final_score = max(0.01, min(0.99, raw_score))

        return EvalResult(
            compilation=comp,
            simulation=sim,
            synthesis=synth,
            final_score=final_score,
            score_breakdown=breakdown,
        )
