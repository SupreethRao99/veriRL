# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
VeriRL EDA Evaluator.

Wraps iverilog, vvp, yosys, and optionally SymbiYosys to evaluate agent-submitted
Verilog. Supports multi-file projects: all methods accept either a single string
(backward-compatible) or a Dict[str, str] mapping filename → source.

Provides:
- Compilation checking via iverilog
- Functional simulation via iverilog + vvp against task testbenches
- Synthesis via yosys for area estimation
- Formal verification via SymbiYosys (optional, graceful degradation)
- Weighted per-task grading across compile / sim / timing / area / formal
"""

import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Union

TOOL_TIMEOUT = 30   # seconds — prevents infinite loops in agent-submitted code
SYNTH_TIMEOUT = 60  # synthesis is slower than compilation
FORMAL_TIMEOUT = 90 # SymbiYosys can be slow on complex designs

# ---------------------------------------------------------------------------
# Result data classes
# ---------------------------------------------------------------------------


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
class FormalResult:
    success: bool           # True if all properties proven
    stdout: str
    stderr: str
    properties_proven: int = 0
    properties_total: int = 0
    counterexample: Optional[str] = None  # description of first failing trace

    @property
    def score(self) -> float:
        if self.properties_total == 0:
            return 0.0
        return self.properties_proven / self.properties_total


@dataclass
class EvalResult:
    compilation: CompilationResult
    simulation: Optional[SimulationResult] = None
    synthesis: Optional[SynthesisResult] = None
    formal: Optional[FormalResult] = None
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
        if self.formal:
            parts.append(
                f"Formal: {self.formal.properties_proven}/{self.formal.properties_total} "
                f"properties proven"
            )
            if self.formal.counterexample:
                parts.append(f"Counterexample:\n{self.formal.counterexample}")
        if self.final_score is not None:
            parts.append(f"Score: {self.final_score:.3f}")
            if self.score_breakdown:
                breakdown_str = ", ".join(
                    f"{k}={v:.2f}" for k, v in self.score_breakdown.items()
                )
                parts.append(f"Breakdown: {breakdown_str}")
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Per-task grading weights
# Tasks with a "formal" key require SymbiYosys; weight is redistributed if
# sby is not installed.
# ---------------------------------------------------------------------------

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
    # --- New tasks ---
    "relu_clip": {
        "compile": 0.10,
        "sim": 0.75,
        "formal": 0.10,
        "area": 0.05,
    },
    "barrel_shifter": {
        "compile": 0.10,
        "sim": 0.80,
        "area": 0.10,
    },
    "register_file": {
        "compile": 0.10,
        "sim": 0.70,
        "formal": 0.10,
        "area": 0.10,
    },
    "ring_buffer": {
        "compile": 0.10,
        "sim": 0.70,
        "area": 0.20,
    },
    "dot_product": {
        "compile": 0.10,
        "sim": 0.60,
        "timing": 0.20,
        "area": 0.10,
    },
    "fir_filter": {
        "compile": 0.10,
        "sim": 0.60,
        "timing": 0.20,
        "area": 0.10,
    },
    "fp16_adder": {
        "compile": 0.05,
        "sim": 0.60,
        "formal": 0.15,
        "area": 0.20,
    },
}

SYSTOLIC_TIMING_LIMIT = 10


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class VerilogEvaluator:
    """
    Evaluates agent-submitted Verilog using real EDA tools.

    Accepts both single-file (str) and multi-file (Dict[str, str]) inputs.
    """

    def __init__(self, timeout: int = TOOL_TIMEOUT):
        self.timeout = timeout
        self._sby_available: Optional[bool] = None  # cached check

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _files_to_dict(
        self, source: Union[str, Dict[str, str]]
    ) -> Dict[str, str]:
        """Normalize single-string source to a filename dict."""
        if isinstance(source, str):
            return {"design.v": source}
        return dict(source)

    def _write_files_to_dir(
        self, files: Dict[str, str], tmpdir: str
    ) -> list:
        """Write all source files to tmpdir. Returns list of absolute paths."""
        paths = []
        for fname, src in files.items():
            # Sanitize filename — no path traversal
            safe_name = Path(fname).name
            fpath = os.path.join(tmpdir, safe_name)
            with open(fpath, "w") as f:
                f.write(src)
            paths.append(fpath)
        return paths

    def _check_sby(self) -> bool:
        """Check once whether sby (SymbiYosys) is installed."""
        if self._sby_available is None:
            self._sby_available = shutil.which("sby") is not None
        return self._sby_available

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compile(
        self, source: Union[str, Dict[str, str]]
    ) -> CompilationResult:
        """Syntax-check Verilog using iverilog. Accepts single file or multi-file dict."""
        files = self._files_to_dict(source)
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = self._write_files_to_dir(files, tmpdir)
            try:
                result = subprocess.run(
                    ["iverilog", "-o", "/dev/null", "-Wall"] + paths,
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

    def simulate(
        self,
        source: Union[str, Dict[str, str]],
        testbench_path: str,
    ) -> SimulationResult:
        """
        Compile agent files together with the task testbench and run with vvp.

        Testbenches use $display("PASS: ...") / $display("FAIL: ...") conventions.
        Counts lines to compute test score. Extracts "CYCLES: N" for timing.
        """
        files = self._files_to_dict(source)
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = self._write_files_to_dir(files, tmpdir)
            sim_binary = os.path.join(tmpdir, "sim.vvp")
            try:
                compile_result = subprocess.run(
                    ["iverilog", "-o", sim_binary] + paths + [testbench_path],
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

    def synthesize(
        self, source: Union[str, Dict[str, str]]
    ) -> SynthesisResult:
        """
        Run Yosys synthesis. Uses 'synth -flatten; stat' for deterministic cell count.
        Accepts single file or multi-file dict.
        """
        files = self._files_to_dict(source)
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = self._write_files_to_dir(files, tmpdir)
            read_cmds = " ".join(f"read_verilog {p};" for p in paths)
            synth_script = f"{read_cmds} synth -flatten; stat"
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
                    success=False, stdout="", stderr="ERROR: synthesis timed out",
                    cell_count=0,
                )

    def formal_verify(
        self,
        source: Union[str, Dict[str, str]],
        properties_path: str,
    ) -> FormalResult:
        """
        Run SymbiYosys formal verification on the agent's design + task properties.

        Returns a FormalResult with properties_proven / properties_total and any
        counterexample trace. Returns a graceful failure if sby is not installed.
        """
        if not self._check_sby():
            return FormalResult(
                success=False,
                stdout="",
                stderr="SymbiYosys (sby) not installed — formal verification skipped.",
                properties_proven=0,
                properties_total=0,
            )

        files = self._files_to_dict(source)
        with tempfile.TemporaryDirectory() as tmpdir:
            design_paths = self._write_files_to_dir(files, tmpdir)
            props_dest = os.path.join(tmpdir, "properties.sv")

            # Copy properties file
            import shutil as _shutil
            _shutil.copy(properties_path, props_dest)

            # Determine top module from properties filename
            props_stem = Path(properties_path).stem  # e.g. "relu_clip_formal"

            # Build sby config
            read_cmds = "\n".join(f"read -formal {p}" for p in design_paths)
            sby_config = f"""\
[options]
mode prove
depth 20

[engines]
smtbmc

[script]
{read_cmds}
read -formal {props_dest}
prep -top {props_stem}

[files]
{chr(10).join(design_paths)}
{props_dest}
"""
            sby_file = os.path.join(tmpdir, "verify.sby")
            with open(sby_file, "w") as f:
                f.write(sby_config)

            try:
                result = subprocess.run(
                    ["sby", "-f", sby_file],
                    capture_output=True,
                    text=True,
                    timeout=FORMAL_TIMEOUT,
                    cwd=tmpdir,
                )
                combined = result.stdout + result.stderr

                # Parse sby output for PASS / FAIL lines
                proven = len(re.findall(r"PASS", combined, re.IGNORECASE))
                failed_matches = re.findall(r"FAIL", combined, re.IGNORECASE)

                # Heuristic: count unique assertion checks mentioned
                assert_checks = re.findall(r"assert\s+\w+", combined)
                total = max(proven + len(failed_matches), len(assert_checks), 1)

                counterexample = None
                if failed_matches:
                    # Extract first counterexample trace (after "Counterexample:")
                    cex_match = re.search(
                        r"(Counterexample.*?)(?=\n\n|\Z)", combined, re.DOTALL
                    )
                    if cex_match:
                        counterexample = cex_match.group(1)[:500]  # cap at 500 chars

                success = (result.returncode == 0) and not failed_matches

                return FormalResult(
                    success=success,
                    stdout=result.stdout[:2000],
                    stderr=result.stderr[:500],
                    properties_proven=proven if success else max(0, proven - len(failed_matches)),
                    properties_total=total,
                    counterexample=counterexample,
                )

            except subprocess.TimeoutExpired:
                return FormalResult(
                    success=False,
                    stdout="",
                    stderr="ERROR: formal verification timed out after 90s",
                    properties_proven=0,
                    properties_total=1,
                )

    def grade(
        self,
        source: Union[str, Dict[str, str]],
        task_id: str,
        testbench_path: str,
        reference_cells: int,
        properties_path: Optional[str] = None,
    ) -> EvalResult:
        """
        Full grading pipeline: compile → simulate → synthesize → [formal] → weighted score.

        Args:
            source: Agent-submitted Verilog (str or Dict[filename, src])
            task_id: Task identifier
            testbench_path: Absolute path to the task testbench file
            reference_cells: Reference synthesis cell count for area scoring
            properties_path: Optional path to .sv formal properties file

        Returns:
            EvalResult with final_score in [0.01, 0.99] and all breakdown values
        """
        files = self._files_to_dict(source)
        primary_src = "\n".join(files.values())

        weights = dict(TASK_WEIGHTS.get(task_id, {"compile": 1.0}))
        breakdown: Dict[str, float] = {k: 0.0 for k in weights}

        # If formal is in weights but sby is unavailable (or no properties file),
        # redistribute formal weight to sim.
        if "formal" in weights:
            formal_avail = bool(properties_path) and self._check_sby()
            if not formal_avail:
                weights["sim"] = weights.get("sim", 0) + weights.pop("formal")
                breakdown.pop("formal", None)
                breakdown = {k: 0.0 for k in weights}

        def _clamp(d: Dict[str, float]) -> Dict[str, float]:
            return {k: max(0.01, min(0.99, v)) for k, v in d.items()}

        if not primary_src.strip():
            return EvalResult(
                compilation=CompilationResult(
                    success=False, stdout="", stderr="Empty submission"
                ),
                final_score=0.01,
                score_breakdown=_clamp(breakdown),
            )

        # Step 1: Compilation
        comp = self.compile(files)
        if not comp.success:
            return EvalResult(
                compilation=comp,
                final_score=0.01,
                score_breakdown=_clamp(breakdown),
            )
        breakdown["compile"] = 0.99

        # Step 2: Simulation
        sim = self.simulate(files, testbench_path)
        if sim.tests_total > 0:
            ratio = sim.tests_passed / sim.tests_total
            breakdown["sim"] = max(0.01, min(ratio, 0.99))

        # Step 3: Synthesis (always run if code compiles)
        synth = self.synthesize(files)

        # Step 4: Timing scoring (task-specific)
        if "timing" in breakdown:
            if task_id == "mac_unit" and synth is not None:
                dff_count = 0
                dff_match = re.search(r"\$dff\s+(\d+)", synth.stdout)
                if dff_match:
                    dff_count = int(dff_match.group(1))
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
                        breakdown["timing"] = max(
                            0.01, min(1.0 - (cycles - SYSTOLIC_TIMING_LIMIT) * 0.2, 0.99)
                        )
                    elif sim.tests_passed > 0:
                        breakdown["timing"] = 0.20
                elif sim.tests_passed > 0:
                    breakdown["timing"] = 0.20

            elif task_id in ("dot_product", "fir_filter"):
                # Verify pipeline registers exist via DFF count
                dff_count = 0
                sdff_match = re.findall(r"\$_SDFF", synth.stdout)
                dff_match = re.search(r"\$dff\s+(\d+)", synth.stdout)
                if sdff_match:
                    dff_count = len(sdff_match)
                elif dff_match:
                    dff_count = int(dff_match.group(1))
                if task_id == "dot_product" and dff_count >= 2:
                    breakdown["timing"] = 0.99
                elif task_id == "fir_filter" and dff_count >= 2:
                    breakdown["timing"] = 0.99
                elif synth.cell_count >= 5:
                    breakdown["timing"] = 0.40

        # Step 5: Area scoring (gated on functional correctness)
        if (
            "area" in breakdown
            and synth is not None
            and synth.success
            and synth.cell_count > 0
            and reference_cells > 0
            and sim.tests_passed > 0
        ):
            ratio = reference_cells / synth.cell_count
            breakdown["area"] = max(0.01, min(ratio, 0.99))

        # Step 6: Formal verification (optional)
        formal_result = None
        if "formal" in breakdown and properties_path:
            formal_result = self.formal_verify(files, properties_path)
            breakdown["formal"] = max(0.01, min(formal_result.score, 0.99))

        breakdown = _clamp(breakdown)

        # Recompute weights for any keys that may have been redistributed
        active_weights = {k: weights[k] for k in weights if k in breakdown}
        weight_sum = sum(active_weights.values())
        if weight_sum > 0:
            norm_weights = {k: v / weight_sum for k, v in active_weights.items()}
        else:
            norm_weights = active_weights

        raw_score = round(sum(norm_weights[k] * breakdown[k] for k in norm_weights), 4)
        final_score = max(0.01, min(0.99, raw_score))

        return EvalResult(
            compilation=comp,
            simulation=sim,
            synthesis=synth,
            formal=formal_result,
            final_score=final_score,
            score_breakdown=breakdown,
        )
