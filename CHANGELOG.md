## [0.4.0] - 2026-04-05

Features
~~~~~~~~

- Add delta bonus reward for test improvement between sim runs — agents now earn +0.15 reward for each test they fix, encouraging iterative debugging over exhaustive exploration (#10)
- Add reference cell target to run_synth feedback — agents now see how their module size compares to the reference implementation (#14)
- Raise systolic array timing limit from 7 to 10 cycles with graceful degradation scoring — allows agents more leeway on hard task without sacrificing correctness signals (#17)

Bug Fixes
~~~~~~~~~

- Fix stale compile_ok and simulation state after write_file — reset internal state so agents cannot observe cached compile/test results from previous code (#11)
- Gate area score on functional correctness — prevent trivially small but incorrect modules from scoring high on area efficiency (#12)
- Ensure all reward values in [0.0, 1.0] per rubric requirement — clamp negative rewards to 0.0 instead of allowing -0.05 minimum (#13)
- Fix DFF detection for MAC unit timing verification — handle both pre-mapped ($dff) and post-mapped ($_SDFF) cells from yosys synthesis (#16)
- Implement correct systolic array reference implementation with proper cycle-gated accumulation and 7-cycle timing; expand test coverage with 5 comprehensive grading tests (#18)
- Fix stdout format compliance — route [WARNING] logs to stderr to keep stdout clean per strict output specification; add OPENAI_API_KEY as primary API credential fallback (#19)

Documentation
~~~~~~~~~~~~~

- Update reward function documentation to reflect delta bonus and capped time penalty; clarify systolic timing requirements and graceful degradation scoring formula (#15)
## [0.1.2] - 2026-04-04
No significant changes.

## [0.2.0] - 2026-04-04

Features
~~~~~~~~

- Automatic releases on every merge to main - each merge now automatically generates changelog, creates version tag, builds Docker images, publishes GitHub release, and deploys to HF Spaces (auto-release)

Bug Fixes
~~~~~~~~~

- Remove colorFrom and colorTo from README metadata - let HuggingFace Spaces use default colors to avoid validation errors (hf-colors)
- Fix HuggingFace Spaces metadata validation - change colorTo from gray to slate (valid HF color) (hf-metadata)

Miscellaneous
~~~~~~~~~~~~~

-  (workflow)
## [0.1.1] - 2026-04-04

Features
~~~~~~~~

- Initial release of VeriRL - Verilog RTL Design Environment with 3 hardware design tasks (MAC unit, AXI-Stream FIFO, 4×4 Systolic Array), real EDA tool grading (iverilog, yosys, vvp), and professional CI/CD pipeline with semantic versioning (init)
- Initial release of VeriRL environment with 3 hardware design tasks, real EDA tool grading, and professional CI/CD pipeline (#2)

Bug Fixes
~~~~~~~~~

- Fix Docker image tag validation error by converting repository name to lowercase in CI/CD and release workflows (ghcr.io requires lowercase image names) (#3)

Documentation
~~~~~~~~~~~~~

- Update README.md documentation for clarity (#1)
## [0.1.0] - 2026-04-04

Features
~~~~~~~~

- Implement complete VeriRL Verilog hardware design environment with 3 tasks (MAC unit, AXI-Stream FIFO, 4×4 systolic array), real EDA tool grading (iverilog, yosys), and baseline inference script (#1)
- Add professional CI/CD pipeline with GitHub Actions, automated Docker builds, semantic versioning via towncrier, and branch protection for production-ready release management (#2)
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

