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

