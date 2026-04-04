# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Features
- Task enumeration and automatic grader validation in inference script
- GitHub Actions CI/CD pipeline with automatic validation, Docker build, and HF Spaces deployment
- Comprehensive pre-submission checklist and deployment guide
- Strict STDOUT format compliance (only [START]/[STEP]/[END] lines)

## [0.2.0] - 2026-04-04

### Features
- Implement complete VeriRL Verilog hardware design environment
  - 3 tasks: MAC unit (easy), AXI-Stream FIFO (medium), 4×4 systolic array (hard)
  - Real EDA tool grading (iverilog, yosys, vvp)
  - Deterministic scoring with compile/sim/timing/area dimensions
  - Dense per-step reward function
  - 22, 34, 76 test assertions per task

### Features
- OpenEnv environment implementation
  - `reset()` / `step()` / `state()` endpoints
  - WebSocket support for persistent sessions
  - Concurrent session support (factory mode)
  - Full environment state tracking

### Features
- Baseline inference script
  - OpenAI Client integration
  - Structured [START]/[STEP]/[END] logging format
  - Task enumeration and grader validation
  - Per-task time budgets (4/6/8 minutes)

### Features
- Docker support
  - Multi-stage build with EDA tools (iverilog, yosys)
  - Automatic health checks
  - FastAPI app on port 8000

### Features
- GitHub Actions CI/CD
  - Automatic validation on every push
  - Docker build testing
  - Automatic deployment to HF Spaces

### Documentation
- Comprehensive README with environment description
- Task specifications with test details
- Setup and deployment instructions
- Baseline scores and interpretation

---

**How to use this changelog:**

1. **During development:** Create fragment files in `.changelog.d/`
   - Example: `.changelog.d/123.feature.md` with feature description

2. **Before release:** Run `towncrier build --version X.Y.Z`
   - Combines all fragments into CHANGELOG.md
   - Removes fragment files
   - Updates version in pyproject.toml

3. **Release:** Use GitHub Actions
   - Go to Actions → Release
   - Enter version (e.g., 0.2.1)
   - Workflow creates tag, generates changelog, publishes release
