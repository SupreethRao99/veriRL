# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick Start

### Install dependencies
```bash
uv sync
```

### Run the server locally
```bash
uvicorn server.app:app --reload
# or
uv run --project . server
# or with custom port
uv run --project . server --port 8001
```

### Build the Docker image
```bash
docker build -t verirl:latest -f server/Dockerfile .
```

### Test environment logic directly (no server needed)
```bash
python3 server/verirl_env_environment.py
```

### Run tests
```bash
pytest
# with coverage
pytest --cov
```

## Development Workflow

This repository enforces a **strict PR-based workflow** with branch protection on `main`.

### 1. Create Feature Branch
```bash
git checkout -b feature/your-feature-name
# or: bugfix/xyz, docs/xyz, refactor/xyz
```

### 2. Make Changes
```bash
# Edit files
vim server/verirl_env_environment.py

# Test locally
pytest
openenv validate
docker build -f server/Dockerfile .
```

### 3. Create Changelog Fragment
For **every change**, create a fragment in `.changelog.d/`:

```bash
# Feature
echo "Added new task type" > .changelog.d/123.feature.md

# Bug fix
echo "Fixed STDOUT format" > .changelog.d/124.bugfix.md

# Documentation
echo "Updated README" > .changelog.d/125.doc.md

# Internal/misc (not shown in changelog)
echo "Code refactoring" > .changelog.d/126.misc.md
```

Fragment format: `.changelog.d/<ID>.<TYPE>.md`
- ID: PR number, commit hash, or unique identifier
- TYPE: feature, bugfix, doc, misc
- Content: One-line description

### 4. Commit & Push
```bash
git add server/ tests/ .changelog.d/
git commit -m "Add feature

- Detailed change 1
- Detailed change 2
- Detailed change 3"

git push origin feature/your-feature-name
```

### 5. Create Pull Request
1. Go to https://github.com/SupreethRao99/veriRL/pulls
2. Click "New pull request"
3. Select: `main` ← `feature/your-feature-name`
4. Fill description and submit

### 6. CI/CD Runs Automatically
CI validates on every PR:
- ✓ `openenv validate` (spec compliance)
- ✓ `docker build` (Dockerfile validation)

Merge is blocked until all checks pass.

### 7. Merge to Main
When approved + checks pass, merge via GitHub UI.

On merge, auto-deploy pipeline runs:
1. Final validation
2. Docker build
3. Push to `ghcr.io/SupreethRao99/veriRL:main`
4. Deploy to HF Spaces

## Release Management

Releases use **towncrier** for automated changelog generation and versioning.

### Creating a Release

1. Go to https://github.com/SupreethRao99/veriRL/actions
2. Click "Release" workflow
3. Click "Run workflow"
4. Enter version (e.g., `0.3.0`)
5. Workflow automatically:
   - Generates `CHANGELOG.md` from fragments
   - Updates version in `pyproject.toml`
   - Creates git tag `v0.3.0`
   - Builds Docker image
   - Publishes to:
     * `ghcr.io/SupreethRao99/veriRL:v0.3.0`
     * `ghcr.io/SupreethRao99/veriRL:latest`
   - Creates GitHub Release with changelog

### Version Numbering (Semantic Versioning)

- `0.1.0` → `0.2.0` = new feature/task (minor bump)
- `0.2.0` → `0.2.1` = bug fix (patch bump)
- `0.2.1` → `1.0.0` = breaking change (major bump)

### Manual Release (if workflow fails)

```bash
# Generate changelog from fragments
towncrier build --version 0.3.0

# Review and commit
git add CHANGELOG.md pyproject.toml .changelog.d/
git commit -m "Release v0.3.0"

# Create tag and push
git tag -a v0.3.0 -m "Release v0.3.0"
git push origin main v0.3.0

# Build and push Docker image
docker build -t ghcr.io/SupreethRao99/veriRL:v0.3.0 -f server/Dockerfile .
docker push ghcr.io/SupreethRao99/veriRL:v0.3.0
```

## Branch Protection

The `main` branch has these protections:
- ✓ **No direct pushes** — all changes via PR
- ✓ **Requires CI checks** — openenv validate + docker build
- ✓ **Requires approval** — recommended 1 approval
- ✓ **Requires up-to-date** — PR must be rebased with main
- ✓ **Prevents force push** — cannot overwrite history
- ✓ **Prevents deletion** — main cannot be deleted

If you try to push directly to main:
```bash
git push origin main
# ERROR: failed to push some refs to ...
# Use a feature branch and PR instead
```

## Container Registry

Docker images are published to GitHub Container Registry (ghcr.io).

### On Merge to Main
```bash
docker pull ghcr.io/SupreethRao99/veriRL:main
docker run -p 8000:8000 ghcr.io/SupreethRao99/veriRL:main
```

### On Release v0.3.0
```bash
docker pull ghcr.io/SupreethRao99/veriRL:v0.3.0
docker pull ghcr.io/SupreethRao99/veriRL:latest  # same as v0.3.0
docker run -p 8000:8000 ghcr.io/SupreethRao99/veriRL:v0.3.0
```

## CI/CD Pipeline

### Workflows

**`.github/workflows/ci-cd.yml`** (automatic on push/PR)
- Triggers: push to main, PR to main
- Jobs:
  1. `validate` — runs `openenv validate`
  2. `docker-build` — tests Docker image builds
  3. `deploy` — (on merge only) pushes image + deploys to HF Spaces

**`.github/workflows/release.yml`** (manual trigger)
- Triggers: workflow_dispatch
- Creates release with versioned Docker image

**`.github/workflows/validate.yml`** (PR checks)
- Triggers: PR to main
- Quick validation (openenv validate, docker build)

## Version Traceability

Every change is trackable:

```
Commit SHA → merged to main → Docker:main pushed → (optional) Release v0.3.0 → Docker:v0.3.0
```

**Changelog fragments** are combined into `CHANGELOG.md` during release:
- Feature → under "Features" section
- Bugfix → under "Bug Fixes" section
- Doc → under "Documentation" section
- Misc → not shown (internal changes only)

## Documentation Files

- **CONTRIBUTING.md** — complete development workflow
- **RELEASES.md** — release management details
- **CHANGELOG.md** — generated changelog
- **.github/SETUP.md** — initial setup instructions
- **.github/BRANCH_PROTECTION.md** — branch protection details
- **.github/DEPLOYMENT.md** — deployment info

## Architecture

This is an **OpenEnv** environment — a framework for exposing RL-style environments (reset/step/state) over HTTP and WebSocket. The package is named `openenv-verirl_env` and installed as `verirl_env`.

**Data flow:**
1. `models.py` — defines `VerirlAction` (input) and `VerirlObservation` (output) as Pydantic models extending OpenEnv base types
2. `server/verirl_env_environment.py` — `VerirlEnvironment` implements the `Environment` interface with `reset()` and `step()` methods; this is the pure logic layer
3. `server/app.py` — wraps `VerirlEnvironment` with `create_app()` from `openenv-core` to produce a FastAPI app exposing REST + WebSocket endpoints
4. `client.py` — `VerirlEnv` extends `EnvClient` and connects via WebSocket; `_step_payload()` serializes actions, `_parse_result()` deserializes observations

**Key design points:**
- `VerirlEnvironment.SUPPORTS_CONCURRENT_SESSIONS = True` enables factory mode — each WebSocket client gets its own environment instance. The `max_concurrent_envs` parameter in `app.py` controls how many simultaneous sessions are allowed (currently set to 10).
- The client uses persistent WebSocket connections (not per-request HTTP) for low latency. When using `VerirlEnv.from_docker_image()`, the client manages the Docker container lifecycle automatically.
- When connecting to an existing server (`VerirlEnv(base_url=...)`), `close()` does NOT stop the server.

**Package layout quirk:** `pyproject.toml` maps the package root (`.`) to `verirl_env` and `server/` to `verirl_env.server`, so imports within the server use relative imports (`from ..models import ...`) with a fallback for direct execution.

## Important Constraints

### STDOUT Format (Strict Compliance Required)

The inference script MUST output ONLY these three line types to stdout:
```
[START] task=<task> env=<benchmark> model=<model>
[STEP] step=<n> action=<action> reward=<0.00> done=<true|false> error=<msg|null>
[END] success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
```

NO other output to stdout (no comments, no validation messages, no summaries).
Errors must be single-line, max 80 chars, no newlines.

### OpenEnv Validation

Must pass `openenv validate` before any release:
```bash
openenv validate
# Should output: [OK] verirl: Ready for multi-mode deployment
```

### Docker Build

Must build successfully:
```bash
docker build -t verirl:test -f server/Dockerfile .
# No errors allowed
```
