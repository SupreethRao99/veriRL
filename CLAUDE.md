# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Install dependencies
```bash
uv sync
```

### Run the server locally
```bash
uvicorn server.app:app --reload
# or
uv run --project . server
# or
uv run --project . server --port 8001
```

### Build the Docker image
```bash
docker build -t verirl_env-env:latest -f server/Dockerfile .
```

### Test environment logic directly (no server needed)
```bash
python3 server/verirl_env_environment.py
```

### Deploy to Hugging Face Spaces
```bash
openenv push
# or with options
openenv push --repo-id my-org/my-env --private
```

### Run tests
```bash
pytest
# with coverage
pytest --cov
```

## Architecture

This is an **OpenEnv** environment — a framework for exposing RL-style environments (reset/step/state) over HTTP and WebSocket. The package is named `openenv-verirl_env` and installed as `verirl_env`.

**Data flow:**
1. `models.py` — defines `VerirlAction` (input) and `VerirlObservation` (output) as Pydantic models extending OpenEnv base types
2. `server/verirl_env_environment.py` — `VerirlEnvironment` implements the `Environment` interface with `reset()` and `step()` methods; this is the pure logic layer
3. `server/app.py` — wraps `VerirlEnvironment` with `create_app()` from `openenv-core` to produce a FastAPI app exposing REST + WebSocket endpoints
4. `client.py` — `VerirlEnv` extends `EnvClient` and connects via WebSocket; `_step_payload()` serializes actions, `_parse_result()` deserializes observations

**Key design points:**
- `VerirlEnvironment.SUPPORTS_CONCURRENT_SESSIONS = True` enables factory mode — each WebSocket client gets its own environment instance. The `max_concurrent_envs` parameter in `app.py` controls how many simultaneous sessions are allowed (currently set to 1).
- The client uses persistent WebSocket connections (not per-request HTTP) for low latency. When using `VerirlEnv.from_docker_image()`, the client manages the Docker container lifecycle automatically.
- When connecting to an existing server (`VerirlEnv(base_url=...)`), `close()` does NOT stop the server.

**Package layout quirk:** `pyproject.toml` maps the package root (`.`) to `verirl_env` and `server/` to `verirl_env.server`, so imports within the server use relative imports (`from ..models import ...`) with a fallback for direct execution.
