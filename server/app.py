# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the VeriRL Verilog hardware design environment.

This module creates an HTTP + WebSocket server that exposes VerirlEnvironment
over endpoints compatible with EnvClient.

Endpoints:
    POST /reset   — start a new episode (pass task_id in JSON body)
    POST /step    — execute one action
    GET  /state   — current episode state
    GET  /schema  — JSON schemas for action / observation / state
    WS   /ws      — persistent WebSocket session (required for multi-step episodes)
    GET  /health  — liveness check
    GET  /tasks   — list available benchmark tasks (custom extension)

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    uv run --project . server
"""

from pathlib import Path

from omegaconf import OmegaConf
from pydantic import BaseModel

# Support both in-repo and standalone imports
try:
    # In-repo imports (when running from OpenEnv repository)
    from openenv.core.env_server.http_server import create_app

    from ..models import VerirlAction, VerirlObservation
    from .verirl_env_environment import _TASK_CONFIGS, VerirlEnvironment
except ImportError:
    # Standalone imports (when environment is standalone with openenv from pip)
    from openenv.core.env_server.http_server import create_app

    from models import VerirlAction, VerirlObservation
    from server.verirl_env_environment import _TASK_CONFIGS, VerirlEnvironment


def _load_max_concurrent_envs() -> int:
    for candidate in [
        Path(__file__).parent.parent / "config.yaml",
        Path("/root/verirl/config.yaml"),
    ]:
        if candidate.exists():
            return int(OmegaConf.load(candidate).server.max_concurrent_envs)
    return 64  # safe default if config not found


class TaskInfo(BaseModel):
    """Metadata for a single benchmark task."""

    id: str
    name: str
    difficulty: str
    max_turns: int
    description: str


# Create the app — pass the class (factory) for concurrent WebSocket session support
app = create_app(
    VerirlEnvironment,
    VerirlAction,
    VerirlObservation,
    env_name="verirl_env",
    max_concurrent_envs=_load_max_concurrent_envs(),
)


@app.get("/tasks", response_model=list[TaskInfo], tags=["Tasks"])
def list_tasks():
    """List all available benchmark tasks with their metadata."""
    return [
        TaskInfo(
            id=config["id"],
            name=config["name"],
            difficulty=config["difficulty"],
            max_turns=config["max_turns"],
            description=config["description"],
        )
        for config in _TASK_CONFIGS
    ]


def main(host: str = "0.0.0.0", port: int = 8000):
    """
    Entry point for direct execution via uv run or python -m.

    Enables running the server without Docker:
        uv run --project . server
        uv run --project . server --port 8001
        python -m verirl_env.server.app
    """
    import argparse

    import uvicorn

    # Parse port from command line if provided
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=port)
    args, _ = parser.parse_known_args()

    uvicorn.run(app, host=host, port=args.port)


if __name__ == "__main__":
    main()
