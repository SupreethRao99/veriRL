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
    GET  /blog    — rendered HTML view of BLOG.md
    GET  /blog/raw — raw BLOG.md markdown source

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    uv run --project . server
"""

from pathlib import Path

from fastapi.responses import HTMLResponse, PlainTextResponse
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


_BLOG_PATH = Path(__file__).parent.parent / "BLOG.md"

_BLOG_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>VeriRL — Training Blog</title>
  <link rel="stylesheet"
        href="https://cdnjs.cloudflare.com/ajax/libs/github-markdown-css/5.5.1/github-markdown-light.min.css">
  <script src="https://cdnjs.cloudflare.com/ajax/libs/marked/12.0.0/marked.min.js"></script>
  <style>
    body {{ box-sizing: border-box; min-width: 200px; max-width: 860px;
            margin: 40px auto; padding: 0 24px; font-family: sans-serif; }}
    .markdown-body img {{ max-width: 100%; }}
  </style>
</head>
<body class="markdown-body">
  <script>
    const raw = {raw_json};
    document.currentScript.insertAdjacentHTML('afterend', marked.parse(raw));
  </script>
</body>
</html>
"""


@app.get("/blog/raw", response_class=PlainTextResponse, tags=["Blog"])
def blog_raw():
    """Return the raw BLOG.md markdown source."""
    if not _BLOG_PATH.exists():
        return PlainTextResponse("Blog not found.", status_code=404)
    return _BLOG_PATH.read_text(encoding="utf-8")


@app.get("/blog", response_class=HTMLResponse, tags=["Blog"])
def blog():
    """Render BLOG.md as a styled HTML page."""
    if not _BLOG_PATH.exists():
        return HTMLResponse("<p>Blog not found.</p>", status_code=404)
    import json
    raw = _BLOG_PATH.read_text(encoding="utf-8")
    html = _BLOG_HTML_TEMPLATE.format(raw_json=json.dumps(raw))
    return HTMLResponse(html)


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
