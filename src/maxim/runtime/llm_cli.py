"""``maxim llm``: operator commands for the local LLM server (#1344 PR 3b).

Only ``reap`` today: report the maxim-recorded server on a port and, at a terminal, kill it once the operator types
its pid (``runtime/llm_spawn_record.py::reap``). Not ``maxim peer llm``, which switches a leader's model.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence


def _default_port() -> int:
    """``MAXIM_AUTO_SPAWN_PORT``, else ``config.json::auto_spawn.port`` (the port the spawner uses)."""
    raw = os.environ.get("MAXIM_AUTO_SPAWN_PORT", "").strip()
    if raw:
        return int(raw)
    from maxim.runtime.config_loader import get_config

    return int(get_config().auto_spawn.port)


def run_llm_subcommand(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="maxim llm", description="Operator commands for the local LLM server.")
    sub = parser.add_subparsers(dest="action", required=True)
    reap_p = sub.add_parser(
        "reap",
        help="clean up a leftover maxim-spawned LLM server (asks you to type its pid; refuses outside a terminal)",
        description=(
            "Report the llama_cpp.server on PORT. If a maxim process recorded spawning it and its identity still "
            "matches, and stdin and stdout are a terminal, type its pid to stop it. Never touches a server maxim did "
            "not record. Exit 0 reaped or nothing recorded, 1 cancelled, 2 refused."
        ),
    )
    reap_p.add_argument("--port", type=int, default=None, help="the server's port (default: auto_spawn.port, 8100)")
    args = parser.parse_args(list(argv))

    from maxim.runtime import llm_spawn_record

    port = args.port if args.port is not None else _default_port()
    if not 1 <= port <= 65535:
        parser.error(f"--port {port} is not a TCP port")
    return llm_spawn_record.reap(port)
