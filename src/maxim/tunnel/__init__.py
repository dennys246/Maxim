"""Cloudflare tunnel integration for Maxim leader-mode clusters.

Provides a `maxim tunnel` subcommand that wraps cloudflared's CLI to set up,
inspect, and manage a tunnel that exposes the auto-spawned llama-cpp-server
to peers outside the local network.

Once configured, leader-mode detection (runtime/leader_mode.py) fires
automatically on subsequent `maxim` startups — no env vars needed.
"""

from maxim.tunnel.cli import run_tunnel_subcommand

__all__ = ["run_tunnel_subcommand"]
