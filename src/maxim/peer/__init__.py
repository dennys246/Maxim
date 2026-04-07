"""Peer-mode helpers for connecting to a Maxim leader.

Symmetric to `maxim tunnel setup` on the leader side: `maxim peer connect`
on a peer machine configures the local Maxim to route inference to a
leader's tunnel URL, stores the API key, verifies connectivity, and
persists the config so future `maxim` runs just work.
"""

from maxim.peer.cli import run_peer_connect_subcommand

__all__ = ["run_peer_connect_subcommand"]
