"""Maxim doctor — environment diagnostics + guided setup.

`maxim doctor` inspects the local system (platform, GPU, LLM server, tunnel,
API key) and prints platform-specific fix suggestions for each failing check.
Designed so a new user running `maxim doctor` gets a complete, actionable
picture of their setup without reading docs.
"""

from maxim.doctor.cli import run_doctor_subcommand, run_peer_subcommand

__all__ = ["run_doctor_subcommand", "run_peer_subcommand"]
