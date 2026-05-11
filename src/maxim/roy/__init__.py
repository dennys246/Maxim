"""Roy harness — persona-convergence crucible utilities.

Currently exposes the ``maxim roy diff`` subcommand, which delegates
to :mod:`maxim.analysis.substrate_diff` for substrate divergence
analysis between two sim_reports session directories.
"""

from __future__ import annotations

from maxim.roy.cli import run_roy_subcommand

__all__ = ["run_roy_subcommand"]
