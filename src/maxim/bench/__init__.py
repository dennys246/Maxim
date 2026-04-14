"""Benchmark harnesses for Maxim.

Each submodule defines one tight-loop benchmark that exercises the LLM
path directly (no sim orchestrator, no agent loop) so timing measurements
are not muddied by workload cadence. Built for Plan 4 A+B (Phase D
observability follow-ups) to close out the LLM-path stability story for
version 0.4.

Submodules:

- ``recovery_time``: fires chat completions in a tight loop against a
  configured peer URL. Used to rigorously measure leader-restart
  recovery time without the sim orchestrator's local-work cadence.

CLI entry: ``maxim bench <subcommand>`` (see ``run_bench_subcommand``
in :mod:`maxim.bench.cli`).
"""

from __future__ import annotations

from maxim.bench.cli import run_bench_subcommand

__all__ = ["run_bench_subcommand"]
