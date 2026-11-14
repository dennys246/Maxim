"""Deterministic fast-suite shards (#1645): this process collects the WHOLE suite and keeps only its shard's files.

Loaded only by ``scripts/run_fast_suite.py`` (``-p tests._shard_plugin --shard k/N``); without ``--shard`` it does
nothing. Collecting everything (instead of passing a file list) keeps the import-time state of every shard identical
to the serial run's: every test module, including one whose tests are all deselected, is imported at collection in
the same order as serially (#1645 review, claim 1 and S4). Only which tests have already RUN in this process differs.

A module goes to ``suite_shards.json``'s index for it, else to ``sha256(path) mod N``, so adding a test file never
moves another one. After collection the plugin writes ``--shard-record`` (JSON): the item set it saw after ``-m``
(the whole selection), the items it kept, and every ``Module`` collector it visited, so the runner can reconcile the
shards as SETS (S3), never as counts.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.run_fast_suite import MAP_REL, MapError, parse_map


def shard_of(path: str, n: int, mapping: dict[str, int]) -> int:
    """The shard of a repo-relative test-module path (POSIX separators)."""
    if path in mapping:
        return mapping[path]
    return int(hashlib.sha256(path.encode("utf-8")).hexdigest(), 16) % n


def load_map(root: Path, n: int) -> dict[str, int]:
    """``scripts/suite_shards.json``'s ``shards`` for ``n``, parsed by the runner's own rules (duplicate keys, indices,
    ``n``: ``scripts/run_fast_suite.py::parse_map``); a malformed map is a usage error (the runner exits non-zero)."""
    try:
        return parse_map((root / MAP_REL).read_text(encoding="utf-8"), n)
    except MapError as e:
        raise pytest.UsageError(str(e)) from e


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("shard", "deterministic fast-suite shards (#1645)")
    group.addoption("--shard", default=None, help="k/N: keep only shard k of N (0-based)")
    group.addoption("--shard-record", default=None, help="where to write this shard's collection record (JSON)")


class _Shard:
    def __init__(self, config: pytest.Config, k: int, n: int, record: Path) -> None:
        self.root = Path(str(config.rootpath))
        self.k, self.n, self.record = k, n, record
        self.mapping = load_map(self.root, n)
        self.modules: list[str] = []
        self.collect_outcomes: dict[str, str] = {}

    def _rel(self, path: Path) -> str:
        return path.resolve().relative_to(self.root.resolve()).as_posix()

    @pytest.hookimpl
    def pytest_collectreport(self, report: pytest.CollectReport) -> None:
        if report.outcome == "passed" and report.nodeid.endswith(".py") and "::" not in report.nodeid:
            self.modules.append(report.nodeid)
        elif report.outcome != "passed":
            # A collector skipped or failed at collection (e.g. a module-level importorskip). Every shard collects
            # the whole suite, so every shard reports it; the runner requires the same set from all N and keeps one.
            self.collect_outcomes[report.nodeid] = report.outcome

    @pytest.hookimpl(trylast=True)  # after -m/-k deselection, so `whole` is the suite's real selection
    def pytest_collection_modifyitems(self, config: pytest.Config, items: list[pytest.Item]) -> None:
        whole = [i.nodeid for i in items]
        keep: list[pytest.Item] = []
        drop: list[pytest.Item] = []
        for item in items:
            (keep if shard_of(self._rel(item.path), self.n, self.mapping) == self.k else drop).append(item)
        if drop:
            config.hook.pytest_deselected(items=drop)
        items[:] = keep
        payload = {
            "_format_version": "1.0",
            "shard": self.k,
            "n": self.n,
            "whole": whole,
            "selected": [i.nodeid for i in keep],
            "modules": sorted(set(self.modules)),
            "module_shards": {m: shard_of(m, self.n, self.mapping) for m in sorted(set(self.modules))},
            "collect_outcomes": dict(sorted(self.collect_outcomes.items())),
        }
        self.record.write_text(json.dumps(payload, indent=0), encoding="utf-8")


def pytest_configure(config: pytest.Config) -> None:
    spec = config.getoption("--shard")
    if spec is None:
        return
    try:
        k_s, n_s = spec.split("/")
        k, n = int(k_s), int(n_s)
    except ValueError as e:
        raise pytest.UsageError(f"--shard {spec!r}: expected k/N") from e
    if not (n >= 1 and 0 <= k < n):
        raise pytest.UsageError(f"--shard {spec!r}: needs N >= 1 and 0 <= k < N")
    record = config.getoption("--shard-record")
    if not record:
        raise pytest.UsageError("--shard needs --shard-record")
    config.pluginmanager.register(_Shard(config, k, n, Path(record)), "maxim-shard")
