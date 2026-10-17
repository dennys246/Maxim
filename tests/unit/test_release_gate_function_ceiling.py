"""#1323: the 1.4.0 gate's function bound (owner decision 2026-10-16).

The gate's three functions (``start_simulation_mode``, ``run_agentic_loop``, ``execute_and_learn``) were held to 200
lines; the owner moved 1.4.0's bound to 1,000 and will reassess 200 (or about 500) for 1.5 at the 1.4 release. The
three ``functions`` entries are retired in ``scripts/release_gates.json`` and this test is the evidence of item
``1323.functions-1000``. Everything it reads comes from the gate file and the lint's own measurement, so the bound
cannot drift away from the gate, and every def in the three files is checked, so a rename plus a short wrapper does
not pass. It stays a strict red gate until the decomposition reaches the bound; then the item records it.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from scripts import lint_release_gates as L

ROOT = Path(__file__).resolve().parents[2]
ITEM_ID = "1323.functions-1000"


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="#1323: start_simulation_mode and run_agentic_loop are still over 1,000 lines",
)
def test_gated_functions_their_split_outs_and_their_files_are_within_the_bound():
    gate = json.loads((ROOT / "scripts" / "release_gates.json").read_text(encoding="utf-8"))["gates"]["1.4.0"]
    item = next(i for i in gate["items"] if i["id"] == ITEM_ID)
    found = re.search(r"at most ([\d,]+) lines", item["scope"])
    assert found, "the bound is read from the item's immutable scope"
    bound = int(found.group(1).replace(",", ""))
    roots = {(f["file"], f["qualname"]) for f in gate["functions"]}
    assert len(roots) == 3

    over = []
    for rel, qualname in sorted(roots):
        lines = L._measure(ROOT, rel, qualname)
        assert lines is not None and lines != -1, f"{rel}::{qualname} must be exactly one def (got {lines})"
        if lines > bound:
            over.append(f"{rel}::{qualname} = {lines}")
    baseline = L.FL.parse_baseline((ROOT / L.FL.BASELINE_REL).read_text(encoding="utf-8"))
    for (rel, qualname), via in sorted(L.descendants(baseline, roots).items()):
        lines = L._measure(ROOT, rel, qualname)
        if lines == -1 or (lines is not None and lines > bound):
            over.append(f"{rel}::{qualname} = {lines} (split or moved out of {via})")
    for rel in sorted({r for r, _ in roots}):
        for qualname, nodes in L.FL.defs_in((ROOT / rel).read_text(encoding="utf-8"), rel).items():
            over += [f"{rel}::{qualname} = {L.FL.span(n)}" for n in nodes if L.FL.span(n) > bound]
    assert not over, f"over {bound} lines: {sorted(set(over))}"
