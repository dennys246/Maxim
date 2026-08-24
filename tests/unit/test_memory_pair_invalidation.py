"""D2 guard — NAc and EC are invalidated together, never one alone.

NAc's ``reward_bias`` / ``cluster_reward_bias`` are keyed by EC node ids
(``runtime/bio_stack.py``, EC construction). Clearing one half leaves the other
referring to nodes that will never be re-allocated — persistence that looks
healthy while every restored bias dangles.

Before this fix ``MEMORY_PATHS`` had no ``ec`` key at all, so EC was
unreachable from the CLI *and* ``--clear-memory all`` wiped NAc while leaving
EC on disk. Every test below fails against the pre-fix table.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from maxim.cli_utils import MEMORY_PAIRS, MEMORY_PATHS, clear_memory, expand_paired_types


def _agent_home(tmp_path: Path, agent: str = "scout") -> Path:
    d = tmp_path / "agents" / agent
    d.mkdir(parents=True)
    for f in ("nac.json", "ec.json", "hippocampus.json"):
        (d / f).write_text("{}", encoding="utf-8")
    return d


def test_ec_is_reachable_from_the_memory_table():
    assert "ec" in MEMORY_PATHS, "EC had no key at all — a stale EC was unclearable (D2)"
    assert any("ec.json" in p for p in MEMORY_PATHS["ec"])


def test_nac_and_ec_are_declared_a_pair():
    assert any({"nac", "ec"} <= set(pair) for pair in MEMORY_PAIRS)


@pytest.mark.parametrize("requested", ["nac", "ec"])
def test_clearing_one_half_clears_the_other(tmp_path, requested):
    home = _agent_home(tmp_path)
    clear_memory(requested, home_dir=str(tmp_path))
    assert not (home / "nac.json").exists(), "NAc survived (D2)"
    assert not (home / "ec.json").exists(), "EC survived — biases now dangle (D2)"


@pytest.mark.parametrize("requested", ["nac", "ec"])
def test_pair_partner_is_reported_in_results(tmp_path, requested):
    _agent_home(tmp_path)
    results = clear_memory(requested, home_dir=str(tmp_path))
    assert {"nac", "ec"} <= set(results), f"partner missing from results: {results}"


def test_clear_all_includes_ec(tmp_path):
    home = _agent_home(tmp_path)
    clear_memory("all", home_dir=str(tmp_path))
    assert not (home / "ec.json").exists(), "'all' left EC behind (D2)"
    assert not (home / "nac.json").exists()


def test_unpaired_types_are_not_widened(tmp_path):
    """The pairing must not become a blanket 'clear everything'."""
    home = _agent_home(tmp_path)
    clear_memory("hippo", home_dir=str(tmp_path))
    assert not (home / "hippocampus.json").exists()
    assert (home / "nac.json").exists(), "clearing hippo must not touch NAc"
    assert (home / "ec.json").exists(), "clearing hippo must not touch EC"


def test_expansion_is_idempotent_and_order_preserving():
    assert expand_paired_types(["nac", "ec"]) == ["nac", "ec"]
    assert expand_paired_types(["ec", "nac"]) == ["ec", "nac"]
    assert expand_paired_types(["hippo", "nac"]) == ["hippo", "nac", "ec"]


def test_pairing_spans_every_agent_directory(tmp_path):
    a = _agent_home(tmp_path, "scout")
    b = _agent_home(tmp_path, "ranger")
    clear_memory("nac", home_dir=str(tmp_path))
    for d in (a, b):
        assert not (d / "nac.json").exists()
        assert not (d / "ec.json").exists(), f"{d.name} kept a stale EC (D2)"
