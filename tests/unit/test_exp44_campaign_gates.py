"""Guards for the Exp 44b campaign runner's capture gate (scripts/exp44/campaign.py).

Pins the annotation-fraction counter against the REAL capture row shape,
produced by the actual ``_digest``/``_append`` code in
capture_paired_prompts.py — not a hand-imagined dict. Motivating bug
(2026-08-10 pilot): the counter read ``has_cluster_bias`` at the row's top
level while the hook nests it under ``world_state``, so annotation_fraction
was structurally 0.0 and the gate failed a healthy capture (30/34 prompts
demonstrably annotated). The verify-the-instrument class: the gate itself
must be tested against the artifact it gates.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[2]


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


capture_mod = _load("_exp44_capture_for_test", "scripts/exp44/capture_paired_prompts.py")
campaign_mod = _load("_exp44_campaign_for_test", "scripts/exp44/campaign.py")


def _real_row(decision_id: int, *, annotated: bool) -> dict:
    """Build a capture row through the REAL _digest, like _append does."""
    ctx = SimpleNamespace(
        body_state=None,
        cluster_bias_annotations=[("green_flame_warm_self", 0.9)] if annotated else None,
    )
    request = SimpleNamespace(context=ctx, triggering_input="x", available_tools=["a"])
    return {
        "decision_id": decision_id,
        "world_state": capture_mod._digest(request, "PROMPT"),
        "prompt_full": "PROMPT",
        "prompt_ablated": "ABLATED",
    }


def test_annotated_rows_counted(tmp_path):
    p = tmp_path / "capture.jsonl"
    rows = [_real_row(i, annotated=(i < 3)) for i in range(4)]
    p.write_text("\n".join(json.dumps(r) for r in rows))
    n_pairs, n_annotated = campaign_mod.capture_stats(p)
    assert n_pairs == 4
    assert n_annotated == 3  # fails with the old top-level read (would be 0)


def test_empty_bias_list_counts_as_annotated(tmp_path):
    """get_agent_tool_biases returns [] (not None) for no-bias agents; the
    producer having RUN is what has_cluster_bias records (is-not-None)."""
    p = tmp_path / "capture.jsonl"
    ctx = SimpleNamespace(body_state=None, cluster_bias_annotations=[])
    request = SimpleNamespace(context=ctx, triggering_input=None, available_tools=[])
    row = {
        "decision_id": 0,
        "world_state": capture_mod._digest(request, "P"),
        "prompt_full": "P",
        "prompt_ablated": "A",
    }
    p.write_text(json.dumps(row))
    assert campaign_mod.capture_stats(p) == (1, 1)


def test_capture_error_rows_are_not_pairs(tmp_path):
    p = tmp_path / "capture.jsonl"
    rows = [
        json.dumps(_real_row(0, annotated=True)),
        json.dumps({"decision_id": 1, "capture_error": "ValueError('x')"}),
        "not json at all",
        "",
    ]
    p.write_text("\n".join(rows))
    assert campaign_mod.capture_stats(p) == (1, 1)


def test_missing_file():
    assert campaign_mod.capture_stats(Path("/nonexistent/capture.jsonl")) == (0, 0)


def test_digest_shape_contract():
    """If capture_paired_prompts ever moves/renames has_cluster_bias, this
    fails here instead of silently zeroing the campaign gate again."""
    row = _real_row(0, annotated=True)
    assert row["world_state"]["has_cluster_bias"] is True
    row2 = _real_row(1, annotated=False)
    assert row2["world_state"]["has_cluster_bias"] is False


stats_mod = _load("_exp44_stats_for_test", "scripts/exp44/stats_counterfactual.py")


class TestPhantomGuard44c:
    """Longest-match ordering with the 44c hearth entities — bare 'hearth' is a
    substring of every hearth twin and must never shadow them."""

    def test_twin_wins_before_base(self):
        assert stats_mod.referenced_flame("green_hearth_b_warm_self") == "green_hearth_b"
        assert stats_mod.referenced_flame("purple_hearth_observe") == "purple_hearth"
        assert stats_mod.referenced_flame("hearth_warm_self") == "hearth"
        assert stats_mod.referenced_flame("green_flame_b_touch") == "green_flame_b"

    def test_phantom_in_collision_world(self):
        world = ("hearth",)
        assert stats_mod.is_phantom("green_flame_warm_self", world) is True
        assert stats_mod.is_phantom("green_hearth_warm_self", world) is True  # twin leak
        assert stats_mod.is_phantom("hearth_warm_self", world) is False
        assert stats_mod.is_phantom("sense_presence", world) is False

    def test_phantom_in_hearth_twin_world(self):
        world = ("green_hearth", "purple_hearth")
        assert stats_mod.is_phantom("green_hearth_warm_self", world) is False
        assert stats_mod.is_phantom("green_hearth_b_warm_self", world) is True
        assert stats_mod.is_phantom("green_flame_warm_self", world) is True
        # bare-hearth reference: not in this world's entity list -> phantom
        assert stats_mod.is_phantom("hearth_warm_self", world) is True


class TestCampaignLock:
    """Single-runner lock (third double-launch incident, 2026-08-10)."""

    def test_fresh_take_and_release_registration(self, tmp_path):
        assert campaign_mod.acquire_campaign_lock(tmp_path) is True
        lock = tmp_path / "campaign.lock"
        assert lock.exists() and int(lock.read_text()) == __import__("os").getpid()
        lock.unlink()  # cleanup (atexit registered, but keep tmp deterministic)

    def test_live_holder_refuses(self, tmp_path, capsys):
        import os

        (tmp_path / "campaign.lock").write_text(str(os.getpid()))  # our own pid = alive
        assert campaign_mod.acquire_campaign_lock(tmp_path) is False
        assert "already holds" in capsys.readouterr().err
        (tmp_path / "campaign.lock").unlink()

    def test_stale_holder_is_replaced(self, tmp_path):
        import os
        import subprocess

        # A pid that definitely exited: spawn-and-wait a trivial child.
        proc = subprocess.Popen(["true"])
        proc.wait()
        (tmp_path / "campaign.lock").write_text(str(proc.pid))
        assert campaign_mod.acquire_campaign_lock(tmp_path) is True
        assert int((tmp_path / "campaign.lock").read_text()) == os.getpid()
        (tmp_path / "campaign.lock").unlink()

    def test_garbage_lock_is_replaced(self, tmp_path):
        import os

        (tmp_path / "campaign.lock").write_text("not-a-pid")
        assert campaign_mod.acquire_campaign_lock(tmp_path) is True
        assert int((tmp_path / "campaign.lock").read_text()) == os.getpid()
        (tmp_path / "campaign.lock").unlink()
