"""Loop-level selection golden: the 1.3.2 decomposition's selection gate (slice 0).

WHY THIS EXISTS. The roadmap (``docs/plans/roadmap_1_3_x.md`` §"The decomposition") makes behaviour
preservation the merge gate for every ``run_agentic_loop`` slice. Until this file the
"byte-identical selection" gate (``test_decision_provenance.py``) was NAc-level only: nothing ran
the LOOP, so an extraction that changed what the loop feeds ``recommend_action`` (clusters, drives,
the Wire-4 fear read, the tick cadence, which proposal reaches the executor) stayed green. Owner
decision 2026-10-05: this loop-level golden is THE selection gate; the NAc-level tests stay.

HOW IT RUNS. ``tests/unit/_loop_harness.py`` (read its docstring): the REAL loop, substrate-primary,
on the canonical builders, single-threaded on a global step clock against an in-process scripted
world. Two arms:
  1. ``shore``: dry land, food draining past the hunger threshold, the NAc seeded with a cluster
     reward on the shore world cluster -- learned bias, sub-threshold idle ticks, then drive-led eat.
  2. ``fear_water``: submerged, Wire-4 ``cluster_fear`` seeded (two saturating ``drive:oxygen``
     writes) on the water cluster; ``flee`` is dead in water (the bridge refuses it, as the live
     pathfinder does) and the loop must select and EXECUTE ``escape_water`` and surface -- the
     Exp 60/61 decision path. Asserted directly AND as part of the golden.

WHAT IS RECORDED. Per substrate tick (the loop's own ``substrate_telemetry.snapshot`` call site):
step, step-clock time, gated flag, proposed tool/params/clusters/confidence, and the
``recommend_action`` decision record(s) for that tick (``nac._emit_recommend_action_event``'s
keywords: score components, runner-up, candidates, consulted bias, active clusters). Every
``executor.execute`` call in order. The loop's LIFECYCLE (slice 1's ground): the hub session start,
the capture worker start, every Hippocampus capture (tool + situation), the state persists, the
consolidation flavour the session ends with, and every 2S-d situation-cue call with its clusters
(the cue is recall-only until 2S-e, so no decision depends on it; the trace pins that the substrate
tick hands the hub's cue the tick's clusters). EC node ids are pinned to a counter and relabelled
by first appearance.

CANONICAL FORM. ``json.dumps(sort_keys=True)`` with every float rounded to ``FLOAT_DECIMALS`` (9).
Every recorded float is IEEE-754 arithmetic in a fixed order, identical across platforms except for
vectorised reductions (the encoder's numpy dot products), which can differ in the last ulp (~1e-16
relative) between CPUs. 9 decimals is seven orders of magnitude above that and three below the
coarsest value a decision reads (NAc scores move in 1e-4 steps) -- tight enough that any real change
shows, loose enough not to pin the ulp. The fixture records the precision it was written at.

REGENERATION RULE (the ``encoder_golden_v1.json`` posture). The fixture is regenerated ONLY from the
pre-change commit, never by pasting the new output: a slice that moves code must pass this test
UNCHANGED. Regenerate with ``python tests/unit/test_agent_loop_selection_golden.py --regen`` on the
commit you mean to pin; the fixture records ``generated_at_commit`` and the regen refuses a dirty
``src/`` tree. An intentional behaviour change (not a pure extraction) regenerates in its own commit
with the diff justified in the PR.

PROVEN ABLE TO FAIL (deletion probes on ``src/``, ``python -B``, slice 0, 2026-10-05):

  ===================================================  =====  ==========  ======  =========
  probe                                                shore  fear_water  escape  lifecycle
  ===================================================  =====  ==========  ======  =========
  NAc argmax + drive-gate tie-break flipped            pass   FAIL        pass    pass
  ``cluster_reward_bias`` term zeroed                  FAIL   FAIL        pass    pass
  Wire-4 fear read disabled (``fear_need = 0.0``)      pass   FAIL        FAIL    FAIL
  ``_start_bio_session`` dropped                       FAIL   FAIL        pass    FAIL
  setup cue resolved as ``NO_SITUATION_CUE``           FAIL   FAIL        pass    FAIL
  tick passes ``situation_cue=NO_SITUATION_CUE``       FAIL   FAIL        pass    FAIL
  substrate cadence ``llm_submit_interval`` 0.5->0.4   FAIL   FAIL        pass    pass
  ===================================================  =====  ==========  ======  =========

("escape" is ``test_fear_water_arm_selects_and_executes_escape_water``, "lifecycle" is
``test_the_loop_opens_its_session_and_cues_the_hub_with_each_ticks_clusters``.)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

if str(Path(__file__).resolve().parents[2]) not in sys.path:  # run as a script (--regen / --emit)
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.unit._loop_harness import (  # noqa: E402
    ARMS,
    ESCAPE,
    FLOAT_DECIMALS,
    REPO_ROOT,
    WrongCheckout,
    assert_this_checkout,
    canonical_json,
    run_arm,
)

FIXTURE = REPO_ROOT / "tests" / "fixtures" / "agent_loop_selection_golden_v1.json"


def _emit_in_subprocess(arm: str, hashseed: str, tmp_path: Path) -> str:
    work = tmp_path / f"sub_{arm}_{hashseed}"
    work.mkdir()
    env = dict(os.environ)
    env.update(
        PYTHONHASHSEED=hashseed,
        PYTHONPATH=os.pathsep.join([str(REPO_ROOT / "src"), str(REPO_ROOT), env.get("PYTHONPATH", "")]).rstrip(
            os.pathsep
        ),
        HOME=str(work / "home"),
        MAXIM_DATA_HOME=str(work / "maxim_home"),
    )
    out = subprocess.run(
        [sys.executable, "-B", str(Path(__file__).resolve()), "--emit", arm, str(work)],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )
    assert out.returncode == 0, out.stderr[-4000:]
    return out.stdout.split("<<<GOLDEN>>>\n", 1)[1]


# ── tests ─────────────────────────────────────────────────────────────────


def _fixture() -> dict[str, Any]:
    fx = json.loads(FIXTURE.read_text())
    assert fx["float_decimals"] == FLOAT_DECIMALS, "the fixture was written at another precision; regenerate"
    return fx


@pytest.mark.timeout(240)
@pytest.mark.parametrize("arm", sorted(ARMS))
def test_loop_selection_matches_the_golden(arm: str, tmp_path: Path) -> None:
    """THE selection gate: the loop's per-tick decisions and executor calls equal the pinned trace,
    and a second in-process run is identical (no hidden state carried between runs)."""
    assert_this_checkout()  # a worktree without PYTHONPATH would otherwise test main's code
    first = canonical_json(run_arm(arm, _mkdir(tmp_path / "a")))
    second = canonical_json(run_arm(arm, _mkdir(tmp_path / "b")))
    assert first == second, "two in-process runs differ (not deterministic): " + _first_difference(
        json.loads(second), json.loads(first)
    )
    expected = canonical_json(_fixture()["arms"][arm])
    assert first == expected, _first_difference(json.loads(first), json.loads(expected))


@pytest.mark.timeout(240)
def test_fear_water_arm_selects_and_executes_escape_water(tmp_path: Path) -> None:
    """The Exp 60/61 path, asserted directly: Wire-4 fear on the water cluster drives the loop to
    SELECT ``escape_water`` while submerged, the executor RUNS it, and the bot surfaces -- and the
    shore arm, with no fear, never selects it."""
    trace = run_arm("fear_water", _mkdir(tmp_path / "w"))
    water = trace["seed_clusters"]["world"]
    escapes = [t for t in trace["ticks"] if t["tool"] == ESCAPE]
    assert escapes, [t["tool"] for t in trace["ticks"]]
    assert escapes[0]["clusters"]["world"] == water, "escape_water was not selected in the water situation"
    rec = escapes[0]["recommend"][-1]
    assert rec["best_tool"] == ESCAPE and rec["passed_gate"] and rec["score_components"]["drive"] > 0.0
    assert any(c["tool"] == ESCAPE and c["success"] for c in trace["executor_calls"])
    assert trace["end_submerged"] is False and "escape_water" in trace["world_actions"]
    shore = run_arm("shore", _mkdir(tmp_path / "s"))
    assert all(t["tool"] != ESCAPE for t in shore["ticks"])


@pytest.mark.timeout(240)
def test_the_loop_opens_its_session_and_cues_the_hub_with_each_ticks_clusters(tmp_path: Path) -> None:
    """Slice 1's ground, asserted directly (it is also in the golden): the loop starts the hub
    session and the capture worker once, ends the session with the flavour the harness asks for
    (``consolidation="full"``), captures what it executes with the situation it acted in, and hands
    the hub's 2S-d situation cue every substrate tick's clusters (not ``NO_SITUATION_CUE``)."""
    trace = run_arm("fear_water", _mkdir(tmp_path / "l"))
    life = trace["lifecycle"]
    assert life["session_start"] == 1 and life["capture_worker_started"] == 1
    assert life["session_end"] == ["full"]
    assert life["persists"] >= 1
    assert life["captures"] and all(c["situation"] for c in life["captures"])
    assert {c["tool"] for c in life["captures"]} >= {ESCAPE}
    cued = [c["clusters"] for c in life["cue_calls"]]
    assert len(cued) == len(trace["ticks"]), "one cue per substrate tick"
    assert cued == [t["recommend"][-1]["current_clusters"] for t in trace["ticks"]]


def test_a_shadowing_install_is_refused(monkeypatch) -> None:
    import maxim

    monkeypatch.setattr(maxim, "__file__", "/elsewhere/site-packages/maxim/__init__.py")
    with pytest.raises(WrongCheckout):
        assert_this_checkout()


def test_the_golden_is_non_vacuous() -> None:
    """The pinned trace exercises what the gate claims to pin: several ticks per arm, a learned-bias
    component and a drive component in the shore arm's decisions, executor calls in both arms."""
    fx = _fixture()["arms"]
    for arm in ARMS:
        assert len(fx[arm]["ticks"]) >= 8 and fx[arm]["executor_calls"], arm
    comps = [r["score_components"] or {} for t in fx["shore"]["ticks"] for r in t["recommend"]]
    assert any(c.get("learned_bias", 0.0) > 0.0 for c in comps)
    assert any(c.get("drive", 0.0) > 0.0 for c in comps)
    assert len({t["tool"] for t in fx["shore"]["ticks"] if t["tool"]}) >= 2


@pytest.mark.timeout(600)
@pytest.mark.parametrize("arm", sorted(ARMS))
def test_golden_is_identical_across_processes_and_hash_seeds(arm: str, tmp_path: Path) -> None:
    """Two fresh interpreters with differing PYTHONHASHSEED produce the pinned trace byte for byte
    (set/dict iteration over hashed strings, import order, a fresh home)."""
    expected = canonical_json(_fixture()["arms"][arm])
    for seed in ("0", "4242"):
        assert _emit_in_subprocess(arm, seed, tmp_path) == expected + "\n", f"PYTHONHASHSEED={seed}"


# ── helpers / regeneration ────────────────────────────────────────────────


def _mkdir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _first_difference(got: Any, want: Any, path: str = "$") -> str:
    if type(got) is not type(want):
        return f"{path}: {got!r} != {want!r}"
    if isinstance(got, dict):
        for k in sorted(set(got) | set(want)):
            if k not in got or k not in want:
                return f"{path}.{k}: present in only one side"
            if got[k] != want[k]:
                return _first_difference(got[k], want[k], f"{path}.{k}")
    if isinstance(got, list):
        for i, (a, b) in enumerate(zip(got, want)):
            if a != b:
                return _first_difference(a, b, f"{path}[{i}]")
        if len(got) != len(want):
            return f"{path}: length {len(got)} != {len(want)}"
    return f"{path}: {got!r} != {want!r}"


def _head_commit() -> str:
    out = subprocess.run(["git", "rev-parse", "--short=12", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True)
    return out.stdout.strip()


def _src_is_clean() -> bool:
    out = subprocess.run(["git", "status", "--porcelain", "--", "src/"], cwd=REPO_ROOT, capture_output=True, text=True)
    return out.returncode == 0 and not out.stdout.strip()


if __name__ == "__main__":  # regeneration / subprocess entry point -- read the module docstring first
    import tempfile

    if len(sys.argv) >= 4 and sys.argv[1] == "--emit":
        print("<<<GOLDEN>>>\n" + canonical_json(run_arm(sys.argv[2], Path(sys.argv[3]))))
        raise SystemExit(0)
    if "--regen" not in sys.argv:
        raise SystemExit(
            "usage: python tests/unit/test_agent_loop_selection_golden.py --regen  (from the commit you mean to pin)"
        )
    if not _src_is_clean():
        raise SystemExit("refusing to regenerate: src/ has uncommitted changes (regenerate from a committed tree)")
    arms = {}
    for name in sorted(ARMS):
        with tempfile.TemporaryDirectory() as d:
            arms[name] = json.loads(canonical_json(run_arm(name, Path(d))))
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    payload = {"generated_at_commit": _head_commit(), "float_decimals": FLOAT_DECIMALS, "arms": arms}
    FIXTURE.write_text(json.dumps(payload, sort_keys=True, indent=1) + "\n")
    print(f"wrote {FIXTURE} at {payload['generated_at_commit']}")
