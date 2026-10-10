"""GL3.B0 characterization (NOT a gate): breach -> pain-publish and breach -> protective-action latency.

WHY IT EXISTS. ``docs/plans/thalamic_relay.md`` §6 "GL3.B0" asks for the latency, in loop PASSES, from
a body breach to its pain publish and to the protective action, "on both arms", FROZEN with the commit
it measured before GL3.B4's gate design pass begins. GL3.B4's gate 3 compares its flag-on latency on
"the damage arm" against THIS frozen number, never a re-measurement. So this test does not judge the
latency; it pins it: a fresh measurement must equal ``tests/fixtures/afferent_latency_characterization_v1.json``
exactly, so any change to the latency (L1/L2/L5 fixes, the tracks, a cadence change) is visible, and the
file is regenerated only deliberately.

THE TWO ARMS (interpretation, stated because §6 does not name them): the two worlds of
``tests/unit/_loop_harness.py`` (the plan's "every scheduler gate runs through this harness"; GL3.B4's
guard tests are "the ``_loop_harness`` arms"), substrate-primary, each carrying one breach:
  * ``damage`` -- the harness's ``shore`` world (dry land, the seeded mine_block reward). Health drops
    from 20 to 8 hp (inside the 14 hp pain band) in the sleep right after pass 0, which selected
    mine_block: a nociceptive breach landing while a proposal is pending, the L1 case GL3.B4 preempts.
    Protective tools: ``flee`` (the innate ``health -> threat`` need).
  * ``drowning`` -- the harness's ``fear_water`` world (submerged) WITHOUT the seeded Wire-4 fear: with
    the seed the agent escapes at the first tick, so no breach ever happens (Exp 60's anticipation), and
    there is nothing to characterize. Oxygen drains 1/s and crosses the 14-bubble pain band at t > 6 s:
    a world-written breach whose protective action (``escape_water``) is LEARNED from the breach itself
    (the drive:oxygen pain writes the fear Wire 4 then reads). Protective tools: ``flee`` (proposed by the
    threat need, but dead in water) and ``escape_water``.

WHAT IS MEASURED. Through the REAL loop on the harness's step clock (the driver is
``test_afferent_red_gates_loop.drive_loop``; read its module docstring). A pass is one loop iteration
(``step_num``; idle passes count). The BREACH PASS is the first pass that starts strictly after the
breach time (the first pass whose synced body is in the band). Per arm:
  * ``publish_*``: the first ``PainBus.publish`` of the breach's failure mode (``drive:health`` /
    ``drive:oxygen``);
  * ``protective_attempt_*``: the first executor call of a protective tool, success or not;
  * ``protective_effective_*``: the first SUCCESSFUL executor call of a protective tool;
  * ``stale_executed``: every executor call between the breach pass and the first protective attempt
    (the action the loop dispatched on a world it no longer saw), with its success (the shore arm's
    seeded ``mine_block`` is proposed without coordinates, so the executor refuses it -- the dispatch,
    not the world effect, is what L1 is about);
each as ``*_pass``, ``*_latency_passes`` (minus the breach pass) and ``*_latency_s`` (step-clock
seconds after the breach pass started; passes differ in length -- an idle pass sleeps 0.05 s, a
running one to the 0.25 s period -- so both are recorded). Floats are rounded to 9 decimals.

TWO READINGS THE NUMBERS DEPEND ON (stated so GL3.B4 does not over-read them):
  * the damage arm's publish latency (+2 passes) depends on the stale ``mine_block`` being REFUSED
    before the tool runs (no coordinates -> "Missing required input"): a refused call reaches no
    tool-coupled ``evaluate_failures``, so the health breach waits for the next substrate tick's
    evaluation; a stale action that really ran could move this number;
  * the drowning arm's +0 is the substrate cadence's PHASE (the breach pass happens to be a
    substrate tick), not a mechanism that publishes a drift breach immediately.

A FROZEN RECORD, NOT A GOLDEN. The plan (§6 GL3.B0) freezes these numbers as GL3.B4's baseline, which
compares against them and never re-measures. So the fixture is never regenerated once ``src/`` moves:
the fixes it describes (#1176 L1, #1177 L2, GL3.B4) will move the numbers, and that is the point. The
equality check therefore runs only while ``src/`` is the tree that was measured (``src_tree``, clean);
on any other tree the test checks that the measurement still runs and still has the frozen record's
shape, and says it is comparing against history. The fixture records ``measured_at_commit``, ``src_tree``
and, as PROVENANCE only (never asserted, so the shared ``_loop_harness.py`` stays free to change),
``instrument_blobs``: the ``git hash-object`` of the harness, the driver and this file at measurement.
``--regen`` refuses a dirty ``src/`` and a tracked instrument file with uncommitted changes; run it only
to (re)create the record on the commit it describes, never to make a failing comparison pass.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

if str(Path(__file__).resolve().parents[2]) not in sys.path:  # run as a script (--regen)
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.unit._loop_harness import REPO_ROOT, _StepClock, assert_this_checkout  # noqa: E402
from tests.unit.test_afferent_red_gates_loop import drive_loop, minecraft_rig  # noqa: E402

FIXTURE = REPO_ROOT / "tests" / "fixtures" / "afferent_latency_characterization_v1.json"
DECIMALS = 9
# The instrument whose code the frozen numbers are a reading of (module docstring, REGENERATION RULE).
INSTRUMENT = (
    "tests/unit/_loop_harness.py",
    "tests/unit/test_afferent_red_gates_loop.py",
    "tests/unit/test_afferent_latency_characterization.py",
)


def blob_hash(rel: str) -> str:
    """``git hash-object <rel>`` computed in-process (sha1 over ``blob <len>\\0<bytes>``)."""
    data = (REPO_ROOT / rel).read_bytes()
    return hashlib.sha1(b"blob %d\x00" % len(data) + data).hexdigest()


def instrument_blobs() -> dict[str, str]:
    return {rel: blob_hash(rel) for rel in INSTRUMENT}


ARMS: dict[str, dict[str, Any]] = {
    "damage": {
        "world": "shore",
        "rig": {"submerged": False, "seed_reward": True, "hurt_after": 0.0},
        "breach": "health 20 -> 8 hp (pain band < 14) in the sleep after pass 0",
        "breach_t": 0.0,
        "failure_mode": "drive:health",
        "protective_tools": ["minecraft_player_flee"],
        "max_steps": 12,
    },
    "drowning": {
        "world": "fear_water (no seeded fear)",
        "rig": {"submerged": True},
        "breach": "oxygen 20 - t bubbles crosses the pain band (< 14) at t > 6 s",
        "breach_t": 6.0,
        "failure_mode": "drive:oxygen",
        "protective_tools": ["minecraft_player_flee", "minecraft_player_escape_water"],
        "max_steps": 110,
    },
}


def _r(x: float) -> float:
    r = round(float(x), DECIMALS)
    return 0.0 if r == 0 else r


def measure_arm(name: str, workdir: Path) -> dict[str, Any]:
    spec = ARMS[name]
    clock = _StepClock()
    trace = drive_loop(minecraft_rig(clock, workdir, **spec["rig"]), workdir, clock=clock, max_steps=spec["max_steps"])
    start = {p["pass"]: p["t"] for p in trace.passes}
    breach = next(p for p in trace.passes if p["t"] > spec["breach_t"])
    b_pass, b_t = breach["pass"], breach["t"]
    tools = set(spec["protective_tools"])

    def stamp(prefix: str, event: dict[str, Any] | None) -> dict[str, Any]:
        if event is None:
            return {f"{prefix}_pass": None, f"{prefix}_latency_passes": None, f"{prefix}_latency_s": None}
        out = {
            f"{prefix}_pass": event["pass"],
            f"{prefix}_latency_passes": event["pass"] - b_pass,
            f"{prefix}_latency_s": _r(start[event["pass"]] - b_t),
        }
        if "tool" in event:
            out[f"{prefix}_tool"] = event["tool"]
        return out

    publish = next((p for p in trace.pains if p["failure_mode"] == spec["failure_mode"] and p["pass"] >= b_pass), None)
    attempt = next((c for c in trace.calls if c["tool"] in tools and c["pass"] >= b_pass), None)
    effective = next((c for c in trace.calls if c["tool"] in tools and c["success"] and c["pass"] >= b_pass), None)
    until = attempt["pass"] if attempt is not None else float("inf")
    stale = [
        {"pass": c["pass"], "tool": c["tool"], "success": c["success"]}
        for c in trace.calls
        if b_pass <= c["pass"] < until and c["tool"] not in tools
    ]
    return {
        "world": spec["world"],
        "breach": spec["breach"],
        "breach_t": _r(spec["breach_t"]),
        "failure_mode": spec["failure_mode"],
        "protective_tools": sorted(tools),
        "max_steps": spec["max_steps"],
        "breach_pass": b_pass,
        "breach_pass_t": _r(b_t),
        **stamp("publish", publish),
        **stamp("protective_attempt", attempt),
        **stamp("protective_effective", effective),
        "stale_executed": stale,
    }


def measure_all(tmp_root: Path) -> dict[str, Any]:
    out = {}
    for name in sorted(ARMS):
        work = tmp_root / name
        work.mkdir(parents=True, exist_ok=True)
        out[name] = measure_arm(name, work)
    return out


# ── the test ──────────────────────────────────────────────────────────────


def _src_is_the_measured_tree(frozen: dict[str, Any]) -> bool:
    """``src/`` is the tree the record was measured on, with no uncommitted change under it."""
    try:
        head_src = _git("rev-parse", "HEAD:src")
        dirty = _git("status", "--porcelain", "--", "src/")
    except (SystemExit, OSError):  # _git raises SystemExit on a git error; OSError when git is absent
        return False
    return head_src == frozen["src_tree"] and not dirty


@pytest.mark.timeout(480)
def test_afferent_latency_against_the_frozen_record(tmp_path: Path) -> None:
    """On the measured tree, a fresh measurement equals the record (it is reproducible). On any other
    tree the record is history (module docstring): the measurement must still run and keep its shape."""
    assert_this_checkout()
    frozen = json.loads(FIXTURE.read_text())
    assert frozen["unit"] == "loop passes" and frozen["float_decimals"] == DECIMALS
    fresh = measure_all(tmp_path)
    assert set(fresh) == set(frozen["arms"])
    if not _src_is_the_measured_tree(frozen):
        for name in sorted(ARMS):
            assert set(fresh[name]) == set(frozen["arms"][name]), name
        return
    for name in sorted(ARMS):
        assert fresh[name] == frozen["arms"][name], (
            f"arm {name!r}: the measured latency moved from the frozen GL3.B0 characterization "
            f"(measured at {frozen['measured_at_commit']}); regenerate ONLY deliberately (module docstring).\n"
            f"fresh:  {json.dumps(fresh[name], sort_keys=True)}\nfrozen: {json.dumps(frozen['arms'][name], sort_keys=True)}"
        )


def test_the_frozen_characterization_is_non_vacuous() -> None:
    """The frozen numbers describe a breach that was published and acted on, in both arms."""
    arms = json.loads(FIXTURE.read_text())["arms"]
    assert set(arms) == set(ARMS)
    for name, arm in arms.items():
        assert arm["publish_pass"] is not None and arm["publish_latency_passes"] >= 0, name
        assert arm["protective_effective_pass"] is not None, name
        assert arm["protective_effective_latency_passes"] >= arm["publish_latency_passes"], name
    assert arms["damage"]["stale_executed"], "the damage arm's breach lands while a proposal is pending (L1)"


# ── regeneration ──────────────────────────────────────────────────────────


def _git(*args: str) -> str:
    out = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} failed: {out.stderr.strip()}")
    return out.stdout.strip()


if __name__ == "__main__":  # regeneration entry point -- read the module docstring first
    import tempfile

    if "--regen" not in sys.argv:
        raise SystemExit(
            "usage: python tests/unit/test_afferent_latency_characterization.py --regen  (from the commit you mean to pin)"
        )
    if _git("status", "--porcelain", "--", "src/"):
        raise SystemExit("refusing to regenerate: src/ has uncommitted changes (measure a committed tree)")
    dirty = [
        line for line in _git("status", "--porcelain", "--", *INSTRUMENT).splitlines() if not line.startswith("??")
    ]
    if dirty:
        raise SystemExit(f"refusing to regenerate: the instrument has uncommitted changes: {dirty}")
    blobs = instrument_blobs()
    for rel, digest in blobs.items():
        if _git("hash-object", rel) != digest:
            raise SystemExit(f"in-process blob hash of {rel} disagrees with git hash-object")
    assert_this_checkout()
    with tempfile.TemporaryDirectory() as d:
        arms = measure_all(Path(d))
    payload = {
        "what": "GL3.B0 characterization: breach -> pain publish and breach -> protective action, substrate-primary",
        "unit": "loop passes",
        "float_decimals": DECIMALS,
        "measured_at_commit": _git("rev-parse", "HEAD"),
        "src_tree": _git("rev-parse", "HEAD:src"),
        "measured_by": "tests/unit/test_afferent_latency_characterization.py --regen",
        "instrument_blobs": blobs,
        "arms": arms,
    }
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(json.dumps(payload, sort_keys=True, indent=1) + "\n")
    print(f"wrote {FIXTURE} at {payload['measured_at_commit']}")
