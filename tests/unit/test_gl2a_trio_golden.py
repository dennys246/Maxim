"""GL2a's record never feeds memory strength: the trio, the tag and the strength stay byte-identical.

Grounding GL2a (``docs/plans/autonomic_layer.md`` §3.1.4, §5.2) adds ``ToolOutput.interoceptive_outcome``
and writes it into ``EncodingSignals.extra["interoception"]``. It must not move what a capture ENCODES
WITH: the existing trio the executor stamps (``drive_pressure_before`` -> ``drive_pressure``,
``drive_relief``, ``pain``) and what the Hippocampus derives from it (``encoding_tag`` ->
``storage_strength``). ``drive_relief`` reaches ``storage_strength``, so refilling it from the record
would be a memory-strength change with its own T1-16 walk.

This pins, per loop capture and in capture order, the trio plus ``encoding_tag`` and
``storage_strength`` over two runs of the REAL loop and capture, read back from the REAL Hippocampus:

- ``cradle``: the scripted cradle sequence (``cool_air.feel`` x2, ``fire_pit.warm_self`` x2,
  ``fire_pit.touch`` on ``infant_humanoid``), driven by ``tests/unit/_cradle_loop_driver.py``;
- ``fear_water``: the Minecraft arm of ``tests/unit/_loop_harness.py`` (the selection golden's arm).

REGENERATION RULE (the ``agent_loop_selection_golden_v1.json`` posture). The fixture is regenerated
ONLY from the pre-change commit, never by pasting new output: GL2a must pass this test UNCHANGED, and
so must the record reverted (the reverse deletion probe: the record never fed the trio). Regenerate with
``python tests/unit/test_gl2a_trio_golden.py --regen`` on the commit you mean to pin; the fixture records
``generated_at_commit`` and the regen refuses a dirty ``src/`` tree. Each arm runs in its own
subprocess (``--emit``), as the selection golden's do, so no import-time binding crosses arms.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

if str(Path(__file__).resolve().parents[2]) not in sys.path:  # run as a script (--regen / --emit)
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "gl2a_trio_golden_v1.json"
ARMS = ("cradle", "fear_water")
CRADLE_SCRIPT = ("cool_air_feel", "cool_air_feel", "fire_pit_warm_self", "fire_pit_warm_self", "fire_pit_touch")
pytestmark = pytest.mark.timeout(300)


def capture_view(trace: Any) -> dict[str, Any]:
    """What one loop capture encoded WITH, and what the Hippocampus derived from it."""
    enc = trace.encoding
    return {
        "tool": trace.tool_name,
        "pain": enc.pain,
        "drive_pressure": enc.drive_pressure,
        "drive_relief": enc.drive_relief,
        "encoding_tag": trace.encoding_tag,
        "storage_strength": trace.storage_strength,
    }


def loop_captures(traces: list[Any]) -> list[dict[str, Any]]:
    return [capture_view(t) for t in traces if getattr(t.encoding, "site", None) == "loop"]


def _emit_cradle(workdir: Path) -> list[dict[str, Any]]:
    from tests.unit._cradle_loop_driver import Step, run_cradle

    mp = pytest.MonkeyPatch()
    try:
        run = run_cradle(
            mp,
            workdir,
            body_ref="bodies/infant_humanoid",
            entity_refs=("items/cradle_cool_air", "items/cradle_fire_pit"),
            script=tuple(Step(t) for t in CRADLE_SCRIPT),
        )
    finally:
        mp.undo()
    assert run.executed == list(CRADLE_SCRIPT), run.executed
    return loop_captures(run.traces)


def _emit_fear_water(workdir: Path) -> list[dict[str, Any]]:
    from maxim.simulation import minecraft_harness
    from tests.unit._loop_harness import run_arm

    built: list[Any] = []
    real_build = minecraft_harness.build_minecraft_aut

    def _build(**kw: Any) -> Any:
        aut = real_build(**kw)
        built.append(aut)
        return aut

    minecraft_harness.build_minecraft_aut = _build
    try:
        run_arm("fear_water", workdir)
    finally:
        minecraft_harness.build_minecraft_aut = real_build
    (aut,) = built
    hippo = aut.bio.hippocampus
    assert hippo.flush(timeout=30.0)
    traces = sorted(
        (m for m in hippo._memories.values() if getattr(m, "capture_seq", None) is not None),
        key=lambda m: m.capture_seq,
    )
    return loop_captures(traces)


def emit(arm: str, workdir: Path) -> str:
    from tests.unit._loop_harness import canonical_json

    captures = {"cradle": _emit_cradle, "fear_water": _emit_fear_water}[arm](workdir)
    return canonical_json({"arm": arm, "captures": captures})


def _run_emit(arm: str) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as work:
        env = dict(os.environ, PYTHONPATH=str(REPO_ROOT / "src"), HOME=work)
        out = subprocess.run(
            [sys.executable, "-B", str(Path(__file__).resolve()), "--emit", arm, work],
            capture_output=True,
            text=True,
            env=env,
            cwd=str(REPO_ROOT),
            timeout=240,
        )
        assert out.returncode == 0, f"--emit {arm} failed:\n{out.stdout[-3000:]}\n{out.stderr[-3000:]}"
        return json.loads(out.stdout.split("<<<GOLDEN>>>\n", 1)[1])


@pytest.mark.parametrize("arm", ARMS)
def test_trio_tag_and_strength_are_byte_identical(arm: str) -> None:
    fixture = json.loads(FIXTURE.read_text())
    assert _run_emit(arm) == fixture["arms"][arm]


def test_the_fixture_pins_real_captures() -> None:
    """Known answer: a fixture of empty arms would pass vacuously."""
    fixture = json.loads(FIXTURE.read_text())
    cradle = fixture["arms"]["cradle"]["captures"]
    assert [c["tool"] for c in cradle] == list(CRADLE_SCRIPT)
    # The scripted sequence's own physics, read by hand from the YAML: warm_self relieves the cold the
    # two feels caused (core_temperature), so the trio carries that relief on both warms.
    assert [dict(c["drive_relief"] or []).get("core_temperature", 0.0) > 0 for c in cradle] == [
        False,
        False,
        True,
        True,
        True,
    ]
    assert fixture["arms"]["fear_water"]["captures"], "the fear_water arm captured nothing"


def _regen() -> None:
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--", "src/"], capture_output=True, text=True, cwd=str(REPO_ROOT)
    ).stdout.strip()
    if dirty:
        sys.exit(f"refusing to regenerate from a dirty src/ tree:\n{dirty}")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=str(REPO_ROOT), check=True
    ).stdout.strip()
    arms = {arm: _run_emit(arm) for arm in ARMS}
    FIXTURE.write_text(json.dumps({"generated_at_commit": commit, "arms": arms}, sort_keys=True, indent=1) + "\n")
    print(f"wrote {FIXTURE} at {commit}")


if __name__ == "__main__":
    if sys.argv[1:2] == ["--regen"]:
        _regen()
    elif sys.argv[1:2] == ["--emit"]:
        print("<<<GOLDEN>>>\n" + emit(sys.argv[2], Path(sys.argv[3])))
    else:
        sys.exit("usage: test_gl2a_trio_golden.py --regen | --emit <arm> <workdir>")
