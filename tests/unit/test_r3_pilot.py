"""The R3 pilot runs OFFLINE end to end against the scripted water bridge before it touches the rig.

Two live runs died in `Pilot.__init__` (a wrong module alias; an anchor key the classroom builder
never writes) — slips an import-only check cannot see and a hand-written anchor stub HID. So this
test builds the anchor with the REAL builder (`water_classroom_geometry` + `water_anchor_record`),
then drives every pilot row against `ScriptedWaterBridge` with short caps: the apparatus row, the
two no-rescue lethal windows (attached: the in-situ fear must surface the bot; detached: no death
on the scripted bridge → capped, named), and a drowning row (capped; the pool cap and the regen
toggle go through the scripted RCON and are restored). What it cannot prove: real damage, real
death/respawn, real timings — the rig's job.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from survival_world import r3_pilot as R  # noqa: E402
from survival_world.scripted_water import ScriptedWaterBridge, ScriptedWaterControl  # noqa: E402
from survival_world.setup_world import water_anchor_record, water_classroom_geometry  # noqa: E402


class _ColumnBridge(ScriptedWaterBridge):
    """In water anywhere in the column above the submerged point (the pilot places at floor+2 too)."""

    def _in_water_locked(self) -> bool:
        a, s = self.anchor, self.submerged
        return abs(a["x"] - s["x"]) < 0.5 and abs(a["z"] - s["z"]) < 0.5 and s["y"] - 0.5 <= a["y"] < s["y"] + 4.0


def _xyz(t: list) -> dict[str, float]:
    return {"x": float(t[0]), "y": float(t[1]), "z": float(t[2])}


@pytest.mark.timeout(300)
def test_pilot_runs_every_row_offline_on_the_scripted_bridge(tmp_path: Path, monkeypatch) -> None:
    geom = water_classroom_geometry(10, 20, depth=5, shore_y=64)
    record = water_anchor_record(geom)  # the SAME writer the builder uses — no hand-typed keys
    record["measured"] = {"t_damage_onset_min_s": 16.067}
    anchor_path = tmp_path / "anchor.json"
    anchor_path.write_text(json.dumps(record))
    srv = _ColumnBridge(shore=_xyz(record["shore"]), submerged=_xyz(record["submerged"]))
    rcon = ScriptedWaterControl(srv)
    sent: list[str] = []
    _orig_command = rcon.command

    def _command(cmd: str) -> str:  # record what the PILOT issues (the fake echoes reads only)
        sent.append(cmd)
        return _orig_command(cmd)

    rcon.command = _command  # type: ignore[method-assign]
    apparatus = {"all_pass": True}
    monkeypatch.setattr(R, "ANCHOR_FILE", anchor_path)
    monkeypatch.setattr(
        R,
        "_load_json",
        lambda p: apparatus if str(p).endswith(str(R.APPARATUS_RECORD)) else json.loads(Path(p).read_text()),
    )
    monkeypatch.setattr(R, "min_pain_edge_s", lambda a: 5.085)
    monkeypatch.setattr(R.C, "RconControl", lambda *a, **k: rcon)
    monkeypatch.setattr(R, "in_process_code_provenance", lambda *a, **k: {"executed_git_hash": "offline"})
    monkeypatch.setattr(R, "LETHAL_CAP_S", 25.0)
    monkeypatch.setattr(R, "DROWN_CAP_S", 5.0)
    monkeypatch.setattr(R, "REST_S", 4.0)
    monkeypatch.setattr(R, "POST_DEATH_S", 1.0)
    out = tmp_path / "pilot.jsonl"
    args = argparse.Namespace(
        rcon_host="h",
        rcon_port=1,
        rcon_password="p",
        username="maxim",
        bridge_host="127.0.0.1",
        bridge_port=srv.port,
        workdir=str(tmp_path / "wd"),
        out=str(out),
        allow_dirty=True,
        only="",
    )
    try:
        pilot = R.Pilot(args)
        # geometry derived from the REAL record: the cap slab sits on the pool's first air layer
        assert pilot.floor == _xyz(record["submerged"])
        sx, _sy, sz = record["submerged"]
        assert pilot.cap_slab == (sx - 1, record["surface_y"], sz - 1, sx + 1, record["surface_y"], sz + 1)
        pilot.row_apparatus()
        pilot.row_lethal("lethal_B", detach_fear=False)
        pilot.row_lethal("lethal_A", detach_fear=True)
        pilot.row_drown("drown_on", regen_on=True)
    finally:
        srv.close()
    rows = {json.loads(ln)["kind"]: json.loads(ln) for ln in out.read_text().splitlines() if ln.strip()}
    assert set(rows) == {"apparatus", "lethal_B", "lethal_A", "drown_on"}
    ap = rows["apparatus"]
    assert ap["refusal"] is None, ap["refusal"]
    assert "difficulty" in ap["rules"] and "naturalRegeneration" in ap["rules"]  # the fake answers "" for difficulty
    assert ap["ascent_from_floor"]["t_surface"] is not None and ap["rest"]["n_ticks"] >= 1
    b = rows["lethal_B"]
    assert b["refusal"] is None, b["refusal"]
    assert b["subscriber_detached"] is False and b["n_pain"] >= 1, b
    assert b["t_surface"] is not None and any(c["tool"].endswith("escape_water") for c in b["calls"]), b["calls"]
    assert b["fear_after"], "the in-situ learner must have booked fear on its own cluster"
    a = rows["lethal_A"]
    assert a["refusal"] is None, a["refusal"]
    assert a["detached_count"] >= 1 and a["fear_after"] == {} and a["calls"] == [], a
    assert a["capped"] is True  # the scripted bridge deals no damage: no innate route, no death → the cap, named
    d = rows["drown_on"]
    assert d["refusal"] is None and d["capped_window"] is True, d
    # the pool cap went on and came off, and regeneration was restored, through RCON — in that order
    assert any(c.startswith("fill ") and c.endswith("minecraft:stone") for c in sent), sent
    uncap = next(c for c in sent if c.startswith("fill ") and c.endswith("minecraft:air"))
    last_restore = len(sent) - 1 - sent[::-1].index("gamerule naturalRegeneration true")  # the row SETS regen first
    assert last_restore > sent.index(uncap), "regen restored AFTER the uncap (the finally order)"
