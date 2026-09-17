"""The R3 pilot must CONSTRUCT offline (a stubbed RCON, a stubbed anchor record, no rig): the first live
run failed at `Pilot.__init__` on a wrong module alias (`survival_world.common` has no `REPO_ROOT`),
which an import-only check cannot see. This builds the object end to end and pins the geometry it
derives from the anchor (the pool floor teleport target and the stone cap slab over the pool's air)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from survival_world import r3_pilot as R  # noqa: E402


class _Rcon:
    def __init__(self, *a, **k):
        self.sent: list[str] = []

    def command(self, cmd: str) -> str:
        self.sent.append(cmd)
        if cmd.startswith("gamerule "):
            return f"Gamerule {cmd.split()[1]} is currently set to: true"
        if cmd == "difficulty":
            return "The difficulty is Normal"
        return ""

    def teleport(self, bot: str, pos: dict) -> None:
        self.sent.append(f"tp {bot} {pos}")


def test_pilot_constructs_offline_and_derives_geometry(tmp_path: Path, monkeypatch) -> None:
    anchor = {
        "submerged": [10, 35, 20],
        "surface": [11, 40, 20],
        "shore": [7, 40, 20],
        "depth": 5,
        "deaths_objective": "exp60_deaths",
        "measured": {"t_damage_onset_min_s": 16.067},
    }
    anchor_path = tmp_path / "anchor.json"
    anchor_path.write_text(json.dumps(anchor))
    apparatus = {"all_pass": True, "w4_escape": {}, "cycles": [{"t_pain_edge": 5.1}, {"t_pain_edge": 5.3}]}
    monkeypatch.setattr(R, "ANCHOR_FILE", anchor_path)
    monkeypatch.setattr(
        R,
        "_load_json",
        lambda p: apparatus if "apparatus" in str(p) or "exp60_water" in str(p) else json.loads(Path(p).read_text()),
    )
    monkeypatch.setattr(R, "min_pain_edge_s", lambda a: 5.085)
    monkeypatch.setattr(R.C, "RconControl", _Rcon)
    monkeypatch.setattr(R, "in_process_code_provenance", lambda *a, **k: {"executed_git_hash": "stub"})
    args = argparse.Namespace(
        rcon_host="h",
        rcon_port=1,
        rcon_password="p",
        username="maxim",
        bridge_host="h",
        bridge_port=2,
        workdir=str(tmp_path / "wd"),
        out=str(tmp_path / "out.jsonl"),
        allow_dirty=True,
        only="",
    )
    pilot = R.Pilot(args)
    assert pilot.floor == {"x": 10.5, "y": 35.0, "z": 20.5}
    assert pilot.cap_slab == (10, 40, 19, 12, 40, 21)
    assert pilot.t_damage_onset == 16.067 and pilot.pain_edge_min == 5.085
    rules = pilot.read_rules()
    assert set(R.EXTRA_RULES) <= set(rules) and rules["difficulty"].endswith("Normal")
    pilot.cap_pool(True)
    assert pilot.rcon.sent[-1] == "fill 10 40 19 12 40 21 minecraft:stone"
    pilot.cap_pool(False)
    assert pilot.rcon.sent[-1].endswith("minecraft:air")
    assert pilot.set_regen(False) == "Gamerule naturalRegeneration is currently set to: true"
