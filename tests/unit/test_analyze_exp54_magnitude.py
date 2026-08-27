"""analyze_exp54_magnitude.py — the producer for the pre-registration's "reported, not
gated" magnitude-choice item (amendment 1 item 8). Pins the pairing rule (actions
answer the stimulus placed on their turn), the _big classification, the direction
sanity check and the 3-bin roll-up on a synthetic mother log."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "analyze_exp54_magnitude.py"
spec = importlib.util.spec_from_file_location("analyze_exp54_magnitude", _SCRIPT)
mag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mag)


def _log(tmp_path: Path, lines: list[str]) -> Path:
    p = tmp_path / "mother_log.jsonl"
    p.write_text("\n".join(json.dumps({"message": m}) for m in lines) + "\n")
    return p


def test_pairs_actions_to_the_turns_stimulus_and_classifies_big(tmp_path):
    log = _log(
        tmp_path,
        [
            "act=act1_early fed=False az_prior=0.0 az_stimulus=-0.7 az_guided=-0.7",
            "Executing: reachy_mini_turn_left_big by MaximAgent params=[]",
            "Executing: reachy_mini_listen by MaximAgent params=[]",  # not a turn: ignored
            "act=act3_consolidating fed=True az_prior=-0.2 az_stimulus=0.4 az_guided=0.4",
            "Executing: reachy_mini_turn_right by MaximAgent params=[]",
            "Executing: reachy_mini_turn_left by MaximAgent params=[]",
            "act=act4_autonomous fed=False az_prior=0.1 az_stimulus=None az_guided=0.1",
            "Executing: reachy_mini_turn_right_big by MaximAgent params=[]",  # no stimulus: dropped
        ],
    )
    rows = mag.pair_turns(log)
    assert [(r["stim"], r["tool"].rsplit("_turn_", 1)[-1], r["big"], r["left"]) for r in rows] == [
        (-0.7, "left_big", True, True),
        (0.4, "right", False, False),
        (0.4, "left", False, True),
    ]
    s = mag.summarize(rows)
    assert s["all_acts"]["by_bin"]["mid"] == {"n": 1, "big": 1, "toward": 1, "big_frac": 1.0, "toward_frac": 1.0}
    assert s["all_acts"]["by_bin"]["near"]["big_frac"] == 0.0 and s["all_acts"]["by_bin"]["near"]["toward_frac"] == 0.5
    assert s["late_acts"]["n"] == 2 and "mid" not in s["late_acts"]["by_bin"]


def test_infant_body_reports_no_big_turns(tmp_path):
    log = _log(
        tmp_path,
        [
            "act=act3_consolidating fed=True az_prior=0.0 az_stimulus=0.9 az_guided=0.9",
            "Executing: infant_operant_turn_right by MaximAgent params=[]",
        ],
    )
    s = mag.summarize(mag.pair_turns(log))
    assert s["late_acts"]["by_bin"]["far"]["big_frac"] == 0.0
