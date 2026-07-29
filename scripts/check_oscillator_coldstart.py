#!/usr/bin/env python
"""SCN oscillator cold-start floor check — outstanding validation for the
channel-split drive-pain fold (docs/plans/deferred/transition_based_drive_pain.md).

WHY THIS EXISTS
---------------
``Body._publish_drive_pain`` also emits a ``TemporalEvent`` into
``TemporalCreditDistributor.record_event`` → ``OscillatorNetwork.observe_event``.
Before the fold, a lingering breach re-published every tick, so the oscillator
saw a *dense* stream of phase samples per drive event type. After the fold,
channel 2 fires on band entry + material re-injury only, so it sees roughly one
sample per breach EPISODE.

That is the better input for circadian phase learning (the per-tick stream
produced a spuriously tight cluster around whichever moment the breach started,
and evicted genuine cross-episode history from the 50-entry ring buffer). But
``OscillatorNetwork.predict_imminence`` has a hard cold-start guard:

    if not phases or len(phases) < 3: return 0.0        # oscillator.py:305-306

so a short sim may now leave ``anticipatory_pre_activate`` at 0.0 imminence for
drive event types that previously cleared the floor on tick-density alone. This
script measures whether that happens on a real body, rather than assuming.

WHAT IT REPORTS
---------------
Per drive event signature (``drive:<sensor>:discomfort`` / ``:deprived``):
the observation count, whether it clears the 3-sample floor, and the imminence
the oscillator would currently predict.

READING THE RESULT
------------------
- **Floor cleared for the drives the run exercised** → the fold costs nothing
  the oscillator was actually getting; record and close the item.
- **Floor NOT cleared** → the honest finding is that per-episode density is too
  sparse for this run length. That is a *real* behavioral delta to document,
  NOT a reason to revert the fold (reverting re-introduces the per-tick pain
  flood the fold exists to kill). The options are: longer runs, or letting the
  oscillator observe drive-value band transitions directly rather than riding
  the pain channel. Do not "fix" it by re-firing pain per tick.

USAGE
-----
    # A) From a completed session's persisted SCN (preferred — real run):
    python scripts/check_oscillator_coldstart.py --session ~/.maxim/sessions/<id>
    python scripts/check_oscillator_coldstart.py --scn-json ~/.maxim/memory/scn.json

    # B) Offline synthetic A/B — how many episodes a body of this shape yields
    #    under episode-density vs the pre-fold tick-density (no LLM, no robot):
    python scripts/check_oscillator_coldstart.py --simulate
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

MIN_OBSERVATIONS = 3  # oscillator.py::predict_imminence cold-start guard


def _find_oscillator_blob(data: Any) -> dict[str, Any] | None:
    """Locate the oscillator payload in an SCN dump (shape-tolerant)."""
    if not isinstance(data, dict):
        return None
    if "event_phases" in data:
        return data
    osc = data.get("oscillator")
    if isinstance(osc, dict):
        return osc
    for value in data.values():
        if isinstance(value, dict):
            found = _find_oscillator_blob(value)
            if found is not None:
                return found
    return None


def _resolve_scn_path(session: Path | None, scn_json: Path | None) -> Path | None:
    """Locate a persisted SCN dump for a session.

    NOTE: sims write to ``~/.maxim/sim_reports/<id>/`` (NOT ``sessions/``), and
    ``aut_scn.json`` only exists for runs made after 2026-07-28 — before that
    ``save_aut_state`` persisted hippocampus/NAc/EC/ATL but *not* SCN, so the
    oscillator was not inspectable from a completed run at all. An older
    session legitimately has no SCN dump; re-run one sub-sim on current code.
    """
    if scn_json is not None:
        return scn_json.expanduser()
    if session is None:
        return None
    base = session.expanduser()
    # Accept either a session dir or a bare session id under the usual roots.
    candidates_roots = [base]
    if not base.exists():
        for root in (Path("~/.maxim/sim_reports").expanduser(), Path("~/.maxim/sessions").expanduser()):
            candidates_roots.append(root / base.name)
    for root in candidates_roots:
        if not root.exists():
            continue
        for name in ("aut_scn.json", "scn.json"):
            p = root / name
            if p.exists():
                return p
        matches = sorted(root.glob("*scn*.json"))
        if matches:
            return matches[0]
    return None


def report_from_scn(path: Path) -> int:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[FAIL] cannot read SCN dump at {path}: {exc}")
        return 2

    blob = _find_oscillator_blob(data)
    if blob is None:
        print(f"[FAIL] no oscillator payload found in {path}")
        print("       (was the oscillator enabled? build_bio_stack enables it by default)")
        return 2

    phases: dict[str, list[float]] = blob.get("event_phases") or {}
    drive_sigs = {k: v for k, v in phases.items() if k.startswith("drive:")}

    print(f"[scn] {path}")
    print(f"[scn] event signatures total={len(phases)}  drive={len(drive_sigs)}")
    if not drive_sigs:
        print("\n[WARN] no drive:* event signatures observed at all.")
        print("       Either no drive breached during the run, or the pain channel")
        print("       never reached the SCN distributor. Check the run exercised a")
        print("       drive (cold body / hunger) before concluding anything about density.")
        return 1

    print(f"\n{'event signature':<44} {'obs':>5} {'floor':>7}")
    print("-" * 60)
    below = []
    for sig, plist in sorted(drive_sigs.items()):
        n = len(plist or [])
        ok = n >= MIN_OBSERVATIONS
        if not ok:
            below.append((sig, n))
        print(f"{sig:<44} {n:>5} {'PASS' if ok else 'BELOW':>7}")

    print()
    if below:
        print(
            f"[RESULT] {len(below)}/{len(drive_sigs)} drive signatures BELOW the {MIN_OBSERVATIONS}-observation floor:"
        )
        for sig, n in below:
            print(f"           {sig}  (n={n})")
        print()
        print("         → anticipatory_pre_activate returns 0.0 imminence for these.")
        print("         → This is a real behavioral delta to DOCUMENT, not to fix by")
        print("           re-firing pain per tick (that re-introduces the flood).")
        return 1

    print(f"[RESULT] all {len(drive_sigs)} drive signatures clear the {MIN_OBSERVATIONS}-observation floor.")
    print("         → per-episode density is sufficient at this run length; close the item.")
    return 0


def simulate() -> int:
    """Offline A/B: episode-density vs pre-fold tick-density, no LLM/robot.

    Drives a real Embodiment through repeated breach/recover cycles and counts
    how many TemporalEvents each emission policy would produce.
    """
    from unittest.mock import MagicMock

    from maxim.embodiment.body import Embodiment
    from maxim.embodiment.spec import _parse_entity

    spec = {
        "name": "probe_body",
        "entity_type": "body",
        "sensors": {
            "cold": {
                "unit": "ratio",
                "range": [0, 1],
                "initial": 0.0,
                "drive": {
                    "drift_mode": "entropic",
                    "drift_direction": "up",
                    "drift_rate": 0.0,
                    "deprivation_threshold": 0.7,
                    "deprivation_pain": 0.3,
                    "satisfaction_threshold": 0.3,
                },
            },
        },
    }
    episodes = 4
    ticks_per_episode = 25

    bus = MagicMock()
    emb = Embodiment(_parse_entity(spec), pain_bus=bus)
    ticks_total = 0
    for _ in range(episodes):
        emb.root.vital_metrics["cold"] = 0.9  # breach
        for _ in range(ticks_per_episode):
            emb.evaluate_failures()
            ticks_total += 1
        emb.root.vital_metrics["cold"] = 0.1  # recover past satisfaction_threshold
        emb.evaluate_failures()
        ticks_total += 1

    post = bus.publish.call_count
    print("[simulate] synthetic breach/recover cycles on a real Embodiment")
    print(f"           episodes={episodes}  ticks/episode={ticks_per_episode}  total evaluate_failures={ticks_total}")
    print()
    print(f"  pre-fold  (per-tick emission):  ~{ticks_total - episodes} TemporalEvents  [flood]")
    print(f"  post-fold (per-episode):         {post} TemporalEvents")
    print()
    if post >= MIN_OBSERVATIONS:
        print(f"[RESULT] {post} >= {MIN_OBSERVATIONS} → {episodes} breach episodes clear the cold-start floor.")
        print("         A run must yield >= 3 genuine breach EPISODES for drive imminence.")
        return 0
    print(f"[RESULT] {post} < {MIN_OBSERVATIONS} → would NOT clear the floor.")
    return 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", type=Path, default=None, help="~/.maxim/sessions/<id> directory")
    ap.add_argument("--scn-json", type=Path, default=None, help="explicit path to a persisted SCN json")
    ap.add_argument("--simulate", action="store_true", help="offline synthetic A/B (no session needed)")
    args = ap.parse_args()

    if args.simulate:
        return simulate()

    path = _resolve_scn_path(args.session, args.scn_json)
    if path is None or not path.exists():
        print("[FAIL] no SCN dump found for that session.")
        print("       Sims write to ~/.maxim/sim_reports/<id>/, and aut_scn.json only")
        print("       exists for runs on code from 2026-07-28 or later — before that")
        print("       save_aut_state persisted hippocampus/NAc/EC/ATL but NOT SCN.")
        print("       Fix: re-run one sub-sim on current code, then re-check. Or use")
        print("       --simulate for the offline A/B, or --scn-json <file> directly.")
        return 2
    return report_from_scn(path)


if __name__ == "__main__":
    sys.exit(main())
