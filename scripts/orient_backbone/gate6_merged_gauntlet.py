#!/usr/bin/env python
"""Gate 6 merged gauntlet — seeds 42+43 through the re-keyed path (1.2 gate 6).

Pre-registration (THE authority on any divergence):
``docs/experiments/protocols/gate6_merged_gauntlet_preregistration.md``.

Empirical half of gate 6: merge the archived Exp 52 taught pairs
(``docs/experiments/data/53_agents/``, SHA-verified against the manifest before
any load) through the SHIPPED ``substrate_merge`` composition, and read the
merged substrate out on Exp 53's own harness (DryReadoutRig — production
body/tools/encode path; exp53's frozen gates computed by exp53's own verdict
code, unchanged). A PRESERVATION claim, not superadditivity: both parents pass
alone (ceiling disclosed in the prereg); the discriminating weight sits on the
mechanical checks and the dangling defect-reproduction arm.

Arms: records A = merged taught x2 (both directions) + merged satiated x2 +
merged no_feed x2 (exp53 Gate I + Gate T, unchanged). Records B = receiver-alone
taught 42/43 + the DANGLING-HALF x2 (both directions; pre-D43 recipe: bare
``nac_merge`` without re-key over the receiver's EC). Mechanical instrument arm:
empty-want donor must read ``biases_rekeyed == 0``.

Frozen decision rule: see :func:`decide_verdict` (mirrors the prereg §Frozen
decision rule verbatim).

In-process harness — imports maxim + exp53's module; spawns nothing.
Gated writes (records under ``docs/experiments/data/``) go through the
in-process provenance preflight (clean tree, this repo's ``maxim``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

ARCHIVE = _REPO_ROOT / "docs" / "experiments" / "data" / "53_agents"
MANIFEST_53 = _REPO_ROOT / "docs" / "experiments" / "data" / "53_agents_manifest.json"
NAC_KEY_SEP = "\x1f"

# Frozen thresholds (pre-registered; changes after first data need a protocol amendment).
PRESERVATION_SLACK = 0.10  # one-trial slack at 12 gated trials
DANGLING_SLACK = 0.10
MIN_REKEYED = 1

MERGE_DIRECTIONS = (("42", "43"), ("43", "42"))  # (receiver, donor)
ARM_NAMES = ("taught", "satiated", "no_feed")


def _refuse(msg: str) -> None:
    print(f"[REFUSED — apparatus] {msg}", file=sys.stderr)
    raise SystemExit(4)


def _preflight(out_path: str | None, allow_dirty: bool):
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))
    import maxim
    from _provenance import DirtyTreeError, ProvenanceError, in_process_code_provenance

    try:
        return in_process_code_provenance(_REPO_ROOT, maxim.__file__, out_path=out_path, allow_dirty=allow_dirty)
    except (ProvenanceError, DirtyTreeError) as exc:
        print(f"[FAIL] gated-record preflight: {exc}", file=sys.stderr)
        raise SystemExit(3)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_archive() -> dict[str, dict]:
    """SHA-verify every archived pair against the Exp 53 manifest; refuse on drift.

    Returns {label: manifest agent entry} for the labels this protocol uses.
    """
    manifest = json.loads(MANIFEST_53.read_text())
    by_label: dict[str, dict] = {}
    needed = {f"{arm}_seed{s}" for arm in ARM_NAMES for s in ("42", "43")}
    for agent in manifest["agents"]:
        if agent["label"] not in needed:
            continue
        pair_dir = ARCHIVE / agent["label"]
        nac, ec = pair_dir / "aut_nac.json", pair_dir / "aut_ec.json"
        if not nac.is_file() or not ec.is_file():
            _refuse(f"archive pair missing for {agent['label']} under {pair_dir}")
        if _sha256(nac) != agent["nac_sha256"] or _sha256(ec) != agent["ec_sha256"]:
            _refuse(f"archive SHA mismatch for {agent['label']} — the evidence files are not the manifested ones")
        by_label[agent["label"]] = {**agent, "nac_path": str(nac), "ec_path": str(ec)}
    missing = needed - set(by_label)
    if missing:
        _refuse(f"manifest lacks needed labels: {sorted(missing)}")
    return by_label


def _load_raw(spec: dict) -> tuple[dict, dict]:
    nac_raw = json.loads(Path(spec["nac_path"]).read_text())
    ec_raw = json.loads(Path(spec["ec_path"]).read_text())
    return nac_raw, ec_raw


def _write_pair(out_dir: Path, nac_state: dict, ec_payload: dict) -> tuple[Path, Path]:
    from maxim.utils.atomic_io import atomic_write_json

    out_dir.mkdir(parents=True, exist_ok=True)
    nac_path, ec_path = out_dir / "aut_nac.json", out_dir / "aut_ec.json"
    atomic_write_json(str(nac_path), nac_state)
    atomic_write_json(str(ec_path), ec_payload)
    return nac_path, ec_path


def _merge_pair(receiver_spec: dict, donor_spec: dict, *, strip_donor_want: bool = False):
    """The shipped composition. Returns (merged_nac_state, merged_ec_payload, result)."""
    from maxim.hivemind.merge import substrate_merge

    import exp53_cross_context_readout as exp53

    nac_r, ec_r = _load_raw(receiver_spec)
    nac_d, ec_d = _load_raw(donor_spec)
    if strip_donor_want:
        nac_d = {**nac_d, "cluster_reward_bias": {}, "cluster_reward_source": {}, "reward_bias": {}}
    result = substrate_merge(
        receiver_nac=nac_r,
        receiver_ec=ec_r.get("substrate_nodes") or {},
        donor_nac=nac_d,
        donor_ec=ec_d.get("substrate_nodes") or {},
        receiver_source=receiver_spec["label"],
        donor_source=donor_spec["label"],
        receiver_agent_id=exp53.AGENT_ID,
    )
    # Envelope keys the merge does not return (e.g. _format_version) come from
    # the receiver's file; every merged payload field overrides (disclosed in
    # the prereg's Apparatus section).
    merged_nac = {**nac_r, **result.nac}
    merged_ec = {**ec_r, "substrate_nodes": result.ec_nodes}
    return merged_nac, merged_ec, result


def _dangling_pair(receiver_spec: dict, donor_spec: dict) -> tuple[dict, dict]:
    """The pre-D43 recipe: bare nac_merge, NO re-key, receiver's EC unchanged."""
    from maxim.hivemind.merge import nac_merge

    nac_r, ec_r = _load_raw(receiver_spec)
    nac_d, _ = _load_raw(donor_spec)
    folded = nac_merge(nac_r, nac_d, left_source=receiver_spec["label"], right_source=donor_spec["label"])
    return {**nac_r, **folded}, ec_r


def _coverage(nac_state: dict, ec_payload: dict) -> tuple[int, int]:
    """(surviving cluster bias keys, keys naming a cluster ABSENT from the EC)."""
    nodes = set((ec_payload.get("substrate_nodes") or {}).keys())
    keys = list((nac_state.get("cluster_reward_bias") or {}).keys())
    dangling = sum(1 for k in keys if len(k.split(NAC_KEY_SEP, 2)) == 3 and k.split(NAC_KEY_SEP, 2)[1] not in nodes)
    return len(keys), dangling


def _agent_entry(label: str, arm: str, seed: int, nac_path: Path, ec_path: Path) -> dict:
    import exp53_cross_context_readout as exp53

    desc = exp53._describe_state(nac_path, ec_path)
    return {
        "arm": arm,
        "seed": seed,
        "exploratory": False,
        "label": label,
        "nac_path": str(nac_path),
        "ec_path": str(ec_path),
        "nac_sha256": _sha256(nac_path),
        "ec_sha256": _sha256(ec_path),
        **desc,
    }


def _write_manifest(path: Path, agents: list[dict]) -> None:
    import exp53_cross_context_readout as exp53

    payload = {
        "_format_version": "1.0",
        "experiment": "gate6_merged_gauntlet",
        "archive": str(ARCHIVE),
        "frozen": {
            "body_ref": exp53.BODY_REF,
            "agent_id": exp53.AGENT_ID,
            "deltas_rad": exp53.DELTAS,
            "targets_az": exp53.TARGETS,
            "trials_per_agent": exp53.TRIALS_PER_AGENT,
            "explore_primary": exp53.EXPLORE_PRIMARY,
            "explore_secondary": exp53.EXPLORE_SECONDARY,
            "protocol": "docs/experiments/protocols/gate6_merged_gauntlet_preregistration.md",
        },
        "agents": agents,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _run_gauntlet(manifest_path: Path, records_path: Path, allow_dirty: bool) -> str:
    """Phase 1 -> phase 2 (both conditions) -> verdict. Returns "ok" or "gate_I_fail".

    exp53's phase-1 run emits the gate_I record itself and returns 6 on a
    COMPUTED Gate-I FAIL — that is a recorded outcome for THIS protocol
    (frozen rule: "any other combination -> FAIL, numbers recorded"), never a
    refusal; phase 2 is skipped (exp53's own stop rule I). Anything else
    non-zero IS a refusal. The final verdict's rc conflates REFUSED(2) with a
    computed FAIL(1), so 0/1 are both accepted and PASS/FAIL is read from the
    gate records by decide_verdict.
    """
    import time as _time

    import exp53_cross_context_readout as exp53

    def _spacer() -> None:
        # exp53's run_id is time.strftime(seconds)+pid, minted per main() call.
        # Two in-process phases inside the same wall-clock second collide into
        # one malformed "run" and the verdict then finds no complete phase-2
        # run (smoke finding, 2026-09-05). One second of spacing restores the
        # per-invocation uniqueness a subprocess caller gets for free.
        _time.sleep(1.1)

    base = ["--manifest", str(manifest_path), "--out", str(records_path), "--dry-run", "--yes", "--settle", "0.1"]
    dirty = ["--allow-dirty"] if allow_dirty else []
    rc1 = exp53.main(["run", "--phase", "1", *base, *dirty])
    if rc1 == 6:
        return "gate_I_fail"
    if rc1 != 0:
        _refuse(f"exp53 phase 1 refused for {manifest_path.name} (rc {rc1})")
    _spacer()
    if exp53.main(["run", "--phase", "2", *base, *dirty]) != 0:
        _refuse(f"exp53 phase 2 refused for {manifest_path.name} (stop rule I?)")
    if exp53.main(["verdict", "--records", str(records_path), *dirty]) not in (0, 1):
        _refuse(f"exp53 verdict refused for {records_path.name}")
    return "ok"


def _gate_records(records_path: Path) -> dict[str, dict]:
    """Latest gate_I / gate_T summary records from an exp53 records file."""
    out: dict[str, dict] = {}
    for line in records_path.read_text().splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("event") in ("gate_I", "gate_T"):
            out[rec["event"]] = rec
    return out


# ---------------------------------------------------------------------------
# The frozen decision rule — the protocol's own verdict function.
# ---------------------------------------------------------------------------


def decide_verdict(m: dict[str, Any]) -> str:
    """gate6-pass iff prereg rules 1-5 all hold; else gate6-fail.

    Refusals (apparatus/stop-rule failures) exit 4 upstream and never reach
    this function — mirroring the prereg: any other combination is FAIL with
    numbers recorded, no threshold motion.
    """
    r1 = m["gate_I_verdict"] == "PASS" and m["gate_T_verdict"] == "PASS"
    r2 = bool(m["merged_directedness"]) and all(
        m["merged_directedness"][d] >= m["receiver_alone_directedness"][d] - PRESERVATION_SLACK
        for d in m["merged_directedness"]
    )
    r3 = all(
        h["biases_dropped"] == 0 and h["biases_rekeyed"] >= MIN_REKEYED and h["dangling_keys"] == 0
        for h in m["merge_health"].values()
    )
    r4 = bool(m["dangling_arms"]) and all(
        a["dangling_keys"] >= 1 and a["directedness"] <= a["receiver_alone"] + DANGLING_SLACK
        for a in m["dangling_arms"].values()
    )
    r5 = m["instrument_empty_want_rekeyed"] == 0
    return "gate6-pass" if (r1 and r2 and r3 and r4 and r5) else "gate6-fail"


# ---------------------------------------------------------------------------


def run(args) -> int:
    out_json = args.json if args.write_experiment_results else None
    provenance = _preflight(out_json, args.allow_dirty)

    specs = _verify_archive()
    work = Path(args.workdir).expanduser().resolve()
    work.mkdir(parents=True, exist_ok=True)

    # --- build merged substrates (records A) ------------------------------
    merge_health: dict[str, dict] = {}
    agents_a: list[dict] = []
    directions: dict[str, tuple[str, str]] = {f"r{r}d{d}": (r, d) for r, d in MERGE_DIRECTIONS}
    for arm in ARM_NAMES:
        for key, (r, d) in directions.items():
            rx, dx = specs[f"{arm}_seed{r}"], specs[f"{arm}_seed{d}"]
            nac, ec, result = _merge_pair(rx, dx)
            n_keys, dangling = _coverage(nac, ec)
            if arm == "taught":
                merge_health[key] = {
                    "biases_rekeyed": result.biases_rekeyed,
                    "biases_dropped": result.biases_dropped,
                    "surviving_keys": n_keys,
                    "dangling_keys": dangling,
                }
            label = f"{arm}_seed{int(r) * 100 + int(d)}"  # e.g. taught_seed4243
            nac_p, ec_p = _write_pair(work / f"merged_{arm}_{key}", nac, ec)
            agents_a.append(_agent_entry(label, arm, int(r) * 100 + int(d), nac_p, ec_p))

    # --- records B: receiver-alone + dangling -----------------------------
    agents_b: list[dict] = []
    for s in ("42", "43"):
        spec = specs[f"taught_seed{s}"]
        agents_b.append(_agent_entry(f"alone_seed{s}", "taught", int(s), Path(spec["nac_path"]), Path(spec["ec_path"])))
    dangling_keys_by_dir: dict[str, int] = {}
    for key, (r, d) in directions.items():
        dn, de = _dangling_pair(specs[f"taught_seed{r}"], specs[f"taught_seed{d}"])
        _, dk = _coverage(dn, de)
        dangling_keys_by_dir[key] = dk
        dn_p, de_p = _write_pair(work / f"dangling_{key}", dn, de)
        agents_b.append(_agent_entry(f"dangling_seed{9900 + int(r)}", "taught", 9900 + int(r), dn_p, de_p))

    # --- mechanical instrument arm (no readout) ---------------------------
    _, _, empty_result = _merge_pair(specs["taught_seed42"], specs["taught_seed43"], strip_donor_want=True)

    # --- readouts ---------------------------------------------------------
    manifest_a, records_a = work / "manifest_A.json", Path(args.records_a)
    manifest_b, records_b = work / "manifest_B.json", Path(args.records_b)
    # exp53's JsonlLog opens without creating parents; the gated data dir may
    # not exist on a fresh checkout (bit the official run, 2026-09-05 — see
    # the protocol's amendment header).
    records_a.parent.mkdir(parents=True, exist_ok=True)
    records_b.parent.mkdir(parents=True, exist_ok=True)
    _write_manifest(manifest_a, agents_a)
    _write_manifest(manifest_b, agents_b)
    _run_gauntlet(manifest_a, records_a, args.allow_dirty)  # gate-I FAIL is read from the records
    status_b = _run_gauntlet(manifest_b, records_b, args.allow_dirty)
    if status_b != "ok":
        # Records B is the comparison baseline: receiver-alone agents failing
        # Gate I contradicts Exp 53's earned result — apparatus, not outcome.
        _refuse("records B phase 1 returned a computed Gate-I FAIL — the baseline apparatus is wrong")

    gates_a, gates_b = _gate_records(records_a), _gate_records(records_b)
    if "gate_I" not in gates_a:
        _refuse("records A lack a gate_I summary record")
    gate_i_verdict = gates_a["gate_I"].get("verdict")
    if gate_i_verdict == "PASS" and "gate_T" not in gates_a:
        _refuse("records A lack a gate_T summary record despite Gate I PASS")
    if (gates_a.get("gate_T") or {}).get("verdict") == "APPARATUS":
        _refuse("exp53 Gate T returned APPARATUS on records A (sign-agreement/spread check)")
    if (gates_b.get("gate_T") or {}).get("verdict") == "APPARATUS":
        _refuse("exp53 Gate T returned APPARATUS on records B (sign-agreement/spread check)")
    per_seed_a = (gates_a.get("gate_T") or {}).get("per_seed") or {}
    per_seed_b = (gates_b.get("gate_T") or {}).get("per_seed") or {}
    try:
        # A computed Gate-I FAIL on records A is a RECORDED outcome (frozen
        # rule: gate6-fail with numbers); its phase 2 never ran, so the
        # behavioral dicts stay empty and rule 1 carries the failure.
        merged_directedness = (
            {k: float(per_seed_a[f"taught_seed{int(r) * 100 + int(d)}"]) for k, (r, d) in directions.items()}
            if gate_i_verdict == "PASS"
            else {}
        )
        receiver_alone = {k: float(per_seed_b[f"alone_seed{r}"]) for k, (r, d) in directions.items()}
        dangling_arms = {
            k: {
                "dangling_keys": dangling_keys_by_dir[k],
                "directedness": float(per_seed_b[f"dangling_seed{9900 + int(r)}"]),
                "receiver_alone": float(per_seed_b[f"alone_seed{r}"]),
            }
            for k, (r, d) in directions.items()
        }
    except KeyError as exc:
        _refuse(f"per-seed directedness missing from gate_T records: {exc}")

    metrics: dict[str, Any] = {
        "gate_I_verdict": gate_i_verdict,
        "gate_T_verdict": (gates_a.get("gate_T") or {}).get("verdict", "NOT-RUN"),
        "gate_T_means": (gates_a.get("gate_T") or {}).get("primary_directedness_by_arm"),
        "merged_directedness": merged_directedness,
        "receiver_alone_directedness": receiver_alone,
        "merge_health": merge_health,
        "dangling_arms": dangling_arms,
        "instrument_empty_want_rekeyed": empty_result.biases_rekeyed,
    }
    verdict = decide_verdict(metrics)

    print(f"gate 6 merged gauntlet — verdict: {verdict}")
    print(
        f"  gate I: {metrics['gate_I_verdict']}   gate T: {metrics['gate_T_verdict']} (means {metrics['gate_T_means']})"
    )
    print(f"  merged vs alone: {merged_directedness} vs {receiver_alone} (floor: alone - {PRESERVATION_SLACK})")
    print(f"  merge health: {merge_health}")
    print(f"  dangling arms: {metrics['dangling_arms']} (defect must reproduce: >=1 dangling key, both directions)")
    print(f"  instrument empty-want rekeyed: {empty_result.biases_rekeyed} (must be 0)")

    if out_json:
        from maxim.utils.atomic_io import atomic_write_json
        from maxim.utils.format_version import with_format_version

        record = {
            "experiment": "gate6_merged_gauntlet",
            "protocol": "docs/experiments/protocols/gate6_merged_gauntlet_preregistration.md",
            "ran_at": datetime.now(timezone.utc).isoformat(),
            "provenance": provenance,
            "constants": {
                "preservation_slack": PRESERVATION_SLACK,
                "dangling_slack": DANGLING_SLACK,
                "min_rekeyed": MIN_REKEYED,
                "merge_directions": MERGE_DIRECTIONS,
            },
            "metrics": metrics,
            "verdict": verdict,
        }
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(out_json, with_format_version(record))
        print(f"  wrote {out_json}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    data_dir = _REPO_ROOT / "docs" / "experiments" / "data" / "gate6_merged_gauntlet"
    stamp = f"{datetime.now(timezone.utc):%Y%m%d}"
    ap.add_argument("--workdir", default="/tmp/gate6_merged_gauntlet", help="derived merged substrates (not committed)")
    ap.add_argument(
        "--records-a",
        default=None,
        help="default: <workdir>/records_A.jsonl; the gated data dir with --write-experiment-results",
    )
    ap.add_argument(
        "--records-b",
        default=None,
        help="default: <workdir>/records_B.jsonl; the gated data dir with --write-experiment-results",
    )
    ap.add_argument("--json", default=str(data_dir / f"gate6_verdict_{stamp}.json"))
    ap.add_argument("--write-experiment-results", action="store_true", help="write the gated verdict record")
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args(argv)
    base = data_dir if args.write_experiment_results else Path(args.workdir)
    if args.records_a is None:
        args.records_a = str(base / "records_A.jsonl")
    if args.records_b is None:
        args.records_b = str(base / "records_B.jsonl")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
