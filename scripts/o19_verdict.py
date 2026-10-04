#!/usr/bin/env python3
"""The O19 re-run verdict: Exp 10 (T1-1), Exp 09 (T3-9) and Exp 63 (T1-16), computed from committed bytes only.

Pre-registrations (read them first; this module implements them and nothing else):
  docs/experiments/protocols/exp10_rerun_2026-09-30_preregistration.md   (Exp 10 campaign 1, key "10": CLOSED)
  docs/experiments/protocols/exp10_rerun_2026-10-02_preregistration.md   (Exp 10 campaign 2, key "10c2")
  docs/experiments/protocols/exp09_rerun_2026-09-30_preregistration.md   (Exp 09, key "09")
  docs/experiments/exp63_carried_recall_prereg.md                        (Exp 63, key "63": a new experiment, not a
                                                                          re-run; its gates are ``exp63_*`` below)

A PROTOCOL key is a CAMPAIGN: its own prereg, data directory and marker namespace (``refs/tags/o19/<key>/``). Its
``experiment`` ("10"/"09"/"63") decides the phases' gates; every gate choice goes through :func:`experiment_of`. A
successor campaign (``supersedes``) opens only after its predecessor's stamped ABORT verdict, pinned by SHA-256, and
at most ``MAX_CAMPAIGNS`` campaigns exist per experiment (owner decisions 2026-10-02; :func:`protocol_problems`,
:func:`successor_problems`).

This module also holds the PROTOCOL (goals, turn caps, flags, model) that ``o19_rerun.py`` runs, so the
verdict's ``verdict_source_sha256`` covers what was run as well as how it was judged.

What the verdict reads, and how each input could lie:
  * the rows file (``rows.jsonl``), read ONCE: its bytes are hashed into the verdict (``stamp_verdict``);
  * the copied session files the rows name, each re-hashed against the row's SHA-256 (a mismatch refuses);
  * the start markers (annotated tags ``refs/tags/o19/<exp>/attempt-<k>-<run_id>``) on ``origin``, the tag ruleset
    that keeps them, and its history (``gh api``, read-only) — an attempt discarded before it was committed is still
    visible as a marker, and removing a marker needs a ruleset change, which its history shows;
  * ``origin/main``'s first-parent history of the rows file — each attempt landed before the next one started;
  * for a successor campaign, its predecessor's closure verdict and its own prereg on ``origin/main`` — both landed
    before its first marker (GitHub's merge time against the rig clock: forgetting is caught, not evasion).
  A failure of any of these REFUSES (exit 2, no verdict written): no status change, the row stays STALE. Rig-clock
  times (row ``ts``, tagger dates) are trusted as the prereg says: the gate catches forgetting, not evasion.

Exits: 0 PASS; 1 FAIL; 4 PARTIAL (Exp 09), NOT SHOWN (Exp 63: complete and conforming, but no decisive turn) or
ABORT -- 4 is every "no status change" outcome; 2 refused; 3 provenance (the imported maxim is not this repo's).
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from collections.abc import Callable
from fractions import Fraction
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
REPO_ROOT = SCRIPTS_DIR.parent

# ── the protocol (frozen by the preregs; a test asserts each value against the prereg text) ─────────────────

MODEL_PROFILE = "mistral-7b"  # what the operator configures (`maxim config set llm.profile mistral-7b`)
# What the report stamps: the router's profile is the NORMALIZED name (owner decision 2026-09-30, prereg amended
# before any data; a test pins it to ``normalize_llm_profile(MODEL_PROFILE)``).
MODEL_PROFILE_STAMPED = "mistral-7b-instruct-v0.2"
MODEL_GGUF = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
N_CTX = 8192
MAX_ATTEMPTS = 3
MARKER_NAMESPACE = "refs/tags/o19"
RULESET_INCLUDE = "refs/tags/o19/**"

EXP10_GOAL_DUNGEON = "escape a dungeon with a sleeping guard"
EXP10_GOAL_GARDEN = "you are in a peaceful garden, enjoy the flowers"
EXP09_GOAL = (
    "You are an adventurer in a dark cave. A dragon attacks you repeatedly with claws and fire breath. The dragon "
    "roars deafeningly. It slams you against the wall. A freezing wind blows through the cave."
)

# One entry per phase: (phase name, goal, turn cap, resumes phase 1?, extra argv, extra env).
EXP10_PHASES = [
    ("baseline", EXP10_GOAL_DUNGEON, 8, False, [], {}),
    ("gate", EXP10_GOAL_DUNGEON, 8, True, [], {}),
    ("negative_transfer", EXP10_GOAL_GARDEN, 5, True, [], {}),
]
# A campaign may be succeeded at most once per experiment (owner decision 2026-10-02: campaign 2 is the last before
# the 1.3.2 cut; another ABORT leaves the row STALE and the cause is investigated). Raising it is an owner decision.
# 2026-10-03 (owner, strict): Exp 10 opens no third campaign; a successor supports a row only on byte-identical
# subject code (#1059; docs/experiments/reproduction.md §13).
MAX_CAMPAIGNS = 2
_KEY = re.compile(r"[0-9a-z]+")
RESERVED_KEYS = frozenset({"preflight"})  # o19_rerun.py's dry-run marker namespace

PROTOCOL: dict[str, dict] = {
    "10": {
        "experiment": "10",
        "scope": "rerun_exp10_o19",
        "kind": "exp10_verdict",
        "prereg": "docs/experiments/protocols/exp10_rerun_2026-09-30_preregistration.md",
        "phases": EXP10_PHASES,
    },
    "10c2": {
        "experiment": "10",
        "scope": "rerun_exp10_o19c2",
        "kind": "exp10_verdict",
        "prereg": "docs/experiments/protocols/exp10_rerun_2026-10-02_preregistration.md",
        "phases": EXP10_PHASES,
        # Campaign 1 closed after one aborted attempt (planning_failed, cause #1042, fixed by #1045 + #1047): owner
        # decision 2026-10-01; its closing verdict was written 2026-10-02 before any O19 script changed (#1049).
        "supersedes": {
            "key": "10",
            "verdict": "docs/experiments/data/rerun_exp10_o19/verdict.json",
            "verdict_sha256": "5da2128968bb4516350aa26680294d9f79966424a518fc37b3dd68c0bf826fbe",
            "owner_decision": "2026-10-01",
            "cause_issue": 1042,
        },
    },
    "09": {
        "experiment": "09",
        "scope": "rerun_exp09_o19",
        "kind": "exp09_verdict",
        "prereg": "docs/experiments/protocols/exp09_rerun_2026-09-30_preregistration.md",
        "phases": [
            (
                "session",
                EXP09_GOAL,
                8,
                False,
                ["--embodiment", "bodies/base_humanoid", "--sim-run-full-turns"],  # amended 2026-10-03, before data
                {"MAXIM_SUBSTRATE_PATH": "1", "MAXIM_BACKEND_TRACE": "1"},
            ),
        ],
    },
    # Exp 63 (T1-16): a NEW experiment on current code (#1060), not a successor of Exp 10's campaigns: it supersedes
    # nothing. Two phases from one fresh data home, both with --sim-run-full-turns; phase 3 dropped (owner decision
    # 2026-10-03).
    "63": {
        "experiment": "63",
        "scope": "63_carried_recall",
        "kind": "exp63_verdict",
        "prereg": "docs/experiments/exp63_carried_recall_prereg.md",
        "phases": [
            ("baseline", EXP10_GOAL_DUNGEON, 8, False, ["--sim-run-full-turns"], {}),
            ("gate", EXP10_GOAL_DUNGEON, 8, True, ["--sim-run-full-turns"], {}),
        ],
    },
}
EXPERIMENTS = frozenset({"10", "09", "63"})


def experiment_of(key: str) -> str:
    """The experiment a campaign key runs ("10", "09" or "63"): every gate choice goes through this, never the key."""
    return PROTOCOL[key]["experiment"]


def closed_keys() -> dict[str, str]:
    """Campaign key -> the key of the campaign that superseded it. A closed campaign starts no attempt. Its closing
    verdict was computed once, before any O19 script changed; afterwards ``--exp <closed key>`` refuses
    (``check_bound``: the scripts differ from its executed commit), by design. What survives is the committed
    verdict, pinned by SHA-256 in its successor."""
    return {p["supersedes"]["key"]: k for k, p in PROTOCOL.items() if "supersedes" in p}


def predecessors(key: str, protocol: dict[str, dict] | None = None) -> list[str]:
    """Every campaign ``key`` succeeds, nearest first, ending at the ROOT campaign ([] for a root)."""
    protocol = PROTOCOL if protocol is None else protocol
    out: list[str] = []
    sup = protocol[key].get("supersedes")
    while isinstance(sup, dict) and sup.get("key") in protocol and sup["key"] not in out and sup["key"] != key:
        out.append(sup["key"])
        sup = protocol[sup["key"]].get("supersedes")
    return out


def _plain(value):
    """``value`` as JSON holds it (a phase tuple reads as a list), for comparing campaign entries."""
    return json.loads(json.dumps(value, sort_keys=True))


def protocol_problems(protocol: dict[str, dict] | None = None) -> list[str]:
    """The structural campaign rules ([] = sound): plain keys, one experiment per chain, a successor names an
    existing predecessor of its own experiment and kind, each campaign is succeeded at most once, at most one open
    campaign per experiment, and at most ``MAX_CAMPAIGNS`` campaigns per experiment."""
    protocol = PROTOCOL if protocol is None else protocol
    problems = []
    for key, p in protocol.items():
        if not _KEY.fullmatch(key) or key in RESERVED_KEYS:
            problems.append(f"campaign key {key!r} is not [0-9a-z]+ or is reserved")
        if p.get("experiment") not in EXPERIMENTS:
            problems.append(f"campaign {key}: experiment {p.get('experiment')!r} is not 10, 09 or 63")
        sup = p.get("supersedes")
        if sup is None:
            continue
        prev = protocol.get(sup.get("key"))
        if prev is None or sup.get("key") == key:
            problems.append(f"campaign {key} supersedes {sup.get('key')!r}, which is not another campaign")
            continue
        if prev.get("experiment") != p.get("experiment") or prev.get("kind") != p.get("kind"):
            problems.append(f"campaign {key} supersedes {sup['key']}, another experiment or kind")
        for field in ("verdict", "verdict_sha256", "owner_decision", "cause_issue"):
            if not sup.get(field):
                problems.append(f"campaign {key}: supersedes.{field} is missing")
        if sup.get("verdict") != f"docs/experiments/data/{prev.get('scope')}/verdict.json":
            problems.append(f"campaign {key}: the closure verdict is not {sup['key']}'s own")
        # #1059 S6: a successor re-runs its predecessor's argv and env exactly (a mechanism switched by a flag or an
        # env var counts as a change of subject; a successor needing a new harness flag is a new experiment).
        if _plain(prev.get("phases")) != _plain(p.get("phases")):
            problems.append(f"campaign {key}'s phases (argv, env) are not {sup['key']}'s")
    # #1059 D1: each verdict kind belongs to exactly one experiment (the gate also holds the map append-only against
    # the merge-base table), so a new experiment id cannot reuse a kind to get a fresh root.
    experiments_of_kind: dict[str, set] = {}
    for p in protocol.values():
        experiments_of_kind.setdefault(p.get("kind"), set()).add(p.get("experiment"))
    for kind, exps in sorted(experiments_of_kind.items(), key=lambda kv: str(kv[0])):
        if len(exps) > 1:
            problems.append(f"verdict kind {kind!r} belongs to more than one experiment: {sorted(map(str, exps))}")
    for field in ("scope", "prereg"):  # each campaign its own data directory and prereg
        values = [p.get(field) for p in protocol.values()]
        if len(set(values)) != len(values):
            problems.append(f"two campaigns share a {field}")
    successors: dict[str, list[str]] = {}
    for key, p in protocol.items():
        if "supersedes" in p:
            successors.setdefault(p["supersedes"].get("key"), []).append(key)
    for prev, keys in successors.items():
        if len(keys) > 1:
            problems.append(f"campaign {prev} is superseded more than once: {sorted(keys)}")
    closed = set(successors)
    by_experiment: dict[str, list[str]] = {}
    for key, p in protocol.items():
        by_experiment.setdefault(p.get("experiment"), []).append(key)
    for exp, keys in by_experiment.items():
        open_keys = [k for k in keys if k not in closed]
        if len(open_keys) != 1:
            problems.append(f"experiment {exp} has {len(open_keys)} open campaigns ({sorted(open_keys)}), not 1")
        if len(keys) > MAX_CAMPAIGNS:
            problems.append(f"experiment {exp} has {len(keys)} campaigns, more than {MAX_CAMPAIGNS}")
    return problems


def successor_problems(key: str, closure: bytes | None, closure_landed: float | None, prereg_landed: float | None,
                       first_marker: float, *, data_root: Path) -> list[str]:  # fmt: skip
    """Why campaign ``key`` may not run or be judged ([] = it may, or it supersedes nothing). ``closure`` is the
    predecessor's verdict as ``origin/main`` holds it, ``*_landed`` when that verdict and this campaign's prereg
    first reached ``origin/main`` (None = never), ``first_marker`` this campaign's first marker's tagger date (or
    now, before one is pushed). Only an ABORT may be succeeded: a PASS, a FAIL and Exp 63's NOT SHOWN are terminal
    (owner decision 2026-10-03, D2: NOT SHOWN is a complete attempt that could not test the claim; the next route is
    a changed ranker and a new experiment, never a successor campaign). ``data_root`` holds every predecessor's data
    directory as ``origin/main`` holds it (``materialize``): a FAILED gate leaked into a predecessor's committed
    phases bars the successor (#1059, :func:`chain_leaked_gate_problems`)."""
    sup = PROTOCOL[key].get("supersedes")
    if sup is None:
        return []
    if closure is None or closure_landed is None:
        return [f"campaign {key}: {sup['key']}'s closure verdict {sup['verdict']} is not on origin/main"]
    problems = chain_leaked_gate_problems(key, data_root)
    if sha256_bytes(closure) != sup["verdict_sha256"]:
        problems.append(f"campaign {key}: {sup['verdict']} is not the pinned closure verdict")
    try:
        verdict = json.loads(closure)
    except ValueError:
        return [*problems, f"campaign {key}: {sup['verdict']} is not JSON"]
    prev = PROTOCOL[sup["key"]]
    if (verdict.get("verdict"), verdict.get("experiment"), verdict.get("kind"), verdict.get("mock")) != (
        "ABORT",
        sup["key"],
        prev["kind"],
        False,
    ):
        problems.append(
            f"campaign {key}: {sup['key']}'s closure verdict is not a real ABORT of {sup['key']} "
            f"({verdict.get('verdict')!r}, {verdict.get('experiment')!r}, mock={verdict.get('mock')!r}); "
            "only an ABORT may be succeeded (PASS, FAIL and NOT SHOWN are terminal)"
        )
    if verdict.get("apparatus_checked") is not True:
        problems.append(f"campaign {key}: {sup['key']}'s closure verdict did not check its apparatus")
    if closure_landed >= first_marker:
        problems.append(f"campaign {key}: the closure verdict reached origin/main after its first marker")
    if prereg_landed is None or prereg_landed >= first_marker:
        problems.append(f"campaign {key}: its prereg did not reach origin/main before its first marker")
    return problems


RUN_LOG = "run_log.jsonl"  # the copied MAXIM_LOG_FILE (stored gzipped)
# The stores a resume restores (``simulation/report.py::RESUME_STORES``; a test pins the two together).
RESUME_STORES = ("hippocampus", "nac", "ec", "atl")
SIM_PORT = 8100  # the sim's local server (MAXIM_AUTO_SPAWN_PORT is not passed to the sims)
# The MAXIM_* environment a sim receives, beyond its protocol's own keys: the harness builds it from this list
# (everything else MAXIM_* is dropped), and C4 requires the recorded set to be exactly this plus the protocol's.
HARNESS_ENV = {"MAXIM_LOG_FILE_MAX_BYTES": "0"}


def required_files(exp: str) -> tuple[str, ...]:
    """The copied files a phase must carry: its report and run log, and Exp 10's and Exp 63's hippocampus store
    (their C2)."""
    if experiment_of(exp) in ("10", "63"):
        return ("report.json", RUN_LOG, "aut_hippocampus.json")
    return ("report.json", RUN_LOG)


def expected_env(exp: str, index: int) -> dict[str, str]:
    return {**HARNESS_ENV, **PROTOCOL[exp]["phases"][index][5]}


def phase_argv(exp: str, index: int, resume_session: str | None) -> list[str]:
    """The sim argv after ``python -m maxim`` for one phase: the original protocol's command, unchanged."""
    _name, goal, cap, resumes, extra, _env = PROTOCOL[exp]["phases"][index]
    argv = ["--sim", goal, "--interactive", "false", "--sim-max-turns", str(cap), *extra]
    if resumes:
        if not resume_session:
            raise ValueError(f"phase {index} resumes phase 1 and needs its session id")
        argv += ["--resume-sim", resume_session]
    return argv


def data_dir(exp: str) -> str:
    return f"docs/experiments/data/{PROTOCOL[exp]['scope']}"


def rows_path(exp: str) -> str:
    return f"{data_dir(exp)}/rows.jsonl"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ── reading committed bytes ───────────────────────────────────────────────────────────────────────────────


class Refusal(Exception):
    """The verdict cannot be established from these bytes: no verdict is written (exit 2)."""


def read_copied(session_dir: Path, name: str) -> bytes:
    """A copied session file's UNCOMPRESSED bytes: ``<name>`` as stored, or ``<name>.gz``."""
    plain, packed = session_dir / name, session_dir / f"{name}.gz"
    if plain.is_file():
        return plain.read_bytes()
    if packed.is_file():
        return gzip.decompress(packed.read_bytes())
    raise FileNotFoundError(f"{session_dir}/{name}")


def log_lines(data: bytes, *, strict: bool = False) -> list[dict]:
    """The run log's JSON lines (compact form: event key ``e``, time ``t``). The LAST line may be torn (the sim was
    still writing); with ``strict``, any other line that is not a JSON object refuses: a log the reader cannot
    fully parse cannot be counted over."""
    out: list[dict] = []
    raw_lines = data.decode("utf-8", errors="replace").splitlines()
    for i, raw in enumerate(raw_lines):
        try:
            rec = json.loads(raw)
        except ValueError:
            rec = None
        if isinstance(rec, dict):
            out.append(rec)
        elif strict and raw.strip() and i != len(raw_lines) - 1:
            raise Refusal(f"run log line {i + 1} is not a JSON object")
    return out


_TURN_ENTER = re.compile(r"Bridge\.send_and_wait ENTER turn=(\d+)\b")


def turn_windows(lines: list[dict]) -> dict[int, tuple[float, float]]:
    """``turn -> [start, end)`` from the ``sim_exec`` ``Bridge.send_and_wait ENTER turn=N`` markers; the last turn
    runs to the end of the log. A turn entered twice keeps its first entry (a retry is the same turn)."""
    starts: dict[int, float] = {}
    for rec in lines:
        if rec.get("e") != "sim_exec":
            continue
        m = _TURN_ENTER.search(str(rec.get("message", "")))
        if m and int(m.group(1)) not in starts:
            starts[int(m.group(1))] = float(rec.get("t", 0.0))
    ordered = sorted(starts.items(), key=lambda kv: kv[1])
    windows: dict[int, tuple[float, float]] = {}
    for i, (turn, start) in enumerate(ordered):
        end = ordered[i + 1][1] if i + 1 < len(ordered) else float("inf")
        windows[turn] = (start, end)
    return windows


def hippocampus_ids(store: dict) -> list[str]:
    return [str(m.get("id")) for m in store.get("memories", []) if isinstance(m, dict) and m.get("id")]


# ── the complete-attempt condition (C1–C4), per phase ────────────────────────────────────────────────────


def complete_problems(
    exp: str, index: int, row: dict, report: dict, phase1_session: str | None, phase1_stores: set[str]
) -> list[str]:
    """Every way this phase's run falls short of the prereg's complete-attempt condition ([] = complete)."""
    name, goal, cap, resumes, extra, env = PROTOCOL[exp]["phases"][index]
    prov = report.get("provenance") or {}
    p: list[str] = []
    files = row.get("files") or {}
    for needed in required_files(exp):
        if needed not in files:
            p.append(
                f"C2 {name}: {needed} was not copied"
                if needed.startswith("aut_")
                else f"C1 {name}: {needed} was not copied"
            )
    # C1: ran to the cap.
    if report.get("finish_reason") != "max_turns":
        p.append(f"C1 {name}: finish_reason {report.get('finish_reason')!r}, not 'max_turns'")
    if not isinstance(report.get("turns"), int) or report["turns"] < cap:
        p.append(f"C1 {name}: turns {report.get('turns')!r} < {cap}")
    # C2: the resume chain (Exp 10's phases 2 and 3; nothing else resumes).
    resume = prov.get("resume")
    if resumes:
        stores = (resume or {}).get("stores") or {}
        if not (resume or {}).get("resume_loaded"):
            p.append(f"C2 {name}: resume_loaded is not true ({resume!r})")
        if (resume or {}).get("resumed_from_session") != phase1_session:
            p.append(
                f"C2 {name}: resumed {(resume or {}).get('resumed_from_session')!r}, not phase 1's {phase1_session!r}"
            )
        if stores.get("hippocampus") != "loaded":
            p.append(f"C2 {name}: stores.hippocampus is {stores.get('hippocampus')!r}, not 'loaded'")
        for store in sorted(phase1_stores):
            if stores.get(store) != "loaded":
                p.append(f"C2 {name}: phase 1 saved aut_{store}.json but it reads {stores.get(store)!r}")
    elif resume is not None:
        p.append(f"C2 {name}: phase resumes nothing, but its report carries a resume stamp")
    # C3: one known, unchanged code tree, the harness's own.
    tree = prov.get("code_tree_sha256")
    harness_tree = (row.get("provenance") or {}).get("code_tree_sha256")
    if prov.get("working_tree_dirty_src_scripts") is not False:
        p.append(f"C3 {name}: working_tree_dirty_src_scripts is {prov.get('working_tree_dirty_src_scripts')!r}")
    if prov.get("code_changed_during_run") is not False:
        p.append(f"C3 {name}: code_changed_during_run is {prov.get('code_changed_during_run')!r}")
    for label, value in (
        ("code_tree_sha256", tree),
        ("end_code_tree_sha256", prov.get("end_code_tree_sha256")),
        ("harness code_tree_sha256", harness_tree),
    ):
        if not isinstance(value, str) or not value or value.startswith("unknown"):
            p.append(f"C3 {name}: {label} is {value!r}")
    if not (tree == prov.get("end_code_tree_sha256") == harness_tree):
        p.append(
            f"C3 {name}: start {tree!r}, end {prov.get('end_code_tree_sha256')!r} and harness {harness_tree!r} differ"
        )
    # C4: the pre-registered apparatus.
    for role in ("language", "aut"):
        if prov.get(f"{role}_profile") != MODEL_PROFILE_STAMPED:
            p.append(f"C4 {name}: {role}_profile {prov.get(f'{role}_profile')!r}, not {MODEL_PROFILE_STAMPED!r}")
        if prov.get(f"{role}_router_n_ctx") != N_CTX:
            p.append(f"C4 {name}: {role}_router_n_ctx {prov.get(f'{role}_router_n_ctx')!r}, not {N_CTX}")
    if prov.get("configured_n_ctx") != N_CTX:
        p.append(f"C4 {name}: configured_n_ctx {prov.get('configured_n_ctx')!r}, not {N_CTX}")
    if prov.get("configured_n_ctx_source") != "config":
        p.append(f"C4 {name}: configured_n_ctx_source {prov.get('configured_n_ctx_source')!r}, not 'config'")
    if report.get("goal") != goal:
        p.append(f"C4 {name}: goal {report.get('goal')!r} is not the prereg's")
    expected_argv = phase_argv(exp, index, phase1_session if resumes else None)
    if row.get("sim_argv") != expected_argv:
        p.append(f"C4 {name}: recorded argv {row.get('sim_argv')!r} is not {expected_argv!r}")
    if (row.get("sim_env") or None) != expected_env(exp, index):
        p.append(f"C4 {name}: the sim's MAXIM_* env {row.get('sim_env')!r} is not {expected_env(exp, index)!r}")
    served = row.get("served_model") or {}
    reads = [r for r in served.get("reads") or [] if r.get("served") is not None]
    if not reads:
        p.append(f"C4 {name}: the served model was never read during the run")
    elif any(r.get("match") is not True for r in reads):
        p.append(f"C4 {name}: a served-model read did not match {MODEL_GGUF} ({reads!r})")
    endpoint = str(report.get("language_endpoint") or "").rstrip("/")
    if not endpoint or str(served.get("url") or "").rstrip("/") != endpoint:
        p.append(f"C4 {name}: served model read at {served.get('url')!r}, but the sim used {endpoint!r}")
    # C4' (Exp 63 only): the retention model the harness read at preflight in the attempt's fresh data home.
    if experiment_of(exp) == "63" and row.get("memory_strategy") != EXP63_MEMORY_STRATEGY:
        p.append(f"C4' {name}: memory_strategy {row.get('memory_strategy')!r}, not {EXP63_MEMORY_STRATEGY!r}")
    return p


# ── the gates ─────────────────────────────────────────────────────────────────────────────────────────────


def exp10_gates(phases: list[dict]) -> dict:
    """P0–P2 and R1–R2 over one complete attempt's three phases (``phases[i]`` = report, store, lines)."""
    base, gate, garden = phases
    ids1 = hippocampus_ids(base["store"])
    n1 = len(ids1)
    out: dict = {"N1": n1}

    def first_trace(lines: list[dict]) -> dict | None:
        return next((r for r in lines if r.get("e") == "enrichment_trace"), None)

    out["P0"] = {"pass": n1 >= 3, "N1": n1}
    p1 = {}
    for label, phase in (("gate", gate), ("negative_transfer", garden)):
        tr = first_trace(phase["lines"])
        size = None if tr is None else tr.get("hippocampus_size")
        p1[label] = {"first_trace_hippocampus_size": size, "pass": isinstance(size, int) and size >= n1}
    out["P1"] = {**p1, "pass": all(v["pass"] for v in p1.values())}
    p2 = {}
    for label, phase in (("gate", gate), ("negative_transfer", garden)):
        missing = sorted(set(ids1) - set(hippocampus_ids(phase["store"])))
        p2[label] = {"missing": len(missing), "pass": not missing}
    out["P2"] = {**p2, "pass": all(v["pass"] for v in p2.values())}

    windows = turn_windows(gate["lines"])
    traces = [r for r in gate["lines"] if r.get("e") == "enrichment_trace" and str(r.get("goal") or "").strip()]
    per_turn = {}
    for turn in range(1, 9):
        window = windows.get(turn)
        in_turn = [r for r in traces if window and window[0] <= float(r.get("t", 0.0)) < window[1]]
        first = in_turn[0] if in_turn else None
        per_turn[turn] = {
            "entered": window is not None,
            "traces": len(in_turn),
            "first_memories": None if first is None else first.get("memories"),
            "first_hippocampus_size": None if first is None else first.get("hippocampus_size"),
        }
    out["R1"] = {"pass": all(v["traces"] >= 1 for v in per_turn.values()), "turns": per_turn}
    out["R2"] = {
        "pass": all(isinstance(v["first_memories"], int) and v["first_memories"] >= 3 for v in per_turn.values())
    }
    out["turns_at_N1"] = [t for t, v in per_turn.items() if v["first_hippocampus_size"] == n1]
    k = None if per_turn[1]["first_hippocampus_size"] is None else per_turn[1]["first_hippocampus_size"] - n1
    out["carried_recall_turn1"] = (
        "not shown (no first-turn trace)"
        if k is None
        else "not shown (the first trace holds fewer memories than phase 1 saved)"
        if k < 0
        else "carried recall shown on the first resume turn"
        if k == 0
        else f"at least {3 - k} of the 3 were carried"
        if k < 3
        else "carried recall is not shown"
    )
    # Negative transfer: NOT MEASURED. Reported, never gating.
    new = [m for m in garden["store"].get("memories", []) if isinstance(m, dict) and m.get("id") not in set(ids1)]
    out["negative_transfer"] = {
        "status": "NOT MEASURED",
        "agent_actions": garden["report"].get("total_actions"),
        "new_memories": len(new),
        "new_memories_mentioning_dungeon_or_guard": sum(
            1 for m in new if re.search(r"dungeon|guard", json.dumps(m).lower())
        ),
    }
    passed = all(out[g]["pass"] for g in ("P0", "P1", "P2", "R1", "R2"))
    out["verdict"] = "PASS" if passed else "FAIL"
    return out


_DAMAGE = re.compile(r"component damage: (\w+)\.integrity\b.*\(source=(\w+)")


def h5_lower_bound(intensity: float, raw: float, n: int) -> Fraction:
    """The reconstructed sensitization factor's lower bound: ``(intensity − 0.0005)(1 + 0.3·n) / raw``, exactly,
    as fractions of the logged decimals (``intensity`` is logged to 3 dp, so the full rounding error is taken off
    before it is amplified)."""
    i, r = Fraction(str(intensity)), Fraction(str(raw))
    return (i - Fraction(5, 10000)) * (1 + Fraction(3, 10) * n) / r


def exp09_gates(phase: dict, log_bytes: bytes) -> dict:
    """H1–H7 over the one session's run log."""
    lines = phase["lines"]
    reflexes = [r for r in lines if r.get("e") == "sim_reflex"]
    out: dict = {}
    flinch = [r for r in reflexes if r.get("reflex") == "attack_flinch"]
    out["H1"] = {"status": "PASS" if flinch else "NOT MET", "attack_flinch_records": len(flinch)}

    components = set()
    for r in lines:
        if r.get("e") != "sim_sem_damage" or r.get("agent_id") != "sim_aut":
            continue
        m = _DAMAGE.search(str(r.get("message", "")))
        if m and m.group(2).startswith("reflex_"):
            components.add(m.group(1))
    out["H2"] = {
        "status": "PASS" if len(components) >= 2 and "legs" in components else "NOT MET",
        "components": sorted(components),
    }
    out["H3"] = {
        "status": "NOT MEASURED",
        "why": "no committed byte carries a published pain signal's reflex source (#1026)",
    }
    seq = [r.get("intensity") for r in flinch]
    out["H4"] = {
        "status": "PASS"
        if len(seq) >= 3 and all(isinstance(x, (int, float)) for x in seq) and seq[-1] < seq[0]
        else "NOT MET",
        "attack_flinch_intensities": seq,
    }
    seen: dict[str, int] = {}
    best = None
    for r in reflexes:
        name = str(r.get("reflex"))
        n = seen.get(name, 0)
        seen[name] = n + 1
        intensity, raw = r.get("intensity"), r.get("raw_intensity")
        if not isinstance(intensity, (int, float)) or not isinstance(raw, (int, float)) or raw <= 0:
            continue
        bound = h5_lower_bound(intensity, raw, n)
        if best is None or bound > best[0]:
            best = (bound, name, n)
    out["H5"] = {
        "status": "PASS" if best is not None and best[0] > 1 else "NOT MET",
        "max_lower_bound": None if best is None else f"{float(best[0]):.6f}",
        "at": None if best is None else {"reflex": best[1], "n": best[2]},
    }
    hits = log_bytes.count(b"auto_attack") + log_bytes.count(b"auto_damage")
    out["H6"] = {"status": "PASS" if hits == 0 else "NOT MET", "occurrences": hits}
    reflex_enrichment = [r for r in lines if r.get("e") == "sim_enrichment" and r.get("system") == "reflex"]
    out["H7"] = {"status": "PASS" if reflex_enrichment else "NOT MET", "records": len(reflex_enrichment)}
    row_metric = ("H1", "H4", "H5", "H6")
    if any(out[h]["status"] != "PASS" for h in row_metric):
        out["verdict"] = "FAIL"
    elif all(out[h]["status"] == "PASS" for h in out):
        out["verdict"] = "PASS"
    else:
        out["verdict"] = "PARTIAL"
    out["not_passed"] = [h for h in ("H1", "H2", "H3", "H4", "H5", "H6", "H7") if out[h]["status"] != "PASS"]
    return out


# ── Exp 63: carried memory takes part in recall (docs/experiments/exp63_carried_recall_prereg.md) ──────────

EXP63_TURNS = 8
EXP63_TOP = 3  # what _query_hippocampus keeps and the formatter renders (memory_ids[:3])
EXP63_GOAL_LIMIT = 5  # recall(query=goal, limit=5)
SIM_AUT_AGENT_ID = "sim_aut"  # config_loader.SIM_AUT_AGENT_ID (a test pins the two together)
EXP63_MEMORY_STRATEGY = "access_based"  # C4': memory.strategy read at preflight in the attempt's fresh data home
# P3: the access bookkeeping a resume may move; every other field of a carried record is byte-equal.
EXP63_MUTABLE_FIELDS = frozenset(
    {
        "access_count",
        "accessed_at",
        "last_scored_at",
        "activation_count",
        "activation_sources",
        "promotion_pressure",
        "access_contexts",
        "long_term",
        "consolidated_at",
    }
)
EXP63_PATHS = ("graph", "goal", "substring")


def _seq(rec: dict) -> int | None:
    s = rec.get("capture_seq")
    return s if isinstance(s, int) and not isinstance(s, bool) else None


def store_records(store) -> dict[str, dict]:
    """A saved Hippocampus store's records by id, in file order (a malformed store reads as empty, never a crash)."""
    memories = store.get("memories") if isinstance(store, dict) else None
    return {
        str(m["id"]): m for m in (memories if isinstance(memories, list) else []) if isinstance(m, dict) and m.get("id")
    }


def _dict(x) -> dict:
    return x if isinstance(x, dict) else {}


def _list(x) -> list:
    return x if isinstance(x, list) else []


# The FROZEN ranker: a stdlib copy of ``maxim.memory.hippocampus_retrieval._rank_by_relevance`` and of
# ``recall(query=...)``'s candidate rule (every record a candidate, compressed included), over the saved store's
# JSON dicts (field names per ``maxim.memory.types`` ``to_dict``). The judge must re-judge the same long after #1064
# changes ``src/``, so it carries its own copy; tests/unit/test_o19_rerun.py pins it behaviourally to the shipped
# ranker (when #1064 lands, that pin is re-pointed at the executed commit's ranker, never deleted).


def ranker_tokens(rec: dict) -> set[str]:
    """The tokens ``_rank_by_relevance`` scores a record by, read from its saved dict."""
    tokens: set[str] = set()
    if rec.get("_compressed", False):  # CompressedMemory (hippocampus_persistence.load_state's marker)
        tokens.update((rec.get("goal") or "").lower().split())
        tokens.add((rec.get("tool_name") or "").lower())
        return tokens
    context = rec.get("context") or {}
    if context.get("active_goal"):
        tokens.update(context["active_goal"].lower().split())
    tokens.add((((rec.get("action") or {}).get("tool_name", "")) or "").lower())
    perception = rec.get("perception") or {}
    for obj in perception.get("detected_objects", []) or []:
        tokens.update(obj.lower().split())
    for person in perception.get("detected_people", []) or []:
        tokens.update(person.lower().split())
    obs_text = perception.get("observations", {}).get("text", "")
    if isinstance(obs_text, str):
        tokens.update(obs_text.lower().split()[:50])
    rationale = perception.get("decision_rationale", "")
    if rationale:
        tokens.update(rationale.lower().split())
    return tokens


def ranker_key(rec: dict, query: str) -> tuple:
    """The sort key ``_rank_by_relevance`` orders by, descending: ``(score, timestamp)``, or ``(timestamp,)`` for a
    query with no tokens (it then sorts by recency alone)."""
    query_tokens = set(query.lower().split())
    if not query_tokens:
        return (rec["timestamp"],)
    return (len(query_tokens & ranker_tokens(rec)) / len(query_tokens), rec["timestamp"])


def rank_by_relevance(records: list[dict], query: str, limit: int) -> list[dict]:
    """The frozen ranker itself, for a given input order (Python's sort is stable, as the shipped one's)."""
    return sorted(records, key=lambda r: ranker_key(r, query), reverse=True)[:limit]


def ranked_groups(records: list[dict], query: str) -> list[list[str]]:
    """The candidates' ids in rank order, as TIE GROUPS: records whose sort key is exactly equal. ``recall`` builds
    its candidates from a set, whose order is not reproducible, so within a group any order is the ranker's."""
    groups: list[list[str]] = []
    last = None
    for rec in sorted(records, key=lambda r: ranker_key(r, query), reverse=True):
        key = ranker_key(rec, query)
        if groups and key == last:
            groups[-1].append(str(rec["id"]))
        else:
            groups.append([str(rec["id"])])
        last = key
    return groups


def top_slots(groups: list[list[str]], n: int) -> tuple[list[list[str]], list[str], int]:
    """The first ``n`` ranks as ``(whole groups, the boundary group, its slots)``: every whole group lies inside the
    first ``n`` in any order; ``slots`` members of the boundary group (any of them) fill the rest."""
    whole: list[list[str]] = []
    left = n
    for group in groups:
        if left == 0:
            break
        if len(group) <= left:
            whole.append(group)
            left -= len(group)
        else:
            return whole, group, left
    return whole, [], 0


def goal_ids_conform(logged_goal: list[str], groups: list[list[str]], seen: set[str], need: int) -> bool:
    """Whether ``logged_goal`` is ``[m for m in top3 if m not in seen][:need]`` for SOME linearisation of the
    ranking (``top3`` = its first ``EXP63_TOP``; ``seen`` = the graph-path ids the goal path dedups against)."""
    if len(set(logged_goal)) != len(logged_goal) or set(logged_goal) & seen:
        return False
    whole, boundary, slots = top_slots(groups, EXP63_TOP)
    i = produced = 0
    for group in whole:
        if produced == need:
            break
        fresh = set(group) - seen
        k = min(len(fresh), need - produced)
        chunk = logged_goal[i : i + k]
        if len(chunk) != k or not set(chunk) <= fresh:
            return False
        i, produced = i + k, produced + k
    if produced < need and boundary:
        fresh = set(boundary) - seen
        rest = logged_goal[i:]
        c, left = len(rest), need - produced
        # ``slots`` boundary members are chosen; s of them are not in ``seen``, for any s in
        # [max(0, slots - |boundary & seen|), min(slots, |fresh|)]; the goal path then keeps min(s, left).
        if not set(rest) <= fresh or c > left or c > min(slots, len(fresh)):
            return False
        if c != left and c < slots - (len(boundary) - len(fresh)):
            return False
        i += c
    return i == len(logged_goal)


def forces_carried(groups: list[list[str]], carried: set[str]) -> bool:
    """R3d: EVERY linearisation's goal top 3 holds a carried id, so the top 3 with the carried ids removed differs
    from all of them (a recall that dropped carried records gives a different answer). Ties are exact only: a
    carried id that only MAY be chosen at the boundary is not decisive."""
    whole, boundary, slots = top_slots(groups, EXP63_TOP)
    if any(set(g) & carried for g in whole):
        return True
    return bool(boundary) and len(set(boundary) - carried) < slots


def _aut_traces(lines: list[dict]) -> list[tuple[int, dict]]:
    """``(line index, record)`` of the AUT's enrichment traces (C5(a): the gates read only ``sim_aut``'s)."""
    return [
        (i, r)
        for i, r in enumerate(lines)
        if r.get("e") == "enrichment_trace" and r.get("agent_id") == SIM_AUT_AGENT_ID
    ]


def turn_starts(lines: list[dict]) -> dict[int, int]:
    """``turn -> `` the line index of its first ``sim_exec`` ``Bridge.send_and_wait ENTER turn=N`` line. Exp 63
    attributes by LINE ORDER (the log's ``t`` is rounded to 0.01 s)."""
    starts: dict[int, int] = {}
    for i, rec in enumerate(lines):
        if rec.get("e") != "sim_exec":
            continue
        m = _TURN_ENTER.search(str(rec.get("message", "")))
        if m and int(m.group(1)) not in starts:
            starts[int(m.group(1))] = i
    return starts


def turn_of(index: int, starts: dict[int, int]) -> int:
    """The turn whose window holds line ``index`` (0 = before any turn entered)."""
    turn, at = 0, -1
    for t, start in starts.items():
        if at < start <= index:
            turn, at = t, start
    return turn


def turn_traces(lines: list[dict]) -> dict[int, tuple[int, dict] | None]:
    """``turn -> `` its trace: the first AUT ``enrichment_trace`` with a non-empty goal in its window."""
    starts = turn_starts(lines)
    out: dict[int, tuple[int, dict] | None] = {t: None for t in range(1, EXP63_TURNS + 1)}
    for i, r in _aut_traces(lines):
        if not str(r.get("goal") or "").strip():
            continue
        t = turn_of(i, starts)
        if t in out and out[t] is None:
            out[t] = (i, r)
    return out


def _holes(trace: dict) -> set[int]:
    return {s for a, b in trace.get("goal_path_holes") or [] for s in range(a, b + 1)}


def visible_ids(trace: dict, records: dict[str, dict]) -> set[str]:
    """V0: phase 2's saved records with ``capture_seq <= horizon`` and not in the trace's holes (C5(d) makes every
    carried record carry its capture_seq; a record without one is never visible here)."""
    h, holes = trace.get("goal_path_horizon"), _holes(trace)
    return {
        i
        for i, rec in records.items()
        if _seq(rec) is not None and _is_int(h) and _seq(rec) <= h and _seq(rec) not in holes
    }


def _is_int(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _seq_c(base_records: dict[str, dict]) -> int:
    return max((s for s in map(_seq, base_records.values()) if s is not None), default=-1)


def exp63_r3a_trace(lines: list[dict], seq_c: int) -> tuple[int, dict] | None:
    """R3a's view: the first GOAL-BEARING phase-2 AUT trace whose horizon is seq_C and whose holes are empty (no new
    memory). An empty-goal trace never ran the goal path, so it is never R3a's (E3)."""
    return next(
        (
            (i, r)
            for i, r in _aut_traces(lines)
            if str(r.get("goal") or "").strip()
            and _is_int(r.get("goal_path_horizon"))
            and r.get("goal_path_horizon") == seq_c
            and r.get("goal_path_holes") == []
        ),
        None,
    )


def _p1(gate: dict, n1: int) -> tuple[object, bool]:
    """P1: phase 2's first AUT trace's hippocampus_size, and whether it holds at least N1."""
    traces = _aut_traces(gate["lines"])
    size = traces[0][1].get("hippocampus_size") if traces else None
    return size, _is_int(size) and size >= n1


def exp63_c5_problems(phases: list[dict], *, goal: str) -> list[str]:
    """C5 (the instrument reads true) and R3a's observability: every failure makes the attempt INCOMPLETE (an
    aborted attempt, never a FAIL). ``phases`` = [baseline, gate], each ``{report, store, lines}``; ``goal`` the
    gate phase's protocol goal."""
    base, gate = phases
    found: dict[str, str] = {}  # the first problem per check (one bad trace is enough to abort)
    # Every read below is type-guarded (E5): a corrupt trace or store makes the attempt incomplete, never a crash.

    def problem(check: str, text: str) -> None:
        found.setdefault(check, f"C5{check} {text}")

    records = store_records(gate["store"])
    base_records = store_records(base["store"])
    carried = set(base_records)
    # (a) the agent: every trace names one; the AUT is sim_aut, as its own deliberation lines say.
    for label, phase in (("baseline", base), ("gate", gate)):
        for r in phase["lines"]:
            if r.get("e") == "enrichment_trace" and not (isinstance(r.get("agent_id"), str) and r["agent_id"]):
                problem("(a)", f"{label}: an enrichment_trace carries no agent_id")
    if not any(r.get("e") == "sim_deliberation" and r.get("agent_id") == SIM_AUT_AGENT_ID for r in gate["lines"]):
        problem("(a)", f"gate: no sim_deliberation line names the AUT {SIM_AUT_AGENT_ID!r}")
    traces = _aut_traces(gate["lines"])
    if not traces:
        problem("(a)", f"gate: no enrichment_trace of {SIM_AUT_AGENT_ID!r}")
    # (b) the shape of every phase-2 AUT trace.
    shaped: list[tuple[int, dict]] = []
    for i, tr in traces:
        ids, paths = tr.get("memory_ids"), tr.get("memory_paths")
        where = f"gate line {i + 1}"
        if not (isinstance(ids, list) and isinstance(paths, list) and all(isinstance(x, str) for x in ids)):
            problem("(b)", f"{where}: memory_ids / memory_paths are not lists of ids")
            continue
        if not (len(ids) == len(paths) == tr.get("memories")) or not all(
            isinstance(p, str) and p in EXP63_PATHS for p in paths
        ):
            problem("(b)", f"{where}: {len(ids)} ids, {len(paths)} paths, memories {tr.get('memories')!r}")
            continue
        if not set(ids) <= set(records):
            problem("(b)", f"{where}: ids {sorted(set(ids) - set(records))} are not in phase 2's saved store")
            continue
        holes = tr.get("goal_path_holes")
        if not _is_int(tr.get("goal_path_horizon")) or not (
            isinstance(holes, list)
            and all(isinstance(h, list) and len(h) == 2 and all(map(_is_int, h)) and h[0] <= h[1] for h in holes)
        ):
            problem("(b)", f"{where}: goal_path_horizon {tr.get('goal_path_horizon')!r} / holes {holes!r} unreadable")
            continue
        if not _is_int(tr.get("hippocampus_size")):
            # on EVERY AUT trace (Arch SF2): P1 reads the first one, so an unreadable size must abort, never FAIL P1
            problem("(b)", f"{where}: hippocampus_size {tr.get('hippocampus_size')!r} is not an integer")
            continue
        if not isinstance(tr.get("goal") or "", str) or (str(tr.get("goal") or "").strip() and tr.get("goal") != goal):
            problem("(b)", f"{where}: goal {tr.get('goal')!r} is not the protocol goal")
            continue
        shaped.append((i, tr))
    # (L) liveness: each turn's trace surfaced min(3, the store) memories.
    for turn, hit in turn_traces(gate["lines"]).items():
        if hit is not None:
            tr = hit[1]
            size = tr.get("hippocampus_size")
            if not _is_int(size) or tr.get("memories") != min(EXP63_TOP, size):
                problem("(L)", f"turn {turn}: memories {tr.get('memories')!r} with hippocampus_size {size!r}")
    # (d) record shape: no compressed record; observations.text a string or absent; every carried record carries its
    # capture_seq (Amendment 1: strict; without it V0 and seq_C are undefined).
    for rid, rec in records.items():
        if rec.get("_compressed", False):
            problem("(d)", f"phase 2's store holds a compressed record ({rid})")
            continue
        perception = rec.get("perception", {})
        obs = perception.get("observations", {}) if isinstance(perception, dict) else None
        if not isinstance(obs, dict) or ("text" in obs and not isinstance(obs["text"], str)):
            problem("(d)", f"record {rid}: perception.observations.text is not a string")
        elif not isinstance(rec.get("timestamp"), (int, float)) or isinstance(rec.get("timestamp"), bool):
            problem("(d)", f"record {rid}: timestamp {rec.get('timestamp')!r} is not a number")
    for rid, rec in base_records.items():
        if _seq(rec) is None:
            problem("(d)", f"carried record {rid} has no integer capture_seq")
    # (c) rendering identity: each id's enrichment activations in phase 2 = its appearances in memory_ids[:3].
    rendered: dict[str, int] = {}
    for _i, tr in traces:
        ids = tr.get("memory_ids") if isinstance(tr.get("memory_ids"), list) else []
        for mid in ids[:EXP63_TOP]:
            rendered[str(mid)] = rendered.get(str(mid), 0) + 1

    def enrichment(rec: dict | None) -> int:
        n = _dict(_dict(rec).get("activation_sources")).get("enrichment", 0)
        return n if _is_int(n) else -(10**9)

    # Over phase 2's records (and anything rendered): a carried id missing from phase 2 is P2's FAIL, not C5's.
    for rid in sorted(set(records) | set(rendered)):
        delta = enrichment(records.get(rid)) - enrichment(base_records.get(rid))
        if delta != rendered.get(rid, 0):
            problem("(c)", f"{rid}: enrichment activations moved {delta}, but it was rendered {rendered.get(rid, 0)}x")
    # (g) graph ids: an object-bearing record (in the saved store: (b); one that landed after the horizon read is
    # allowed, as a goal id is, and joins L, E4); (s) no substring id on a goal-bearing trace.
    for i, tr in shaped:
        for mid, path in zip(tr["memory_ids"], tr["memory_paths"]):
            if path == "graph":
                objects = _list(_dict(records[mid].get("perception")).get("detected_objects"))
                if not objects:
                    problem("(g)", f"gate line {i + 1}: graph id {mid} has no detected_objects")
            if path == "substring" and str(tr.get("goal") or "").strip():
                problem("(s)", f"gate line {i + 1}: a goal-bearing trace carries substring id {mid}")
    # R3a's observability: a goal-bearing phase-2 AUT trace saw the store before any new memory. Only when P1 and P2
    # hold (A2): after a total restore failure no trace can see seq_C, and P1/P2 (independent bytes) FAIL instead.
    restored = _p1(gate, len(carried))[1] and carried <= set(records)
    r3a = exp63_r3a_trace(gate["lines"], _seq_c(base_records))
    if restored and r3a is None:
        problem(
            "(R3a)", "no goal-bearing phase-2 AUT trace has horizon seq_C with no holes: reachability is not observable"
        )
    elif restored and r3a is not None and set(_list(r3a[1].get("memory_ids"))) - set(base_records):
        # Only carried records were visible at its horizon read, so an id outside C landed between that read and the
        # recall: the view is contaminated by the capture race, an instrument fault, never the claim's FAIL.
        problem("(R3a)", f"gate line {r3a[0] + 1}: R3a's trace surfaced an id that landed during the query")
    return list(found.values())


def _qualifying_carried(groups: list[list[str]], carried: set[str], records: dict[str, dict], goal: str) -> set[str]:
    """The carried ids that rank STRICTLY above at least one visible non-carried record (D1): a carried id that only
    fills a slot because fewer than 3 new records are visible does not qualify."""
    new = [m for g in groups for m in g if m not in carried]
    if not new:
        return set()
    lowest_new = min(ranker_key(records[m], goal) for m in new)
    return {m for g in groups for m in g if m in carried and ranker_key(records[m], goal) > lowest_new}


def shown_goal_ids(groups: list[list[str]], seen: set[str], need: int, last: set[str]) -> list[str]:
    """The goal-path ids shown under ONE linearisation: within each tie group the ids in ``last`` go last (and are
    left out of a boundary group whenever its slots allow), then ``[m for m in top3 if m not in seen][:need]``. With
    ``last`` = the qualifying carried ids this is the ordering that shows as few of them as any ordering can."""
    whole, boundary, slots = top_slots(groups, EXP63_TOP)
    top = [m for g in whole for m in sorted(g, key=lambda m: m in last)]
    top += sorted(boundary, key=lambda m: m in last)[:slots]
    return [m for m in top if m not in seen][:need]


def exp63_decisive(
    groups: list[list[str]], graph: list[str], carried: set[str], records: dict[str, dict], goal: str
) -> bool:
    """R3d (D1): in EVERY valid tie ordering, a carried id is among the 3 memories SHOWN
    (``dedup(graph ids + goal top 3)[:3]``) from the goal path AND ranks strictly above a visible non-carried record.
    Graph ids that fill the slots (the goal path never ran) are not decisive: no goal id is then shown (``need``
    is 0); a carried graph id is cue-dependent, not the ranking, and does not count."""
    q = _qualifying_carried(groups, carried, records, goal)
    return bool(set(shown_goal_ids(groups, set(graph), max(0, EXP63_TOP - len(graph)), q)) & q)


def exp63_turn(tr: dict, records: dict[str, dict], carried: set[str], goal: str) -> dict:
    """R3' and R3d for one turn's trace (C5 has held: its shape is sound)."""
    ids, paths = list(tr.get("memory_ids") or []), list(tr.get("memory_paths") or [])
    v0 = visible_ids(tr, records)
    # L: logged ids (goal or graph, E4) outside V0 that are in the saved store: they landed between the horizon read
    # and the recall.
    late = [m for m in ids if m not in v0 and m in records]
    pool = [records[m] for m in records if m in v0 or m in late]
    groups = ranked_groups(pool, goal)
    graph = [m for m, p in zip(ids, paths) if p == "graph"]
    logged_goal = [m for m, p in zip(ids, paths) if p == "goal"]
    order_ok = paths == ["graph"] * len(graph) + ["goal"] * len(logged_goal)  # the path order: graph, then goal
    if len(graph) >= EXP63_TOP:
        conforms = order_ok and len(ids) == EXP63_TOP and not logged_goal  # the goal path never ran
    else:
        conforms = order_ok and goal_ids_conform(logged_goal, groups, set(graph), EXP63_TOP - len(graph))
    canonical = [m for g in groups for m in sorted(g)][:EXP63_TOP]  # one linearisation, for the descriptive counts
    recomputed = list(dict.fromkeys(graph + canonical))[:EXP63_TOP]
    scores = {m: ranker_key(records[m], goal)[0] for m in v0 | set(late)}
    best_carried = max((scores[m] for m in scores if m in carried), default=None)
    whole, boundary, slots = top_slots(groups, EXP63_TOP)
    possible = any(set(g) & carried for g in whole) or bool(slots and set(boundary) & carried)
    return {
        "conforms": conforms,
        "decisive": exp63_decisive(groups, graph, carried, records, goal),
        "visible": len(v0),
        "late": late,
        "observed_carried": len(set(ids) & carried),
        "recomputed_carried_canonical": len(set(recomputed) & carried),
        # A8: whether the goal top 3 holds a carried id in every tie ordering ("forced"), in some ("possible"), or none.
        "carried_in_goal_top3": "forced" if forces_carried(groups, carried) else "possible" if possible else "none",
        "best_carried_score": best_carried,
        "new_at_or_above_best_carried": None
        if best_carried is None
        else sum(1 for m, s in scores.items() if m not in carried and s >= best_carried),
        "graph_ids": len(graph),
    }


def exp63_gates(phases: list[dict], *, goal: str) -> dict:
    """P0–P3, R1, R3a, R3' and R3d over one COMPLETE attempt's two phases (C5 has held), and the verdict."""
    base, gate = phases
    base_records, records = store_records(base["store"]), store_records(gate["store"])
    carried = set(base_records)
    n1, seq_c = len(carried), _seq_c(base_records)
    out: dict = {"N1": n1, "seq_C": seq_c}
    out["P0"] = {"pass": n1 >= 3, "N1": n1}
    size, p1 = _p1(gate, n1)
    out["P1"] = {"first_trace_hippocampus_size": size, "pass": p1}
    missing = sorted(carried - set(records))
    out["P2"] = {"missing": missing, "pass": not missing}

    def fields(rec: dict) -> dict[str, str]:
        return {k: json.dumps(x, sort_keys=True) for k, x in rec.items() if k not in EXP63_MUTABLE_FIELDS}

    changed = {
        rid: sorted(set(fields(base_records[rid]).items()) ^ set(fields(records[rid]).items()))
        for rid in sorted(carried & set(records))
    }
    changed = {rid: sorted({k for k, _x in diff}) for rid, diff in changed.items() if diff}
    early = sorted(
        rid for rid, rec in records.items() if rid not in carried and (_seq(rec) is None or _seq(rec) <= seq_c)
    )
    out["P3"] = {"changed": changed, "new_not_after_seq_C": early, "pass": not changed and not early}

    starts = turn_starts(gate["lines"])
    per_turn = turn_traces(gate["lines"])
    out["R1"] = {
        "pass": all(hit is not None for hit in per_turn.values()),
        "turns": sorted(t for t, h in per_turn.items() if h),
    }
    r3a = exp63_r3a_trace(gate["lines"], seq_c)
    r3a_turn = None if r3a is None else turn_of(r3a[0], starts)
    r3a_ids = [] if r3a is None else list(r3a[1].get("memory_ids") or [])
    out["R3a"] = {
        "turn": r3a_turn,
        "ids": r3a_ids,
        "pass": r3a is not None and len(r3a_ids) == min(EXP63_TOP, n1) and set(r3a_ids) <= carried,
    }
    turns = {t: None if hit is None else exp63_turn(hit[1], records, carried, goal) for t, hit in per_turn.items()}
    out["R3prime"] = {
        "pass": all(x is not None and x["conforms"] for x in turns.values()),
        "nonconforming": [t for t, x in turns.items() if x is None or not x["conforms"]],
    }
    decisive = [
        t for t, x in turns.items() if x is not None and x["decisive"] and r3a_turn is not None and t > r3a_turn
    ]
    out["R3d"] = {"decisive_turns": decisive, "pass": bool(decisive)}
    # A8: the first turn whose goal top 3 no longer MUST hold a carried id, and the first where it no longer CAN.
    leave = {
        level: next((t for t, x in sorted(turns.items()) if x is not None and x["carried_in_goal_top3"] in below), None)
        for level, below in (("forced", ("possible", "none")), ("possible", ("none",)))
    }
    recall_calls = 0
    for r in gate["lines"]:
        text = json.dumps(r)
        recall_calls += "memory_recall" in text and any(m in text for m in carried)
    out["descriptive"] = {
        "status": "DESCRIPTIVE (never gating)",
        "turns": turns,
        "carried_leave_goal_top3_turn": leave,
        "memory_recall_lines_naming_carried_ids": recall_calls,
    }
    gating = ("P0", "P1", "P2", "P3", "R1", "R3a", "R3prime")
    if not all(out[g]["pass"] for g in gating):
        out["verdict"] = "FAIL"
    else:
        out["verdict"] = "PASS" if out["R3d"]["pass"] else "NOT SHOWN"
    out["not_passed"] = [g for g in (*gating, "R3d") if not out[g]["pass"]]
    return out


# ── #1059: a FAILED gate leaked into an aborted predecessor's committed phases bars its successor ───────────
# D2 (the design pass's fix): the gates a committed PREFIX of an attempt's phases decides, per experiment: the number of
# leading ``ok`` phase rows -> the gates computable from them. A prefix of another length cannot be judged, and a
# leaked phase that cannot be judged BARS the successor (strict: the design pass's recommendation, adopted
# 2026-10-04). ``P1.gate`` / ``P2.gate``: Exp 10's P1 and P2 over the gate phase only (the garden phase never ran).
# Deliberately stricter than the preregs' FAIL lines: Exp 09's H2/H7 and Exp 63's R3d decide PARTIAL / NOT SHOWN, not
# FAIL, yet a leak of any of them bars the successor too (fail closed, intentional: a leaked non-PASS is still a
# result seen before the successor was opened).
LEAK_GATES: dict[str, dict[int, tuple[str, ...]]] = {
    "10": {1: ("P0",), 2: ("P0", "P1.gate", "P2.gate", "R1", "R2"), 3: ("P0", "P1", "P2", "R1", "R2")},
    "09": {1: ("H1", "H2", "H4", "H5", "H6", "H7")},  # H3 is NOT MEASURED by design (#1026), never a leak
    "63": {1: ("P0",), 2: ("P0", "P1", "P2", "P3", "R1", "R3a", "R3prime", "R3d")},
}
# What reading or gating a committed phase may raise: each is "cannot be judged", which bars (never a crash).
_UNJUDGEABLE = (Refusal, ValueError, KeyError, TypeError, IndexError, AttributeError, OSError, ZeroDivisionError)


def _gate_passes(key: str, phases: list[dict]) -> dict[str, bool]:
    """``gate -> passed`` for every gate a prefix of ``phases`` (1..all of the experiment's phases) can decide."""
    experiment = experiment_of(key)
    if experiment == "10":
        g = exp10_gates([*phases, *[phases[-1]] * (3 - len(phases))])  # padding: only LEAK_GATES' names are read
        out = {name: g[name]["pass"] for name in ("P0", "P1", "P2", "R1", "R2")}
        out.update({"P1.gate": g["P1"]["gate"]["pass"], "P2.gate": g["P2"]["gate"]["pass"]})
        return out
    if experiment == "09":
        g = exp09_gates(phases[0], phases[0]["log_bytes"])
        return {name: g[name]["status"] == "PASS" for name in ("H1", "H2", "H3", "H4", "H5", "H6", "H7")}
    if len(phases) == 1:
        return {"P0": len(store_records(phases[0]["store"])) >= 3}
    g = exp63_gates(phases, goal=PROTOCOL[key]["phases"][1][1])
    return {name: g[name]["pass"] for name in LEAK_GATES["63"][2]}


def _committed_phase(data_root: Path, row: dict) -> dict:
    """One committed ``ok`` phase's bytes as the gates read them, each copied file re-hashed against its row."""
    session, files = row.get("session_id"), row.get("files")
    if not isinstance(session, str) or not session or "/" in session or session in (".", ".."):
        raise Refusal(f"session_id {session!r} is not one plain path component")
    if not isinstance(files, dict):
        raise Refusal("files is not a mapping")
    sdir = data_root / session
    for name, digest in files.items():
        if sha256_bytes(read_copied(sdir, name)) != digest:
            raise Refusal(f"{session}/{name}: SHA-256 differs from its row")
    log_bytes = read_copied(sdir, RUN_LOG)
    store = json.loads(read_copied(sdir, "aut_hippocampus.json")) if "aut_hippocampus.json" in files else {}
    report = json.loads(read_copied(sdir, "report.json"))
    return {"report": report, "store": store, "lines": log_lines(log_bytes, strict=True), "log_bytes": log_bytes}


def leaked_gate_problems(pred_key: str, rows: list[dict], data_root: Path) -> list[str]:
    """Why campaign ``pred_key``'s committed rows bar a successor ([] = they do not): over each attempt's leading
    ``ok`` phases (``data_root`` = its data directory), every gate :data:`LEAK_GATES` says they decide must pass. A
    prefix that cannot be judged bars too. Pure over the bytes (no git, no network)."""
    try:
        attempts = attempts_from_rows(rows)
    except Refusal as exc:
        return [f"campaign {pred_key}: its rows cannot be read ({exc}), so a leaked phase cannot be ruled out"]
    problems = []
    for run_id, attempt_rows in attempts.items():
        prefix = []
        for i, row in enumerate(attempt_rows):
            if row.get("phase_index") != i or row.get("status") != "ok":
                break
            prefix.append(row)
        if not prefix:
            continue
        label = f"campaign {pred_key}, attempt {str(run_id)[:12]}"
        names = LEAK_GATES[experiment_of(pred_key)].get(len(prefix))
        if names is None:
            problems.append(f"{label}: {len(prefix)} committed phases decide no known gate set (cannot be judged)")
            continue
        try:
            passes = _gate_passes(pred_key, [_committed_phase(data_root, row) for row in prefix])
            failed = [name for name in names if passes[name] is not True]
        except _UNJUDGEABLE as exc:
            problems.append(f"{label}: its committed phases cannot be judged ({type(exc).__name__}: {exc})"[:300])
            continue
        if failed:
            problems.append(f"{label}: a FAILED gate leaked into its committed phases: {failed}")
    return problems


def chain_leaked_gate_problems(key: str, data_root: Path) -> list[str]:
    """:func:`leaked_gate_problems` over every campaign ``key`` succeeds, read from ``data_root/<scope>/``. Each
    predecessor's ``verdict.json`` there must be the closure its successor pins by SHA-256, and its ``rows.jsonl`` the
    bytes that closure judged (``data_sha256``): a predecessor's rows edited after its closure cannot launder a leak."""
    problems = []
    succ = key
    for pred in predecessors(key):
        sup, succ = PROTOCOL[succ]["supersedes"], pred
        pdir = data_root / PROTOCOL[pred]["scope"]
        try:
            closure = (pdir / "verdict.json").read_bytes()
            raw = (pdir / "rows.jsonl").read_bytes()
            if sha256_bytes(closure) != sup["verdict_sha256"]:
                raise ValueError("its verdict.json is not the pinned closure")
            pinned = json.loads(closure)
            if pinned.get("data") != rows_path(pred) or sha256_bytes(raw) != pinned.get("data_sha256"):
                raise ValueError("its rows are not the bytes its pinned closure judged")
            rows = [json.loads(ln) for ln in raw.decode("utf-8").splitlines() if ln.strip()]
        except (OSError, ValueError, AttributeError) as exc:
            problems.append(f"campaign {pred}: its rows cannot be read as its pinned closure judged them "
                            f"({str(exc)[:120]}), so a leaked phase cannot be ruled out")  # fmt: skip
            continue
        problems += leaked_gate_problems(pred, rows, pdir)
    return problems


# ── #1059: the subject (owner decision 2026-10-04) -- byte-identical between a root and its successors ───────────
# The harness's preflight copy (o19_rerun.check_campaign); the evidence gate holds its own constants
# (_evidence_records.SUBJECT_PATHS / SUBJECT_EXCLUDED: a test pins the two equal) and is the authority. Installed
# library versions, llama.cpp, the model and encoder weights and ``.python-version`` are outside it: a disclosed gap
# (mechanization backlog M28). Not subject: ``scripts/exp44/capture_paired_prompts.py``, which
# ``simulation/orchestrator.py`` loads only when ``MAXIM_EXP44_CAPTURE_LOG`` is set; an O19 sim never has it (the
# harness drops every MAXIM_* key but HARNESS_ENV and the protocol's, and C4 pins the recorded env to exactly those).
SUBJECT_PATHS = (
    "src/maxim",
    "pyproject.toml",
    "uv.lock",
    "poetry.lock",
    "pdm.lock",
    "Pipfile.lock",
    "scenarios",
    "data",
)
SUBJECT_EXCLUDED = frozenset({"src/maxim/utils/function_length_baseline.json"})  # non-runtime lint data


def subject_listing(commit: str) -> list[str]:
    """``mode type oid\tpath`` for every subject entry at ``commit`` (literal paths, no rename or attribute magic)."""
    out = _git("--literal-pathspecs", "ls-tree", "-r", "-z", "--full-tree", commit, "--", *SUBJECT_PATHS)
    return sorted(e for e in out.split("\0") if e and e.partition("\t")[2] not in SUBJECT_EXCLUDED)


def campaign_commits_on_main(key: str) -> tuple[set[str], list[str]]:
    """Every commit an attempt of a campaign ``key`` succeeds ran on (its pinned closure verdict's rows' executed
    commits and its markers' peeled commits, as ``origin/main`` holds them), and the problems reading them."""
    commits: set[str] = set()
    problems: list[str] = []
    succ = key
    for pred in predecessors(key):
        sup = PROTOCOL[succ]["supersedes"]
        succ = pred
        closure = _git_bytes("show", f"origin/main:{sup['verdict']}")
        if closure is None or sha256_bytes(closure) != sup["verdict_sha256"]:
            problems.append(f"campaign {pred}: its pinned closure verdict is not on origin/main")
            continue
        try:
            verdict = json.loads(closure)
            rows = _git_bytes("show", f"origin/main:{verdict.get('data')}")
            if rows is None or sha256_bytes(rows) != verdict.get("data_sha256"):
                problems.append(f"campaign {pred}: its closure verdict's rows are not on origin/main")
                continue
            lines = [json.loads(ln) for ln in rows.splitlines() if ln.strip()]
            found = {
                (r.get("provenance") or {}).get("executed_git_hash")
                for r in lines
                if r.get("record_kind") == "harness_row"  # as the gate's _executed reads them
            }
            found |= {m.get("peeled") for m in (verdict.get("apparatus") or {}).get("markers") or []}
        except (ValueError, AttributeError, TypeError) as exc:
            problems.append(f"campaign {pred}: its closure records cannot be read ({type(exc).__name__})")
            continue
        found.discard(None)
        if not found:
            problems.append(f"campaign {pred}: no executed commit is recorded")
        commits |= found
    return commits, problems


def subject_problems(key: str, head: str) -> list[str]:
    """Why an attempt of successor ``key`` at ``head`` would run on another subject than its predecessors' ([] = the
    subject is byte-identical, or ``key`` supersedes nothing). Harness preflight only: the evidence gate decides."""
    if not predecessors(key):
        return []
    commits, problems = campaign_commits_on_main(key)
    try:
        want = subject_listing(head)
    except subprocess.CalledProcessError:
        return [*problems, f"the subject at {head[:12]} cannot be listed"]
    for commit in sorted(commits, key=str):
        try:
            have = subject_listing(commit)
        except subprocess.CalledProcessError:
            problems.append(f"a predecessor's executed commit {str(commit)[:12]} cannot be read")
            continue
        if have != want:
            first = sorted(set(have) ^ set(want))[0].partition("\t")[2]
            problems.append(
                f"campaign {key}: the subject at {head[:12]} differs from a predecessor's executed commit "
                f"{commit[:12]} (first: {first}); a successor needs byte-identical subject code (#1059)"
            )
    return problems


def materialize(ref: str, keys: list[str], dest: Path) -> Path:
    """Write each campaign's data directory as ``ref`` holds it under ``dest/<scope>/`` (no symlinks); returns
    ``dest``, the ``data_root`` :func:`successor_problems` reads."""
    for key in keys:
        rel = data_dir(key)
        out = subprocess.run(["git", "ls-tree", "-r", "-z", "--full-tree", ref, "--", rel],
                             cwd=REPO_ROOT, capture_output=True, check=True).stdout  # fmt: skip
        for entry in out.split(b"\0"):
            if not entry:
                continue
            meta, _, path = entry.partition(b"\t")
            mode, kind, oid = meta.split()
            if kind != b"blob" or mode == b"120000":
                continue
            target = dest / PROTOCOL[key]["scope"] / path.decode()[len(rel) + 1 :]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(subprocess.run(["git", "cat-file", "blob", oid.decode()], cwd=REPO_ROOT,
                                              capture_output=True, check=True).stdout)  # fmt: skip
    return dest


# ── attempts, markers, the ruleset, ordering ─────────────────────────────────────────────────────────────


def attempts_from_rows(rows: list[dict]) -> dict[str, list[dict]]:
    """Harness rows grouped by attempt (the harness run id), each in phase order."""
    out: dict[str, list[dict]] = {}
    for row in rows:
        if row.get("record_kind") != "harness_row":
            continue
        run_id = (row.get("provenance") or {}).get("harness_run_id")
        if not run_id:
            raise Refusal("a harness row carries no provenance.harness_run_id")
        out.setdefault(run_id, []).append(row)
    for run_rows in out.values():
        run_rows.sort(key=lambda r: r.get("phase_index", 0))
    return out


_MARKER = re.compile(r"attempt-(\d+)-([0-9a-f]{32})$")


def parse_markers(exp: str, ls_remote: str) -> dict[str, dict]:
    """``git ls-remote --tags origin`` lines for this exp's namespace -> ``run_id -> {k, ref, object, peeled}``.
    A tag with no ``^{}`` line is lightweight (no tagger date): refused."""
    prefix = f"{MARKER_NAMESPACE}/{exp}/"
    objects: dict[str, str] = {}
    peeled: dict[str, str] = {}
    for line in ls_remote.splitlines():
        if not line.strip():
            continue
        sha, ref = line.split()
        if not ref.startswith(prefix):
            continue
        if ref.endswith("^{}"):
            peeled[ref[: -len("^{}")]] = sha
        else:
            objects[ref] = sha
    markers: dict[str, dict] = {}
    for ref, sha in sorted(objects.items()):
        m = _MARKER.search(ref)
        if not m or ref != f"{prefix}{m.group(0)}":
            raise Refusal(f"marker {ref} is not attempt-<k>-<32-hex run id>")
        if ref not in peeled:
            raise Refusal(f"marker {ref} is a lightweight tag (no tagger date)")
        run_id = m.group(2)
        if run_id in markers:
            raise Refusal(f"two markers name run id {run_id}")
        markers[run_id] = {"k": int(m.group(1)), "ref": ref, "object": sha, "peeled": peeled[ref]}
    ks = sorted(m["k"] for m in markers.values())
    if ks != list(range(1, len(ks) + 1)):
        raise Refusal(f"marker k values {ks} are not unique and 1..n with no gaps")
    if len(ks) > MAX_ATTEMPTS:
        raise Refusal(f"{len(ks)} markers: more than {MAX_ATTEMPTS} attempts")
    return markers


def ruleset_problems(
    rulesets: list[dict],
    details: dict[int, dict],
    history: dict[int, list[dict]],
    first_marker_ts: float,
    parse_ts: Callable[[str], float],
) -> list[str]:
    """Why the tag ruleset does not keep the markers, or [] (prereg (a)). ``rulesets`` is the repo's ruleset list,
    ``details``/``history`` the full ruleset and its ``/history`` by id; ``first_marker_ts`` the earliest tagger date."""
    matches = []
    for rs in rulesets:
        d = details.get(rs.get("id"))
        if not d or d.get("target") != "tag":
            continue
        cond = (d.get("conditions") or {}).get("ref_name") or {}
        if cond.get("include") == [RULESET_INCLUDE] and not cond.get("exclude"):
            matches.append(d)
    if len(matches) != 1:
        return [f"{len(matches)} tag rulesets include exactly {RULESET_INCLUDE!r}; need exactly one"]
    d = matches[0]
    p: list[str] = []
    if d.get("enforcement") != "active":
        p.append(f"ruleset enforcement is {d.get('enforcement')!r}, not 'active'")
    types = {r.get("type") for r in d.get("rules") or []}
    for needed in ("deletion", "update"):
        if needed not in types:
            p.append(f"ruleset does not carry the {needed!r} rule")
    if d.get("bypass_actors") != []:
        p.append(f"ruleset bypass_actors is {d.get('bypass_actors')!r}, not []")
    if d.get("current_user_can_bypass") != "never":
        p.append(f"current_user_can_bypass is {d.get('current_user_can_bypass')!r}, not 'never'")
    stamps = [("created_at", d.get("created_at")), ("updated_at", d.get("updated_at"))]
    stamps += [(f"history {h.get('version_id')} updated_at", h.get("updated_at")) for h in history.get(d.get("id"), [])]
    for label, value in stamps:
        try:
            when = parse_ts(str(value))
        except ValueError:
            p.append(f"ruleset {label} {value!r} is unreadable")
            continue
        if when >= first_marker_ts:
            p.append(f"ruleset {label} {value} is not before the first marker")
    return p


def ordering_problems(order: list[dict]) -> list[str]:
    """``order`` = the attempts in k order, each ``{run_id, start, landed, has_rows, first_ts}``: ``start`` is its
    marker's tagger date (the declared start), ``landed`` when its rows first reached ``main`` (None = never). An
    attempt with rows must be on main before the next attempt's marker, and its rows start no earlier than its own
    marker. A marker with no rows (the harness died after the push) is an aborted attempt with nothing to order."""
    p = []
    for a, nxt in zip(order, order[1:]):
        if a["has_rows"] and (a["landed"] is None or a["landed"] >= nxt["start"]):
            p.append(
                f"attempt {a['run_id']}'s rows reached main at {a['landed']}, not before attempt {nxt['run_id']}'s "
                f"marker ({nxt['start']})"
            )
    for a in order:
        if a["has_rows"] and (a["first_ts"] is None or a["first_ts"] < a["start"]):
            p.append(f"attempt {a['run_id']}'s first row ({a['first_ts']}) precedes its own marker ({a['start']})")
    return p


def history_problems(versions: list[bytes], judged: bytes) -> list[str]:
    """The rows file is append-only on ``main``: each first-parent version (oldest first) is a byte prefix of the
    next, and the newest is exactly the bytes judged. A dropped or edited attempt fails here."""
    if not versions:
        return ["the rows file is not on origin/main"]
    p = [f"version {i + 1} of the rows file on main is not a prefix of version {i + 2}" for i, (a, b) in
         enumerate(zip(versions, versions[1:])) if not b.startswith(a)]  # fmt: skip
    if versions[-1] != judged:
        p.append("the rows file judged is not origin/main's")
    return p


def landing_times(history: list[tuple[str, float, bytes]]) -> dict[str, float]:
    """``run_id -> `` the commit time of the first first-parent version of the rows file that holds its rows."""
    landed: dict[str, float] = {}
    for _sha, when, data in history:
        for line in data.decode("utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            rid = (json.loads(line).get("provenance") or {}).get("harness_run_id")
            if rid and rid not in landed:
                landed[rid] = when
    return landed


# ── main ──────────────────────────────────────────────────────────────────────────────────────────────────


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout


def _git_bytes(*args: str) -> bytes | None:
    out = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True)
    return out.stdout if out.returncode == 0 else None


def _is_ancestor(commit: str, of: str) -> bool:
    return subprocess.run(["git", "merge-base", "--is-ancestor", commit, of], cwd=REPO_ROOT).returncode == 0


def _on_first_parent(commit: str, ref: str) -> bool:
    """Whether ``commit`` landed on ``ref`` itself (its first-parent history), not only on a branch merged into it."""
    return commit in _git("rev-list", "--first-parent", ref).split()


def _gh_json(path: str):
    out = subprocess.run(["gh", "api", path], cwd=REPO_ROOT, capture_output=True, text=True)
    if out.returncode != 0:
        raise Refusal(f"gh api {path} failed (an unreadable ruleset refuses, never skips): {out.stderr.strip()[:300]}")
    return json.loads(out.stdout)


def _gh_list(path: str) -> list:
    """One page of up to 100; a full page refuses (a list the verdict cannot see whole is not read)."""
    items = _gh_json(f"{path}{'&' if '?' in path else '?'}per_page=100")
    if not isinstance(items, list) or len(items) >= 100:
        raise Refusal(f"gh api {path}: not a list, or 100+ entries (unpaginated)")
    return items


def _iso(value: str) -> float:
    from datetime import datetime

    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def rows_history(rel_rows: str) -> list[tuple[str, float, bytes]]:
    """The rows file at each first-parent commit of ``origin/main`` that touched it, oldest first (a deletion
    reads as empty bytes)."""
    out = []
    for line in reversed(_git("log", "--first-parent", "--format=%H %cI", "origin/main", "--", rel_rows).splitlines()):
        sha, when = line.split()
        out.append((sha, _iso(when), _git_bytes("show", f"{sha}:{rel_rows}") or b""))
    return out


def landed_on_main(path: str, want_sha256: str | None = None) -> float | None:
    """When ``path`` first reached ``origin/main`` with the bytes it must have (SHA-256 ``want_sha256``, default the
    bytes ``origin/main`` holds now): the first first-parent commit there whose blob matches (GitHub's merge time),
    or None. Dated by the bytes, not the path: an earlier, different file at the path does not count."""
    if want_sha256 is None:
        now = _git_bytes("show", f"origin/main:{path}")
        if now is None:
            return None
        want_sha256 = sha256_bytes(now)
    for line in _git("log", "--first-parent", "--reverse", "--format=%H %cI", "origin/main", "--", path).splitlines():
        sha, when = line.split()
        if sha256_bytes(_git_bytes("show", f"{sha}:{path}") or b"") == want_sha256:
            return _iso(when)
    return None


def check_apparatus(exp: str, rel_rows: str, judged: bytes, attempts: dict[str, list[dict]]) -> tuple[list[dict], dict]:
    """The remote half: markers, the ruleset, the rows file's history on ``origin/main`` and each attempt's code on
    main. Returns the attempts in k order (``[{run_id, k, rows}]``, rows [] for a marker with none) and the
    apparatus record the verdict stamps. Raises :class:`Refusal`."""
    _git("fetch", "--quiet", "origin", "main", f"+{MARKER_NAMESPACE}/{exp}/*:{MARKER_NAMESPACE}/{exp}/*")
    markers = parse_markers(exp, _git("ls-remote", "--tags", "origin", f"{MARKER_NAMESPACE}/{exp}/*"))
    if not markers:
        raise Refusal("no start markers on origin")
    if set(attempts) - set(markers):
        raise Refusal(f"rows name attempts with no start marker: {sorted(set(attempts) - set(markers))}")
    for run_id, m in markers.items():
        fields = _git("for-each-ref", "--format=%(objecttype) %(objectname) %(taggerdate:unix)", m["ref"]).split()
        if len(fields) != 3 or fields[0] != "tag" or fields[1] != m["object"] or not fields[2].isdigit():
            raise Refusal(f"marker {m['ref']} is not the annotated tag origin lists ({fields})")
        if _git("rev-parse", f"{m['ref']}^{{commit}}").strip() != m["peeled"]:
            raise Refusal(f"marker {m['ref']} peels to another commit locally")
        m["tagger_date"] = float(fields[2])
        executed = {(r.get("provenance") or {}).get("executed_git_hash") for r in attempts.get(run_id, [])}
        if executed and executed != {m["peeled"]}:
            raise Refusal(f"attempt {run_id} ran on {sorted(map(str, executed))}, not its marker's commit")
        if not _is_ancestor(m["peeled"], "origin/main"):
            raise Refusal(f"attempt {run_id} ran on {m['peeled']}, which is not on origin/main")
    first_marker = min(m["tagger_date"] for m in markers.values())
    nwo = _gh_json("repos/{owner}/{repo}")["full_name"]
    rulesets = _gh_list(f"repos/{nwo}/rulesets?includes_parents=false")
    details = {rs["id"]: _gh_json(f"repos/{nwo}/rulesets/{rs['id']}") for rs in rulesets}
    history = {rid: _gh_list(f"repos/{nwo}/rulesets/{rid}/history") for rid in details}
    problems = ruleset_problems(rulesets, details, history, first_marker, _iso)
    sup = PROTOCOL[exp].get("supersedes")
    succession = None
    if sup is not None:
        closure_landed = landed_on_main(sup["verdict"], sup["verdict_sha256"])
        prereg_landed = landed_on_main(PROTOCOL[exp]["prereg"])
        with tempfile.TemporaryDirectory() as tmp:
            problems += successor_problems(
                exp,
                _git_bytes("show", f"origin/main:{sup['verdict']}"),
                closure_landed,
                prereg_landed,
                first_marker,
                data_root=materialize("origin/main", predecessors(exp), Path(tmp)),
            )
        succession = {**sup, "closure_landed": closure_landed, "prereg_landed": prereg_landed}
    rows_hist = rows_history(rel_rows)
    problems += history_problems([data for _s, _w, data in rows_hist], judged)
    landed = landing_times(rows_hist)
    ordered = sorted(markers.items(), key=lambda kv: kv[1]["k"])
    order = [
        {
            "run_id": rid,
            "start": m["tagger_date"],
            "landed": landed.get(rid),
            "has_rows": bool(attempts.get(rid)),
            "first_ts": min((r["ts"] for r in attempts.get(rid, [])), default=None),
        }
        for rid, m in ordered
    ]
    problems += ordering_problems(order)
    if problems:
        raise Refusal("; ".join(problems))
    kept = [
        d
        for d in details.values()
        if (d.get("conditions") or {}).get("ref_name", {}).get("include") == [RULESET_INCLUDE]
    ]
    record = {
        "markers": [{**m, "run_id": rid} for rid, m in ordered],
        "ruleset": {
            "id": kept[0].get("id"),
            "created_at": kept[0].get("created_at"),
            "updated_at": kept[0].get("updated_at"),
            "history_versions": [h.get("version_id") for h in history.get(kept[0].get("id"), [])],
        },
        "rows_history": [{"commit": sha, "committed_at": when} for sha, when, _d in rows_hist],
        "landed": landed,
        "succession": succession,
    }
    return [{"run_id": rid, "k": m["k"], "rows": attempts.get(rid, [])} for rid, m in ordered], record


def judge(exp: str, attempts_in_order: list[dict], data_root: Path) -> dict:
    """Re-hash every copied file, decide each attempt's completeness from its committed bytes, and gate the
    first complete one. Pure over the bytes (no git, no network). Raises :class:`Refusal` on an integrity break."""
    listing = []
    deciding = None
    for attempt in attempts_in_order:
        rows = attempt["rows"]
        if deciding is not None:
            # The harness refuses a new attempt once the file holds a complete one; an attempt after it would let
            # the result be chosen.
            raise Refusal(f"attempt {attempt['run_id']} follows the complete attempt {deciding[0]['run_id']}")
        if [r.get("phase_index") for r in rows] != list(range(len(rows))):
            raise Refusal(f"attempt {attempt['run_id']}'s phase rows are not 0..n-1 exactly")
        entry = {"run_id": attempt["run_id"], "k": attempt.get("k"), "complete": False, "problems": []}
        listing.append(entry)
        if not rows:
            entry["problems"].append("a start marker with no rows: an aborted attempt")
            continue
        phases = []
        phase1_session = None
        phase1_stores: set[str] = set()
        for row in rows:
            index = row["phase_index"]
            session = row.get("session_id")
            if row.get("status") != "ok" or not session:
                entry["problems"].append(f"phase {index} row ended {row.get('status')!r}: {row.get('reason')}")
                break
            sdir = data_root / session
            files = row.get("files") or {}
            for name, digest in files.items():
                try:
                    data = read_copied(sdir, name)
                except FileNotFoundError as exc:
                    raise Refusal(f"{sdir}/{name}: named by its row but not committed") from exc
                if sha256_bytes(data) != digest:
                    raise Refusal(f"{sdir}/{name}: SHA-256 differs from its row")
            missing = [n for n in required_files(exp) if n not in files]
            if missing:
                entry["problems"].append(f"phase {index}: {missing} not copied")
                break
            report = json.loads(read_copied(sdir, "report.json"))
            if index == 0:
                phase1_session = session
                phase1_stores = {s for s in RESUME_STORES if f"aut_{s}.json" in files}
            entry["problems"] += complete_problems(exp, index, row, report, phase1_session, phase1_stores)
            log_bytes = read_copied(sdir, RUN_LOG)
            store = json.loads(read_copied(sdir, "aut_hippocampus.json")) if "aut_hippocampus.json" in files else {}
            phases.append(
                {"report": report, "store": store, "lines": log_lines(log_bytes, strict=True), "log_bytes": log_bytes}
            )
        if len(phases) != len(PROTOCOL[exp]["phases"]) and not entry["problems"]:
            entry["problems"].append(f"{len(phases)} of {len(PROTOCOL[exp]['phases'])} phases recorded")
        if experiment_of(exp) == "63" and len(phases) == len(PROTOCOL[exp]["phases"]):
            # C5 and R3a's observability, from the committed traces and stores: an instrument fault is not data.
            entry["problems"] += exp63_c5_problems(phases, goal=PROTOCOL[exp]["phases"][1][1])
        entry["complete"] = not entry["problems"]
        if entry["complete"]:
            deciding = (entry, phases)  # the first complete attempt decides
    out: dict = {"attempts": listing, "experiment": exp}  # the CAMPAIGN key ("10c2"), as the gate and successor read it
    if predecessors(exp):
        # #1059 D2: emitted so the gate's re-judge with this (bound) judge reproduces the bar; a successor verdict
        # whose bound judge does not emit it supports nothing. Never compared with the record (o19_difference).
        out["leaked_gates"] = chain_leaked_gate_problems(exp, data_root.parent)
    if deciding is None:
        out["verdict"] = "ABORT"
        out["deciding_attempt"] = None
        return out
    entry, phases = deciding
    out["deciding_attempt"] = entry["run_id"]
    if experiment_of(exp) == "63":
        gates = exp63_gates(phases, goal=PROTOCOL[exp]["phases"][1][1])
    else:
        gates = exp10_gates(phases) if experiment_of(exp) == "10" else exp09_gates(phases[0], phases[0]["log_bytes"])
    out["gates"] = gates
    out["verdict"] = gates["verdict"]
    return out


# The files that define what an attempt ran and how it is judged: identical at every executed commit, on main and
# at the verdict's own commit, or the verdict refuses (a post-data change to the gate is a new experiment).
BOUND_FILES = ("scripts/o19_verdict.py", "scripts/o19_rerun.py")

EXIT = {"PASS": 0, "FAIL": 1, "PARTIAL": 4, "NOT SHOWN": 4, "ABORT": 4}


def bound_blobs(exp: str, commits: list[str]) -> dict[str, dict[str, str | None]]:
    """``path -> {commit -> blob sha}`` for the prereg and the two O19 scripts."""
    out: dict[str, dict[str, str | None]] = {}
    for path in (PROTOCOL[exp]["prereg"], *BOUND_FILES):
        out[path] = {}
        for commit in commits:
            blob = _git_bytes("rev-parse", f"{commit}:{path}")
            out[path][commit] = blob.decode().strip() if blob else None
    return out


def check_bound(exp: str, rows: list[dict]) -> dict:
    """The verdict runs on ``main``, and the prereg and both O19 scripts are byte-identical at every executed commit,
    on ``origin/main`` and at the verdict's own commit. Returns what the verdict stamps; raises :class:`Refusal`."""
    head = _git("rev-parse", "HEAD").strip()
    if not _on_first_parent(head, "origin/main"):  # as the evidence gate requires (#1050)
        raise Refusal(f"the verdict runs at {head}, which is not on origin/main's first-parent history")
    executed = sorted({(r.get("provenance") or {}).get("executed_git_hash") for r in rows} - {None})
    blobs = bound_blobs(exp, [*executed, "origin/main", head])
    for path, by_commit in blobs.items():
        if None in by_commit.values() or len(set(by_commit.values())) != 1:
            raise Refusal(f"{path} differs between the executed commits, origin/main and the verdict: {by_commit}")
    return {"bound_files": {path: next(iter(b.values())) for path, b in blobs.items()}, "verdict_commit": head}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", required=True, choices=sorted(PROTOCOL))
    ap.add_argument("--data", required=True, help="the rows file (docs/experiments/data/rerun_exp<NN>_o19/rows.jsonl)")
    ap.add_argument("--json", required=True, help="where the verdict is written")
    ap.add_argument("--write-experiment-results", action="store_true")
    ap.add_argument(
        "--offline",
        action="store_true",
        help="skip the markers/ruleset/history/blob checks; mock rows only, never written as evidence",
    )
    args = ap.parse_args(argv)
    if problems := protocol_problems():
        print(f"REFUSED: the campaign table is unsound: {problems}", file=sys.stderr)
        return 2
    if args.offline and args.write_experiment_results:
        print("REFUSED: an --offline verdict is a smoke and never replaces committed evidence", file=sys.stderr)
        return 2

    from _provenance import (  # noqa: PLC0415
        DirtyTreeError,
        ProvenanceError,
        any_not_stamped_real,
        evidence_out_path,
        in_process_code_provenance,
        stamp_verdict,
    )

    import maxim  # noqa: PLC0415

    exp = args.exp
    data = Path(args.data).resolve()
    try:
        out_path = evidence_out_path(REPO_ROOT, args.json, write_experiment_results=args.write_experiment_results)
        provenance = in_process_code_provenance(REPO_ROOT, maxim.__file__, out_path=out_path)
    except (ProvenanceError, DirtyTreeError) as exc:
        print(f"PROVENANCE: {exc}", file=sys.stderr)
        return 3
    extra: dict = {"apparatus_checked": not args.offline}
    try:
        data_bytes = data.read_bytes()  # read once: the verdict and its hash see the same bytes
        rows = [json.loads(ln) for ln in data_bytes.decode().splitlines() if ln.strip()]
        mock = any_not_stamped_real(rows)
        attempts = attempts_from_rows(rows)
        if args.offline:
            if not mock:
                raise Refusal("--offline skips the apparatus checks and is for mock rows only")
            ordered = [
                {"run_id": rid, "k": i + 1, "rows": rs}
                for i, (rid, rs) in enumerate(sorted(attempts.items(), key=lambda kv: min(r["ts"] for r in kv[1])))
            ]
        else:
            try:
                rel_rows = data.relative_to(REPO_ROOT).as_posix()
            except ValueError as exc:
                raise Refusal(f"rows file {data} is outside the repo") from exc
            if rel_rows != rows_path(exp):
                raise Refusal(f"rows file {rel_rows} is not the prereg's {rows_path(exp)}")
            ordered, extra["apparatus"] = check_apparatus(exp, rel_rows, data_bytes, attempts)
            extra.update(check_bound(exp, rows))
        report = judge(exp, ordered, data.parent)
    except Refusal as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    except (subprocess.CalledProcessError, OSError, ValueError, KeyError, TypeError) as exc:
        # Anything the verdict could not read is a refusal (no status change), never a FAIL (exit 1).
        print(f"REFUSED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    report.update(extra)
    report["verdict_source_sha256"] = sha256_bytes(Path(__file__).read_bytes())
    stamp_verdict(
        report,
        repo_root=REPO_ROOT,
        kind=PROTOCOL[exp]["kind"],
        data=data,
        data_bytes=data_bytes,
        scope={"all_rows": True},
        mock=mock,
    )
    report["provenance"] = provenance
    from maxim.utils.atomic_io import atomic_write_json  # noqa: PLC0415
    from maxim.utils.format_version import with_format_version  # noqa: PLC0415

    out_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out_path, with_format_version(report), indent=2)
    print(json.dumps({k: report[k] for k in ("verdict", "deciding_attempt", "mock")}, indent=2))
    return EXIT[report["verdict"]]


if __name__ == "__main__":
    raise SystemExit(main())
