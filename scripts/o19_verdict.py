#!/usr/bin/env python3
"""The O19 re-run verdict: Exp 10 (T1-1), Exp 09 (T3-9), Exp 63 and Exp 63b (T1-16), computed from committed bytes
only.

Pre-registrations (read them first; this module implements them and nothing else):
  docs/experiments/protocols/exp10_rerun_2026-09-30_preregistration.md   (Exp 10 campaign 1, key "10": CLOSED)
  docs/experiments/protocols/exp10_rerun_2026-10-02_preregistration.md   (Exp 10 campaign 2, key "10c2")
  docs/experiments/protocols/exp09_rerun_2026-09-30_preregistration.md   (Exp 09, key "09")
  docs/experiments/exp63_carried_recall_prereg.md                        (Exp 63, key "63": a new experiment, not a
                                                                          re-run; its gates are ``exp63_*`` below)
  docs/experiments/exp63b_carried_recall_prereg.md                       (Exp 63b, key "63b": T1-16's re-run on the
                                                                          content-token ranker and cue path, a new
                                                                          experiment id; its gates are ``exp63b_*``)

A PROTOCOL key is a CAMPAIGN: its own prereg, data directory and marker namespace (``refs/tags/o19/<key>/``). Its
``experiment`` ("10"/"09"/"63"/"63b"/"63c") decides the phases' gates; every gate choice goes through :func:`experiment_of`. A
successor campaign (``supersedes``) opens only after its predecessor's stamped ABORT verdict, pinned by SHA-256, and
at most ``MAX_CAMPAIGNS`` campaigns exist per experiment (owner decisions 2026-10-02; :func:`protocol_problems`,
:func:`successor_problems`).

A FAILED gate seen in an aborted attempt's committed phases bars later support: a predecessor campaign's bars its
successor (#1059, :func:`chain_leaked_gate_problems`), and, for campaign keys outside :data:`PRE_1079_KEYS`, an earlier
attempt's (or an earlier start marker with no rows) bars the deciding attempt of the same campaign (#1079,
:func:`within_campaign_leaks`): the verdict reports it as ``within_campaign_leaks`` and the evidence gate refuses.

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

Exits: 0 PASS; 1 FAIL; 4 PARTIAL (Exp 09), NOT SHOWN (Exp 63 and 63b: complete and conforming, but no decisive turn) or
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
from urllib.parse import urlsplit

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
# A campaign's data directory is ONE path component under docs/experiments/data/: the evidence gate places each
# campaign's directory under its last component and the leaked-gate bar reads ``data_root/<scope>`` (#1081 item 3).
_SCOPE = re.compile(r"[0-9a-z_]+")
RESERVED_KEYS = frozenset({"preflight"})  # o19_rerun.py's dry-run marker namespace
# #1079 (owner decision 2026-10-08): the campaigns whose verdicts predate the within-campaign leaked-gate bar. Each has
# a verdict on main and a frozen data directory (no attempt can ever be added under them), so their judge() output
# stays exactly what it was before #1079: no ``within_campaign_leaks`` field. Not a trust point: the evidence gate
# holds no key list and refuses a deciding attempt with k != 1 whose bound judge omits the field, so adding a NEW key
# here fails closed; removing one of the four changes its output and fails the judge-edit re-judge (half B).
PRE_1079_KEYS = frozenset({"10", "10c2", "09", "63"})

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
    # Exp 63b (T1-16's re-run, #1476): "63" closed on PASS and only an ABORT may be succeeded, so the re-run on the
    # content-token ranker and the cue path is a NEW experiment id with Exp 63's argv (its prereg's protocol section).
    "63b": {
        "experiment": "63b",
        "scope": "63b_carried_recall",
        "kind": "exp63b_verdict",
        "prereg": "docs/experiments/exp63b_carried_recall_prereg.md",
        "phases": [
            ("baseline", EXP10_GOAL_DUNGEON, 8, False, ["--sim-run-full-turns"], {}),
            ("gate", EXP10_GOAL_DUNGEON, 8, True, ["--sim-run-full-turns"], {}),
        ],
    },
    # Exp 63c (T1-16 under a scripted narrator, #1476 after 63b's NOT SHOWN): a NEW experiment with 63b's goal and
    # model, each phase's lines from --sim-script (#1628). Its base prereg (63b's, whose gates it inherits) is bound by
    # hash with its own (``base_preregs``: bound_blobs, and the evidence gate's bound-file set).
    "63c": {
        "experiment": "63c",
        "scope": "63c_scripted_recall",
        "kind": "exp63c_verdict",
        "prereg": "docs/experiments/exp63c_scripted_recall_prereg.md",
        "base_preregs": ["docs/experiments/exp63b_carried_recall_prereg.md"],
        "phases": [
            (
                "baseline",
                EXP10_GOAL_DUNGEON,
                8,
                False,
                ["--sim-script", "docs/experiments/protocols/exp63c/phase1_script.yaml"],
                {},
            ),
            (
                "gate",
                EXP10_GOAL_DUNGEON,
                8,
                True,
                ["--sim-script", "docs/experiments/protocols/exp63c/phase2_script.yaml"],
                {},
            ),
        ],  # fmt: skip
        "scenario_content_sha256": [
            "dc2ec6db636333d5ed940bb0e258384eb2edfbae9a0e96c0a3881ff63e94529c",
            "eabc0b96d589bd1247cacef35713cb764a1f44a41256455197578acc34338775",
        ],
        "script_lines": [
            [
                "You lie in a cold damp alcove. The guard is asleep in the dungeon and you wish to escape.",
                "A brass ladle hangs from a copper hook beside the guard's chair.",
                "The guard hid a small knife beneath the straw mattress.",
                "The cell hatch by the guard has a cracked padlock whose chain squeaks.",
                "Water drips from a crevice in the north bricks into a drain grate in the dungeon.",
                "The guard is still sleeping in the dungeon.",
                "What comes to mind?",
                "Sum up your intention in a sentence.",
            ],
            [
                "You are back in the dungeon and the guard is still sleeping.",
                "Is a silver bell ringing in the guard's chapel?",
                "Where does the brass ladle hang by the guard?",
                "What is beneath the guard's straw mattress?",
                "Does the padlock on the cell hatch by the guard squeak?",
                "Is a purple banner draped in the dungeon?",
                "Where does water from the crevice drip in the dungeon?",
                "The guard stirs. What do you do now?",
            ],
        ],
        # OD-c4: the one GGUF S2' admits, by SHA-256 (63b's committed ``identity``); the harness re-reads it at preflight.
        "gguf_sha256": "3e0039fd0273fcbebb49228943b17831aadd55cbcbf56f0af00499be2040ccf9",
    },
}
EXPERIMENTS = frozenset({"10", "09", "63", "63b", "63c"})


def experiment_of(key: str) -> str:
    """The experiment a campaign key runs ("10", "09", "63", "63b" or "63c"): every gate choice goes through this,
    never the key."""
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
    """The structural campaign rules ([] = sound): plain keys, one-component scopes, one experiment per chain, a
    successor names an existing predecessor of its own experiment and kind, each campaign is succeeded at most once,
    at most one open campaign per experiment, and at most ``MAX_CAMPAIGNS`` campaigns per experiment."""
    protocol = PROTOCOL if protocol is None else protocol
    problems = []
    for key, p in protocol.items():
        if not _KEY.fullmatch(key) or key in RESERVED_KEYS:
            problems.append(f"campaign key {key!r} is not [0-9a-z]+ or is reserved")
        if p.get("experiment") not in EXPERIMENTS:
            problems.append(f"campaign {key}: experiment {p.get('experiment')!r} is not 10, 09, 63, 63b or 63c")
        base = p.get("base_preregs", [])
        others = {q.get("prereg") for k, q in protocol.items() if k != key}
        if not isinstance(base, list) or not all(
            isinstance(b, str) and b in others and b != p.get("prereg") for b in base
        ):
            problems.append(f"campaign {key}: base_preregs {base!r} is not a list of other campaigns' preregs")
        if not isinstance(p.get("scope"), str) or not _SCOPE.fullmatch(p["scope"]):
            problems.append(f"campaign {key}: scope {p.get('scope')!r} is not one path component [0-9a-z_]+")
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
    for key, p in protocol.items():  # #1079: both leaked-gate bars read every prefix length of every campaign
        phases = p.get("phases") if isinstance(p.get("phases"), list) else []
        if set(LEAK_GATES.get(p.get("experiment"), {})) != set(range(1, len(phases) + 1)):
            problems.append(f"campaign {key}: LEAK_GATES does not cover each of its {len(phases)} phase counts")
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
    """The copied files a phase must carry: its report and run log, and Exp 10's, 63's, 63b's and 63c's hippocampus
    store (their C2)."""
    if experiment_of(exp) in ("10", "63", "63b", "63c"):
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


def _endpoint_origin(url) -> str:
    """``scheme://host[:port]`` of an endpoint, or ``""``: the form a sim report records since #997. A stdlib copy
    of ``maxim.utils.http::url_origin`` (this script imports no ``maxim``); a parity test pins the two. C4
    reduces BOTH sides with it, so a record from before #997 (which kept ``/v1``) compares as it did."""
    try:
        parts = urlsplit(str(url or "").strip())
        port = parts.port
    except ValueError:
        return ""
    host = parts.hostname
    if parts.scheme.lower() not in ("http", "https") or not host or "%" in host or "\\" in host:
        return ""
    host = f"[{host}]" if ":" in host else host
    return f"{parts.scheme.lower()}://{host}" + (f":{port}" if port is not None else "")


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
    # C1: ran to the cap (Exp 63c: the script ran out, `completed`, at the pinned content).
    if experiment_of(exp) == "63c":
        p += exp63c_c1_problems(index, report, name)
    else:
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
    endpoint = _endpoint_origin(report.get("language_endpoint"))
    if not endpoint or _endpoint_origin(served.get("url")) != endpoint:
        p.append(
            f"C4 {name}: served model read at {_endpoint_origin(served.get('url'))!r}, but the sim used {endpoint!r}"
        )
    # C4' (Exp 63, 63b and 63c): the retention model the harness read at preflight in the attempt's fresh data home.
    if experiment_of(exp) in ("63", "63b", "63c") and row.get("memory_strategy") != EXP63_MEMORY_STRATEGY:
        p.append(f"C4' {name}: memory_strategy {row.get('memory_strategy')!r}, not {EXP63_MEMORY_STRATEGY!r}")
    # C4'' (Exp 63b and 63c): the evidence gate's B0 checks over the phase's sim record (b0_problems).
    if experiment_of(exp) in ("63b", "63c"):
        sims = row.get("sims")
        if not isinstance(sims, list) or len(sims) != 1 or not isinstance(sims[0], dict):
            p.append(f"C4'' {name}: the row does not carry exactly one sim record")
        else:
            p += [f"C4'' {text}" for text in b0_problems(sims[0], name)]
    if experiment_of(exp) == "63c":  # S2' (OD-c4/OD-c5): everything served in the phase was the pinned GGUF
        p += exp63c_serving_problems(row, name)
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


# ── Exp 63b (T1-16's re-run as a new experiment: docs/experiments/exp63b_carried_recall_prereg.md) ───────────────
# Additive: nothing Exp 63 reads changes, so "63"'s committed verdict still reproduces (gate half B). Its own frozen
# port of the content-token ranker (G2), ONE simulator for R3' and R3d (G1), R2', and the widened C5.

EXP63B_PATHS = ("graph", "cue", "goal", "substring")
EXP63B_CUE_LIMIT = 5  # recall(query=text, limit=5); the cue path stops once EXP63_TOP are surfaced
EXP63B_TIE_BOUND = 10_000  # G1: past this many tie orderings on a turn the attempt is incomplete (C5), never a guess
EXP63B_STORE_FILE = "aut_hippocampus.json"
EXP63B_DIGEST_EVENT = "hippocampus_store_digest"  # I1's line (memory/hippocampus_persistence.py::_log_store_digest)
_TEXT_LEN = re.compile(r"\btext_len=(\d+)\b")

# G2, the FROZEN port: stdlib copies of ``maxim.memory.text`` (``key_tokens`` with ``normalize_tokens``,
# ``_lemmatize``, ``settle_key``, ``text_values``, ``content_tokens``, ``same_words``), of
# ``maxim.utils.content_safety``'s frame removal, of ``memory.types``' ``text_fields`` / ``content_tokens`` /
# ``input_text`` / ``copy_key`` for both record kinds, and of ``hippocampus_retrieval``'s ``_rank_by_relevance`` and
# ``distinct_copies``, over the saved store's JSON dicts. Exp 63b ran (e841abca) before #1620 changed the shipped
# content keys, so tests/unit/test_o19_rerun.py pins THIS keying against the vendored executed copy
# (tests/fixtures/exp63b/executed_text.py), never against the shipped code; the rest of the port (frames, record
# fields, ranker, collapse) is unchanged since e841abca and stays pinned against the shipped code. Exp 63c's keying is
# its own port below (``c_key_tokens``, pinned against the shipped ``memory/text.py``). Every function that keys text
# takes the keying as a REQUIRED ``key=`` (``keying_of``), so a site that forgets it is a TypeError, never the other
# experiment's keys. A saved store whose records the loader would rewrite (a 1.0 ``ToolOutput(`` result) is C5(d).

_P_STOP_WORDS = frozenset(
    {
        "a",
        "about",
        "after",
        "against",
        "all",
        "also",
        "although",
        "am",
        "an",
        "and",
        "another",
        "any",
        "are",
        "as",
        "at",
        "bad",
        "be",
        "because",
        "been",
        "before",
        "being",
        "best",
        "both",
        "but",
        "by",
        "can",
        "could",
        "did",
        "do",
        "does",
        "doing",
        "during",
        "even",
        "few",
        "for",
        "from",
        "good",
        "had",
        "has",
        "have",
        "having",
        "he",
        "her",
        "here",
        "hers",
        "him",
        "his",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "just",
        "less",
        "many",
        "may",
        "me",
        "might",
        "mine",
        "more",
        "most",
        "much",
        "must",
        "my",
        "myself",
        "no",
        "not",
        "now",
        "of",
        "on",
        "one",
        "only",
        "onto",
        "or",
        "other",
        "our",
        "ours",
        "over",
        "shall",
        "she",
        "should",
        "so",
        "some",
        "still",
        "such",
        "than",
        "that",
        "the",
        "their",
        "theirs",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "though",
        "through",
        "to",
        "too",
        "two",
        "under",
        "unless",
        "us",
        "very",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "while",
        "who",
        "why",
        "will",
        "with",
        "would",
        "yes",
        "you",
        "your",
        "yours",
    }
)
_P_METADATA_KEYS = frozenset(
    {
        "_skill_name",
        "bio_response",
        "detections_count",
        "entity_source",
        "has_maxim",
        "modality",
        "novelty",
        "perceived_intensity",
        "salience",
        "source",
        "spatial_source",
    }
)
_P_PUNCT_STRIP = ".,;:!?\"'`()[]{}<>—–-"

_P_FRAMED_REGION = re.compile(r"<<TOOL_OUTPUT id=([0-9a-f]+) .*?<</TOOL_OUTPUT id=\1>>", re.DOTALL)
_P_FRAME_MARKER = re.compile(r"<<(/?)TOOL_OUTPUT id=([0-9a-f]+)(?: [^\n]*?)?>>")


def _p_has_vowel(word: str) -> bool:
    return any(c in "aeiou" for c in word)


def _p_lemmatize(word: str) -> str:
    if word.endswith("ing") and len(word) > 4:
        stem = word[:-3]
        doubled = len(stem) >= 2 and stem[-1] == stem[-2]
        if doubled:
            stem = stem[:-1]
            if len(stem) >= 3 and _p_has_vowel(stem):
                return stem
        else:
            if len(stem) >= 4 and _p_has_vowel(stem):
                return stem
        return word
    if word.endswith("ed") and len(word) > 3:
        if word.endswith("eed"):
            return word
        if word.endswith("ied") and len(word) > 4:
            return word[:-3] + "y"
        stem = word[:-2]
        if len(stem) >= 2 and _p_has_vowel(stem):
            return stem
        stem_d = word[:-1]
        if stem_d.endswith("e") and len(stem_d) >= 3:
            return stem_d
        return word
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        return word[:-1]
    return word


def _p_normalize_tokens(text: str) -> list[str]:
    result: list[str] = []
    for raw in re.split(r"[\s_'’]+", text.lower()):
        w = raw.strip(_P_PUNCT_STRIP)
        if not w or w in _P_STOP_WORDS or len(w) < 3:
            continue
        result.append(_p_lemmatize(w))
    return result


def _p_settle_key(token: str) -> str:
    for _ in range(4):
        again = _p_normalize_tokens(token)
        if not again or again == [token]:
            return token
        token = again[0]
    return token


def p_key_tokens(text: str) -> list[str]:
    """``maxim.memory.text.key_tokens`` as Exp 63b executed it (e841abca, before #1620)."""
    return [_p_settle_key(t) for t in _p_normalize_tokens(text)]


# Exp 63c's keying: ``maxim.memory.text`` after #1620 (PR #1624, d40eb4c0). ``normalize_tokens`` and ``_lemmatize`` are
# frozen (unchanged by #1620, so the 63b copies above serve); ``settle_key`` converges a word's regular inflections
# (``_converge``, guarded by ``_kept``) under the two length caps. tests/unit/test_o19_rerun.py pins it against the
# shipped module on fuzz words, and the harness refuses before the marker when it disagrees with the sim interpreter's
# ``key_tokens`` (``o19_rerun.check_keying``).
_C_CONVERGE_MAX_LEN = 40
_C_SETTLE_MAX_LEN = 64


def _c_kept(token: str) -> bool:
    again = _p_normalize_tokens(token)
    return bool(again) and again[0] not in _P_STOP_WORDS


def _c_converge(token: str) -> str:
    if len(token) > _C_CONVERGE_MAX_LEN or not (token.isascii() and token.isalpha()):
        return token
    if token.endswith("ing") and len(token) > 4:
        stem = token[:-3]
        if len(stem) >= 3 and _p_has_vowel(stem) and _c_kept(stem):
            token = stem
    if len(token) >= 4 and token[-1] == token[-2] and token[-1] not in "aeiouys" and _c_kept(token[:-1]):
        token = token[:-1]
    if len(token) >= 4 and token[-1] == "e" and token[-2] not in "aeioy" and _c_kept(token[:-1]):
        token = token[:-1]
    return token


def _c_settle_key(token: str) -> str:
    if len(token) > _C_SETTLE_MAX_LEN:
        for _ in range(4):
            again = _p_normalize_tokens(token)
            if not again or again == [token]:
                return token
            token = again[0]
        return token
    while True:
        while True:
            again = _p_normalize_tokens(token)
            if not again or again == [token]:
                break
            token = again[0]
        converged = _c_converge(token)
        if converged == token:
            return token
        token = converged


def c_key_tokens(text: str) -> list[str]:
    """``maxim.memory.text.key_tokens`` after #1620 (Exp 63c's keying)."""
    return [_c_settle_key(t) for t in _p_normalize_tokens(text)]


# The keying each experiment's recall was keyed with: the one place a judge picks it (an experiment without a port of
# its own raises, never borrows one).
KEYINGS = {"63b": p_key_tokens, "63c": c_key_tokens}


def keying_of(exp: str):
    """The content-key function Exp ``exp``'s port judges with (``KEYINGS``); :class:`Refusal` for a key that is no
    campaign or an experiment with no port of its own (never another's)."""
    if exp not in PROTOCOL or experiment_of(exp) not in KEYINGS:
        raise Refusal(f"campaign {exp!r} has no content-key port")
    return KEYINGS[experiment_of(exp)]


def _p_unbalanced_markers(text: str) -> tuple[list, list]:
    open_by_id: dict = {}
    orphan_openers: list = []
    orphan_closers: list = []
    for m in _P_FRAME_MARKER.finditer(text):
        closing, frame_id = m.group(1) == "/", m.group(2)
        if not closing:
            if frame_id in open_by_id:
                orphan_openers.append(open_by_id[frame_id])
            open_by_id[frame_id] = m
        elif open_by_id.pop(frame_id, None) is None:
            orphan_closers.append(m)
    return sorted([*orphan_openers, *open_by_id.values()], key=lambda m: m.start()), orphan_closers


def _p_partial_marker_start(text: str) -> int | None:
    at = text.rfind("<<")
    if at < 0 or ">>" in text[at:]:
        return None
    fragment = text[at:]
    for head in ("<<TOOL_OUTPUT", "<</TOOL_OUTPUT"):
        if head.startswith(fragment) or fragment.startswith(head):
            return at
    return None


def _p_drop_unbalanced_frames(text: str) -> str:
    """``content_safety.drop_unbalanced_frames``' text (its count is not read here)."""
    tail = _p_partial_marker_start(text)
    if tail is not None:
        text = text[:tail]
    openers, closers = _p_unbalanced_markers(text)
    if not openers and not closers:
        return text.rstrip("\n") if tail is not None else text
    start = closers[-1].end() if closers else 0
    end = openers[0].start() if openers else len(text)
    kept = text[start:end] if start < end else ""
    return kept.strip("\n")


def _p_text_values(value, *, depth: int = 3):
    if isinstance(value, str):
        if "TOOL_OUTPUT" in value:
            value = _p_drop_unbalanced_frames(_P_FRAMED_REGION.sub("", value))
        if value:
            yield value
    elif depth > 0 and isinstance(value, dict):
        for key, inner in value.items():
            if key not in _P_METADATA_KEYS:
                yield from _p_text_values(inner, depth=depth - 1)
    elif depth > 0 and isinstance(value, (list, tuple)):
        for inner in value:
            yield from _p_text_values(inner, depth=depth - 1)


def p_content_tokens(*values, key) -> set[str]:
    """``maxim.memory.text.content_tokens`` under the keying ``key`` (``keying_of``)."""
    tokens: set[str] = set()
    for value in values:
        for text in _p_text_values(value):
            tokens.update(key(text))
    return tokens


def p_same_words(text: str) -> str:
    """``maxim.memory.text.same_words``."""
    import unicodedata  # noqa: PLC0415

    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _p_parts(rec: dict) -> tuple[dict, dict, dict, dict, dict]:
    return tuple(_dict(rec.get(k)) for k in ("perception", "context", "decision", "action", "outcome"))  # type: ignore[return-value]


def p_record_tokens(rec: dict, *, key) -> frozenset[str]:
    """A saved record's ``content_tokens()``: its ``text_fields`` and labels (an episode), or what compression kept
    (a compressed record), read with the defaults ``from_dict`` reads them with."""
    if rec.get("_compressed", False):
        return frozenset(
            p_content_tokens(
                rec.get("heard") or "", rec.get("goal"), rec.get("active_goal"), rec.get("tool_name", ""), key=key
            )
        )
    p, c, d, a, o = _p_parts(rec)
    return frozenset(
        p_content_tokens(
            p.get("cli_input"),
            p.get("transcript"),
            p.get("observations", {}),
            p.get("decision_rationale", ""),
            d.get("reasoning"),
            o.get("result"),
            o.get("error"),
            c.get("active_goal"),
            a.get("tool_name", ""),
            p.get("detected_objects", []),
            p.get("detected_people", []),
            key=key,
        )
    )


def p_input_text(rec: dict) -> str:
    """A saved record's ``input_text``."""
    if rec.get("_compressed", False):
        return rec.get("heard") or ""
    p = _p_parts(rec)[0]
    for value in (p.get("cli_input"), p.get("transcript")):
        if isinstance(value, str) and value.strip():
            return value
    observations = p.get("observations", {})
    if not isinstance(observations, dict):
        return ""
    for key in ("cli_input", "transcript", "text"):
        value = observations.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def p_copy_key(rec: dict) -> tuple | None:
    """A saved record's ``copy_key()``: the same words, tool and outcome (a compressed record keeps no result)."""
    words = p_same_words(p_input_text(rec))
    if not words:
        return None
    if rec.get("_compressed", False):
        return (words, repr((rec.get("tool_name", "") or "", rec.get("success"), {}, None)))
    _p, _c, _d, a, o = _p_parts(rec)
    return (words, repr((a.get("tool_name", "") or "", o.get("success"), o.get("result"), o.get("error"))))


def _p_rows_sha256(rows) -> str:
    return sha256_bytes(json.dumps(rows, separators=(",", ":")).encode("utf-8"))


def p_store_digest(records: dict[str, dict], *, key) -> tuple[str, dict[str, str]]:
    """I1's digest of a saved store (``hippocampus_persistence.store_digest``) and each record's 16-hex row hash."""
    rows = sorted([rid, sorted(p_record_tokens(rec, key=key)), repr(p_copy_key(rec))] for rid, rec in records.items())
    return _p_rows_sha256(rows), {row[0]: _p_rows_sha256(row)[:16] for row in rows}


def _copy_class(rec: dict, rid: str) -> tuple:
    """``distinct_copies``' identity: a record's copy key, or its own id when it heard nothing (always kept)."""
    key = p_copy_key(rec)
    return ("id", rid) if key is None else ("copy", key)


def p_ranked_groups(records: dict[str, dict], ids: list[str], query: str, *, key) -> list[list[str]]:
    """``_rank_by_relevance``'s order over ``ids`` as TIE GROUPS (exact ``(score, timestamp)`` equality; a candidate
    set's order is not reproducible, so within a group any order is the ranker's). A zero score is not ranked (D1);
    a query with no content key ranks nothing."""
    query_tokens = set(key(query))
    if not query_tokens:
        return []
    keyed = []
    for rid in ids:
        overlap = query_tokens & p_record_tokens(records[rid], key=key)
        if overlap:
            keyed.append(((len(overlap) / len(query_tokens), records[rid]["timestamp"]), rid))
    keyed.sort(key=lambda kv: kv[0], reverse=True)
    groups: list[list[str]] = []
    last = None
    for rank, rid in keyed:
        if groups and rank == last:
            groups[-1].append(rid)
        else:
            groups.append([rid])
        last = rank
    return groups


def _groups_needed(groups: list[list[str]], classes: dict[str, tuple], limit: int, skip=frozenset()) -> int:
    """How many leading groups decide the collapsed top ``limit`` (ids in ``skip`` removed): the first of each copy
    class survives whatever the order, so once the classes seen reach ``limit`` no later group can enter."""
    seen: set = set()
    for n, group in enumerate(groups, 1):
        seen.update(classes[m] for m in group if m not in skip)
        if len(seen) >= limit:
            return n
    return len(groups)


def _ordering_count(groups: list[list[str]], upto: int) -> int:
    import math  # noqa: PLC0415

    count = 1
    for group in groups[:upto]:
        count *= math.factorial(len(group))
    return count


def _orderings(groups: list[list[str]], upto: int):
    """Every linearisation of the first ``upto`` tie groups (``_ordering_count`` of them)."""
    import itertools  # noqa: PLC0415

    for combo in itertools.product(*(itertools.permutations(g) for g in groups[:upto])):
        yield [m for part in combo for m in part]


def p_distinct(order: list[str], classes: dict[str, tuple], limit: int) -> list[str]:
    """``distinct_copies(order)[:limit]``."""
    kept: list[str] = []
    seen: set = set()
    for m in order:
        if classes[m] not in seen:
            kept.append(m)
            seen.add(classes[m])
            if len(kept) == limit:
                break
    return kept


def p_simulate(graph: list[str], cue: list[str], goal: list[str] | None) -> tuple[list[str], list[str]]:
    """``_query_hippocampus_traced``'s merge, line for line: the graph ids, then the cue path's (collapsed top 5,
    stopping at 3 surfaced), then, only under 3, the goal path's collapsed top 3 (no refill); ids dedup across paths.
    ``goal`` None = no goal (the path does not run). Returns ``(ids, paths)`` as the trace logs them."""
    ids: list[str] = []
    paths: list[str] = []
    seen: set[str] = set()

    def add(m: str, path: str) -> None:
        if m not in seen:
            seen.add(m)
            ids.append(m)
            paths.append(path)

    for m in graph:
        add(m, "graph")
    if len(ids) < EXP63_TOP:
        for m in cue:
            if len(ids) >= EXP63_TOP:
                break
            add(m, "cue")
    if len(ids) < EXP63_TOP and goal is not None:
        for m in goal[:EXP63_TOP]:
            add(m, "goal")
    return ids[:EXP63_TOP], paths[:EXP63_TOP]


def _nongoal_score(rec: dict, nongoal: set[str], *, key) -> int:
    return len(nongoal & p_record_tokens(rec, key=key))


def exp63b_turn(
    tr: dict, records: dict[str, dict], carried: set[str], goal: str, *, key, decide: bool = True, explain: bool = False
) -> dict:
    """R3' and R3d for one trace, by ONE mechanism (G1): every tie ordering of the cue and goal rankings over the pool
    V0 + L less the trace's ``own_ids`` is run through :func:`p_simulate`; R3' holds when one reproduces the log, R3d
    when EVERY one that does is decisive (G4). ``over_bound`` = more orderings than ``EXP63B_TIE_BOUND`` (C5).
    ``decide=False`` (C5 and the smoke) computes R3' and the descriptive counts only: ``decisive`` is None.
    ``explain=True`` (Exp 63c's report; never set by 63b, whose output stays byte-identical) adds ``orders``: for
    EVERY valid tie order, its displaced ids, their non-goal scores and every ``[carried, displaced, carried score,
    displaced score]`` pair; and ``shown_cue``, the logged cue-path ids as ``[id, carried, non-goal score, ts]``."""
    ids, paths = list(tr.get("memory_ids") or []), list(tr.get("memory_paths") or [])
    own = {m for m in _list(tr.get("own_ids")) if isinstance(m, str)}
    cue_text = tr.get("cue_text") if isinstance(tr.get("cue_text"), str) else ""
    v0 = visible_ids(tr, records)
    late = [m for m in ids if m not in v0 and m in records]
    late_goal = {m for m, p in zip(ids, paths) if p == "goal" and m in late}  # landed after the cue path's recall
    goal_pool = [m for m in records if (m in v0 or m in late) and m not in own]
    cue_pool = [m for m in goal_pool if m not in late_goal]
    classes = {m: _copy_class(records[m], m) for m in goal_pool}
    graph = [m for m, p in zip(ids, paths) if p == "graph"]
    cue_groups = p_ranked_groups(records, cue_pool, cue_text, key=key)
    goal_groups = p_ranked_groups(records, goal_pool, goal, key=key) if goal else []
    cue_upto = max(
        _groups_needed(cue_groups, classes, EXP63B_CUE_LIMIT),
        _groups_needed(cue_groups, classes, EXP63B_CUE_LIMIT, skip=frozenset(carried)),
    )
    goal_upto = _groups_needed(goal_groups, classes, EXP63_TOP)
    count = _ordering_count(cue_groups, cue_upto) * _ordering_count(goal_groups, goal_upto)
    out: dict = {"orderings": count, "over_bound": count > EXP63B_TIE_BOUND, "visible": len(v0), "late": late}
    if out["over_bound"]:
        return {**out, "conforms": False, "decisive": False}
    goal_results = {
        tuple(p_distinct(o, classes, EXP63B_CUE_LIMIT)[:EXP63_TOP]) for o in _orderings(goal_groups, goal_upto)
    }
    logged = (ids, paths)
    cue_orders = list(_orderings(cue_groups, cue_upto))
    valid_cue = set()
    for r in {tuple(p_distinct(o, classes, EXP63B_CUE_LIMIT)) for o in cue_orders}:
        if any(p_simulate(graph, list(r), list(g) if goal else None) == logged for g in goal_results):
            valid_cue.add(r)
    out["conforms"] = bool(valid_cue)
    shown_cue = [m for m, p in zip(ids, paths) if p == "cue"]
    nongoal = set(key(cue_text)) - set(key(goal))
    carried_cue = [m for m in shown_cue if m in carried]
    graph_r = [m for m in graph if m not in carried]

    def decisive_under(order: list[str]) -> bool:
        reduced = p_distinct([m for m in order if m not in carried], classes, EXP63B_CUE_LIMIT)
        r_ids, r_paths = p_simulate(graph_r, reduced, None)
        displaced = [m for m, p in zip(r_ids, r_paths) if p == "cue" and m not in ids]
        return any(
            _nongoal_score(records[c], nongoal, key=key) > _nongoal_score(records[d], nongoal, key=key)
            for c in carried_cue
            for d in displaced
        )

    valid_orders = [o for o in cue_orders if tuple(p_distinct(o, classes, EXP63B_CUE_LIMIT)) in valid_cue]
    if explain:
        orders = []
        for order in valid_orders:  # every valid tie order (bounded by EXP63B_TIE_BOUND above)
            reduced = p_distinct([m for m in order if m not in carried], classes, EXP63B_CUE_LIMIT)
            r_ids, r_paths = p_simulate(graph_r, reduced, None)
            displaced = [m for m, p in zip(r_ids, r_paths) if p == "cue" and m not in ids]
            scores = {m: _nongoal_score(records[m], nongoal, key=key) for m in (*carried_cue, *displaced)}
            orders.append(
                {
                    "displaced": displaced,
                    "displaced_scores": {d: scores[d] for d in displaced},
                    "pairs": [[c, d, scores[c], scores[d]] for c in carried_cue for d in displaced],
                }
            )
        out["orders"] = orders
        out["shown_cue"] = [
            [m, m in carried, _nongoal_score(records[m], nongoal, key=key), records[m].get("timestamp")]
            for m in shown_cue
            if m in records
        ]
    out["decisive"] = (
        bool(out["conforms"] and nongoal and carried_cue and all(map(decisive_under, valid_orders))) if decide else None
    )
    canonical_goal = sorted(goal_results)[0] if goal_results else ()
    first_of: dict[tuple, str] = {}
    for m in (m for g in cue_groups for m in sorted(g)):  # one linearisation, for the descriptive counts
        first_of.setdefault(classes[m], m)
    hidden = sorted(
        m for m in cue_pool if m in carried and first_of.get(classes[m], m) != m and first_of[classes[m]] not in carried
    )
    heard = {p_same_words(p_input_text(records[m])) for m in ids if m in records}
    out.update(
        {
            "paths": paths,
            "observed_carried": len(set(ids) & carried),
            "carried_share": (len(set(ids) & carried) / len(ids)) if ids else None,
            "cue_nongoal_keys": sorted(nongoal),
            "cue_hits_on_nongoal_keys": {
                m: _nongoal_score(records[m], nongoal, key=key) for m in shown_cue if m in records
            },
            "cue_added_over_goal": [m for m in shown_cue if m not in canonical_goal],
            "distinct_heard_texts": len(heard - {""}),
            "goal_fill_win": [m for m, p in zip(ids, paths) if p == "goal" and m in carried],
            "carried_hidden_by_a_newer_copy": hidden,
        }
    )
    return out


def turn_traces_63b(lines: list[dict]) -> dict[int, tuple[int, dict] | None]:
    """``turn -> `` its trace (I1): the first goal-bearing AUT trace in the turn's window whose non-empty
    ``own_ids[0]`` differs from the previous turn's trace's (a turn's later traces inherit its id, and a slow one can
    log after the next ENTER)."""
    starts = turn_starts(lines)
    by_turn: dict[int, list[tuple[int, dict]]] = {}
    for i, r in _aut_traces(lines):
        own = r.get("own_ids")
        if str(r.get("goal") or "").strip() and isinstance(own, list) and own and isinstance(own[0], str) and own[0]:
            by_turn.setdefault(turn_of(i, starts), []).append((i, r))
    out: dict[int, tuple[int, dict] | None] = {}
    previous = None
    for turn in range(1, EXP63_TURNS + 1):
        out[turn] = next((hit for hit in by_turn.get(turn, []) if hit[1]["own_ids"][0] != previous), None)
        if out[turn] is not None:
            previous = out[turn][1]["own_ids"][0]
    return out


def digest_lines(lines: list[dict], op: str) -> list[dict]:
    """I1's digest lines for the AUT's store file (``op`` = ``load`` or ``save``)."""
    return [
        r
        for r in lines
        if r.get("e") == EXP63B_DIGEST_EVENT and r.get("op") == op and r.get("file") == EXP63B_STORE_FILE
    ]


def exp63b_r3a(per_turn: dict, turns: dict, records: dict[str, dict], carried: set[str]) -> dict:
    """G3: R3a's view is the first turn's trace, in turn order, with empty holes whose visible non-carried records are
    all its own (``own_ids``). ``status``: ``held``; ``unobservable`` (R3' conforms and it surfaced nothing);
    ``failed`` (a surfaced non-late id outside C, or R3' does not conform there); ``no_view`` and ``late`` (C5)."""
    view = None
    for turn in sorted(per_turn):
        hit = per_turn[turn]
        if hit is None or hit[1].get("goal_path_holes") != []:
            continue
        tr = hit[1]
        if visible_ids(tr, records) - carried <= set(_list(tr.get("own_ids"))):
            view = (turn, hit)
            break
    if view is None:
        return {"turn": None, "ids": [], "status": "no_view", "pass": False}
    turn, (_i, tr) = view
    ids = [m for m in _list(tr.get("memory_ids")) if isinstance(m, str)]
    v0 = visible_ids(tr, records)
    outside = [m for m in ids if m not in carried]
    if any(m in v0 for m in outside):
        status = "failed"  # a surfaced non-late id outside C FAILs, even beside a late one (G3)
    elif outside:
        status = "late"  # only late ids outside C: incomplete (C5), whatever R3' reads
    elif not (turns.get(turn) or {}).get("conforms"):
        status = "failed"
    elif not ids:
        status = "unobservable"
    else:
        status = "held"
    return {"turn": turn, "ids": ids, "status": status, "pass": status == "held"}


# C4'': the evidence gate's B0 checks (``_evidence_records.judge_b0_fields`` + ``judge_b0_router``), restated over a
# phase's sim record so that a complete attempt the gate would refuse never decides (and strands the campaign). The
# gate reads the schema from the code that ran; a sim record here must carry the schema report.py writes (>= 2). A
# test pins the two to the same answer.
B0_IDENTITY_SOURCES = frozenset({"local_file", "server_reported", "provider_name"})
B0_OUTCOMES = ("success", "exhausted", "budget_reject", "raised")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def b0_problems(sim: dict, label: str) -> list[str]:
    p: list[str] = []
    schema = sim.get("provenance_schema")
    if type(schema) is not int or schema < 2:
        return [f"{label}: provenance_schema {schema!r} is not a B0 schema (>= 2)"]
    for key, want in (("concurrent_sims", False), ("sampling_complete", True), ("model_changed_during_run", False)):
        if sim.get(key) is not want:
            p.append(f"{label}: {key} is {sim.get(key)!r}")
    if sim.get("aut_model_attribution") != "not_established":
        p.append(f"{label}: aut_model_attribution {sim.get('aut_model_attribution')!r}")
    if not isinstance(sim.get("scenario_sha256"), str) or not _HEX64.match(sim["scenario_sha256"]):
        p.append(f"{label}: no scenario_sha256")
    index = sim.get("session_index")
    if index is not None and (type(index) is not int or index < 0):
        p.append(f"{label}: session_index {index!r}")
    roles = [k[: -len("_profile")] for k in sim if isinstance(k, str) and k.endswith("_profile") and sim[k]]
    if not roles:
        return p
    calls = sim.get("llm_calls")
    if not isinstance(calls, dict):
        return [*p, f"{label}: ran an LLM role without llm_calls"]
    for role in roles:
        serial = sim.get(f"{role}_router_serial")
        if type(serial) is not int or str(serial) not in calls:
            p.append(f"{label}: {role}_router_serial {serial!r} is not a router in llm_calls")
    for serial, r in calls.items():
        p += _b0_router_problems(r, f"{label} router {serial}")
    return p


def _b0_router_problems(r, label: str) -> list[str]:
    counts = r.get("counts") if isinstance(r, dict) else None
    entries = r.get("entries") if isinstance(r, dict) else None
    if not isinstance(counts, dict) or not isinstance(entries, list):
        return [f"{label}: malformed"]
    if any(type(counts.get(k)) is not int or counts[k] < 0 for k in ("started", *B0_OUTCOMES)):
        return [f"{label}: malformed counts {counts!r}"]
    p: list[str] = []
    if counts["started"] != sum(counts[o] for o in B0_OUTCOMES):
        p.append(f"{label}: {counts['started']} started, outcomes {counts!r}")
    ns = [e.get("n") if isinstance(e, dict) else None for e in entries]
    if any(type(n) is not int or n <= 0 for n in ns) or sum(ns) != counts["success"]:  # type: ignore[arg-type]
        return [*p, f"{label}: entries do not sum to its {counts['success']} successes"]
    for e in entries:
        source, identity = e.get("identity_source"), e.get("identity")
        if source not in B0_IDENTITY_SOURCES:
            p.append(f"{label}: identity_source {source!r} is not established")
        elif source == "local_file" and not (isinstance(identity, str) and _HEX64.match(identity)):
            p.append(f"{label}: local_file identity {identity!r} is not a sha256")
        elif not (isinstance(identity, str) and identity):
            p.append(f"{label}: no identity")
        if type(e.get("seed_honoured")) is not bool:
            p.append(f"{label}: seed_honoured {e.get('seed_honoured')!r}")
    return p


def _exp63b_shape_problems(tr: dict, records: dict[str, dict], goal: str, *, key) -> str | None:
    """C5(b) for one AUT trace: Exp 63's shape, the cue path accepted, and I1's fields, each with its identity check
    (``query_text`` is ``cue_text`` cut at 120, ``keywords`` the port's keys of it)."""
    ids, paths = tr.get("memory_ids"), tr.get("memory_paths")
    if not (isinstance(ids, list) and isinstance(paths, list) and all(isinstance(x, str) for x in ids)):
        return "memory_ids / memory_paths are not lists of ids"
    if not (len(ids) == len(paths) == tr.get("memories")) or not all(
        isinstance(p, str) and p in EXP63B_PATHS for p in paths
    ):
        return f"{len(ids)} ids, {len(paths)} paths, memories {tr.get('memories')!r}"
    if not set(ids) <= set(records):
        return f"ids {sorted(set(ids) - set(records))} are not in phase 2's saved store"
    holes = tr.get("goal_path_holes")
    if not _is_int(tr.get("goal_path_horizon")) or not (
        isinstance(holes, list)
        and all(isinstance(h, list) and len(h) == 2 and all(map(_is_int, h)) and h[0] <= h[1] for h in holes)
    ):
        return f"goal_path_horizon {tr.get('goal_path_horizon')!r} / holes {holes!r} unreadable"
    if not _is_int(tr.get("hippocampus_size")):
        return f"hippocampus_size {tr.get('hippocampus_size')!r} is not an integer"
    if not isinstance(tr.get("goal") or "", str) or (str(tr.get("goal") or "").strip() and tr.get("goal") != goal):
        return f"goal {tr.get('goal')!r} is not the protocol goal"
    own, cue = tr.get("own_ids"), tr.get("cue_text")
    if not (isinstance(own, list) and all(isinstance(m, str) for m in own)):
        return f"own_ids {own!r} is not a list of ids"
    if not isinstance(cue, str) or type(tr.get("recall_failed")) is not bool:
        return f"cue_text / recall_failed missing ({type(cue).__name__}, {tr.get('recall_failed')!r})"
    if tr.get("query_text") != cue[:120]:
        return "query_text is not cue_text's first 120 characters"
    if tr.get("keywords") != list(dict.fromkeys(key(cue)))[:8]:
        return f"keywords {tr.get('keywords')!r} are not the port's keys of cue_text"
    return None


def _enter_text_lens(lines: list[dict]) -> dict[int, int]:
    """``turn -> `` the ``text_len`` on its first ``Bridge.send_and_wait ENTER`` line (C5(i))."""
    out: dict[int, int] = {}
    for rec in lines:
        if rec.get("e") != "sim_exec":
            continue
        message = str(rec.get("message", ""))
        m, n = _TURN_ENTER.search(message), _TEXT_LEN.search(message)
        if m and n and int(m.group(1)) not in out:
            out[int(m.group(1))] = int(n.group(1))
    return out


def _digest_problem(phase_label: str, line: dict, records: dict[str, dict], *, key) -> str | None:
    """C5(d): a save line's digest against the port's recomputation from the same phase's saved bytes, naming the
    first record whose row differs."""
    digest, per_record = p_store_digest(records, key=key)
    if line.get("sha256") == digest and line.get("records") == len(records):
        return None
    logged = {str(x[0]): x[1] for x in _list(line.get("record_sha256")) if isinstance(x, list) and len(x) == 2}
    first = next((rid for rid in sorted(set(per_record) | set(logged)) if per_record.get(rid) != logged.get(rid)), None)
    return (
        f"{phase_label}: the save digest is not the saved bytes' ({line.get('records')!r} records logged, "
        f"{len(records)} saved; first differing record {first!r})"
    )


def exp63b_c5_problems(phases: list[dict], *, goal: str, key, r3a_view: bool = True) -> list[str]:
    """C5 for Exp 63b (G5), R3a's view and the tie bound: every failure makes the attempt INCOMPLETE (an aborted
    attempt, never a FAIL). ``phases`` = [baseline, gate], each ``{report, store, lines}``. Type-guarded throughout: a
    corrupt trace or store aborts, never crashes. ``r3a_view=False`` (the smoke, which never reads R3a) skips R3a's
    view."""
    base, gate = phases
    found: dict[str, str] = {}

    def problem(check: str, text: str) -> None:
        found.setdefault(check, f"C5{check} {text}")

    records = store_records(gate["store"])
    base_records = store_records(base["store"])
    carried = set(base_records)
    # (a) the agent, as Exp 63.
    for label, phase in (("baseline", base), ("gate", gate)):
        for r in phase["lines"]:
            if r.get("e") == "enrichment_trace" and not (isinstance(r.get("agent_id"), str) and r["agent_id"]):
                problem("(a)", f"{label}: an enrichment_trace carries no agent_id")
    if not any(r.get("e") == "sim_deliberation" and r.get("agent_id") == SIM_AUT_AGENT_ID for r in gate["lines"]):
        problem("(a)", f"gate: no sim_deliberation line names the AUT {SIM_AUT_AGENT_ID!r}")
    traces = _aut_traces(gate["lines"])
    if not traces:
        problem("(a)", f"gate: no enrichment_trace of {SIM_AUT_AGENT_ID!r}")
    # (b) every phase-2 AUT trace's shape and I1's fields; I1's digest lines.
    shaped: list[tuple[int, dict]] = []
    for i, tr in traces:
        bad = _exp63b_shape_problems(tr, records, goal, key=key)
        if bad:
            problem("(b)", f"gate line {i + 1}: {bad}")
        else:
            shaped.append((i, tr))
    loads = digest_lines(gate["lines"], "load")
    if len(loads) != 1:
        problem("(b)", f"gate: {len(loads)} restore digest lines for {EXP63B_STORE_FILE}, not 1 (R2' reads it)")
    saves = {label: digest_lines(phase["lines"], "save") for label, phase in (("baseline", base), ("gate", gate))}
    for label, lines in saves.items():
        if not lines:
            problem("(b)", f"{label}: no save digest line for {EXP63B_STORE_FILE}")
    per_turn = turn_traces_63b(gate["lines"])
    shaped_at = {i for i, _tr in shaped}
    turn_hits = {t: hit for t, hit in per_turn.items() if hit is not None and hit[0] in shaped_at}
    # (L) liveness reads recall_failed and nothing else.
    for turn, (_i, tr) in sorted(turn_hits.items()):
        if tr.get("recall_failed") is True:
            problem("(L)", f"turn {turn}: the recall failed (recall_failed)")
    # (d) record shape (Exp 63's), widened: no legacy ToolOutput repr, and each save digest is its saved bytes'.
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
    for label, store in (("baseline", base_records), ("gate", records)):
        for rid, rec in store.items():
            result = _dict(rec.get("outcome")).get("result")
            if isinstance(result, str) and result.startswith("ToolOutput("):
                problem("(d)", f"{label} record {rid}: a legacy ToolOutput( result the loader would rewrite")
        lines = saves[label]
        if lines:
            bad = _digest_problem(label, lines[-1], store, key=key)
            if bad:
                problem("(d)", bad)
    # (c) rendering identity, as Exp 63.
    rendered: dict[str, int] = {}
    for _i, tr in traces:
        ids = tr.get("memory_ids") if isinstance(tr.get("memory_ids"), list) else []
        for mid in ids[:EXP63_TOP]:
            rendered[str(mid)] = rendered.get(str(mid), 0) + 1

    def enrichment(rec: dict | None) -> int:
        n = _dict(_dict(rec).get("activation_sources")).get("enrichment", 0)
        return n if _is_int(n) else -(10**9)

    for rid in sorted(set(records) | set(rendered)):
        delta = enrichment(records.get(rid)) - enrichment(base_records.get(rid))
        if delta != rendered.get(rid, 0):
            problem("(c)", f"{rid}: enrichment activations moved {delta}, but it was rendered {rendered.get(rid, 0)}x")
    # (g) graph ids carry objects (all shaped traces); (s) no substring id on a TURN's trace (G5).
    for i, tr in shaped:
        for mid, path in zip(tr["memory_ids"], tr["memory_paths"]):
            if path == "graph" and not _list(_dict(records[mid].get("perception")).get("detected_objects")):
                problem("(g)", f"gate line {i + 1}: graph id {mid} has no detected_objects")
    for turn, (i, tr) in sorted(turn_hits.items()):
        if "substring" in tr["memory_paths"]:
            problem("(s)", f"turn {turn} (gate line {i + 1}): its trace carries a substring id (not ported)")
    # (i) identity: each turn's own capture is a saved non-carried visible record of the words the cue queried, and
    # the cue is as long as the text the bridge sent (both independent of the pipeline).
    text_lens = _enter_text_lens(gate["lines"])
    for turn, (i, tr) in sorted(turn_hits.items()):
        own0, cue = tr["own_ids"][0], tr["cue_text"]
        rec = records.get(own0)
        if rec is None or own0 in carried or own0 not in visible_ids(tr, records):
            problem("(i)", f"turn {turn}: own_ids[0] {own0!r} is not a saved, new, visible record")
        elif p_same_words(p_input_text(rec)) != p_same_words(cue):
            problem("(i)", f"turn {turn}: own_ids[0]'s words are not the cue's")
        elif text_lens.get(turn) != len(cue):
            problem("(i)", f"turn {turn}: cue_text has {len(cue)} characters, the bridge sent {text_lens.get(turn)!r}")
    # The tie bound (G1) on every turn's trace, and R3a's view, read only over a sound store and sound traces: a
    # malformed record or trace has already made the attempt incomplete above, and must not crash the reading.
    if any(k.startswith(("C5(b)", "C5(d)")) for k in found.values()):
        return list(found.values())
    turns = {}
    for turn, (i, tr) in sorted(turn_hits.items()):
        turns[turn] = exp63b_turn(tr, records, carried, goal, key=key, decide=False)
        if turns[turn]["over_bound"]:
            problem("(t)", f"turn {turn}: {turns[turn]['orderings']} tie orderings, over {EXP63B_TIE_BOUND}")
    # R3a's view and a late id in it (G3), only while P1 and P2 hold.
    restored = _p1(gate, len(carried))[1] and carried <= set(records)
    if r3a_view and restored and len(turn_hits) == len(per_turn):
        r3a = exp63b_r3a(per_turn, turns, records, carried)
        if r3a["status"] == "no_view":
            problem("(R3a)", "no turn's trace has empty holes and only its own records new: R3a has no view")
        elif r3a["status"] == "late":
            problem("(R3a)", f"turn {r3a['turn']}: R3a's trace surfaced an id that landed during the query")
    return list(found.values())


def _store_gates(base: dict, gate: dict) -> dict:
    """P0-P3 over a two-phase attempt's stores and phase 2's first AUT trace: a COPY of Exp 63's (``exp63_gates``
    stays byte-identical, so "63"'s committed verdict re-judges the same under the gate's half B)."""
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
    return out


def exp63b_gates(phases: list[dict], *, goal: str, key) -> dict:
    """P0-P3, R1, R2', R3a, R3' and R3d over one attempt's two phases, and the verdict (the prereg's table). Also
    computed over an aborted attempt's committed phases for the leaked-gate bars, where a gate reads passed only in its
    held state (R3a unobservable, no view or a late id, and no decisive turn each read not passed)."""
    base, gate = phases
    base_records, records = store_records(base["store"]), store_records(gate["store"])
    carried = set(base_records)
    out = _store_gates(base, gate)
    per_turn = turn_traces_63b(gate["lines"])
    out["R1"] = {
        "pass": all(hit is not None for hit in per_turn.values()),
        "turns": sorted(t for t, h in per_turn.items() if h),
    }
    loads = digest_lines(gate["lines"], "load")
    want = p_store_digest(base_records, key=key)[0]
    out["R2prime"] = {
        "restore_digest": loads[0].get("sha256") if len(loads) == 1 else None,
        "phase1_saved_digest": want,
        "pass": len(loads) == 1 and loads[0].get("sha256") == want and loads[0].get("records") == len(base_records),
    }
    turns = {
        t: None if hit is None else exp63b_turn(hit[1], records, carried, goal, key=key) for t, hit in per_turn.items()
    }
    out["R3prime"] = {
        "pass": all(x is not None and x["conforms"] for x in turns.values()),
        "nonconforming": [t for t, x in turns.items() if x is None or not x["conforms"]],
    }
    out["R3a"] = exp63b_r3a(per_turn, turns, records, carried)
    r3a_turn = out["R3a"]["turn"] if out["R3a"]["status"] == "held" else None  # unobservable: no decisive turn
    decisive = [
        t for t, x in turns.items() if x is not None and x["decisive"] and r3a_turn is not None and t > r3a_turn
    ]
    out["R3d"] = {"decisive_turns": decisive, "pass": bool(decisive)}
    records_per_input = (len(records) - len(base_records)) / EXP63_TURNS
    out["descriptive"] = {
        "status": "DESCRIPTIVE (never gating)",
        "turns": turns,
        "new_records_per_input": records_per_input,
    }
    holding = ("P0", "P1", "P2", "P3", "R1", "R2prime", "R3prime")
    if not all(out[g]["pass"] for g in holding) or out["R3a"]["status"] == "failed":
        out["verdict"] = "FAIL"
    elif out["R3a"]["status"] == "held" and out["R3d"]["pass"]:
        out["verdict"] = "PASS"
    else:
        out["verdict"] = "NOT SHOWN"
    out["not_passed"] = [g for g in (*holding, "R3a", "R3d") if not out[g]["pass"]]
    return out


# ── Exp 63c (T1-16 under a scripted narrator: docs/experiments/exp63c_scripted_recall_prereg.md) ─────────────────────
# Additive: 63b's gates run unchanged under 63c's keying, then 63c's changes: R3d searches the cue turns only, the
# controls are armed or contaminated, a decisive pair must share a cue non-goal key, C1 reads ``completed`` and the
# content pins, and S2' checks every serving field. The judge is stdlib-only, so PROTOCOL["63c"] holds the 16 script
# lines and the GGUF's SHA-256 as literals; a test pins the lines to the loader's texts and each phase's content hash
# to its pin. Every function that keys text takes the keying (``keying_of``), never a hard-coded port. The process
# session's rulings on the frozen text (S1-S5): docs/experiments/rationale/exp63c-scripted-recall/build-rulings.md.

EXP63C_SCRIPTS = tuple(phase[4][1] for phase in PROTOCOL["63c"]["phases"])  # each phase's ``--sim-script`` path
EXP63C_LINES = tuple(tuple(lines) for lines in PROTOCOL["63c"]["script_lines"])
# ``serving_provenance._content_sha256([load_campaign_turns(script)])``, the report's ``scenario_content_sha256``.
EXP63C_CONTENT_PINS = tuple(PROTOCOL["63c"]["scenario_content_sha256"])
# S2' (OD-c4/OD-c5): the GGUF every call names, by its SHA-256 (63b's committed identity; re-read at preflight).
EXP63C_GGUF_SHA256 = PROTOCOL["63c"]["gguf_sha256"]
EXP63C_CUE_TURNS = {3: "A", 4: "B", 5: "C", 7: "D"}  # phase-2 turn -> the fact it cues
EXP63C_CONTROL_TURNS = (2, 6)
EXP63C_DISTRACTOR_LINE = 6  # phase 1's goal-words distractor (1-based)
EXP63C_FACT_LINES = {"A": 2, "B": 3, "C": 4, "D": 5}  # fact -> its phase-1 line (1-based)
EXP63C_ANSWER_KEYS = {  # stored as keys (prereg §Vocabulary): c_key_tokens on d40eb4c0
    "A": ("copper", "chair"),
    "B": ("knif", "smal"),
    "C": ("chain", "crack"),
    "D": ("drain", "grat", "north"),
}
EXP63C_GOAL_KEYS = frozenset(
    {"dungeon", "escap", "guard", "sleep"}
)  # the prereg's frozen table; a test binds key(goal)
EXP63C_VERDICT_TIME_ONLY = "C5(k)/(q): verdict-time only (§Smoke heading)"  # what the smoke prints in their place


def _served_basename(served) -> str | None:
    return served.replace("\\", "/").rsplit("/", 1)[-1] if isinstance(served, str) and served else None


def _read_ok(r: dict) -> bool:
    return isinstance(r.get("served"), str) and bool(r["served"]) and not r.get("error")


def _served_span(row: dict) -> tuple[list[dict], list[dict]]:
    """``(judged, unjudged)`` harness ``/v1/models`` reads: judged from the first OK read to the last OK read, the
    rest (the server starting before, stopping after) reported. Computed from the raw reads; the harness labels none."""
    reads = [r for r in _list((row.get("served_model") or {}).get("reads")) if isinstance(r, dict)]
    ok = [i for i, r in enumerate(reads) if _read_ok(r)]
    if not ok:
        return [], reads
    return reads[ok[0] : ok[-1] + 1], reads[: ok[0]] + reads[ok[-1] + 1 :]


def exp63c_serving_problems(row: dict, name: str) -> list[str]:
    """S2' over one phase row (a C4-class check): every ``llm_calls`` entry on every router serial is the peer
    backend's local file with the pinned SHA-256 and a ``served_id`` whose basename is the GGUF (empty fails); a
    router serial with no entries passes only when it served nothing (``counts.success == 0``), and the phase must
    carry at least one entry; every harness read from the first OK read to the last names the GGUF (no OK read at
    all fails; a failed read inside the span fails). A stdlib comparator, never ``_served_model_matches``."""
    p: list[str] = []
    sims = row.get("sims")
    sim = sims[0] if isinstance(sims, list) and len(sims) == 1 and isinstance(sims[0], dict) else {}
    calls = sim.get("llm_calls")
    if not isinstance(calls, dict) or not calls:
        return [f"S2' {name}: no llm_calls accounting"]
    n_entries = 0
    for serial, router in sorted(calls.items()):
        router = router if isinstance(router, dict) else {}
        entries = router.get("entries")
        if not isinstance(entries, list):
            p.append(f"S2' {name}: router {serial} has no entries list")
            continue
        if not entries:
            if (router.get("counts") or {}).get("success") != 0:
                p.append(f"S2' {name}: router {serial} served calls but records no entry")
            continue
        for e in entries:
            n_entries += 1
            e = e if isinstance(e, dict) else {}
            fields = (e.get("backend_kind"), e.get("identity_source"), e.get("identity"))
            if fields != ("peer", "local_file", EXP63C_GGUF_SHA256):
                p.append(f"S2' {name}: router {serial} entry serves {fields!r}, not the pinned peer GGUF")
            if _served_basename(e.get("served_id")) != MODEL_GGUF:
                p.append(f"S2' {name}: router {serial} entry served_id {e.get('served_id')!r} is not {MODEL_GGUF}")
    if not n_entries:
        p.append(f"S2' {name}: no call entry on any router serial")
    judged, _unjudged = _served_span(row)
    if not judged:
        p.append(f"S2' {name}: no harness /v1/models read returned a served id")
    for r in judged:
        if not _read_ok(r) or _served_basename(r.get("served")) != MODEL_GGUF:
            p.append(
                f"S2' {name}: a harness read inside the served span names {r.get('served')!r} ({r.get('error')!r})"
            )
    return p


def exp63c_c1_problems(index: int, report: dict, name: str) -> list[str]:
    """C1 for Exp 63c: the run ends ``completed`` (the precampaign runner's own end; never a relabelled guard event),
    ``turns == 8``, and the report's ``scenario_content_sha256`` is the phase's pin."""
    p: list[str] = []
    if report.get("finish_reason") != "completed":
        p.append(f"C1 {name}: finish_reason {report.get('finish_reason')!r}, not 'completed'")
    if report.get("turns") != EXP63_TURNS:
        p.append(f"C1 {name}: turns {report.get('turns')!r}, not {EXP63_TURNS}")
    content = (report.get("provenance") or {}).get("scenario_content_sha256")
    if content != EXP63C_CONTENT_PINS[index]:
        p.append(f"C1 {name}: scenario_content_sha256 {content!r} is not the pinned script's")
    return p


def _exp63c_nongoal(text: str, goal: str, *, key) -> set[str]:
    return set(key(text)) - set(key(goal))


def exp63c_control_status(turn: int, carried_records: dict[str, dict], goal: str, *, key) -> str:
    """``armed`` when no carried record holds any of the control cue's non-goal keys (a decisive armed control is an
    instrument fault, C5(k)), else ``contaminated`` (reported, never C5). The carried-store rule only (ruling S4)."""
    nongoal = _exp63c_nongoal(EXP63C_LINES[1][turn - 1], goal, key=key)
    held = any(nongoal & p_record_tokens(rec, key=key) for rec in carried_records.values())
    return "contaminated" if held else "armed"


def _exp63c_turns(gate: dict, records: dict[str, dict], carried: set[str], goal: str, *, key) -> dict:
    per_turn = turn_traces_63b(gate["lines"])
    return {
        t: None if hit is None else exp63b_turn(hit[1], records, carried, goal, key=key, explain=True)
        for t, hit in per_turn.items()
    }


def exp63c_c5_problems(phases: list[dict], *, goal: str, key, verdict_time: bool, r3a_view: bool = True) -> list[str]:
    """C5 for Exp 63c: 63b's (G5) under 63c's keying, plus (i') each turn's ``cue_text`` and ENTER ``text_len`` are
    its script line exactly, in both phases; and, at verdict time only, (k) an armed control reads decisive and (q) a
    decisive pair whose carried record shares none of the cue's non-goal keys (G4 condition 3 cannot hold without one;
    a guard against a broken port, checked in EVERY valid order). ``verdict_time`` is required: False (the smoke,
    which never computes R3d: ruling S1) skips (k) and (q) and computes no decisive status."""
    found = exp63b_c5_problems(phases, goal=goal, key=key, r3a_view=r3a_view)
    for index, phase in enumerate(phases):
        label = ("baseline", "gate")[index]
        lens = _enter_text_lens(phase["lines"])
        for turn, hit in turn_traces_63b(phase["lines"]).items():
            line = EXP63C_LINES[index][turn - 1]
            if hit is not None and hit[1].get("cue_text") != line:
                found.append(f"C5(i') {label} turn {turn}: cue_text is not script line {turn}")
            if lens.get(turn) != len(line):
                found.append(
                    f"C5(i') {label} turn {turn}: the bridge sent {lens.get(turn)!r} characters, not {len(line)}"
                )
    if not verdict_time or any(x.startswith(("C5(b)", "C5(d)", "C5(t)")) for x in found):
        return found  # (k) and (q) read only a sound store and sound traces, and only at verdict time
    base, gate = phases
    base_records, records = store_records(base["store"]), store_records(gate["store"])
    carried = set(base_records)
    for turn, x in sorted(_exp63c_turns(gate, records, carried, goal, key=key).items()):
        if x is None or not x.get("decisive"):
            continue
        if turn in EXP63C_CONTROL_TURNS and exp63c_control_status(turn, base_records, goal, key=key) == "armed":
            found.append(f"C5(k) turn {turn}: an armed control reads decisive")
        nongoal = _exp63c_nongoal(EXP63C_LINES[1][turn - 1], goal, key=key)
        for o in x.get("orders", []):
            winners = {c for c, _d, sc, sd in o["pairs"] if sc > sd}
            if not any(nongoal & p_record_tokens(records[c], key=key) for c in winners):
                found.append(f"C5(q) turn {turn}: no decisive carried record shares a cue non-goal key")
                break
    return found


def _by_window(lines: list[dict], pick) -> dict[int, int]:
    """How many of ``lines`` that ``pick`` selects fall in each turn's window (by log order, ``turn_starts``)."""
    starts = turn_starts(lines)
    counts = {t: 0 for t in range(1, EXP63_TURNS + 1)}
    for i, r in enumerate(lines):
        if pick(r):
            t = turn_of(i, starts)
            if t in counts:
                counts[t] += 1
    return counts


def _exp63c_report(
    turn: int, x: dict, tr: dict, records: dict[str, dict], carried: set[str], base_records: dict, goal: str, *, key
) -> dict:
    """The prereg's per-cue-turn report (descriptive, never gating). The pool and the loads count the new records the
    turn's recall could see (V0 less carried and the turn's own capture). Echo load is over the cue's non-goal keys
    (the frozen replay's measure, ruling S2), the answer-key load over the answer keys. ``contested`` is any valid
    order's best pair (ruling S3 i); ``uncontested_literal`` holds when NO displaced id in ANY valid order holds a cue
    non-goal key (ruling S3 ii, what the PASS ledger wording states)."""
    fact = EXP63C_CUE_TURNS[turn]
    cue = EXP63C_LINES[1][turn - 1]
    nongoal = _exp63c_nongoal(cue, goal, key=key)
    cue_keys = set(key(cue))
    fact_text = p_same_words(EXP63C_LINES[0][EXP63C_FACT_LINES[fact] - 1])
    planted = {m for m, rec in base_records.items() if p_same_words(p_input_text(rec)) == fact_text}
    own = set(_list(tr.get("own_ids")))
    new = [m for m in visible_ids(tr, records) if m not in carried and m not in own and m in records]
    tokens = {m: p_record_tokens(records[m], key=key) for m in new}
    orders = []
    for o in x.get("orders", []):
        winning = [pr for pr in o["pairs"] if pr[2] > pr[3]]
        best = max(winning, key=lambda pr: (pr[2] - pr[3], pr[2]), default=None)
        orders.append({"best_pair": best, "score": None if best is None else f"{best[2]} v {best[3]}"})
    bests = [o["best_pair"] for o in orders if o["best_pair"] is not None]
    surfaced_carried = [m for m in _list(tr.get("memory_ids")) if m in carried and m in records]
    answer = set(EXP63C_ANSWER_KEYS[fact])
    out = {
        "decisive": bool(x.get("decisive")),
        "orders": orders,
        "score": orders[0]["score"] if orders else None,
        "contested": any(b[3] >= 1 for b in bests) if bests else None,
        "uncontested_literal": all(s == 0 for o in x.get("orders", []) for s in o["displaced_scores"].values()),
        "displaceable_pool": sum(1 for m in new if cue_keys & tokens[m]),
        "winner_is_planted_fact": bool(bests) and all(b[0] in planted for b in bests),
        "completion": any(answer <= p_record_tokens(records[m], key=key) for m in surfaced_carried),
        "cue_nongoal_keys": sorted(nongoal),
        "echo_load": {k: sum(1 for m in new if k in tokens[m]) for k in sorted(nongoal)},
        "answer_key_load": {k: sum(1 for m in new if k in tokens[m]) for k in sorted(answer)},
    }
    if not out["decisive"]:  # what won and why: the logged cue path, carried or new, its non-goal score and time
        out["what_won"] = x.get("shown_cue", [])
    return out


def exp63c_gates(phases: list[dict], *, goal: str, key) -> dict:
    """63b's gates and verdict table under 63c's keying, with R3d searched on the cue turns only (3, 4, 5, 7) after a
    held R3a, the controls' status, and the descriptive report. The controls (2, 6) and the closing turn never count."""
    out = exp63b_gates(phases, goal=goal, key=key)
    base, gate = phases
    base_records, records = store_records(base["store"]), store_records(gate["store"])
    carried = set(base_records)
    turns = _exp63c_turns(gate, records, carried, goal, key=key)
    per_turn = turn_traces_63b(gate["lines"])
    r3a_turn = out["R3a"]["turn"] if out["R3a"]["status"] == "held" else None
    decisive = [
        t
        for t in sorted(EXP63C_CUE_TURNS)
        if (turns.get(t) or {}).get("decisive") and r3a_turn is not None and t > r3a_turn
    ]
    out["R3d"] = {"decisive_turns": decisive, "pass": bool(decisive), "searched_turns": sorted(EXP63C_CUE_TURNS)}
    report = {
        t: _exp63c_report(t, turns[t], per_turn[t][1], records, carried, base_records, goal, key=key)
        for t in sorted(EXP63C_CUE_TURNS)
        if turns.get(t) is not None and per_turn.get(t) is not None
    }
    distractor = p_same_words(EXP63C_LINES[0][EXP63C_DISTRACTOR_LINE - 1])
    distractor_ids = {m for m, rec in base_records.items() if p_same_words(p_input_text(rec)) == distractor}
    surfaced = {t: {m for m in _list(hit[1].get("memory_ids"))} for t, hit in per_turn.items() if hit is not None}
    out["descriptive"] = {
        **out["descriptive"],
        "k": f"{len(decisive)} of {len(EXP63C_CUE_TURNS)}",
        "cue_turns": report,
        "uncontested": bool(decisive) and all(report[t]["uncontested_literal"] for t in decisive),
        "contested_turns": [t for t in decisive if not report[t]["uncontested_literal"]],
        "controls": {t: exp63c_control_status(t, base_records, goal, key=key) for t in EXP63C_CONTROL_TURNS},
        "distractor_surfaced": any(ids & distractor_ids for ids in surfaced.values()),
        "hippocampus_size_by_turn": {t: hit[1].get("hippocampus_size") for t, hit in per_turn.items() if hit},
        "imagination_events_by_turn": {
            label: _by_window(phase["lines"], lambda r: r.get("e") == "sim_imagination")
            for label, phase in (("baseline", base), ("gate", gate))
        },
        "turns": turns,
    }
    held = out["verdict"] != "FAIL"
    out["verdict"] = "FAIL" if not held else ("PASS" if r3a_turn is not None and out["R3d"]["pass"] else "NOT SHOWN")
    out["not_passed"] = [g for g in out["not_passed"] if g != "R3d"] + ([] if out["R3d"]["pass"] else ["R3d"])
    return out


# ── #1059 / #1079: a FAILED gate leaked into an aborted attempt's committed phases ─────────────────────────────
# The map below serves TWO bars: #1059 (an aborted PREDECESSOR campaign's attempt bars its successor,
# :func:`chain_leaked_gate_problems`) and #1079 (an earlier aborted attempt of the SAME campaign bars its deciding
# attempt, :func:`within_campaign_leaks`; new campaign keys only, :data:`PRE_1079_KEYS`).
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
    "63b": {1: ("P0",), 2: ("P0", "P1", "P2", "P3", "R1", "R2prime", "R3a", "R3prime", "R3d")},
    "63c": {1: ("P0",), 2: ("P0", "P1", "P2", "P3", "R1", "R2prime", "R3a", "R3prime", "R3d")},
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
    if experiment not in ("63", "63b", "63c"):
        raise ValueError(f"no gates for experiment {experiment!r}")  # unjudgeable, which bars: never another's gates
    if len(phases) == 1:
        return {"P0": len(store_records(phases[0]["store"])) >= 3}
    goal = PROTOCOL[key]["phases"][1][1]
    if experiment == "63c":
        g = exp63c_gates(phases, goal=goal, key=keying_of(key))
    elif experiment == "63b":
        g = exp63b_gates(phases, goal=goal, key=keying_of(key))
    else:
        g = exp63_gates(phases, goal=goal)
    return {name: g[name]["pass"] for name in LEAK_GATES[experiment][2]}


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


def _attempt_leak(key: str, label: str, attempt_rows: list[dict], data_root: Path) -> tuple[object, str] | None:
    """One attempt's leaked-gate reading, shared by both bars (#1059 between campaigns, #1079 within one): over the
    attempt's leading ``ok`` phases (``data_root`` = its campaign's data directory), every gate :data:`LEAK_GATES`
    says they decide must pass. Returns None (no committed ``ok`` prefix, or every decided gate passed), else
    ``(status, prose)``: ``status`` is the SORTED failed gate names or ``"unjudgeable"`` (a prefix that cannot be
    judged bars too), the structured form #1079's verdict field carries; ``prose`` (``label`` first) is for humans
    and may embed exception text, so it is never compared."""
    prefix = []
    for i, row in enumerate(attempt_rows):
        if row.get("phase_index") != i or row.get("status") != "ok":
            break
        prefix.append(row)
    if not prefix:
        return None
    names = LEAK_GATES[experiment_of(key)].get(len(prefix))
    if names is None:
        return "unjudgeable", f"{label}: {len(prefix)} committed phases decide no known gate set (cannot be judged)"
    try:
        passes = _gate_passes(key, [_committed_phase(data_root, row) for row in prefix])
        failed = [name for name in names if passes[name] is not True]
    except _UNJUDGEABLE as exc:
        return "unjudgeable", f"{label}: its committed phases cannot be judged ({type(exc).__name__}: {exc})"[:300]
    if failed:
        return sorted(failed), f"{label}: a FAILED gate leaked into its committed phases: {failed}"
    return None


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
        leak = _attempt_leak(pred_key, f"campaign {pred_key}, attempt {str(run_id)[:12]}", attempt_rows, data_root)
        if leak is not None:
            problems.append(leak[1])
    return problems


def within_campaign_leaks(
    key: str, attempts_in_order: list[dict], deciding_run_id: str | None, data_root: Path
) -> tuple[list[list], list[str]]:
    """#1079 (owner decision 2026-10-08, REFUSE): the leaked-gate bar WITHIN one campaign. Over every attempt BEFORE
    the deciding one (all of them when none decides), in k order: an attempt whose committed ``ok`` prefix shows a
    FAILED (or unjudgeable) :data:`LEAK_GATES` gate is listed, and so is an earlier start marker with NO rows (D1:
    rows never committed cannot be ruled out, so a hard-killed attempt bars too). An attempt the judge found
    incomplete for C1-C5 is read the same way (strict, N1): its committed phases were still seen. Returns
    ``(structured, prose)``: ``structured`` = ``[run_id, k, status]`` per listed attempt, ``status`` the sorted failed
    gate names, ``"unjudgeable"`` or ``"rowless"`` (the verdict's ``within_campaign_leaks``, compared by the gate);
    ``prose`` the human lines (never compared and never persisted: it may embed exception text). ``data_root`` is the
    campaign's own directory, the one :func:`judge` reads. Pure over the bytes (no git, no network)."""
    structured: list[list] = []
    prose: list[str] = []
    for attempt in attempts_in_order:
        run_id = attempt["run_id"]
        if run_id == deciding_run_id:
            break
        label = f"campaign {key}, attempt {attempt.get('k')} ({str(run_id)[:12]})"
        if not attempt["rows"]:
            leak: tuple[object, str] | None = (
                "rowless",
                f"{label}: a start marker whose rows never reached main (cannot be judged, #1079 D1)",
            )
        else:
            leak = _attempt_leak(key, label, attempt["rows"], data_root)
        if leak is not None:
            structured.append([run_id, attempt.get("k"), leak[0]])
            prose.append(leak[1])
    return structured, prose


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
    first complete one. Pure over the bytes (no git, no network). Raises :class:`Refusal` on an integrity break.

    For a campaign key outside :data:`PRE_1079_KEYS` it also reports ``within_campaign_leaks`` (#1079): each attempt
    before the deciding one (every attempt when none decides) whose committed phases show a FAILED or unjudgeable
    gate, or that is a start marker with no rows (D1: a rowless earlier marker cannot be judged, so it bars). The
    verdict and its exit code do not change; the evidence gate refuses the support."""
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
        if experiment_of(exp) in ("63", "63b", "63c") and len(phases) == len(PROTOCOL[exp]["phases"]):
            # C5 and R3a's observability, from the committed traces and stores: an instrument fault is not data.
            goal = PROTOCOL[exp]["phases"][1][1]
            if experiment_of(exp) == "63c":
                entry["problems"] += exp63c_c5_problems(phases, goal=goal, key=keying_of(exp), verdict_time=True)
            elif experiment_of(exp) == "63b":
                entry["problems"] += exp63b_c5_problems(phases, goal=goal, key=keying_of(exp))
            else:
                entry["problems"] += exp63_c5_problems(phases, goal=goal)
        entry["complete"] = not entry["problems"]
        if entry["complete"]:
            deciding = (entry, phases)  # the first complete attempt decides
    out: dict = {"attempts": listing, "experiment": exp}  # the CAMPAIGN key ("10c2"), as the gate and successor read it
    if predecessors(exp):
        # #1059 D2: emitted so the gate's re-judge with this (bound) judge reproduces the bar; a successor verdict
        # whose bound judge does not emit it supports nothing. Never compared with the record (o19_difference).
        out["leaked_gates"] = chain_leaked_gate_problems(exp, data_root.parent)
    if exp not in PRE_1079_KEYS:
        # #1079: the bar within this campaign. The structured field is compared with the record by the gate
        # (o19_difference) and refuses a deciding attempt it lists after; the prose is for humans only (main() prints
        # it to stderr and drops it before the verdict is stamped and written: S1).
        out["within_campaign_leaks"], out["within_campaign_leak_notes"] = within_campaign_leaks(
            exp, attempts_in_order, None if deciding is None else deciding[0]["run_id"], data_root
        )
    if deciding is None:
        out["verdict"] = "ABORT"
        out["deciding_attempt"] = None
        return out
    entry, phases = deciding
    out["deciding_attempt"] = entry["run_id"]
    experiment = experiment_of(exp)
    if experiment == "63c":
        gates = exp63c_gates(phases, goal=PROTOCOL[exp]["phases"][1][1], key=keying_of(exp))
    elif experiment == "63b":
        gates = exp63b_gates(phases, goal=PROTOCOL[exp]["phases"][1][1], key=keying_of(exp))
    elif experiment == "63":
        gates = exp63_gates(phases, goal=PROTOCOL[exp]["phases"][1][1])
    elif experiment == "10":
        gates = exp10_gates(phases)
    elif experiment == "09":
        gates = exp09_gates(phases[0], phases[0]["log_bytes"])
    else:
        raise Refusal(f"no gates for experiment {experiment!r}")  # never another experiment's
    out["gates"] = gates
    out["verdict"] = gates["verdict"]
    return out


# The files that define what an attempt ran and how it is judged: identical at every executed commit, on main and
# at the verdict's own commit, or the verdict refuses (a post-data change to the gate is a new experiment).
BOUND_FILES = ("scripts/o19_verdict.py", "scripts/o19_rerun.py")

EXIT = {"PASS": 0, "FAIL": 1, "PARTIAL": 4, "NOT SHOWN": 4, "ABORT": 4}


def bound_blobs(exp: str, commits: list[str]) -> dict[str, dict[str, str | None]]:
    """``path -> {commit -> blob sha}`` for the prereg, its base preregs and the two O19 scripts."""
    out: dict[str, dict[str, str | None]] = {}
    for path in (PROTOCOL[exp]["prereg"], *PROTOCOL[exp].get("base_preregs", []), *BOUND_FILES):
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
    # #1079 N3: the verdict reports (stderr), the gate refuses. The prose may embed host paths and exception text, so
    # it is never persisted (design pass S1): only the structured ``within_campaign_leaks`` reaches verdict.json.
    for note in report.pop("within_campaign_leak_notes", None) or []:
        print(
            f"WITHIN-CAMPAIGN LEAK (#1079; the evidence gate refuses this verdict's support): {note}", file=sys.stderr
        )
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
