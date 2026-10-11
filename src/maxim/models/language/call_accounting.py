"""Which models answered a sim's LLM calls, per router object (baseline floors B0).

Every :class:`~maxim.models.language.router.LLMRouter` takes a process-unique serial at construction and is
registered here with zero counts, so a router that served no call and a router the report never saw cannot
look alike. Each dispatch through ``LLMRouter._complete_text_locked`` records one start and exactly one
terminal outcome; a success also records the served call (:class:`ServedCall`).
The store is module-level and holds plain data, so the counts of a router built outside the report's view
(ExecAgent's private router, the CLI's auto-curate router) or garbage-collected before report time survive.

A sim reports the DELTA between a snapshot at its start and one at report time (:func:`delta`), so calls from
an earlier sim in the same process, or before the sim began, are outside it. Two sims running at once in one
process share this store and cannot be separated; ``simulation/serving_provenance.py`` stamps such overlap.

Accounting never changes what a dispatch does: :func:`record_outcome` never raises, and a call it cannot
record is left without an outcome, which reads as an incomplete window. The store grows by one small entry
per router built, for the process's lifetime.

What this does NOT establish: which model served the agent under test. Calls are attributed to routers, not
to agents, and today part of the AUT runs on the narrator's router (#1228); see
docs/plans/baseline_floors_b0.md §B0b. Calls that bypass the router (``simulation_generator.py`` and
``interactive.py``'s ``LLMAgent`` continuation call ``backend.complete`` directly) are not counted.
"""

from __future__ import annotations

import itertools
import logging
import math
import threading
from dataclasses import dataclass, fields
from typing import Any

# The terminal outcomes of one dispatch. A dispatch that ends without one leaves its router's counts
# unbalanced (started > sum of outcomes), which the report stamps as incomplete.
OUTCOMES = ("success", "exhausted", "budget_reject", "raised")


@dataclass(frozen=True)
class ServedCall:
    """The call a successful dispatch made: which provider answered and the model the router requested of it, through which kind of backend
    and endpoint (``backend_kind`` ``"llama"`` = the in-process llama backend, ``endpoint`` its model path;
    ``"peer"`` = a Maxim/llama.cpp server, ``endpoint`` its base URL, ``served_id`` its ``/v1/models`` id as
    cached then; else ``"other"``), the backend's seed and whether it honours it, the backend's quantization
    when it loads the file itself, and the call's temperature. Every field is part of the entry key, so a
    served id discovered between calls splits the entry rather than merging. Runtime-ephemeral (never
    persisted as a dataclass)."""

    provider_key: str
    model: str
    backend_kind: str
    endpoint: str
    served_id: str
    quantization: str | None
    seed: int | None
    seed_honoured: bool
    temp: float

    def key(self) -> tuple[Any, ...]:
        return (
            self.provider_key,
            self.model,
            self.backend_kind,
            self.endpoint,
            self.served_id,
            self.quantization,
            self.seed,
            self.seed_honoured,
            self.temp,
        )


logger = logging.getLogger(__name__)

_LOCK = threading.Lock()
_SERIALS = itertools.count(1)
# serial -> {"entries": {served-call key: n}, "counts": {"started": n, <outcome>: n}}
_ROUTERS: dict[int, dict[str, dict[Any, int]]] = {}


def _zero_counts() -> dict[Any, int]:
    return {"started": 0, **{o: 0 for o in OUTCOMES}}


def register_router() -> int:
    """A new router's serial, registered with zero counts."""
    with _LOCK:
        serial = next(_SERIALS)
        _ROUTERS[serial] = {"entries": {}, "counts": _zero_counts()}
        return serial


def record_start(serial: int) -> None:
    with _LOCK:
        router = _ROUTERS.get(serial)
        if router is not None:
            router["counts"]["started"] += 1


def record_outcome(serial: int, outcome: str | None, served: ServedCall | None) -> None:
    """One dispatch's terminal outcome. Never raises: ``None``, an unknown outcome, a success without its served
    call, or a non-finite temperature (it would break key equality) records nothing, so the dispatch shows as
    started without an outcome (an incomplete window), and the last three are logged."""
    if outcome is None:
        return
    if outcome not in OUTCOMES or (outcome == "success" and (served is None or not math.isfinite(served.temp))):
        logger.warning("call accounting: router %s dispatch %r not recorded (served=%r)", serial, outcome, served)
        return
    with _LOCK:
        router = _ROUTERS.get(serial)
        if router is None:
            return
        router["counts"][outcome] += 1
        if served is not None and outcome == "success":
            k = served.key()
            router["entries"][k] = router["entries"].get(k, 0) + 1


def snapshot() -> dict[int, dict[str, dict[Any, int]]]:
    """A deep copy of every router's counts and entries."""
    with _LOCK:
        return {s: {"entries": dict(r["entries"]), "counts": dict(r["counts"])} for s, r in _ROUTERS.items()}


def delta(start: dict[int, Any], end: dict[int, Any]) -> tuple[dict[int, Any], list[str]]:
    """The calls between two snapshots, and the problems that make it untrustworthy.

    Iterates ``end``'s serials with a zero baseline for any ``start`` lacks (a router built mid-sim is kept),
    and the same for entries within a serial. A serial or entry in ``start`` but missing from ``end``, or any
    negative difference, is a problem: the store only grows. Entries whose difference is 0 are dropped, so a
    model that served only before the sim is not in the result.
    """
    problems: list[str] = []
    out: dict[int, Any] = {}
    for serial in start:
        if serial not in end:
            problems.append(f"router {serial} vanished between snapshots")
    for serial, now in end.items():
        before = start.get(serial, {"entries": {}, "counts": {}})
        counts = {k: now["counts"].get(k, 0) - before["counts"].get(k, 0) for k in now["counts"]}
        for k in before["entries"]:
            if k not in now["entries"]:
                problems.append(f"router {serial}: entry {k!r} vanished between snapshots")
        entries = {}
        for k, n in now["entries"].items():
            d = n - before["entries"].get(k, 0)
            if d < 0:
                problems.append(f"router {serial}: entry {k!r} fell by {-d}")
            elif d > 0:
                entries[k] = d
        if any(v < 0 for v in counts.values()):
            problems.append(f"router {serial}: a count fell between snapshots")
        out[serial] = {"entries": entries, "counts": counts}
    return out, problems


def check_balanced(snap: dict[int, Any], *, only: set[int] | None = None) -> list[str]:
    """Per router (or per router in ``only``): started equals the sum of terminal outcomes, and the entries'
    calls sum to the successes."""
    problems = []
    for serial, r in snap.items():
        if only is not None and serial not in only:
            continue
        counts = r["counts"]
        if counts.get("started", 0) != sum(counts.get(o, 0) for o in OUTCOMES):
            problems.append(f"router {serial}: {counts.get('started', 0)} started, outcomes {counts}")
        if sum(r["entries"].values()) != counts.get("success", 0):
            problems.append(f"router {serial}: entries sum to {sum(r['entries'].values())}, successes {counts}")
    return problems


def window_problems(start: dict[int, Any], end: dict[int, Any], window: dict[int, Any]) -> list[str]:
    """Completeness of a sim's window. Every router with activity in it must balance in the window AND at both
    snapshots: a dispatch in flight at the start and another in flight at the end would otherwise cancel out
    in the window. A router idle in the window is not judged (an older imbalance is not this sim's)."""
    active = {s for s, r in window.items() if any(r["counts"].values())}
    return check_balanced(window, only=active) + check_balanced(start, only=active) + check_balanced(end, only=active)


KEY_FIELDS = tuple(f.name for f in fields(ServedCall))


def entry_dict(key: tuple[Any, ...], n: int) -> dict[str, Any]:
    """A served-call key and its count as a JSON-ready dict."""
    return {**dict(zip(KEY_FIELDS, key)), "n": n}
