"""What a memory was encoded WITH: the importance signals at capture (memory-strength Phase 2b).

The plan's encoding tag (``S0 = s_base * (1 + k * tag)``, a noisy-OR over each signal's deviation
from its own baseline) needs the signals that were actually present when a trace formed. This
module is the record of them -- nothing more. The tag itself, its baselines and its constants belong
to the Phase 2c strength strategy, which reads these values; here they are only captured.

Every signal is REQUIRED and ``float | None``: a capture site must say, per signal, either what it
measured or that it measured nothing. ``None`` is "no measurement here" and is never read as a
signal; a constant a code path stamps on every capture (the 0.5 salience default, a hard-coded
novelty) is not a measurement and is passed as ``None``. ``unmeasured(site)`` is the explicit
all-``None`` declaration, so a site that has nothing still says so out loud.

``site`` names which capture site produced the trace (``ENCODING_SITES``, closed): one failing tool
call can produce several traces (the loop's, a reflection's, a pain capture's), and Phase 2c must
count one event once.

``drive_pressure`` and ``drive_relief`` are PER DRIVE (Phase 2b-ii), each a sorted tuple of
``(drive, value)`` -- a mapping would be mutable inside a frozen record. Pressure is what the body
was pushing for when the action was taken; relief is how much of what each drive COULD give the
action actually gave. Their keys are also what Phase 2c gates on: a starving stretch must tag the
traces whose actions touched hunger, not everything that happened while hungry.

CC3: frozen and persisted on the episode, path (a) with REQUIRED fields by design -- the required
fields are the ``TypeError`` that makes a forgotten signal loud. Forward-compat lives in the loader:
``from_dict`` tolerates missing keys (``None``) and keeps unknown ones in ``extra`` (JSON values
only, never colliding with a declared field), written back on save.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any

_SIGNALS = ("salience", "novelty", "surprise", "pain")
_PER_DRIVE = ("drive_pressure", "drive_relief")

# Which capture site produced a trace. Closed: a typo cannot open a silent new bucket.
ENCODING_SITES: frozenset[str] = frozenset(
    {
        "loop",  # the agent loop's per-action capture (runtime/bio_integration.py)
        "memory_agent",  # MemoryAgent percept / tool-result / goal captures
        "pain_bus",  # the pain-memory subscriber
        "reflexion",  # a verbal self-critique after a surprising failure
        "engram",  # a cerebellar motor engram (Dormant)
        "observation",  # Hippocampus.store_observation (NPC turns)
        "api",  # a direct call: the public API, scripts, tests
    }
)


class EncodingContractError(TypeError):
    """A capture did not say what it was encoded with (or said it in the wrong type).

    A contract break, never a runtime hiccup: capture paths that swallow ordinary failures re-raise
    exactly this, and nothing else they would not have raised before.
    """


@dataclass(frozen=True, slots=True)
class EncodingSignals:
    """The importance signals present when one trace was captured (all required; ``None`` = none)."""

    site: str  # which capture site (ENCODING_SITES)
    salience: float | None  # percept salience, when a producer MEASURED it
    novelty: float | None  # 1 - familiarity, when measured against a reference set
    surprise: float | None  # |RPE| of the captured invocation's outcome (``ToolOutput.rpe``)
    pain: float | None  # NOCICEPTIVE pain intensity carried by this capture
    drive_pressure: tuple[tuple[str, float], ...] | None  # what the body was pushing for, BEFORE
    drive_relief: tuple[tuple[str, float], ...] | None  # how much of each drive's possible relief it gave
    extra: dict[str, Any] = field(default_factory=dict, compare=False, hash=False)

    def __post_init__(self) -> None:
        if self.site not in ENCODING_SITES:
            raise ValueError(f"unknown encoding site {self.site!r}; expected one of {sorted(ENCODING_SITES)}")
        for name in _SIGNALS:
            value = getattr(self, name)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"encoding signal {name!r} must be a float or None, got {value!r}")
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"encoding signal {name!r} must be in [0, 1], got {value!r}")
            object.__setattr__(self, name, float(value))
        for name in _PER_DRIVE:
            pairs = getattr(self, name)
            if pairs is None:
                continue
            if not isinstance(pairs, tuple) or any(not isinstance(p, tuple) or len(p) != 2 for p in pairs):
                raise TypeError(f"{name!r} must be a tuple of (drive, value) pairs or None, got {pairs!r}")
            drives = [str(d) for d, _ in pairs]
            if drives != sorted(drives) or len(set(drives)) != len(drives):
                raise ValueError(f"{name!r} must name each drive once, sorted: {drives}")
            checked: list[tuple[str, float]] = []
            for drive, value in pairs:
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise TypeError(f"{name}[{drive!r}] must be a float, got {value!r}")
                if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                    raise ValueError(f"{name}[{drive!r}] must be in [0, 1], got {value!r}")
                checked.append((str(drive), float(value)))
            object.__setattr__(self, name, tuple(checked))
        collisions = set(self.extra) & {"site", *_SIGNALS, *_PER_DRIVE}
        if collisions:
            raise ValueError(f"extra keys collide with declared fields: {sorted(collisions)}")
        try:
            json.dumps(self.extra)
        except (TypeError, ValueError) as e:
            raise ValueError(f"extra must hold JSON values only: {e}") from None
        object.__setattr__(self, "extra", dict(self.extra))

    @classmethod
    def unmeasured(cls, site: str) -> EncodingSignals:
        """The explicit declaration that this capture site measured no importance signal."""
        return cls(
            site=site, salience=None, novelty=None, surprise=None, pain=None, drive_pressure=None, drive_relief=None
        )

    def measured(self) -> tuple[str, ...]:
        """Names of the signals this capture actually measured."""
        return tuple(name for name in (*_SIGNALS, *_PER_DRIVE) if getattr(self, name) is not None)

    def to_dict(self) -> dict[str, Any]:
        per_drive = {
            name: (dict(getattr(self, name)) if getattr(self, name) is not None else None) for name in _PER_DRIVE
        }
        return {
            "site": self.site,
            **{name: getattr(self, name) for name in _SIGNALS},
            **per_drive,
            **self.extra,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EncodingSignals:
        known = {name: data.get(name) for name in _SIGNALS}
        for name in _PER_DRIVE:
            raw = data.get(name)
            known[name] = tuple(sorted((str(k), v) for k, v in raw.items())) if isinstance(raw, dict) else raw
        extra = {k: v for k, v in data.items() if k not in _SIGNALS and k not in _PER_DRIVE and k != "site"}
        # A record without its site is malformed, not "api": a guess would misattribute the trace.
        return cls(site=data["site"] if "site" in data else "", **known, extra=extra)


def require_encoding(encoding: Any) -> EncodingSignals:
    """The capture doors' check: exactly an ``EncodingSignals``, or ``EncodingContractError``."""
    if not isinstance(encoding, EncodingSignals):
        raise EncodingContractError(f"encoding must be EncodingSignals, got {type(encoding).__name__}")
    return encoding


__all__ = ["ENCODING_SITES", "EncodingContractError", "EncodingSignals", "require_encoding"]
