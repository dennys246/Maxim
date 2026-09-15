"""Shared helpers for the survival-world scripts (instrument check, Phase-1 harness).

Extracted at the three-copies threshold: `break3_smoke._food`, `r2_learned_bias._food`
and the instrument check's light read were near-identical sync-then-read helpers, and
the Phase-1 harness would have made a fourth. The retired/dormant siblings keep their
local copies (their code is frozen apparatus); new survival scripts use these.
"""

from __future__ import annotations

import re
import time
from typing import Any, Callable


class InstrumentError(RuntimeError):
    """The apparatus (bridge/RCON/world), not the thing measured, failed."""


def sync_snapshot(aut: Any) -> dict[str, Any] | None:
    """Sync world truth into the body and return a COPY of vital_metrics.

    Returns None when the sync wrote nothing (no bridge state yet, or the
    bridge died and ``latest_state()`` is empty) — a None here means "no fresh
    truth", never "sensor reads zero", so callers can separate instrument
    failure from measurement.
    """
    if aut.backend.sync_world_sensors() <= 0:
        return None
    vm = getattr(aut.executor.embodiment.root, "vital_metrics", {}) or {}
    return dict(vm)


def read_vital(aut: Any, key: str) -> float | None:
    """Sync and read ONE vital metric as float; None on no-sync or absent key."""
    vm = sync_snapshot(aut)
    if vm is None or key not in vm:
        return None
    try:
        return float(vm[key])
    except (TypeError, ValueError):
        return None


def settle_until(
    aut: Any,
    predicate: Callable[[dict[str, Any]], bool],
    *,
    timeout_s: float,
    poll_s: float = 0.25,
) -> dict[str, Any] | None:
    """Poll fresh snapshots until ``predicate(vital_metrics)`` holds.

    The settle-until-reflected pattern (exp56 ``common.settle_until_reflected``,
    born from a measured stale-snapshot mis-read): an RCON command takes a
    server tick to apply and the bridge snapshot interval to surface, so a
    blind sleep-then-read can sample the PREVIOUS condition. Returns the first
    snapshot satisfying the predicate, or None on timeout (no-sync polls count
    as not-yet-settled, not as data).
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        vm = sync_snapshot(aut)
        if vm is not None and predicate(vm):
            return vm
        time.sleep(poll_s)
    return None


_POS_TOKEN = re.compile(r"(-?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)d")


def bot_pos(rcon: Any, username: str) -> tuple[float, float, float]:
    """Absolute bot position via RCON ``data get entity <bot> Pos``.

    Matches full Java doubles INCLUDING scientific notation — Double.toString
    prints e.g. ``5.0E-5d`` near zero, and a digits-only pattern would silently
    extract ``-5`` from it (a corrupt anchor that teleports the bot to the
    wrong place while every later number looks plausible).
    """
    resp = rcon.command(f"data get entity {username} Pos")
    nums = _POS_TOKEN.findall(resp)
    if len(nums) != 3:
        raise InstrumentError(f"could not parse bot Pos from RCON reply: {resp!r}")
    x, y, z = (float(n) for n in nums)
    return x, y, z


def build_dark_box(rcon: Any, cx: int, y0: int, cz: int) -> None:
    """Build a roofed 5x5x5 stone shell (interior air) centred on (cx, cz).

    The fill reply is CHECKED: an unloaded chunk or failed fill would otherwise
    leave the "dark" teleport target as open air at altitude (the bot falls),
    while the check happily reads sky light.
    """
    resp = rcon.command(f"fill {cx - 2} {y0} {cz - 2} {cx + 2} {y0 + 4} {cz + 2} minecraft:stone hollow")
    # "No blocks were filled" CONTAINS "filled" — test the failure reply explicitly
    # (a vacuous guard; caught by the Exp 60 chunk-i executor lens).
    if "filled" not in resp.lower() or "no blocks were filled" in resp.lower():
        raise InstrumentError(f"dark-box fill did not confirm: {resp!r}")


def remove_dark_box(rcon: Any, cx: int, y0: int, cz: int) -> None:
    """Remove the shell built by :func:`build_dark_box` (same coords)."""
    rcon.command(f"fill {cx - 2} {y0} {cz - 2} {cx + 2} {y0 + 4} {cz + 2} minecraft:air")


def make_fresh_encoder(aut: Any) -> Any:
    """The production encoder wiring (ec + atl + nac), fresh config.

    Production (``agent_loop``) builds ``SensorEncoder(ec, atl=..., nac=...)`` —
    the NAc supplies threshold overrides during encode and the ATL receives
    activation side effects. Passing both keeps "the same path production uses"
    literally true; on a fresh substrate the NAc is empty so overrides are nil.
    """
    from maxim.similarity.encoder import SensorEncoder, SensorEncoderConfig

    return SensorEncoder(ec=aut.bio.ec, atl=aut.bio.atl, nac=aut.bio.nac, config=SensorEncoderConfig())
