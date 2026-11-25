"""One probe of a Reachy Mini daemon, REST/TCP only (#1806 PR 0).

``probe_daemon`` is the single copy of the probe that ``maxim doctor`` (``doctor/checks.py::_check_one_robot``),
``maxim-diagnostics`` (``utils/reachy_diagnostics.py``) and ``ReachyMiniController.connect``'s pre-checks used to
carry each on their own. Each consumer keeps its own policy over the one :class:`ProbeOutcome`: doctor grades it
into warn/fail rows, the controller treats a failed status read as non-fatal (and takes the #1704 pre-connect
``media_released`` from it), and :func:`refusal` is the one strict reading.

The sequence mirrors the hardware-validated one (``docs/embodiment/reachy_mini/troubleshooting.md``):

1. **name resolution** -- an IPv4 literal is used as is; a name is resolved ONCE to IPv4 (``gethostbyname``, which
   sidesteps the getaddrinfo IPv6-first flake), and every later stage dials that literal;
2. **TCP :8000** -- the daemon's single WS-era control port (FastAPI + ``/ws/sdk``);
3. **``GET /api/daemon/status``** -- through ``utils/http.py::fetch_url(address_policy="private", proxy="never")``,
   so the robot is dialled directly (never through a proxy that could answer for a robot it never reached) and a
   public address is refused on the address actually dialled (#824/#921); ``deadline_s`` bounds the whole request
   (#1715);
4. **SDK/daemon era + minor-version match** against the locally installed ``reachy-mini`` package metadata;
5. **daemon ``state``**.

Nothing here opens ``ReachyMini(``, starts a tunnel or sends a command: the probe never actuates.

The status fields carried for later PRs (``hardware_id``, ``wlan_ip``, the sim flags, ``robot_name``, ``version``)
are UNTRUSTED daemon text: they are type-checked and bounded (:func:`_bounded`) before they are stored, and nothing
here treats them as identity. Identity checks are #1806 PR 1, which extends :func:`refusal`.
"""

from __future__ import annotations

import ipaddress
import socket
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any, Literal

from maxim.exceptions import MaximConnectionError

if TYPE_CHECKING:
    from maxim.hardware.config import RobotConfig

DAEMON_PORT = 8000
STATUS_PATH = "/api/daemon/status"
_MAX_FIELD_LEN = 128

ProbeStage = Literal["unsupported", "resolve", "tcp", "status"]
EraVerdict = Literal["match", "minor_drift", "era_mismatch", "sdk_missing", "unknown"]


@dataclass(frozen=True)
class ProbeOutcome:
    """What one :func:`probe_daemon` call found. Runtime-only, never persisted today.

    CC3 path (a) (docs/agents/persistence-config.md §3): every field is defaulted and ``extra`` is the escape hatch,
    because #1806 PR 1 carries this outcome into the device registry's preflight and the controller's step 1, and
    later PRs add fields. ``__post_init__`` refuses an ``extra`` key that collides with a declared field.

    ``failure`` is the first stage that failed (``None``: the status was read). ``address`` is the IPv4 literal the
    TCP and status stages dialled (``None`` when resolution failed). ``error`` is a bounded, printable description
    of the failure. The status fields are ``None`` unless the daemon reported a value of the expected JSON type.
    """

    target: str = ""
    port: int = DAEMON_PORT
    address: str | None = None
    failure: ProbeStage | None = None
    error: str | None = None
    # GET /api/daemon/status, type-checked and bounded (untrusted daemon text)
    version: str | None = None
    state: str | None = None
    media_released: bool | None = None
    hardware_id: str | None = None
    wlan_ip: str | None = None
    simulation_enabled: bool | None = None
    mockup_sim_enabled: bool | None = None
    robot_name: str | None = None
    # era / minor-version match: (major, minor) of the local SDK and the daemon
    sdk_version: tuple[int, int] | None = None
    daemon_version: tuple[int, int] | None = None
    era: EraVerdict = "unknown"
    extra: dict[str, Any] = field(default_factory=dict, hash=False, compare=False)

    def __post_init__(self) -> None:
        clash = set(self.extra) & {f.name for f in fields(self)}
        if clash:
            raise ValueError(f"ProbeOutcome.extra keys collide with declared fields: {sorted(clash)}")

    @property
    def reachable(self) -> bool:
        """The daemon port accepted a TCP connection (the status may still be unreadable)."""
        return self.failure is None or self.failure == "status"

    @property
    def status_read(self) -> bool:
        return self.failure is None


class DaemonStatusUnreadable(MaximConnectionError):
    """``GET /api/daemon/status`` gave no JSON object (a pre-1.5 daemon, a moved endpoint, a refused address).

    Under the stable ``maxim.exceptions.MaximConnectionError`` (review A7), so a caller catching connection
    failures catches it too.
    """


# ─── the seams (tests patch these module attributes; production calls them through the module) ───


def parse_major_minor(version: str) -> tuple[int, int] | None:
    """``"1.8.3"`` -> ``(1, 8)``; ``None`` for anything unparseable."""
    try:
        parts = version.split(".")
        return (int(parts[0]), int(parts[1]))
    except (ValueError, IndexError, AttributeError):
        return None


def installed_sdk_version() -> tuple[int, int] | None:
    """(major, minor) of the locally installed reachy-mini SDK, or ``None`` when it is not installed.

    Package metadata, not ``__version__``: pre-1.5 (zenoh-era) SDKs have no ``__version__`` attribute, so metadata
    is the only probe that works across the v1.5.0 pivot.
    """
    from importlib.metadata import PackageNotFoundError, version

    try:
        return parse_major_minor(version("reachy-mini"))
    except PackageNotFoundError:
        return None


_IPV6_NOT_PROBED = "IPv6 host: not probed (the daemon probe is IPv4 only)"


def _is_ipv6_literal(name: str) -> bool:
    try:
        ipaddress.IPv6Address(name.strip("[]"))
    except ValueError:
        return False
    return True


def _has_ipv6_address(name: str) -> bool:
    """Whether ``name`` resolves to an IPv6 address (it failed the IPv4 lookup): an AAAA-only host."""
    try:
        return bool(socket.getaddrinfo(name, None, socket.AF_INET6))
    except OSError:
        return False


def resolve_ipv4(name: str, *, timeout: float) -> str:
    """``name`` as one canonical IPv4 literal: an IPv4 literal is returned as is, a name is resolved once.

    Raises ``OSError`` (``socket.gaierror``) when the name does not resolve.
    """
    try:
        return str(ipaddress.IPv4Address(name))
    except ValueError:
        pass  # not an IPv4 literal: resolve the name below
    old_timeout = socket.getdefaulttimeout()
    socket.setdefaulttimeout(timeout)
    try:
        return socket.gethostbyname(name)
    finally:
        socket.setdefaulttimeout(old_timeout)


def tcp_connect_error(address: str, port: int, *, timeout: float) -> str | None:
    """``None`` when ``address:port`` accepts a TCP connection, else a short description of why not."""
    try:
        with socket.create_connection((address, port), timeout=timeout):
            return None
    except socket.timeout:
        return f"connection timeout after {timeout:.1f}s"
    except ConnectionRefusedError:
        return "connection refused"
    except OSError as e:
        return _bounded(f"{type(e).__name__}: {e}") or type(e).__name__


def read_daemon_status(address: str, port: int, *, deadline_s: float) -> dict[str, Any]:
    """The daemon's ``/api/daemon/status`` JSON object, read directly (never through a proxy) from a private address.

    Raises :class:`DaemonStatusUnreadable` for a transport failure, a non-2xx status, a body that is not JSON, or
    JSON that is not an object.
    """
    from maxim.utils import http as maxim_http

    url = f"http://{address}:{port}{STATUS_PATH}"
    try:
        body = maxim_http.fetch_url(
            url, timeout=deadline_s, deadline_s=deadline_s, address_policy="private", proxy="never"
        ).json()
    except (maxim_http.HTTPError, ValueError) as e:
        raise DaemonStatusUnreadable(_bounded(f"{type(e).__name__}: {e}") or type(e).__name__) from e
    if not isinstance(body, dict):
        raise DaemonStatusUnreadable(f"status body is a JSON {type(body).__name__}, not an object")
    return body


# ─── the probe ───


def _bounded(value: Any) -> str | None:
    """A daemon-reported string, made safe to store and log: printable only, at most ``_MAX_FIELD_LEN`` chars."""
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        return None
    text = "".join(ch if ch.isprintable() else "?" for ch in str(value))[:_MAX_FIELD_LEN]
    return text or None


def _flag(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def _era(sdk: tuple[int, int] | None, daemon: tuple[int, int] | None) -> EraVerdict:
    """The v1.5.0 zenoh->WebSocket pivot rule doctor has always applied, now in one place."""
    if sdk is None:
        return "sdk_missing"
    if daemon is None:
        return "unknown"
    if (sdk < (1, 5)) != (daemon < (1, 5)) or sdk < (1, 5):
        return "era_mismatch"
    if daemon != sdk:
        return "minor_drift"
    return "match"


def probe_daemon(host_or_ip: str, *, port: int = DAEMON_PORT, deadline_s: float) -> ProbeOutcome:
    """Probe the Reachy daemon at ``host_or_ip``: resolve -> TCP ``port`` -> status -> era -> state.

    ``deadline_s`` bounds EACH network stage on its own (the lookup, the TCP connect, the status read), as the
    three copies this replaced did: one shared budget let a slow lookup starve the later stages, so a slow
    resolver read as "daemon port unreachable" (review E2). A name lookup itself is not interruptible, the same
    residual ``fetch_url`` states for DNS. The probe is IPv4 only: an IPv6 literal, or a name with only an IPv6
    address, is ``failure="unsupported"`` (never reported as "does not resolve"; review E1). Never raises for a
    network outcome: every failure is recorded as ``ProbeOutcome.failure`` + ``error``.
    """
    if not (isinstance(deadline_s, (int, float)) and deadline_s > 0):
        raise ValueError(f"deadline_s must be a positive number of seconds, got {deadline_s!r}")
    target = str(host_or_ip)
    sdk = installed_sdk_version()
    if _is_ipv6_literal(target):
        return ProbeOutcome(target=target, port=port, failure="unsupported", error=_IPV6_NOT_PROBED)
    try:
        address = resolve_ipv4(target, timeout=deadline_s)
    except OSError as e:
        if _has_ipv6_address(target):
            return ProbeOutcome(target=target, port=port, failure="unsupported", error=_IPV6_NOT_PROBED)
        return ProbeOutcome(
            target=target, port=port, failure="resolve", error=_bounded(f"{target} does not resolve: {e}")
        )
    tcp_error = tcp_connect_error(address, port, timeout=deadline_s)
    if tcp_error is not None:
        return ProbeOutcome(target=target, port=port, address=address, failure="tcp", error=tcp_error)
    try:
        status = read_daemon_status(address, port, deadline_s=deadline_s)
    except DaemonStatusUnreadable as e:
        return ProbeOutcome(target=target, port=port, address=address, failure="status", error=str(e), sdk_version=sdk)
    version = _bounded(status.get("version"))
    daemon_mm = parse_major_minor(version) if version else None
    return ProbeOutcome(
        target=target,
        port=port,
        address=address,
        version=version,
        state=_bounded(status.get("state")),
        media_released=_flag(status.get("media_released")),
        hardware_id=_bounded(status.get("hardware_id")),
        wlan_ip=_bounded(status.get("wlan_ip")),
        simulation_enabled=_flag(status.get("simulation_enabled")),
        mockup_sim_enabled=_flag(status.get("mockup_sim_enabled")),
        robot_name=_bounded(status.get("robot_name")),
        sdk_version=sdk,
        daemon_version=daemon_mm,
        era=_era(sdk, daemon_mm),
    )


def refusal(outcome: ProbeOutcome, entry: RobotConfig | None) -> str | None:
    """Why a strict consumer must NOT go on to connect to this daemon, or ``None`` when nothing refuses.

    PR 0 refuses: an unresolvable name or a closed port (unreachable), an unreadable status, a missing local SDK,
    an era mismatch or minor-version drift, an unreadable daemon version, and a daemon ``state`` other than
    ``running`` (WS connects are refused with 1013 then). ``maxim doctor`` reports ok only when this is ``None``.

    Extension point (#1806 PR 1): the identity checks against ``entry``'s pin (a null ``hardware_id`` never
    matches), the sim/mockup flags for a hardware-pinned entry, and ``wlan_ip`` belong HERE, after the state check,
    so every strict consumer (the ``embody`` preflight, the controller's step 1) gets them from one function. PR 0
    reads nothing from ``entry``: it has no pin to compare yet.
    """
    if outcome.failure == "unsupported":
        return f"not probed: {outcome.error}"
    if outcome.failure == "resolve":
        return f"unreachable: {outcome.error or outcome.target + ' does not resolve'}"
    if outcome.failure == "tcp":
        return f"unreachable: daemon port {outcome.address}:{outcome.port} ({outcome.error or 'closed'})"
    if outcome.failure == "status":
        return f"daemon status unreadable at {outcome.address}:{outcome.port} ({outcome.error or 'no detail'})"
    if outcome.era == "sdk_missing":
        return "no local reachy-mini SDK installed"
    sdk = f"{outcome.sdk_version[0]}.{outcome.sdk_version[1]}" if outcome.sdk_version else "?"
    if outcome.era == "era_mismatch":
        return f"era mismatch: daemon {outcome.version} vs local SDK {sdk}.x (the v1.5.0 zenoh->WebSocket pivot)"
    if outcome.era == "minor_drift":
        return f"minor-version mismatch: daemon {outcome.version} vs local SDK {sdk}.x"
    if outcome.era == "unknown":
        return f"daemon version unreadable ({outcome.version!r})"
    if outcome.state != "running":
        return f"daemon state is {outcome.state!r}, not 'running'"
    return None


__all__ = [
    "DAEMON_PORT",
    "DaemonStatusUnreadable",
    "ProbeOutcome",
    "installed_sdk_version",
    "parse_major_minor",
    "probe_daemon",
    "read_daemon_status",
    "refusal",
    "resolve_ipv4",
    "tcp_connect_error",
]
