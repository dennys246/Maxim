"""Value types for the device registry (#1806).

They live in ``hardware/`` so the runtime and ``embodied_runtime`` can import them without importing the
``maxim.devices`` write verbs (approach note §3b). PR 1a ships :class:`DevicePin`, the typed robots.yaml
``hardware_id:`` field; the resolution and identity types arrive with the PRs that use them.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from typing import Any, ClassVar

# The daemon's ``hardware_id`` (``reachy_mini.utils.hardware_id.get_hardware_id``): 16 lowercase hex characters.
HARDWARE_ID_PATTERN = re.compile(r"[0-9a-f]{16}")

# The literal the writer emits for an entry with no pin (a hand-written legacy file may also omit the key).
UNPINNED_LITERAL = "unpinned"


class _ReadOnlyDict(dict[str, Any]):
    """A dict whose mutators raise, so the shared ``DevicePin.UNPINNED`` cannot leak a write into every entry.

    A ``dict`` subclass rather than ``MappingProxyType`` so ``copy.deepcopy``, ``pickle`` and ``dataclasses.asdict``
    keep working on a ``DevicePin`` and any ``RobotConfig`` holding one (#1806 PR 1a delta review D1). It guards
    against accidental writes only: a deliberate base-class call (``dict.__setitem__(extra, ...)``) still mutates.
    """

    def _refuse(self, *_args: Any, **_kwargs: Any) -> Any:
        raise TypeError("DevicePin.extra is read-only")

    __setitem__ = __delitem__ = __ior__ = _refuse
    clear = pop = popitem = setdefault = update = _refuse

    def __reduce__(self) -> tuple[type, tuple[dict[str, Any]]]:
        return (type(self), (dict(self),))


@dataclass(frozen=True)
class DevicePin:
    """A robots.yaml entry's identity pin: the daemon ``hardware_id`` it must report, or none.

    ``DevicePin.UNPINNED`` is the no-pin sentinel; an entry's pin is never ``None``. An unpinned pin matches
    nothing, so a daemon that reports a null id can never pass as any entry (§4.3 step 1). The id is identity,
    not authentication.

    Forward-compat (CC3): path (a), defaults on every field plus an ``extra`` escape hatch excluded from
    equality and hashing; ``extra`` keys may not collide with field names, and it is read-only.
    """

    hardware_id: str | None = None
    extra: Mapping[str, Any] = field(default_factory=dict, hash=False, compare=False)

    UNPINNED: ClassVar[DevicePin]

    def __post_init__(self) -> None:
        clash = set(self.extra) & {f.name for f in fields(self)}
        if clash:
            raise ValueError(f"DevicePin.extra keys collide with declared fields: {sorted(clash)}")
        # Read-only, so the shared UNPINNED sentinel's extra can never leak a write into every unpinned entry.
        object.__setattr__(self, "extra", _ReadOnlyDict(self.extra))
        if self.hardware_id is not None and (
            not isinstance(self.hardware_id, str) or HARDWARE_ID_PATTERN.fullmatch(self.hardware_id) is None
        ):
            raise ValueError(f"hardware_id must be 16 lowercase hex characters, got {self.hardware_id!r}")

    @property
    def is_pinned(self) -> bool:
        return self.hardware_id is not None

    def matches(self, reported: object) -> bool:
        """True only for a pinned id equal to ``reported``; an unpinned pin matches nothing, ``None`` included."""
        return self.hardware_id is not None and reported == self.hardware_id

    def __str__(self) -> str:
        return self.hardware_id if self.hardware_id is not None else UNPINNED_LITERAL


DevicePin.UNPINNED = DevicePin()
