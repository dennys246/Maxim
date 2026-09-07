"""Static Oasis registry — the known-Oases list backing ``maxim hive`` (1.2 P2P Slice C).

A small operator-owned JSON file at ``~/.config/maxim/hive.json`` naming the Oases
this node knows about: for each, a URL, the Queen public keys that sign its releases
(the trust anchors — public, not secret), and the substrate domains subscribed to.

Layer-1 discipline (persistence-config brief): this is DECLARATIVE operator intent,
written ONLY by the operator-explicit ``maxim hive add`` / ``hive remove`` verbs —
never by runtime code as a side effect. It is deliberately a JSON file (not a
hand-rolled YAML dialect like ``mesh.yml``, whose parser is frozen and must not be
extended) written through the canonical ``atomic_write_json`` + ``with_format_version``,
and it holds NO secrets — Queen keys are public verification anchors, and the bearer
token used to talk to a remote Oasis is supplied per-invocation, never stored here.

Entries are plain dicts (shape ``{name, url, queen_keys: {identity: pubkey_b64}, domains:
[...]}``), validated defensively on read; a present-but-corrupt registry fails loud
rather than being silently reset.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from maxim.tunnel.keys import key_file_path
from maxim.utils.atomic_io import atomic_write_json
from maxim.utils.format_version import check_format_version, with_format_version

logger = logging.getLogger(__name__)

_REGISTRY_FILE_TYPE = "oasis_registry"


class HiveRegistryError(Exception):
    """A registry operation was refused (bad input, or a corrupt registry file)."""


def registry_path() -> Path:
    """The ``~/.config/maxim/hive.json`` path (XDG/Windows-aware, beside the mesh key)."""
    return key_file_path("hive.json")


def _validate_add(name: str, url: str, queen_keys: dict[str, str]) -> None:
    if not name or not isinstance(name, str):
        raise HiveRegistryError("oasis name must be a non-empty string")
    if not isinstance(url, str) or not (url.startswith("http://") or url.startswith("https://")):
        raise HiveRegistryError(f"oasis url must be an http(s) URL, got {url!r}")
    for identity, pubkey in queen_keys.items():
        if not identity or not isinstance(identity, str) or not isinstance(pubkey, str) or not pubkey:
            raise HiveRegistryError(f"queen key must be IDENTITY=PUBKEY_B64, got {identity!r}={pubkey!r}")


class HiveRegistry:
    """Read/write the ``hive.json`` Oasis registry."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else registry_path()

    def _load(self) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        # A present-but-unreadable registry is NOT treated as empty (that would let
        # the next add overwrite the operator's whole known-Oases list). Fail loud.
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise HiveRegistryError(
                f"registry {self.path} is present but unreadable ({exc}); refusing to overwrite it — "
                "inspect/repair the file by hand"
            ) from exc
        if not isinstance(data, dict) or not isinstance(data.get("oases", []), list):
            raise HiveRegistryError(
                f"registry {self.path} is malformed (expected an object with an 'oases' list); "
                "refusing to overwrite it — inspect/repair the file by hand"
            )
        check_format_version(data, _REGISTRY_FILE_TYPE, log=logger)
        return [o for o in data.get("oases", []) if isinstance(o, dict)]

    def _save(self, oases: list[dict[str, Any]]) -> None:
        atomic_write_json(str(self.path), with_format_version({"oases": oases}))

    def list_oases(self) -> list[dict[str, Any]]:
        return self._load()

    def get(self, name: str) -> dict[str, Any] | None:
        return next((o for o in self._load() if o.get("name") == name), None)

    def add(
        self,
        name: str,
        url: str,
        *,
        queen_keys: dict[str, str] | None = None,
        domains: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        """Add (or replace, by name) an Oasis. Operator-explicit write only."""
        keys = dict(queen_keys or {})
        _validate_add(name, url, keys)
        entry = {"name": name, "url": url, "queen_keys": keys, "domains": list(domains)}
        oases = [o for o in self._load() if o.get("name") != name]
        oases.append(entry)
        self._save(oases)
        return entry

    def remove(self, name: str) -> bool:
        """Remove an Oasis by name; returns True if one was removed."""
        oases = self._load()
        kept = [o for o in oases if o.get("name") != name]
        if len(kept) == len(oases):
            return False
        self._save(kept)
        return True
