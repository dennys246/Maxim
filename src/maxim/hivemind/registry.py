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


POLICY_FIELDS = ("allow_unsigned", "inherent_trust", "trusted_sources")


def trust_policy(entry: dict[str, Any]) -> dict[str, Any]:
    """The per-Oasis CONSUMER trust policy, with defaults for older registry files.

    The 1.2 Slice D policy surface — assembled over the shipped ``ingest_bundle``
    hooks, no new merge code:

    - ``allow_unsigned`` (default False) — accept releases this Oasis offers that
      are NOT signed by a registered Queen key. Default trust is Queen-only.
      **This flag disables signature verification for that Oasis's release
      stream** — it admits a tampered ex-Queen release whose signature was
      stripped, not merely "unsigned" content. It is NOT a subscription to the
      server's ``experimental/`` tier: no client function or endpoint reads that
      tier in 1.2 (``hive pull`` reads only ``GET /v1/substrate/releases``).
      The flag is reachable because ``list_releases`` serves whatever sits in a
      release directory, so a lenient or compromised Oasis can offer unsigned
      content even though our own ``oasis publish`` refuses to create it.
    - ``inherent_trust`` (default False) — admit the decay-exempt inherent
      ("safety floor") bias class. Applied ONLY to Queen-verified releases and
      only for the contributor ids this policy already trusts.
    - ``trusted_sources`` (default empty = any contributor the Queen signed) —
      an operator ``contributor_id`` allow-list. When non-empty it is passed to
      ingest as the V1 ``trusted_sources`` set, so the refusal is enforced by the
      pipeline and not merely by this CLI.

    Malformed policy values FAIL LOUD rather than coercing: ``bool("false")`` is
    ``True``, so a hand-edited or template-generated ``"allow_unsigned": "false"``
    would silently invert a safety default. A registry that cannot be read as
    written is an operator problem, not something to guess at.
    """
    policy: dict[str, Any] = {}
    for field in ("allow_unsigned", "inherent_trust"):
        value = entry.get(field, False)
        if not isinstance(value, bool):
            raise HiveRegistryError(
                f"registry field {field!r} must be a JSON boolean (true/false), got {value!r} — "
                "refusing to guess at a trust setting"
            )
        policy[field] = value
    sources = entry.get("trusted_sources", [])
    if sources is None:
        sources = []
    if not isinstance(sources, list) or any(not isinstance(s, str) or not s for s in sources):
        raise HiveRegistryError(
            f"registry field 'trusted_sources' must be a list of non-empty strings, got {sources!r}"
        )
    policy["trusted_sources"] = list(sources)
    return policy


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
        """Add (or update, by name) an Oasis. Operator-explicit write only.

        Re-adding an existing name MERGES onto the existing entry: the trust
        policy, any Queen keys not being replaced, and any field a future version
        wrote are preserved. Silently resetting a trust grant (or the Queen keys
        that make Queen-only verification possible) because the operator
        corrected a URL would be a footgun — and dropping the keys while an
        ``allow_unsigned`` grant survives would silently degrade the posture from
        "Queen-only + escape hatch" to "admit anything". Change policy with
        :meth:`set_trust`.

        Returns the stored entry; ``warnings`` on the returned dict is not
        persisted — callers surface it. A URL change while a loosening grant is
        active is worth telling the operator about: the remote identity moved.
        """
        keys = dict(queen_keys or {})
        _validate_add(name, url, keys)
        oases = self._load()
        existing = next((o for o in oases if o.get("name") == name), None)
        entry: dict[str, Any] = dict(existing or {})
        entry.update({"name": name, "url": url})
        if keys:
            entry["queen_keys"] = keys
        else:
            entry.setdefault("queen_keys", {})
        if domains:
            entry["domains"] = list(domains)
        else:
            entry.setdefault("domains", [])
        # Normalize/validate the policy that is carried forward.
        entry.update(trust_policy(entry))
        oases = [o for o in oases if o.get("name") != name]
        oases.append(entry)
        self._save(oases)
        return entry

    def set_trust(
        self,
        name: str,
        *,
        allow_unsigned: bool | None = None,
        inherent_trust: bool | None = None,
        trusted_sources: list[str] | None = None,
    ) -> dict[str, Any]:
        """Set the consumer trust policy for a registered Oasis (operator-explicit).

        Only the fields passed are changed; ``None`` leaves a field alone.
        """
        oases = self._load()
        entry = next((o for o in oases if o.get("name") == name), None)
        if entry is None:
            raise HiveRegistryError(f"no registered oasis named {name!r} (see `maxim hive list`)")
        policy = trust_policy(entry)
        if allow_unsigned is not None:
            policy["allow_unsigned"] = bool(allow_unsigned)
        if inherent_trust is not None:
            policy["inherent_trust"] = bool(inherent_trust)
        if trusted_sources is not None:
            for source in trusted_sources:
                if not source or not isinstance(source, str):
                    raise HiveRegistryError(f"trusted source must be a non-empty string, got {source!r}")
            policy["trusted_sources"] = list(trusted_sources)
        entry.update(policy)
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
