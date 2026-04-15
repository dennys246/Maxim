"""Shared install core for ``maxim peer install`` verbs.

Plan 4 C3.3 (fold commit). Originally `_install_on_target` lived in
:mod:`maxim.peer.cli` alongside ``_cmd_install``. The architecture-lens
pre-merge review flagged that the mesh-aware ``maxim peer --node <name>
install`` verb's lazy import from ``peer/cli.py`` inverted the coupling
direction (mesh-specific module depending on the general peer verb
dispatcher) and that the same pattern would repeat three more times
across the C3.5/C3.6 verbs. The fix was to move the shared HTTP body
into its own leaf module so:

- :mod:`maxim.peer.cli` imports from here
- :mod:`maxim.peer.mesh_cli` imports from here
- No circular dependency between the two CLI dispatcher modules

This module is the **single source of truth** for:

- ``POST <base>/v1/admin/install`` wire-level shape (endpoint, headers,
  JSON body, timeout policy)
- ``GET <base>/v1/debug/install-status`` polling endpoint shape
- The :data:`KNOWN_EXTRAS` classifier set
- The 401/403/404/generic error classification + exit code contract
- The :data:`_INSTALL_POLL_DEADLINE_S` polling deadline constant
- The :data:`_INSTALL_POST_TIMEOUT` and :data:`_INSTALL_POLL_TIMEOUT`
  timeout policies

A CI grep in ``.github/workflows/test.yml`` enforces that
``/v1/admin/install`` appears only in this file + its test file. Any
new caller MUST import :func:`install_on_target` from here — do NOT
inline a copy. This matches the precedent set by
``_MaximPeerBackend.complete_with_usage`` (Plan 3 R2.5) and
``write_mesh_config`` (Plan 4 C3.1).
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


# Extras that map to pymaxim[extra_name] in pyproject.toml. Keeping this
# as the single source of truth for the classifier — the parametrized
# test `TestKnownExtrasClassification` in test_peer_install.py asserts
# every element of this set round-trips through classify_install_tokens
# into the extras list, so adding a new pyproject extra without
# updating this set would fail CI.
KNOWN_EXTRAS: frozenset[str] = frozenset(
    {
        "semantic",
        "llm-llama",
        "llm-server",
        "llm-torch",
        "llm-anthropic",
        "llm-openai",
        "vision",
        "audio",
        "reachy",
        "comms",
        "search",
        "temporal",
        "training",
        "tts",
        "yolo",
        "database",
    }
)


# Polling deadline: the leader's /v1/admin/install can take up to 10
# minutes for a large extra with heavy C-extension builds (torch,
# scipy). Named constant so C3.5's --node update / --node restart can
# reuse it without copying the magic number.
_INSTALL_POLL_DEADLINE_S: float = 600.0


def _normalize_url_for_probe_cache(url: str) -> str:
    """Canonical form used when clearing the probe cache.

    The probe cache in :mod:`maxim.runtime.probe_cache` is keyed by
    whatever ``lane_configs[name].remote_url`` happened to hold when
    the cache was written — no normalization on write. When the mesh
    verb calls :func:`_clear_probe_cache` with a URL resolved from
    ``mesh.yml::nodes``, the shape may not match the cache key byte-
    for-byte (trailing slash, ``/v1`` suffix, etc.), and the clear
    silently becomes a no-op → next startup trusts a stale entry.

    This normalizer strips trailing slashes and a single trailing
    ``/v1`` so the clear path can do a tolerant match. The actual
    tolerant-match logic lives in :func:`probe_cache.clear_cache_for_url`
    — this function is exported here so callers can build the same
    canonical form without importing the runtime package.
    """
    if not url:
        return ""
    base = url.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    return base


def _looks_like_url(token: str) -> bool:
    """True if ``token`` is a URL rather than a package/extra name.

    Pre-C3.3 used ``token.startswith("http")`` which false-positives
    on real PyPI packages: ``httpx``, ``http-client``, ``httpie``. The
    C3.3 pre-merge review caught this (Executor #2, cross-reviewed
    against the cli.py arg parser too). Fix: require ``"://"`` in the
    token, matching the standard scheme-authority URL shape.
    """
    return "://" in token


def classify_install_tokens(
    raw_tokens: list[str],
) -> tuple[list[str], list[str]]:
    """Split caller-supplied install tokens into known extras + raw pip
    packages.

    Plan 4 C3.3: factored out of ``_cmd_install`` so the mesh-aware
    ``maxim peer --node <name> install <extras>`` verb produces the
    same wire-level body shape from a different argv parser. Empty /
    whitespace-only tokens are dropped silently — matches the
    pre-C3.3 behavior asserted by ``test_peer_install.py``.
    """
    extras: list[str] = []
    packages: list[str] = []
    for token in raw_tokens:
        token = token.strip()
        if not token:
            continue
        if token in KNOWN_EXTRAS:
            extras.append(token)
        else:
            packages.append(token)
    return extras, packages


def install_on_target(
    url: str,
    key: str | None,
    extras: list[str],
    packages: list[str],
) -> int:
    """POST an install request to a target URL and poll if async.

    Plan 4 C3.3 fold: moved from :mod:`maxim.peer.cli` to break the
    mesh_cli → cli coupling inversion (Architecture review finding #1).

    **Single source of truth** for the wire-level install request.
    Two callers:

    - :func:`maxim.peer.cli._cmd_install` — resolves URL from a
      positional ``http*`` arg or ``peer.yml``, then calls this core.
    - :func:`maxim.peer.mesh_cli._run_node_install` — resolves URL +
      key from ``mesh.yml::nodes`` by name, drains the node, calls
      this core, then resumes.

    ``url`` may include a trailing ``/v1`` — it gets stripped for the
    endpoint composition but passed unmodified to
    :func:`_clear_probe_cache` via normalize-on-lookup in
    :func:`probe_cache.clear_cache_for_url`, so the probe cache key
    is cleared even when its shape diverges from the mesh.yml URL.

    The function makes no assumptions about whether the caller has
    drained the target node first — that's the caller's
    responsibility. On the success path, the function **does** mutate
    shared state by calling :func:`_clear_probe_cache(url)` so the
    next startup re-probes the freshly-installed leader instead of
    trusting a cached pre-install entry.

    Exit code contract:

    - ``0`` — install succeeded (sync ``ok`` or async polled ``ok``)
    - ``1`` — install failed (any HTTP error, non-terminal response,
      or generic exception)
    """
    from maxim.utils import http as _http

    # Lazy import: peer.cli imports from install_core at module load,
    # so we can't go back the other way at module scope without a
    # circular. _clear_probe_cache is a thin wrapper we can safely
    # pull in at call time.
    from maxim.peer.cli import _clear_probe_cache

    base = url.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]

    endpoint = f"{base}/v1/admin/install"
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"

    desc_parts = []
    if extras:
        desc_parts.append(f"extras: {', '.join(extras)}")
    if packages:
        desc_parts.append(f"packages: {', '.join(packages)}")
    print(f"Installing on leader ({base}): {'; '.join(desc_parts)}...")

    try:
        resp = _http.fetch_url(
            endpoint,
            method="POST",
            headers=headers,
            json={"extras": extras, "packages": packages},
            timeout=_http.TimeoutPolicy(connect_s=3.0, read_s=30.0, total_s=33.0),
        )
        data = resp.json()
    except _http.HTTPError as e:
        body_text = ""
        if e.response is not None:
            body_text = e.response.text[:500]
        if e.status == 403:
            print(
                "Install disabled on leader. Set MAXIM_ALLOW_REMOTE_UPDATE=1.",
                file=sys.stderr,
            )
        elif e.status == 401:
            print("Authentication failed. Check API key.", file=sys.stderr)
        elif e.status == 404:
            print(
                "Leader does not support remote install (update leader first).",
                file=sys.stderr,
            )
        else:
            print(f"Install failed (HTTP {e.status}): {body_text}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Install failed: {e}", file=sys.stderr)
        return 1

    status = data.get("status", "unknown")
    if status not in ("started", "ok"):
        print(f"  Install failed: {data.get('error', 'unknown error')}", file=sys.stderr)
        return 1

    if status == "ok":
        # Synchronous completion (leader responded before timeout)
        print("  Installed successfully.")
        _clear_probe_cache(url)
        return 0

    # status == "started" — async install. Poll for completion.
    print("  Install started on leader. Waiting for completion...")
    poll_endpoint = f"{base}/v1/debug/install-status"
    poll_headers: dict[str, str] = {}
    if key:
        poll_headers["Authorization"] = f"Bearer {key}"

    import time

    deadline = time.time() + _INSTALL_POLL_DEADLINE_S
    last_output_len = 0
    while time.time() < deadline:
        time.sleep(5)
        try:
            poll_resp = _http.fetch_url(
                poll_endpoint,
                method="GET",
                headers=poll_headers,
                timeout=_http.TimeoutPolicy(connect_s=2.0, read_s=10.0, total_s=12.0),
            )
            poll_data = poll_resp.json()
        except Exception:
            print("    (polling...)", end="\r")
            continue

        poll_status = poll_data.get("status", "unknown")

        # Show new pip output lines as they appear
        pip_out = poll_data.get("pip_output", "")
        if len(pip_out) > last_output_len:
            new_text = pip_out[last_output_len:]
            for line in new_text.strip().splitlines()[-3:]:
                line = line.strip()
                if line:
                    print(f"    {line}")
            last_output_len = len(pip_out)

        if poll_status == "ok":
            installed = poll_data.get("installed", [])
            print(f"  Installed successfully: {', '.join(installed)}")
            _clear_probe_cache(url)
            return 0

        if poll_status == "error":
            print(f"  Install failed: {poll_data.get('error', 'unknown')}", file=sys.stderr)
            return 1

        elapsed = int(time.time() - (deadline - _INSTALL_POLL_DEADLINE_S))
        print(f"    Installing... ({elapsed}s)", end="\r")

    print(
        f"\n  Timed out waiting for install ({int(_INSTALL_POLL_DEADLINE_S / 60)} min). Check leader logs.",
        file=sys.stderr,
    )
    return 1


__all__ = [
    "KNOWN_EXTRAS",
    "_INSTALL_POLL_DEADLINE_S",
    "_looks_like_url",
    "_normalize_url_for_probe_cache",
    "classify_install_tokens",
    "install_on_target",
]
