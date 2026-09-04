"""Console web-bundle resolution + the cross-repo UI contract check.

The Console UI is built in **maxim-pulse** and VENDORED into this package at
release time (``scripts/vendor_console_ui.py`` copies a built bundle into
``maxim/console/ui_dist/``, which ships as package data). That is what makes a
plain ``pip install pymaxim[console] && maxim serve`` show a working Console
with no flag and no config.

Resolution order for the bundle — the standard precedence, one layer deeper:

    ``--ui-dist`` flag  >  ``config.json::console.ui_dist``  >  PACKAGED bundle

A source checkout has no packaged bundle (``ui_dist/`` is .gitignore'd), so
developers keep pointing at their local ``apps/console/dist`` and nothing
changes for them.

**The contract check.** Each bundle carries a ``maxim-ui.json`` written by the
pulse build: ``{target, app_version, contract_version, commit}``. The
``contract_version`` names the OpenAPI facade generation the bundle's
``FacadeClient`` was generated against. If it disagrees with the backend's
:data:`CONSOLE_CONTRACT_VERSION`, the UI may call endpoints this server does
not have (or miss ones it does) — a confusing, silent-ish failure that shows up
as blank panels rather than an error. We WARN loudly at startup and serve
anyway: refusing to boot over a version string would be worse for a local tool,
and a mismatch is often benign (an additive field).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# The OpenAPI facade generation this backend speaks. Bumped DELIBERATELY when
# the wire contract changes in a way a stale bundle would notice (an endpoint
# removed/renamed, a required field added, an envelope reshaped). This is also
# the FastAPI app's ``version`` — one source of truth so they cannot drift.
#
# CHANGELOG (bump on any change a stale bundle would notice):
#   0.1.0 — initial facade (models/diagnose + the typed 501 seam stubs).
#   0.2.0 — #438: /api/campaigns and /api/events/subscribe-frame added;
#           ConsoleEvent gained REQUIRED tier/seq/message; RunAccepted gained
#           a "completed" status and a `reply` field; RunRequest.input became
#           meaningful for adventure. A 0.1.0 bundle predates all of it.
#   0.3.0 — /api/identity + IdentityResponse/SeamStatus; /ws opens with an
#           `identity` frame; /api/run mode="rest" went live; /api/diagnose
#           now returns one row PER CHECK (~69) with extra.group/extra.fix
#           instead of one blank row per group. A 0.2.0 client cannot know
#           any of it exists.
#   0.4.0 — bearer auth (hardening PR 2): every /api/* route, /docs,
#           /openapi.json and /ws now require the console token
#           (securitySchemes.consoleToken; 401 shape documented); NEW
#           unauthenticated GET /api/hello + HelloResponse for skew detection
#           and the login screen; browser /ws carries the token as the
#           `maxim.bearer.<token>` subprotocol beside `maxim-console-v1`.
#           A 0.3.0 client sees 401s with a self-explaining detail string —
#           and since pulse `console-auth-040`, its OWN skew screens: a stale
#           0.3.0 bundle served by a 0.4.0 backend renders pulse's "This
#           backend predates contract 0.4.0" / version-skew screen, not
#           blank panels, so the startup WARN below is the operator's
#           breadcrumb, not the only symptom.
#
# ADDITIVE CHANGES COUNT. This was learned twice: #438 shipped two endpoints
# and a reshaped envelope at 0.1.0, and the identity surface itself first
# shipped at 0.2.0 — the very blindness the 0.1.0→0.2.0 bump was meant to end.
# `test_schema_surface_matches_the_recorded_contract` now fails when the
# OpenAPI surface moves without this number moving, so the rule is enforced
# rather than remembered.
CONSOLE_CONTRACT_VERSION = "0.4.0"

#: Manifest filename the pulse build writes into every bundle.
UI_MANIFEST_NAME = "maxim-ui.json"

#: Where a vendored bundle lands inside the installed package.
_PACKAGED_UI_DIST = Path(__file__).parent / "ui_dist"


def packaged_ui_dist() -> Path | None:
    """The vendored bundle shipped in this wheel, or ``None`` in a checkout.

    "Present" means the directory exists AND has an ``index.html`` — an empty
    or half-copied directory must not shadow the clearer "no UI installed"
    page.
    """
    index = _PACKAGED_UI_DIST / "index.html"
    return _PACKAGED_UI_DIST if index.is_file() else None


def resolve_ui_dist(cli_value: str | None, config_value: str | None) -> Path | None:
    """Pick the bundle to serve: CLI > config > packaged > none."""
    for candidate in (cli_value, config_value):
        if candidate:
            return Path(candidate).expanduser()
    return packaged_ui_dist()


def read_ui_manifest(ui_dist: Path | str) -> dict[str, Any] | None:
    """Read a bundle's ``maxim-ui.json``; ``None`` if absent or unreadable.

    Absent stopped being normal with pulse ``console-auth-040``: every pulse
    build path stamps a manifest, so a served bundle without one is a pulse
    build-path bug (or a foreign bundle). :func:`check_ui_contract` WARNs on
    that case; this reader stays judgment-free.
    """
    path = Path(ui_dist) / UI_MANIFEST_NAME
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError):
        logger.warning("Console UI manifest at %s is unreadable — skipping the contract check", path, exc_info=True)
        return None
    return data if isinstance(data, dict) else None


def check_ui_contract(ui_dist: Path | str | None) -> str | None:
    """Warn if the bundle was built against a different facade contract —
    or carries no manifest at all (post-``console-auth-040``, always a pulse
    build-path bug: every pulse build stamps ``maxim-ui.json``).

    Returns the warning message (for tests / callers that want to surface
    it), or ``None`` when the bundle matches or there is no bundle. Never
    raises — a version string must not stop a local tool from booting.
    """
    if ui_dist is None:
        return None
    manifest = read_ui_manifest(ui_dist)
    if manifest is None:
        if (Path(ui_dist) / UI_MANIFEST_NAME).is_file():
            return None  # present but unreadable — the reader already warned
        if not (Path(ui_dist) / "index.html").is_file():
            return None  # no servable bundle here — nothing to check
        message = (
            f"Console UI bundle at {ui_dist} has no {UI_MANIFEST_NAME} — its facade contract "
            f"cannot be checked (this server speaks {CONSOLE_CONTRACT_VERSION!r}). Every "
            f"maxim-pulse build stamps one, so this is a build-path bug or a foreign bundle; "
            f"rebuild with pnpm build, or expect silent skew."
        )
        logger.warning("%s", message)
        return message
    bundle_contract = str(manifest.get("contract_version") or "")
    if not bundle_contract or bundle_contract == CONSOLE_CONTRACT_VERSION:
        return None
    message = (
        f"Console UI contract mismatch: the bundle was built against facade "
        f"contract {bundle_contract!r} but this server speaks "
        f"{CONSOLE_CONTRACT_VERSION!r}. Parts of the UI may not work. Rebuild "
        f"the bundle from a matching maxim-pulse (pnpm gen:facade && pnpm build), "
        f"or upgrade/downgrade pymaxim to match."
    )
    logger.warning("%s", message)
    return message
