#!/usr/bin/env python3
"""Outbound network surface of ``src/maxim/``: HTTP clients only in utils/http.py, raw dials and egress SDKs pinned.

CLAUDE.md's invariant "HTTP call sites must use ``maxim/utils/http.py``" was guarded only by a grep for
``urllib.request.urlopen`` (kept: CLAUDE.md cites it). That grep could not see the urllib3 pool
``tools/internet_search.py::_search_duckduckgo_lite`` fetched through, which skipped the #825
``max_bytes`` cap, the caller's timeout and the proxy settings. This AST lint replaces the grep's
reach with three rules:

A. **HTTP client libraries** (``urllib3``, ``requests``, ``httpx``, ``httpcore``, ``aiohttp``,
   ``http.client``, ``urllib.request``, and their submodules) are used nowhere but
   ``utils/http.py``. A use is: ``import M``/``from M import ...``; ``from urllib import request``
   (parent + name); any attribute chain spelling a banned dotted module (``urllib.request.urlopen``
   works without importing it once something else loaded it); a string-literal first argument naming
   a banned module to ``importlib.import_module``/``__import__``/``require_optional_dependency``/
   ``optional_dependency_available``/``warn_optional_fallback``. No allow-list: the count is zero.
   ``http.server`` and other inbound servers are out of scope.
B. **Raw connections** -- CALLS to ``socket.socket``/``create_connection``/``socketpair``/
   ``create_server``, ``asyncio.open_connection``, any ``*.wrap_socket`` and any
   ``*.create_connection`` attribute call (aliases resolved) -- are pinned per file in
   ``RAW_DIAL_PINS``, exact both ways: a higher count fails (a new raw dial), a lower one fails
   (lower the pin). Annotations, docstrings and bare ``import socket`` (DNS, gethostname) are not counted.
C. **Egress-capable SDKs** that speak HTTP themselves and cannot route through utils/http are a
   register: ``EGRESS_SDK_IMPORTERS`` pins which files import each (rule A's import forms), exact
   both ways. ``tests/integration/test_offline_latch.py::CENSUS_SITES`` is the behavioural register.

Catches FORGETTING, not evasion: a non-literal ``import_module(name)``, ``getattr(socket, "socket")``,
ctypes, an SDK missing from rule C's list, and subprocess egress (docker, git, cloudflared, pip)
escape it. ``scripts/`` is out of scope, matching the CLAUDE.md invariant.

Exits: 0 clean; 1 a violation (stderr); 2 a file that does not parse.
"""

from __future__ import annotations

import ast
import sys
from collections.abc import Iterator
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = Path("src/maxim")
CANONICAL = "src/maxim/utils/http.py"

HTTP_CLIENT_MODULES = ("urllib3", "requests", "httpx", "httpcore", "aiohttp", "http.client", "urllib.request")

# The functions that import a module named by their first argument.
_DYNAMIC_IMPORTERS = frozenset(
    {
        "import_module",
        "__import__",
        "require_optional_dependency",
        "optional_dependency_available",
        "warn_optional_fallback",
    }
)

_SOCKET_DIALS = frozenset({"socket", "create_connection", "socketpair", "create_server"})

# file -> (count, why). Exact both ways.
RAW_DIAL_PINS: dict[str, tuple[int, str]] = {
    "src/maxim/doctor/checks.py": (3, "TCP port probes of the local LLM/console servers"),
    "src/maxim/doctor/platform_detect.py": (1, "UDP connect to learn the local IP; sends nothing"),
    "src/maxim/hardware/reachy/controller.py": (1, "TCP reachability probe of the robot daemon"),
    "src/maxim/runtime/leader_proxy.py": (1, "TCP probe that the proxied local server is listening"),
    "src/maxim/runtime/local_server_spawner.py": (1, "TCP probe that the spawned server bound its LAN port"),
    "src/maxim/simulation/minecraft.py": (1, "Minecraft RCON client (not HTTP)"),
    "src/maxim/simulation/minecraft_harness.py": (1, "fake RCON server for the harness"),
    "src/maxim/utils/net.py": (1, "UDP connect to learn the local IP; sends nothing"),
    "src/maxim/utils/reachy_diagnostics.py": (1, "TCP reachability probe of the robot daemon"),
}

# SDK -> the files that import it. Exact both ways.
EGRESS_SDK_IMPORTERS: dict[str, frozenset[str]] = {
    "anthropic": frozenset({"src/maxim/api.py", "src/maxim/models/language/anthropic_backend.py"}),
    "openai": frozenset({"src/maxim/api.py", "src/maxim/models/language/openai_backend.py"}),
    "ddgs": frozenset({"src/maxim/tools/internet_search.py"}),
    "huggingface_hub": frozenset({"src/maxim/models/language/transformers_backend.py"}),
    "transformers": frozenset({"src/maxim/models/language/transformers_backend.py"}),
    "sentence_transformers": frozenset(
        {
            "src/maxim/similarity/encoder.py",
            "src/maxim/similarity/semantic.py",
            "src/maxim/models/vision/clip_encoder.py",
        }
    ),
    "faster_whisper": frozenset({"src/maxim/inference/transcribe_audio.py"}),
    "twilio": frozenset({"src/maxim/comms/api.py", "src/maxim/comms/channels/twilio_channel.py"}),
    "ultralytics": frozenset({"src/maxim/models/vision/ultralytics_engine.py"}),
    "reachy_mini": frozenset(
        {
            "src/maxim/embodiment/audio_localization.py",
            "src/maxim/hardware/reachy/controller.py",
            "src/maxim/hardware/reachy/motion_path.py",
        }
    ),
}


def _in(module: str, roots: tuple[str, ...] | frozenset[str]) -> str | None:
    """The root ``module`` is, or is a submodule of; None otherwise."""
    for root in roots:
        if module == root or module.startswith(root + "."):
            return root
    return None


def _dotted(node: ast.AST) -> str | None:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def imported_modules(tree: ast.AST) -> Iterator[tuple[int, str]]:
    """Every module a file imports, by any of rule A's spellings except attribute chains."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            yield node.lineno, node.module
            for alias in node.names:
                yield node.lineno, f"{node.module}.{alias.name}"
        elif isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else func.id if isinstance(func, ast.Name) else None
            if name not in _DYNAMIC_IMPORTERS:
                continue
            # The module is the first positional argument, or the keyword form (import_module(name=...)).
            candidates = node.args[:1] + [kw.value for kw in node.keywords if kw.arg in ("name", "import_name")]
            for arg in candidates:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    yield node.lineno, arg.value


def http_client_uses(text: str) -> list[tuple[int, str]]:
    """Rule A: (line, banned module) for every use of an HTTP client library."""
    tree = ast.parse(text)
    found = {(ln, root) for ln, mod in imported_modules(tree) if (root := _in(mod, HTTP_CLIENT_MODULES))}
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and (dotted := _dotted(node)):
            root = _in(dotted, HTTP_CLIENT_MODULES)
            if root and "." in root:  # a bare name chain (urllib3.x) needs an import, caught above
                found.add((node.lineno, root))
    return sorted(found)


def _socket_aliases(tree: ast.AST) -> tuple[set[str], set[str]]:
    """(names bound to the socket/asyncio modules, names bound to a dial function itself)."""
    modules, functions = {"socket", "asyncio"}, set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in ("socket", "asyncio") and alias.asname:
                    modules.add(alias.asname)
        elif isinstance(node, ast.ImportFrom) and node.module == "socket":
            for alias in node.names:
                if alias.name in _SOCKET_DIALS:
                    functions.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module == "asyncio":
            for alias in node.names:
                if alias.name == "open_connection":
                    functions.add(alias.asname or alias.name)
    return modules, functions


def raw_dial_sites(text: str) -> list[tuple[int, str]]:
    """Rule B: (line, spelling) for every raw-connection CALL."""
    tree = ast.parse(text)
    modules, functions = _socket_aliases(tree)
    sites = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id in functions:
            sites.append((node.lineno, func.id))
        elif isinstance(func, ast.Attribute):
            owner = func.value.id if isinstance(func.value, ast.Name) else None
            if func.attr in ("wrap_socket", "create_connection"):
                sites.append((node.lineno, f"{owner or '<expr>'}.{func.attr}"))
            elif owner in modules and func.attr in _SOCKET_DIALS | {"open_connection"}:
                sites.append((node.lineno, f"{owner}.{func.attr}"))
    return sorted(sites)


def egress_sdks(text: str) -> set[str]:
    """Rule C: the egress-capable SDKs a file imports."""
    tree = ast.parse(text)
    return {root for _, mod in imported_modules(tree) if (root := _in(mod, frozenset(EGRESS_SDK_IMPORTERS)))}


def lint(repo_root: Path = REPO_ROOT) -> int:
    errors: list[str] = []
    dials: dict[str, int] = {}
    importers: dict[str, set[str]] = {sdk: set() for sdk in EGRESS_SDK_IMPORTERS}
    for path in sorted((repo_root / SRC).rglob("*.py")):
        rel = path.relative_to(repo_root).as_posix()
        try:
            text = path.read_text(encoding="utf-8")
            uses, sites, sdks = http_client_uses(text), raw_dial_sites(text), egress_sdks(text)
        except (SyntaxError, UnicodeDecodeError) as exc:
            print(f"lint_http_surface: cannot parse {rel}: {exc}", file=sys.stderr)
            return 2
        if rel != CANONICAL:
            errors += [
                f"{rel}:{ln}: uses {mod} -- outbound HTTP goes through maxim.utils.http (fetch_url / get / post)"
                for ln, mod in uses
            ]
            if sites:
                dials[rel] = len(sites)
        for sdk in sdks:
            importers[sdk].add(rel)

    for rel in sorted(set(dials) | set(RAW_DIAL_PINS)):
        have, (pinned, _why) = dials.get(rel, 0), RAW_DIAL_PINS.get(rel, (0, ""))
        if have > pinned:
            errors.append(
                f"{rel}: {have} raw socket dial(s), pinned at {pinned} -- HTTP goes through maxim.utils.http; "
                "a non-HTTP dial is pinned in RAW_DIAL_PINS with its reason"
            )
        elif have < pinned:
            errors.append(f"{rel}: {have} raw socket dial(s), pinned at {pinned} -- lower the pin in RAW_DIAL_PINS")
    for sdk, pinned_files in EGRESS_SDK_IMPORTERS.items():
        for rel in sorted(importers[sdk] - pinned_files):
            errors.append(f"{rel}: imports the egress-capable SDK {sdk} -- add it to EGRESS_SDK_IMPORTERS")
        for rel in sorted(pinned_files - importers[sdk]):
            errors.append(f"{rel}: no longer imports {sdk} -- remove it from EGRESS_SDK_IMPORTERS[{sdk!r}]")

    print(
        f"lint_http_surface: {sum(dials.values())} raw dial(s) in {len(dials)} file(s); "
        f"{sum(map(len, importers.values()))} egress-SDK import site(s)"
    )
    for err in errors:
        print(f"ERROR: {err}", file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(lint())
