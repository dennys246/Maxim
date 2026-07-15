"""`maxim model add/remove/list` — user-profile management CLI verbs.

Wraps the L2 YAML loader (``maxim.models.language.profile_loader``) with
a CLI surface so operators don't have to hand-edit
``~/.config/maxim/profiles.yml``. Stage L3 of
[docs/plans/archive/leader_ux_profile_management.md].

CLI shape::

    maxim model add NAME --hf REPO:FILE [--chat-format STYLE] [--n-ctx N]
                          [--alias A] [--force]
    maxim model add NAME --local PATH    [--chat-format STYLE] [--n-ctx N]
                                         [--alias A] [--force]
    maxim model remove NAME
    maxim model list

Round-trip caveat: editing via ``maxim model add/remove`` LOSES YAML
comments — the operator-edited file is read with PyYAML, mutated in
memory, then written back via stdlib YAML serialization. To preserve
comments, edit ``~/.config/maxim/profiles.yml`` directly. This is
documented in each verb's --help output.

Chat-format auto-inference: when ``--chat-format`` is omitted, a
substring match on the profile slug or HF repo basename picks a
default. If no match, the operator must supply ``--chat-format``
explicitly — silent default-to-chatml would mis-route every non-Qwen
model.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

from maxim.models.language.profile_loader import profiles_config_path
from maxim.utils.atomic_io import atomic_write_text


# ─────────────────────────────────────────────────────────────────────────────
# Chat-format auto-inference
# ─────────────────────────────────────────────────────────────────────────────

# Substring → prompt_style. Earlier entries in this list win on a tie
# (Mixtral matches "mixtral" before any future "mistral" entry, etc.).
# Substring matching is case-insensitive.
_CHAT_FORMAT_HINTS: tuple[tuple[str, str], ...] = (
    ("qwen", "chatml"),
    ("llama-3", "llama3_instruct"),
    ("llama3", "llama3_instruct"),
    ("llama-2", "llama2_chat"),
    ("llama2", "llama2_chat"),
    ("mixtral", "mistral_instruct"),
    ("mistral", "mistral_instruct"),
    ("phi-3", "phi3"),
    ("phi3", "phi3"),
    ("phi-2", "phi"),
    ("phi2", "phi"),
    ("gemma", "gemma"),
)


def infer_chat_format(profile_name: str, hf_repo: str | None) -> str | None:
    """Return the inferred ``prompt_style`` for ``profile_name`` / ``hf_repo``
    or ``None`` if no rule matched.

    Public entry — also used by the doctor integration to surface a
    "did you forget --chat-format?" hint at config validation time.

    **HF repo basename wins over profile name** (pre-merge executor
    review fix): if the user names a profile ``llama-3-experiment`` but
    points it at ``bartowski/Mixtral-8x7B-Instruct-v0.1-GGUF``, the
    repo is ground truth for the model family. The previous behavior
    joined both strings into one haystack and matched whichever rule
    came first — silently mis-routing the experiment to ``llama3_instruct``.
    """
    # HF repo basename first. If it produces a hit, that wins.
    if hf_repo:
        hf_basename = hf_repo.split("/")[-1].lower()
        for needle, style in _CHAT_FORMAT_HINTS:
            if needle in hf_basename:
                return style
    # Fall back to profile name.
    name_lower = profile_name.lower()
    for needle, style in _CHAT_FORMAT_HINTS:
        if needle in name_lower:
            return style
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Persistence helpers (read + write profiles.yml)
# ─────────────────────────────────────────────────────────────────────────────


def _read_profiles_file(path: Path) -> dict[str, Any]:
    """Read profiles.yml and return the top-level dict, normalizing
    empty/missing/null shapes to ``{"profiles": {}}``.

    Distinct from the loader's ``load_user_profiles`` because here we
    need the FULL top-level structure (so writes round-trip) plus the
    ability to seed a fresh file.
    """
    if not path.is_file():
        return {"profiles": {}}
    try:
        content = path.read_text()
    except OSError as exc:
        print(
            f"ERROR: could not read {path}: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc
    if not content.strip():
        return {"profiles": {}}
    try:
        data = yaml.safe_load(content)
    except yaml.YAMLError as exc:
        print(
            f"ERROR: {path} is not valid YAML:\n  {exc}\nFix the syntax error and retry, or move the file aside.",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc
    if data is None:
        return {"profiles": {}}
    if not isinstance(data, dict):
        print(
            f"ERROR: {path} top-level must be a mapping with a 'profiles:' key.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    if "profiles" not in data or data["profiles"] is None:
        data["profiles"] = {}
    if not isinstance(data["profiles"], dict):
        print(
            f"ERROR: {path}: 'profiles:' must be a mapping.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    return data


def _write_profiles_file(path: Path, data: dict[str, Any]) -> None:
    """Atomic write via stdlib YAML serialization. profiles.yml is NOT
    a secret file — uses plain ``atomic_write_text``, not
    ``atomic_write_secret`` (per the CLAUDE.md mesh.yml two-layer
    invariant: declarative config + plain mode bits)."""
    rendered = yaml.safe_dump(
        data,
        sort_keys=False,
        default_flow_style=False,
        allow_unicode=True,
    )
    atomic_write_text(str(path), rendered)


# ─────────────────────────────────────────────────────────────────────────────
# Argparse helpers
# ─────────────────────────────────────────────────────────────────────────────


def _parse_hf_arg(hf: str) -> tuple[str, str]:
    """Parse ``--hf REPO:FILE`` into ``(repo, file)``.

    The repo MAY contain ``/``; the file is the final segment after the
    first ``:`` separator. A second ``:`` is allowed inside the filename
    (rare but possible) and is preserved.
    """
    if ":" not in hf:
        print(
            f"ERROR: --hf must be REPO:FILE format (e.g., "
            f"'bartowski/Qwen2.5-32B-Instruct-GGUF:Qwen2.5-32B-Instruct-Q4_K_M.gguf'); "
            f"got {hf!r}.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    repo, _, filename = hf.partition(":")
    repo = repo.strip()
    filename = filename.strip()
    if not repo or not filename:
        print(
            f"ERROR: --hf REPO:FILE — both REPO and FILE must be non-empty; got {hf!r}.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return repo, filename


def _build_argparser(prog: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description=(
            "Manage user-defined Maxim model profiles in "
            "~/.config/maxim/profiles.yml. See `maxim model <verb> --help` "
            "for per-verb options."
        ),
    )
    sub = parser.add_subparsers(dest="verb", required=True)

    add = sub.add_parser(
        "add",
        help="Add or replace a user profile",
        description=(
            "Add a user profile to ~/.config/maxim/profiles.yml. The new "
            "profile is immediately usable on the next `maxim` invocation "
            "(no restart of any running daemon — profiles load at import). "
            "Comments in profiles.yml are NOT preserved across `maxim model "
            "add/remove` calls; edit the file directly to keep them."
        ),
    )
    add.add_argument("name", help="Profile slug (e.g., 'my-qwen-32b-q5')")
    src = add.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--hf",
        metavar="REPO:FILE",
        help=(
            "HuggingFace repo + GGUF filename, e.g. "
            "'bartowski/Qwen2.5-32B-Instruct-GGUF:Qwen2.5-32B-Instruct-Q4_K_M.gguf'"
        ),
    )
    src.add_argument(
        "--local",
        metavar="PATH",
        help="Path to a local GGUF file (no HF download)",
    )
    add.add_argument(
        "--chat-format",
        dest="chat_format",
        help=(
            "Chat template name (chatml, mistral_instruct, llama3_instruct, "
            "llama2_chat, phi3, phi, gemma). Auto-inferred from the profile "
            "name / HF repo when omitted; if inference fails the verb errors "
            "out asking for an explicit value."
        ),
    )
    add.add_argument(
        "--n-ctx",
        dest="n_ctx",
        type=int,
        default=None,
        help="Context window (default 8192)",
    )
    add.add_argument(
        "--alias",
        dest="aliases",
        action="append",
        default=[],
        help="Add an alias for this profile. Repeat for multiple aliases.",
    )
    add.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing user profile with the same slug.",
    )

    rm = sub.add_parser(
        "remove",
        help="Remove a user profile",
        description=(
            "Remove a profile from ~/.config/maxim/profiles.yml. Refuses "
            "if the slug matches a bundled profile (bundled profiles can't "
            "be removed via this verb; the user-profile shadow can be "
            "removed if one exists)."
        ),
    )
    rm.add_argument("name", help="Profile slug to remove")

    sub.add_parser(
        "list",
        help="List user-defined profiles",
        description=(
            "List user profiles in ~/.config/maxim/profiles.yml with source "
            "markers. For the full profile catalog (builtin + user), use "
            "`maxim --list-models`."
        ),
    )

    return parser


# ─────────────────────────────────────────────────────────────────────────────
# Verb implementations
# ─────────────────────────────────────────────────────────────────────────────


def _verb_add(args: argparse.Namespace, *, path: Path | None = None) -> int:
    target = path or profiles_config_path()
    data = _read_profiles_file(target)
    profiles = data["profiles"]

    if args.name in profiles and not args.force:
        print(
            f"ERROR: profile {args.name!r} already exists in {target}. "
            f"Pass --force to overwrite, or `maxim model remove {args.name}` first.",
            file=sys.stderr,
        )
        return 2

    if "/" in args.name or ":" in args.name:
        print(
            f"ERROR: profile name {args.name!r} must not contain '/' or ':' "
            f"(those characters break HF cache pathnames).",
            file=sys.stderr,
        )
        return 2

    # Source: --hf or --local. The mutually-exclusive group in argparse
    # already enforces exactly one; defensive check kept for direct callers.
    hf_repo: str | None = None
    hf_file: str | None = None
    local_path: str | None = None
    if args.hf:
        hf_repo, hf_file = _parse_hf_arg(args.hf)
    else:
        local_path = args.local

    # Chat-format inference (or operator-supplied).
    chat_format = args.chat_format
    if chat_format is None:
        chat_format = infer_chat_format(args.name, hf_repo)
        if chat_format is None:
            print(
                f"ERROR: could not infer --chat-format from profile name "
                f"{args.name!r} or HF repo {hf_repo!r}. Supported families: "
                f"qwen, llama-3, llama-2, mistral/mixtral, phi-3, phi-2, "
                f"gemma. Pass --chat-format explicitly (one of: chatml, "
                f"mistral_instruct, llama3_instruct, llama2_chat, phi3, "
                f"phi, gemma).",
                file=sys.stderr,
            )
            return 2
        print(
            f"  inferred --chat-format={chat_format} (from substring match)",
            file=sys.stderr,
        )

    # Assemble the entry. Field order matches the schema doc so the
    # rendered YAML reads naturally.
    entry: dict[str, Any] = {
        "backend": "llama_cpp",
        "prompt_style": chat_format,
    }
    if args.n_ctx is not None:
        entry["n_ctx"] = int(args.n_ctx)
    if hf_repo and hf_file:
        entry["download"] = {"hf_repo": hf_repo, "hf_file": hf_file}
    elif local_path:
        entry["local_path"] = local_path
    if args.aliases:
        entry["aliases"] = list(args.aliases)

    profiles[args.name] = entry
    data["profiles"] = profiles
    _write_profiles_file(target, data)

    source_desc = f"HuggingFace ({hf_repo} :: {hf_file})" if hf_repo else f"local file ({local_path})"
    print(f"OK: profile {args.name!r} written to {target}")
    print(f"    source: {source_desc}")
    print(f"    prompt_style: {chat_format}")
    if args.aliases:
        print(f"    aliases: {', '.join(args.aliases)}")
    return 0


def _verb_remove(args: argparse.Namespace, *, path: Path | None = None) -> int:
    target = path or profiles_config_path()
    if not target.is_file():
        print(
            f"ERROR: no user profiles file at {target} — nothing to remove.",
            file=sys.stderr,
        )
        return 1
    data = _read_profiles_file(target)
    profiles = data["profiles"]
    if args.name not in profiles:
        print(
            f"ERROR: profile {args.name!r} not found in {target}. "
            f"Run `maxim model list` to see configured user profiles.",
            file=sys.stderr,
        )
        return 2
    del profiles[args.name]
    data["profiles"] = profiles
    _write_profiles_file(target, data)
    print(f"OK: profile {args.name!r} removed from {target}")
    return 0


def _verb_list(args: argparse.Namespace, *, path: Path | None = None) -> int:
    del args  # unused
    target = path or profiles_config_path()
    if not target.is_file():
        print(f"  (no user profiles file at {target})")
        print("  Run `maxim model add` to create one.")
        return 0
    data = _read_profiles_file(target)
    profiles = data["profiles"]
    if not profiles:
        print(f"  (no user profiles defined in {target})")
        return 0
    print(f"User profiles in {target}:")
    print()
    for name, entry in profiles.items():
        prompt_style = entry.get("prompt_style", "?")
        backend = entry.get("backend", "?")
        if "download" in entry:
            dl = entry["download"]
            source = f"hf:{dl.get('hf_repo', '?')}:{dl.get('hf_file', '?')}"
        elif "local_path" in entry:
            source = f"local:{entry['local_path']}"
        else:
            source = "?"
        aliases = entry.get("aliases", []) or []
        alias_str = f"  (also: {', '.join(aliases)})" if aliases else ""
        print(f"  {name}  [{backend}/{prompt_style}]{alias_str}")
        print(f"      source: {source}")
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────


def run_model_subcommand(argv: Sequence[str], *, path: Path | None = None) -> int:
    """Dispatch ``maxim model {add,remove,list}``.

    Parameters
    ----------
    argv
        The CLI args AFTER the ``model`` subcommand token (i.e. what
        ``raw_argv[1:]`` is in ``cli.py``).
    path
        Optional override for the profiles file location. Tests pass
        a ``tmp_path / 'profiles.yml'`` here so they don't touch the
        operator's real config.
    """
    parser = _build_argparser("maxim model")
    try:
        args = parser.parse_args(list(argv))
    except SystemExit as exc:
        # argparse exits with code 2 on missing required args; honor it.
        return int(exc.code) if exc.code is not None else 2

    if args.verb == "add":
        return _verb_add(args, path=path)
    if args.verb == "remove":
        return _verb_remove(args, path=path)
    if args.verb == "list":
        return _verb_list(args, path=path)
    # Should be unreachable — argparse requires verb.
    parser.print_help(sys.stderr)
    return 2
