"""``maxim config`` CLI verbs (C2 of config_unification.md).

Surface:

  maxim config get                    — full config + sources
  maxim config get <field-path>       — one field with source marker
  maxim config get llm.profile        — nested via dot path
  maxim config path                   — print resolved config.json path
  maxim config list                   — human-readable summary, all
                                        effective fields with sources
  maxim config set <field-path> <val> — atomic write via config_writer
  maxim config edit                   — open $EDITOR on the file
  maxim config downgrade              — after running an older build: keep the settings it
                                        knows, set the rest aside in config.preserved.json (#974)
  maxim config restore-preserved      — after upgrading: review and restore set-aside settings
                                        (interactive terminal only)

Exit codes:
  0 success
  1 environmental failure (write permission, lock failure, etc.)
  2 operator error (unknown field, bad value type, missing arg)

The verb dispatcher is :func:`run_config_subcommand`, called from
``cli.py`` when ``argv[0] == "config"``.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from collections.abc import Sequence

from maxim.exceptions import ConfigurationError
from maxim.runtime.config_loader import (
    MaximConfig,
    _FIELD_TO_ENV,
    config_path,
    load_config,
    resolve_setting,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────


def run_config_subcommand(argv: Sequence[str]) -> int:
    """Dispatch ``maxim config <verb> [args...]`` subcommands."""
    if not argv or argv[0] in ("-h", "--help"):
        _print_usage()
        return 0 if argv else 2

    verb = argv[0]
    rest = list(argv[1:])

    if verb == "get":
        return _cmd_get(rest)
    if verb == "set":
        return _cmd_set(rest)
    if verb == "path":
        return _cmd_path(rest)
    if verb == "list":
        return _cmd_list(rest)
    if verb == "edit":
        return _cmd_edit(rest)
    if verb == "downgrade":
        return _cmd_downgrade(rest)
    if verb == "restore-preserved":
        return _cmd_restore_preserved(rest)

    print(f"Unknown config verb: {verb}", file=sys.stderr)
    _print_usage()
    return 2


def _print_usage() -> None:
    print("Usage: maxim config <verb> [args...]")
    print()
    print("Verbs:")
    print("  get [field-path]     — print full config + sources, or one field")
    print("  set <field> <value>  — atomic write to config.json")
    print("  path                 — print the resolved config.json path")
    print("  list                 — show every effective field + source marker")
    print("  edit                 — open $EDITOR on the config file")
    print("  downgrade            — keep the settings this build knows; set a newer file's others aside")
    print("  restore-preserved    — review and restore set-aside settings (interactive terminal only)")
    print()
    print("Field paths: dot-separated, e.g. llm.profile, lanes.large.remote_url")
    print()
    print(f"Resolved config file: {config_path()}")


# ─────────────────────────────────────────────────────────────────────────────
# get / list — read-only
# ─────────────────────────────────────────────────────────────────────────────


def _cmd_get(argv: list[str]) -> int:
    """``maxim config get [field-path]``."""
    if argv and argv[0] in ("-h", "--help"):
        print("Usage: maxim config get [field-path]")
        print("  No arg → print every field, value, and source")
        print("  field-path (e.g. llm.profile) → print just that field with source")
        return 0

    config = load_config()

    if not argv:
        return _print_all_fields(config, as_json=False)

    field_path = argv[0]
    if field_path not in _FIELD_TO_ENV:
        print(
            f"Unknown field path: {field_path!r}\nValid paths:\n  " + "\n  ".join(sorted(_FIELD_TO_ENV.keys())),
            file=sys.stderr,
        )
        return 2

    try:
        value, source = resolve_setting(field_path, config=config)
    except ConfigurationError as e:
        print(f"✗ {e}", file=sys.stderr)
        return 2

    print(f"{field_path}: {_format_value(value)}  [source={source}]")
    return 0


def _cmd_list(argv: list[str]) -> int:
    """``maxim config list`` — alias for ``get`` with no args. Supports --json."""
    if argv and argv[0] in ("-h", "--help"):
        print("Usage: maxim config list [--json]")
        return 0

    as_json = "--json" in argv
    config = load_config()
    return _print_all_fields(config, as_json=as_json)


def _print_all_fields(config: MaximConfig, *, as_json: bool) -> int:
    """Render every absorbed field with its effective value + source."""
    rows: list[tuple[str, object, str]] = []
    error_count = 0
    for field_path in sorted(_FIELD_TO_ENV.keys()):
        try:
            value, source = resolve_setting(field_path, config=config)
        except ConfigurationError as e:
            rows.append((field_path, f"ERROR: {e}", "error"))
            error_count += 1
            continue
        rows.append((field_path, value, source))

    if as_json:
        payload = {
            "config_path": str(config_path()),
            "fields": [{"path": p, "value": v, "source": s} for p, v, s in rows],
        }
        print(json.dumps(payload, indent=2, default=str))
        return 1 if error_count else 0

    # Human-readable
    print(f"Resolved config: {config_path()}")
    print()
    max_path_len = max(len(p) for p, _, _ in rows)
    for path, value, source in rows:
        marker = {"cli": "✓", "env": "⚠", "config": "✓", "default": " ", "error": "✗"}.get(source, " ")
        print(f"  {marker} {path:<{max_path_len}}  {_format_value(value):<32}  [source={source}]")
    print()
    print("Sources: cli (per-invocation) > env (MAXIM_*) > config (config.json) > default (schema)")
    return 1 if error_count else 0


def _format_value(value: object) -> str:
    if value is None:
        return "<unset>"
    if isinstance(value, str):
        return value if value else "<empty>"
    return repr(value)


# ─────────────────────────────────────────────────────────────────────────────
# path
# ─────────────────────────────────────────────────────────────────────────────


def _cmd_path(argv: list[str]) -> int:
    """``maxim config path`` — print the resolved config.json path."""
    if argv and argv[0] in ("-h", "--help"):
        print("Usage: maxim config path")
        return 0
    print(config_path())
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# set — writes via the canonical writer module
# ─────────────────────────────────────────────────────────────────────────────


def _cmd_set(argv: list[str]) -> int:
    """``maxim config set <field-path> <value>``."""
    if argv and argv[0] in ("-h", "--help"):
        print("Usage: maxim config set <field-path> <value>")
        print()
        print("Field paths: dot-separated. Use 'maxim config list' for the full set.")
        print()
        print("Boolean: use true/false (also accepts 1/0/yes/no/on/off).")
        print("Null: pass the literal string 'null' or '-' to clear the field.")
        return 0

    if len(argv) < 2:
        print("Usage: maxim config set <field-path> <value>", file=sys.stderr)
        return 2

    field_path = argv[0]
    raw_value = argv[1]

    if field_path not in _FIELD_TO_ENV:
        print(
            f"Unknown field path: {field_path!r}\nUse `maxim config list` to see valid paths.",
            file=sys.stderr,
        )
        return 2

    from maxim.runtime.config_writer import set_field, _apply_field_to_config

    # Allow operators to clear a field by passing "null" or "-"
    if raw_value in ("null", "-"):
        try:
            current = load_config()
            new = _apply_field_to_config(current, field_path, None)
            from maxim.runtime.config_writer import write_config

            write_config(new)
        except ConfigurationError as e:
            print(f"✗ {e}", file=sys.stderr)
            return 2
        except OSError as e:
            print(f"✗ failed to write config: {e}", file=sys.stderr)
            return 1
        print(f"✓ cleared {field_path}")
        return 0

    try:
        set_field(field_path, raw_value)
    except ConfigurationError as e:
        print(f"✗ {e}", file=sys.stderr)
        return 2
    except OSError as e:
        print(f"✗ failed to write config: {e}", file=sys.stderr)
        return 1

    print(f"✓ set {field_path} = {raw_value}")
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# edit — opens $EDITOR
# ─────────────────────────────────────────────────────────────────────────────


def _cmd_edit(argv: list[str]) -> int:
    """``maxim config edit`` — open ``$EDITOR`` on the config file.

    Creates the file with a minimal default skeleton if absent so the
    operator sees the expected shape. Re-validates the file after the
    editor exits — a parse error blocks the save with a recovery hint
    (no auto-revert; the operator keeps their edit and can fix it).
    """
    if argv and argv[0] in ("-h", "--help"):
        print("Usage: maxim config edit")
        print("  Opens $EDITOR (or $VISUAL) on the config.json file.")
        print("  Validates the file after edit; warns on parse error.")
        return 0

    target = config_path()
    target.parent.mkdir(parents=True, exist_ok=True)

    if not target.is_file():
        # Seed with a minimal skeleton so the operator sees the shape
        from maxim.runtime.config_writer import write_config

        write_config(MaximConfig())

    editor = os.environ.get("VISUAL") or os.environ.get("EDITOR") or "vi"

    try:
        rc = subprocess.call([editor, str(target)])
    except FileNotFoundError:
        print(
            f"✗ editor not found: {editor!r}. Set $EDITOR or $VISUAL.",
            file=sys.stderr,
        )
        return 1

    if rc != 0:
        print(f"✗ editor exited with code {rc}", file=sys.stderr)
        return rc

    # Validate after edit
    try:
        load_config(target)
    except ConfigurationError as e:
        print(
            f"⚠ saved, but the file has a validation error:\n  {e}\n"
            f"Fix it via `maxim config edit` again or `maxim config set`.",
            file=sys.stderr,
        )
        return 2

    print(f"✓ saved {target}")
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# downgrade / restore-preserved — a newer config.json after a downgrade (#974)
# ─────────────────────────────────────────────────────────────────────────────


def _safe(text: object, limit: int = 120) -> str:
    """Escape and truncate a name or value before it is printed (injection review, #974)."""
    from maxim.runtime.config_writer import safe_display

    return safe_display(text, limit)


def _interactive_terminal() -> bool:
    """A person at a terminal: both stdin and stdout are TTYs. An agent's shell tool is not one."""
    return sys.stdin.isatty() and sys.stdout.isatty()


def _cmd_downgrade(argv: list[str]) -> int:
    """``maxim config downgrade`` — rewrite a newer config.json for this build (#974)."""
    if argv and argv[0] in ("-h", "--help"):
        print("Usage: maxim config downgrade")
        print("  For a config.json written by a NEWER Maxim: keeps the settings this build knows and moves")
        print("  the rest to config.preserved.json (never applied). Restore them after upgrading with")
        print("  `maxim config restore-preserved`.")
        return 0
    if argv:
        print(f"✗ unexpected argument {_safe(argv[0], 40)}", file=sys.stderr)
        return 2
    from maxim.runtime.config_writer import downgrade_config

    try:
        result = downgrade_config()
    except ConfigurationError as e:
        print(f"✗ {_safe(str(e), 400)}", file=sys.stderr)
        return 2
    except OSError as e:
        print(f"✗ failed to write config: {_safe(str(e), 300)}", file=sys.stderr)
        return 1
    print(f"✓ {_safe(str(result.config), 200)} now holds the settings this build knows.")
    print(f"  Set aside in {_safe(str(result.sidecar), 200)} (not applied):")
    for name in result.preserved:
        print(f"    - {_safe(name, 80)}")
    print("  After upgrading, run `maxim config restore-preserved` to review and restore them.")
    return 0


def _cmd_restore_preserved(argv: list[str]) -> int:
    """``maxim config restore-preserved`` — review and restore set-aside settings (#974).

    Interactive only: it refuses without a terminal and has no flag to skip the confirmation, so the
    CLI never restores unattended. That makes a restore no EASIER than editing config.json directly; it
    is not a barrier against a same-user process, which could allocate a terminal or edit the file.
    """
    if argv and argv[0] in ("-h", "--help"):
        print("Usage: maxim config restore-preserved")
        print("  Shows each set-aside setting this build now knows (current -> preserved, security-relevant")
        print("  ones flagged) and restores them only if you confirm. Requires an interactive terminal.")
        return 0
    if argv:
        print(f"✗ unexpected argument {_safe(argv[0], 40)} (there is no way to skip the confirmation)", file=sys.stderr)
        return 2
    if not _interactive_terminal():
        print("✗ restore-preserved needs an interactive terminal: a person must review the changes", file=sys.stderr)
        return 2

    from maxim.runtime.config_writer import RestoreRow, restore_preserved

    def confirm(rows: list[RestoreRow]) -> bool:
        print("These set-aside settings are known to this build:")
        for row in rows:
            flag = "  [SECURITY-RELEVANT]" if row.security_relevant else ""
            print(f"  {_safe(row.path, 60)}: {_safe(row.current, 60)} -> {_safe(row.preserved, 60)}{flag}")
            note = f"; replaces {row.superseded} older set-aside value(s)" if row.superseded else ""
            print(
                f"      set aside from format {_safe(row.source_format_version, 12)} at {_safe(row.preserved_at, 25)}"
                f"{note}"
            )
        if any(row.security_relevant for row in rows):
            print("Security-relevant settings can widen what an agent may do or where Maxim reads from.")
        return input(f"Restore these {len(rows)} setting(s)? [y/N] ").strip().lower() in ("y", "yes")

    try:
        restored = restore_preserved(confirm=confirm)
    except ConfigurationError as e:
        # A validation error can quote a preserved value: escaped and bounded like everything else.
        print(f"✗ {_safe(str(e), 400)}", file=sys.stderr)
        return 2
    except OSError as e:
        print(f"✗ failed to write config: {_safe(str(e), 300)}", file=sys.stderr)
        return 1
    if not restored:
        print("Nothing restored.")
        return 0
    print(f"✓ restored {len(restored)} setting(s).")
    return 0


__all__ = ["run_config_subcommand"]
