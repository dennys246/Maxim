"""Positive and negative controls for scripts/lint_config_schema_append_only.py (#856)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "lint_cfg_schema", REPO / "scripts" / "lint_config_schema_append_only.py"
)
lint = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lint)  # type: ignore[union-attr]


def _doc(**versions) -> str:
    return json.dumps({"_comment": "x", **versions})


def test_adding_a_new_version_is_allowed() -> None:
    ok, _ = lint.verdict(_doc(**{"1.1": ["a"]}), _doc(**{"1.1": ["a"], "1.2": ["a", "b"]}))
    assert ok


def test_editing_an_existing_version_fails() -> None:
    """The failure mode it exists for: appending the new path to the current entry."""
    ok, msg = lint.verdict(_doc(**{"1.1": ["a"]}), _doc(**{"1.1": ["a", "b"]}))
    assert not ok and "1.1" in msg


def test_removing_an_existing_version_fails() -> None:
    ok, _ = lint.verdict(_doc(**{"1.1": ["a"]}), _doc(**{"1.2": ["a"]}))
    assert not ok


def test_the_comment_may_change_and_a_missing_base_is_not_applicable() -> None:
    assert lint.verdict(_doc(**{"1.1": ["a"]}), json.dumps({"_comment": "new", "1.1": ["a"]}))[0]
    assert lint.verdict("", _doc(**{"1.1": ["a"]}))[0]
