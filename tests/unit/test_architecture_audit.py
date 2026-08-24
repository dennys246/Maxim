"""Tests for the architecture audit tool."""

from __future__ import annotations

import dataclasses
import json

import pytest

from maxim.utils.audit import (
    BASELINE_PATH,
    DISPOSITION_ACCEPTED,
    DISPOSITION_TYPING_ONLY,
    DISPOSITION_UNREVIEWED,
    LAYER_RULES,
    SCOPE_FUNCTION_LOCAL,
    SCOPE_MODULE,
    SCOPE_TYPE_CHECKING,
    AuditViolation,
    BaselineEntry,
    BaselineFormatError,
    StaleEntry,
    audit_architecture,
    compare_to_baseline,
    format_diff,
    load_baseline,
    parse_baseline,
    render_baseline,
    target_layer,
)


class TestAuditViolationFrozen:
    def test_frozen_instance(self):
        v = AuditViolation(
            file="agents/bad.py",
            line=1,
            layer="agents",
            imported_module="maxim.tools.base",
            rule="agents must_not_import tools",
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            v.line = 99  # type: ignore[misc]


class TestForbiddenImport:
    def test_detect_forbidden_import(self, tmp_path):
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        (agents_dir / "__init__.py").write_text("")
        (agents_dir / "bad.py").write_text("from maxim.tools.base import SomeTool\n")

        violations = audit_architecture(src_root=tmp_path)

        assert len(violations) == 1
        v = violations[0]
        assert v.layer == "agents"
        assert v.imported_module == "maxim.tools.base"
        assert v.rule == "agents must_not_import tools"

    def test_allow_valid_import(self, tmp_path):
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        (agents_dir / "__init__.py").write_text("")
        (agents_dir / "good.py").write_text("from maxim.memory.types import MemoryRecord\n")

        violations = audit_architecture(src_root=tmp_path)

        assert violations == []

    def test_detect_bare_import(self, tmp_path):
        tools_dir = tmp_path / "tools"
        tools_dir.mkdir()
        (tools_dir / "__init__.py").write_text("")
        (tools_dir / "bad.py").write_text("from agents.bus import WorkingMemoryEntry\n")

        violations = audit_architecture(src_root=tmp_path)

        assert len(violations) == 1
        v = violations[0]
        assert v.layer == "tools"
        assert v.imported_module == "agents.bus"
        assert v.rule == "tools must_not_import agents"


class TestImportShapes:
    """Shapes a one-line rewrite could use to slip past the gate."""

    def _run(self, tmp_path, body):
        d = tmp_path / "memory"
        d.mkdir()
        (d / "__init__.py").write_text("")
        (d / "bad.py").write_text(body)
        return audit_architecture(src_root=tmp_path)

    def test_from_package_import_layer(self, tmp_path):
        (v,) = self._run(tmp_path, "from maxim import tools\n")
        assert v.rule == "memory must_not_import tools"
        assert v.names == ("tools",)

    def test_from_package_import_two_layers_is_two_findings(self, tmp_path):
        vs = self._run(tmp_path, "from maxim import tools, agents, similarity\n")
        assert sorted(v.imported_module for v in vs) == ["maxim.agents", "maxim.tools"]

    def test_relative_import_of_sibling_layer(self, tmp_path):
        (v,) = self._run(tmp_path, "from .. import tools\n")
        assert v.rule == "memory must_not_import tools"

    def test_relative_import_of_sibling_module(self, tmp_path):
        (v,) = self._run(tmp_path, "from ..agents.bus import EdgeType\n")
        assert v.imported_module == "maxim.agents.bus"  # canonical: same key as the absolute spelling
        assert v.names == ("EdgeType",)

    def test_relative_and_absolute_spellings_share_a_key(self, tmp_path):
        d = tmp_path / "memory"
        d.mkdir()
        (d / "__init__.py").write_text("")
        (d / "a.py").write_text("from maxim.agents.bus import EdgeType\n")
        (d / "b.py").write_text("from ..agents.bus import EdgeType\n")
        va, vb = audit_architecture(src_root=tmp_path)
        assert va.imported_module == vb.imported_module

    def test_relative_import_within_own_layer_is_fine(self, tmp_path):
        assert self._run(tmp_path, "from . import episode\nfrom .episode import Episode\n") == []

    def test_names_are_recorded(self, tmp_path):
        (v,) = self._run(tmp_path, "from maxim.agents.bus import DependencyGraph, EdgeType\n")
        assert v.names == ("DependencyGraph", "EdgeType")

    def test_plain_import_records_module_as_name(self, tmp_path):
        (v,) = self._run(tmp_path, "import maxim.agents.bus\n")
        assert v.names == ("maxim.agents.bus",)

    def test_target_layer(self):
        assert target_layer("maxim.tools.base") == "tools"
        assert target_layer("tools.base") == "tools"
        assert target_layer("maxim") is None
        assert target_layer("numpy") is None


class TestEdgeCases:
    def test_empty_directory(self, tmp_path):
        violations = audit_architecture(src_root=tmp_path)
        assert violations == []

    def test_syntax_error_skipped(self, tmp_path):
        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()
        (memory_dir / "__init__.py").write_text("")
        (memory_dir / "broken.py").write_text("def foo(\n")

        violations = audit_architecture(src_root=tmp_path)
        assert violations == []


class TestLayerRulesCoverage:
    EXPECTED_LAYERS = {
        "agents",
        "planning",
        "environment",
        "tools",
        "memory",
        "runtime",
        "skills",
        "bridges",
    }

    def test_all_layers_present(self):
        assert set(LAYER_RULES.keys()) == self.EXPECTED_LAYERS


class TestScopeDetection:
    """Each finding carries WHERE the import sits — the baseline keys on it."""

    def _write(self, tmp_path, body):
        d = tmp_path / "memory"
        d.mkdir()
        (d / "__init__.py").write_text("")
        (d / "mod.py").write_text(body)
        return audit_architecture(src_root=tmp_path)

    def test_module_level(self, tmp_path):
        (v,) = self._write(tmp_path, "from maxim.agents.bus import EdgeType\n")
        assert v.scope == SCOPE_MODULE

    def test_function_local(self, tmp_path):
        (v,) = self._write(tmp_path, "def f():\n    from maxim.agents.bus import EdgeType\n    return EdgeType\n")
        assert v.scope == SCOPE_FUNCTION_LOCAL

    def test_type_checking_block(self, tmp_path):
        body = "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    from maxim.agents.bus import EdgeType\n"
        (v,) = self._write(tmp_path, body)
        assert v.scope == SCOPE_TYPE_CHECKING

    def test_typing_attribute_form(self, tmp_path):
        body = "import typing\nif typing.TYPE_CHECKING:\n    from maxim.agents.bus import EdgeType\n"
        (v,) = self._write(tmp_path, body)
        assert v.scope == SCOPE_TYPE_CHECKING

    def test_else_branch_of_type_checking_is_runtime(self, tmp_path):
        body = (
            "from typing import TYPE_CHECKING\n"
            "if TYPE_CHECKING:\n    pass\n"
            "else:\n    from maxim.agents.bus import EdgeType\n"
        )
        (v,) = self._write(tmp_path, body)
        assert v.scope == SCOPE_MODULE

    def test_type_checking_inside_function_is_typing_only(self, tmp_path):
        body = (
            "from typing import TYPE_CHECKING\n"
            "def f():\n    if TYPE_CHECKING:\n        from maxim.agents.bus import EdgeType\n"
        )
        (v,) = self._write(tmp_path, body)
        assert v.scope == SCOPE_TYPE_CHECKING

    def test_same_module_two_scopes_are_two_keys(self, tmp_path):
        body = (
            "from typing import TYPE_CHECKING\n"
            "if TYPE_CHECKING:\n    from maxim.agents.bus import EdgeType\n"
            "def f():\n    from maxim.agents.bus import EdgeType\n    return EdgeType\n"
        )
        vs = self._write(tmp_path, body)
        assert len({v.baseline_key for v in vs}) == 2


def _entry(
    file="maxim/memory/mod.py", module="maxim.agents.bus", scope=SCOPE_MODULE, count=1, symbols=("EdgeType",), **kw
):
    kw.setdefault("disposition", DISPOSITION_TYPING_ONLY if scope == SCOPE_TYPE_CHECKING else DISPOSITION_ACCEPTED)
    kw.setdefault("rationale", "reviewed")
    return BaselineEntry(
        file=file,
        imported_module=module,
        rule="memory must_not_import agents",
        scope=scope,
        count=count,
        symbols=symbols,
        **kw,
    )


def _violation(file="maxim/memory/mod.py", module="maxim.agents.bus", scope=SCOPE_MODULE, line=1, names=("EdgeType",)):
    return AuditViolation(
        file=file,
        line=line,
        layer="memory",
        imported_module=module,
        rule="memory must_not_import agents",
        scope=scope,
        names=names,
    )


class TestBaselineCompare:
    def test_exact_match_is_ok(self):
        e = _entry()
        diff = compare_to_baseline([_violation()], {e.key: e})
        assert diff.ok
        assert diff.accepted_count == 1

    def test_new_key_is_added(self):
        diff = compare_to_baseline([_violation()], {})
        assert not diff.ok
        assert [v.line for v in diff.added] == [1]

    def test_count_excess_reports_only_the_surplus(self):
        e = _entry(count=1)
        # Order-independent: the highest line is the surplus regardless of input order.
        diff = compare_to_baseline([_violation(line=9), _violation(line=3)], {e.key: e})
        assert [v.line for v in diff.added] == [9]
        assert diff.accepted_count == 1

    def test_scope_escalation_is_an_addition(self):
        e = _entry(scope=SCOPE_TYPE_CHECKING)
        diff = compare_to_baseline([_violation(scope=SCOPE_MODULE)], {e.key: e})
        assert len(diff.added) == 1
        assert len(diff.stale) == 1

    def test_removed_debt_is_stale(self):
        e = _entry(count=2)
        diff = compare_to_baseline([_violation()], {e.key: e})
        assert not diff.ok
        assert diff.stale == (StaleEntry(e, 1, ()),)

    def test_symbol_widening_is_an_addition(self):
        e = _entry(symbols=("EdgeType",))
        diff = compare_to_baseline([_violation(names=("EdgeType", "AgentBus"))], {e.key: e})
        assert [v.names for v in diff.added] == [("EdgeType", "AgentBus")]
        assert not diff.stale

    def test_widening_with_count_excess_names_the_right_newcomer(self):
        # count=1, accepted {EdgeType}; live: line 3 imports AgentBus, line 9 imports EdgeType.
        # The accepted occurrence fills the allowance; line 3 is the newcomer.
        e = _entry(count=1, symbols=("EdgeType",))
        diff = compare_to_baseline([_violation(line=3, names=("AgentBus",)), _violation(line=9)], {e.key: e})
        assert [v.line for v in diff.added] == [3]
        assert diff.accepted_count == 1

    def test_widened_occurrence_is_not_counted_as_accepted(self):
        e = _entry(symbols=("EdgeType",))
        diff = compare_to_baseline([_violation(names=("EdgeType", "AgentBus"))], {e.key: e})
        assert diff.accepted_count == 0

    def test_symbol_shrink_is_stale(self):
        e = _entry(symbols=("DependencyGraph", "EdgeType"))
        diff = compare_to_baseline([_violation(names=("EdgeType",))], {e.key: e})
        assert diff.stale == (StaleEntry(e, 1, ("DependencyGraph",)),)
        assert not diff.added

    def test_move_between_files_hints_in_report(self):
        e = _entry(file="maxim/memory/old.py")
        diff = compare_to_baseline([_violation(file="maxim/memory/new.py")], {e.key: e})
        text = format_diff(diff, baseline_path="b.json")
        assert "a move?" in text

    def test_unreviewed_disposition_fails(self):
        e = _entry(disposition=DISPOSITION_UNREVIEWED)
        diff = compare_to_baseline([_violation()], {e.key: e})
        assert diff.unreviewed == (e,)
        assert not diff.ok

    def test_empty_rationale_fails(self):
        e = _entry(rationale="   ")
        diff = compare_to_baseline([_violation()], {e.key: e})
        assert diff.unreviewed == (e,)

    def test_format_diff_names_every_bucket(self):
        e = _entry(count=2, disposition=DISPOSITION_UNREVIEWED)
        diff = compare_to_baseline(
            [_violation(module="maxim.agents.modality", names=("SubstrateModality",))], {e.key: e}
        )
        text = format_diff(diff, baseline_path="b.json")
        new_i, stale_i, unrev_i = text.index("NEW"), text.index("STALE"), text.index("UNREVIEWED")
        assert new_i < stale_i < unrev_i
        assert "maxim.agents.modality" in text[new_i:stale_i]
        assert e.key in text[stale_i:unrev_i] and e.key in text[unrev_i:]

    def test_render_preserves_review_and_marks_new_unreviewed(self):
        e = _entry(rationale="kept")
        doc = render_baseline([_violation(), _violation(module="maxim.agents.modality")], {e.key: e})
        by_mod = {d["imported_module"]: d for d in doc["entries"]}
        assert by_mod["maxim.agents.bus"]["rationale"] == "kept"
        assert by_mod["maxim.agents.modality"]["disposition"] == DISPOSITION_UNREVIEWED
        assert by_mod["maxim.agents.bus"]["symbols"] == ["EdgeType"]
        assert "_comment" not in doc
        assert "_comment" in render_baseline([], comment="keep me")
        # The rendered document round-trips through the parser.
        parsed = parse_baseline(doc)
        assert set(parsed) == {v.baseline_key for v in (_violation(), _violation(module="maxim.agents.modality"))}


class TestBaselineFormat:
    def test_wrong_version_rejected(self):
        with pytest.raises(BaselineFormatError):
            parse_baseline({"baseline_format_version": 99, "entries": []})

    def test_duplicate_key_rejected(self):
        raw = {
            "file": "f.py",
            "imported_module": "maxim.agents.bus",
            "rule": "r",
            "scope": SCOPE_MODULE,
            "count": 1,
            "symbols": ["x"],
            "disposition": DISPOSITION_ACCEPTED,
            "rationale": "x",
        }
        with pytest.raises(BaselineFormatError, match="duplicate"):
            parse_baseline({"baseline_format_version": 1, "entries": [raw, dict(raw)]})

    def test_unknown_scope_rejected(self):
        raw = {
            "file": "f.py",
            "imported_module": "m",
            "rule": "r",
            "scope": "weird",
            "count": 1,
            "symbols": ["x"],
            "disposition": DISPOSITION_ACCEPTED,
            "rationale": "x",
        }
        with pytest.raises(BaselineFormatError, match="scope"):
            parse_baseline({"baseline_format_version": 1, "entries": [raw]})

    @pytest.mark.parametrize("symbols", ["EdgeType", [], None])
    def test_symbols_must_be_a_non_empty_list(self, symbols):
        raw = {
            "file": "maxim/memory/f.py",
            "imported_module": "maxim.agents.bus",
            "rule": "memory must_not_import agents",
            "scope": SCOPE_MODULE,
            "count": 1,
            "symbols": symbols,
            "disposition": DISPOSITION_ACCEPTED,
            "rationale": "x",
        }
        with pytest.raises(BaselineFormatError, match="symbols"):
            parse_baseline({"baseline_format_version": 1, "entries": [raw]})

    @pytest.mark.parametrize("entries", [None, "x", {"a": 1}])
    def test_entries_must_be_a_list(self, entries):
        with pytest.raises(BaselineFormatError):
            parse_baseline({"baseline_format_version": 1, "entries": entries})

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_baseline(tmp_path / "nope.json")

    def test_invalid_json_is_a_format_error(self, tmp_path):
        bad = tmp_path / "b.json"
        bad.write_text("{not json")
        with pytest.raises(BaselineFormatError, match="invalid JSON"):
            load_baseline(bad)

    @pytest.mark.parametrize(
        ("scope", "disposition"),
        [(SCOPE_MODULE, DISPOSITION_TYPING_ONLY), (SCOPE_TYPE_CHECKING, DISPOSITION_ACCEPTED)],
    )
    def test_disposition_must_match_scope(self, scope, disposition):
        # A reviewer cannot "accept" a runtime edge by mislabelling it typing-only.
        raw = {
            "file": "maxim/memory/f.py",
            "imported_module": "maxim.agents.bus",
            "rule": "memory must_not_import agents",
            "scope": scope,
            "count": 1,
            "symbols": ["x"],
            "disposition": disposition,
            "rationale": "x",
        }
        with pytest.raises(BaselineFormatError, match="does not match scope"):
            parse_baseline({"baseline_format_version": 1, "entries": [raw]})


class TestRealCodebase:
    """THE D19 regression gate: the live audit must match the reviewed baseline.

    Permanently-red audits detect nothing. This suite fails on (a) any finding
    not in ``architecture_baseline.json``, (b) any baseline entry the code no
    longer justifies, and (c) any entry nobody reviewed. Verified to fail in place
    against a synthetic new import (`from maxim import tools`), a widened accepted
    import, a deleted baseline entry, and an inflated count before landing
    (2026-08-24).
    """

    @pytest.fixture(scope="class")
    def diff(self):
        return compare_to_baseline(audit_architecture(), load_baseline())

    def test_baseline_file_ships_with_the_package(self):
        assert BASELINE_PATH.is_file()
        data = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
        assert data["baseline_format_version"] == 1
        assert data["reviewed"], "baseline must record its review date"

    def test_no_findings_outside_the_baseline(self, diff):
        assert not diff.added, format_diff(diff)

    def test_no_stale_baseline_entries(self, diff):
        assert not diff.stale, format_diff(diff)

    def test_every_baseline_entry_is_reviewed(self, diff):
        assert not diff.unreviewed, format_diff(diff)

    def test_baseline_rules_match_the_auditor(self):
        # A hand-edited entry whose rule text drifted from LAYER_RULES would
        # silently stop matching nothing (keys ignore rule) — pin it here.
        for entry in load_baseline().values():
            layer, _, target = entry.rule.partition(" must_not_import ")
            assert target_layer(entry.imported_module) == target, entry.key
            assert target in LAYER_RULES[layer]["must_not_import"], entry.key
            assert entry.file.startswith(f"maxim/{layer}/"), entry.key
            assert entry.symbols, entry.key
            # A bare-module import would accept every attribute — the symbol check must have teeth.
            assert not any("." in sym for sym in entry.symbols), entry.key
