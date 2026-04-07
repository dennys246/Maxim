"""Tests for the architecture audit tool."""

from __future__ import annotations

import dataclasses

import pytest

from maxim.utils.audit import LAYER_RULES, AuditViolation, audit_architecture


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


class TestRealCodebase:
    def test_runs_without_crash(self):
        violations = audit_architecture()
        assert isinstance(violations, list)
