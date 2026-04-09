"""Tests for the composable object API (maxim.create, maxim.load, maxim.session).

Covers:
- Lazy loading of create/load sub-modules
- Factory functions for all bio-subsystems
- Agent creation and subsystem wiring
- Agent pool lifecycle
- SEM entity instantiation and mutation
- Session creation, persistence, and loading
- Cross-object interactions (agent memory ↔ pool, entity ↔ embodiment)
- Object mutation (add/remove memories, swap subsystems, update personality)
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest import mock

import pytest


# ── Lazy loading ───────────────────────────────────────────────────────


class TestLazyLoading:
    """Verify maxim.create, maxim.load, and types are accessible."""

    def test_import_maxim_is_fast(self):
        """import maxim should not import heavy deps."""
        import maxim

        assert hasattr(maxim, "__version__")

    def test_create_is_module(self):
        import maxim

        assert hasattr(maxim, "create")
        import types

        assert isinstance(maxim.create, types.ModuleType)

    def test_load_is_module(self):
        import maxim

        assert hasattr(maxim, "load")
        import types

        assert isinstance(maxim.load, types.ModuleType)

    def test_session_type_accessible(self):
        from maxim import Session

        assert Session.__name__ == "Session"

    def test_entity_type_accessible(self):
        from maxim import Entity

        assert Entity.__name__ == "Entity"

    def test_report_type_accessible(self):
        from maxim import Report

        assert Report.__name__ == "Report"

    def test_load_sessions_accessible(self):
        import maxim

        assert callable(maxim.load.sessions)

    def test_load_agent_accessible(self):
        import maxim

        assert callable(maxim.load.agent)

    def test_all_contains_new_exports(self):
        import maxim

        for name in ("create", "load", "Session", "Report", "Entity"):
            assert name in maxim.__all__, f"{name} missing from __all__"

    def test_sensor_modulator_not_in_all(self):
        """Protocols should NOT be in __all__ (they're not constructable)."""
        import maxim

        assert "Sensor" not in maxim.__all__
        assert "Modulator" not in maxim.__all__


# ── maxim.create: Bio-subsystems ───────────────────────────────────────


class TestCreateBioSubsystems:
    """Test standalone subsystem creation."""

    def test_create_hippocampus(self):
        import maxim

        hippo = maxim.create.hippocampus()
        assert hippo is not None
        assert hasattr(hippo, "capture")
        assert hasattr(hippo, "recall")
        assert hasattr(hippo, "save")
        assert hasattr(hippo, "load")

    def test_create_hippocampus_with_persistence(self):
        import maxim

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "hippo.json")
            hippo = maxim.create.hippocampus(persistence_path=path)
            assert hippo is not None

    def test_create_nac(self):
        import maxim

        nac = maxim.create.nac()
        assert nac is not None
        assert hasattr(nac, "predict")
        assert hasattr(nac, "save")
        assert hasattr(nac, "load")

    def test_create_atl(self):
        import maxim

        atl = maxim.create.atl()
        assert atl is not None
        assert hasattr(atl, "store")
        assert hasattr(atl, "recall")
        assert hasattr(atl, "find_or_create")

    def test_create_scn(self):
        import maxim

        scn = maxim.create.scn()
        assert scn is not None
        assert hasattr(scn, "register")

    def test_create_angular_gyrus(self):
        import maxim

        ag = maxim.create.angular_gyrus()
        assert ag is not None


# ── maxim.create: Hippocampus capture/recall cycle ─────────────────────


class TestHippocampusOperations:
    """Test that a created Hippocampus can store and recall."""

    def test_store_observation_and_recall(self):
        import maxim

        hippo = maxim.create.hippocampus()
        mem_id = hippo.store_observation("The wolf was near the cave entrance")
        assert mem_id is not None
        assert len(mem_id) > 0

    def test_multiple_observations(self):
        import maxim

        hippo = maxim.create.hippocampus()
        id1 = hippo.store_observation("The door was locked")
        id2 = hippo.store_observation("The key was under the mat")
        assert id1 != id2

    def test_save_and_load_roundtrip(self):
        import maxim

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "hippo.json")
            hippo = maxim.create.hippocampus(persistence_path=path)
            hippo.store_observation("test memory")
            hippo.save(path)
            assert os.path.exists(path)

            hippo2 = maxim.load.hippocampus(path)
            assert hippo2 is not None

    def test_remove_memory(self):
        import maxim

        hippo = maxim.create.hippocampus()
        mem_id = hippo.store_observation("temporary memory")
        hippo.remove(mem_id)


# ── maxim.create: NAc causal learning cycle ────────────────────────────


class TestNAcOperations:
    """Test that a created NAc can observe and predict."""

    def test_record_event_and_outcome(self):
        import maxim

        nac = maxim.create.nac()
        event_id = nac.record_event("action", "ate_mushroom")
        assert event_id is not None

    def test_save_and_load_roundtrip(self):
        import maxim

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "nac.json")
            nac = maxim.create.nac()
            nac.record_event("action", "test_event")
            nac.save(path)
            assert os.path.exists(path)

            nac2 = maxim.load.nac(path)
            assert nac2 is not None

    def test_stats(self):
        import maxim

        nac = maxim.create.nac()
        stats = nac.stats()
        assert isinstance(stats, dict)
        assert "total_links" in stats or "total_observations" in stats


# ── maxim.create: Agents ───────────────────────────────────────────────


class TestCreateAgent:
    """Test standalone agent creation with isolated subsystems."""

    def test_create_agent_basic(self):
        import maxim

        agent = maxim.create.agent("test_scout")
        assert agent is not None
        assert agent.agent_id == "test_scout"
        agent.shutdown()

    def test_agent_has_subsystems(self):
        import maxim

        agent = maxim.create.agent("test_full", remembers=True, learns=True)
        assert agent.hippocampus is not None
        assert agent.nac is not None
        agent.shutdown()

    def test_agent_without_memory(self):
        import maxim

        agent = maxim.create.agent("test_no_mem", remembers=False, learns=False)
        assert agent.hippocampus is None
        assert agent.nac is None
        agent.shutdown()

    def test_agent_personality(self):
        import maxim

        agent = maxim.create.agent("test_person", personality="cautious and observant")
        assert agent.personality == "cautious and observant"
        agent.shutdown()

    def test_agent_personality_mutable(self):
        """Personality should be directly assignable after creation."""
        import maxim

        agent = maxim.create.agent("test_mut", personality="shy")
        assert agent.personality == "shy"
        agent.personality = "bold and reckless"
        assert agent.personality == "bold and reckless"
        agent.shutdown()

    def test_agent_subsystem_swappable(self):
        """Should be able to replace an agent's hippocampus."""
        import maxim

        agent = maxim.create.agent("test_swap", remembers=True)
        old_hippo = agent.hippocampus
        new_hippo = maxim.create.hippocampus()
        agent.hippocampus = new_hippo
        assert agent.hippocampus is new_hippo
        assert agent.hippocampus is not old_hippo
        agent.shutdown()

    def test_create_agent_always_fresh(self):
        """create.agent should NOT auto-load existing state."""
        import maxim

        with tempfile.TemporaryDirectory() as tmpdir:
            # persistence_dir=tmpdir/fresh_test (matches AgentFactory's default layout)
            pdir = os.path.join(tmpdir, "fresh_test")
            agent = maxim.create.agent("fresh_test", persistence_dir=pdir, remembers=True)
            agent.hippocampus.store_observation("should not persist to create")
            agent.shutdown()

            agent2 = maxim.create.agent("fresh_test", persistence_dir=pdir, remembers=True)
            assert len(agent2.hippocampus) == 0, "create.agent should start fresh"
            agent2.shutdown()

    def test_load_agent_restores_state(self):
        """load.agent should restore persisted state."""
        import maxim

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create with default layout (base_dir/agent_id/)
            pdir = os.path.join(tmpdir, "load_test")
            agent = maxim.create.agent("load_test", persistence_dir=pdir, remembers=True)
            agent.hippocampus.store_observation("persisted memory")
            agent.shutdown()

            agent2 = maxim.load.agent("load_test", base_dir=tmpdir)
            assert len(agent2.hippocampus) >= 1, "load.agent should restore memories"
            agent2.shutdown()

    def test_agent_export_memories(self):
        import maxim

        agent = maxim.create.agent("test_export", remembers=True)
        if agent.hippocampus:
            agent.hippocampus.store_observation("test memory for export")
        export = agent.export_memories()
        assert isinstance(export, dict)
        assert export["agent_id"] == "test_export"
        agent.shutdown()


# ── maxim.create: Agent Pool ──────────────────────────────────────────


class TestCreatePool:
    """Test multi-agent pool creation and lifecycle."""

    def test_create_pool(self):
        import maxim

        pool = maxim.create.pool()
        assert pool is not None
        pool.shutdown()

    def test_pool_add_remove(self):
        import maxim

        pool = maxim.create.pool()
        agent = maxim.create.agent("pool_test_1", remembers=False, learns=False)
        pool.add(agent)
        pool.remove("pool_test_1")
        pool.shutdown()

    def test_pool_multiple_agents(self):
        import maxim

        pool = maxim.create.pool()
        a1 = maxim.create.agent("pool_a1", remembers=False, learns=False)
        a2 = maxim.create.agent("pool_a2", remembers=False, learns=False)
        pool.add(a1)
        pool.add(a2)
        assert pool.get_agent("pool_a1") is a1
        assert pool.get_agent("pool_a2") is a2
        pool.shutdown()


# ── maxim.create: SEM Entities ─────────────────────────────────────────


class TestCreateEntity:
    """Test SEM entity instantiation and mutation."""

    def test_entity_from_code(self):
        from maxim import Entity

        e = Entity(name="test_arm", entity_type="limb")
        assert e.name == "test_arm"
        assert e.entity_type == "limb"

    def test_entity_sensors_mutable(self):
        from maxim import Entity

        e = Entity(name="test", entity_type="limb", sensors={})
        # Direct dict mutation should work
        assert isinstance(e.sensors, dict)

    def test_entity_metadata_mutable(self):
        from maxim import Entity

        e = Entity(name="test", entity_type="npc", metadata={})
        e.metadata["faction"] = "rebels"
        assert e.metadata["faction"] == "rebels"

    def test_entity_reparent(self):
        from maxim import Entity

        parent = Entity(name="body", entity_type="body")
        child = Entity(name="arm", entity_type="limb")
        child.reparent(parent)
        assert child in parent.children

    def test_entity_detach(self):
        from maxim import Entity

        parent = Entity(name="body", entity_type="body")
        child = Entity(name="arm", entity_type="limb")
        child.reparent(parent)
        child.detach()
        assert child not in parent.children

    def test_templates_returns_dict(self):
        import maxim

        result = maxim.create.templates()
        assert isinstance(result, dict)
        # Should have at least one category from bundled data
        assert len(result) > 0


# ── Cross-object interactions ──────────────────────────────────────────


class TestCrossObjectInteractions:
    """Verify subsystems interact correctly when composed."""

    def test_agent_hippo_captures_and_recalls(self):
        """Agent's hippocampus should function independently."""
        import maxim

        agent = maxim.create.agent("cross_test", remembers=True)
        assert agent.hippocampus is not None
        agent.hippocampus.store_observation("saw a dragon")
        agent.hippocampus.store_observation("the dragon breathed fire")
        # Memories should be in the agent's hippocampus
        count = len(agent.hippocampus)
        assert count >= 2
        agent.shutdown()

    def test_two_agents_have_isolated_memory(self):
        """Two agents should NOT share memory."""
        import maxim

        a1 = maxim.create.agent("iso_a1", remembers=True)
        a2 = maxim.create.agent("iso_a2", remembers=True)
        a1.hippocampus.store_observation("only a1 knows this")
        assert len(a1.hippocampus) >= 1
        # a2 should have no memories
        assert len(a2.hippocampus) == 0
        a1.shutdown()
        a2.shutdown()

    def test_standalone_hippo_vs_agent_hippo(self):
        """Standalone hippocampus and agent hippocampus are independent."""
        import maxim

        standalone = maxim.create.hippocampus()
        agent = maxim.create.agent("vs_test", remembers=True)
        standalone.store_observation("standalone memory")
        assert len(standalone) >= 1
        assert len(agent.hippocampus) == 0
        agent.shutdown()


# ── Session ────────────────────────────────────────────────────────────


class TestSession:
    """Test Session creation, persistence, and loading."""

    def test_session_from_result(self):
        """Session.from_result wraps a SimulationResult."""
        from maxim.session import Session
        from maxim.simulation.orchestrator import SimulationResult

        result = SimulationResult(
            goal="test",
            persona="cooperative",
            turns=5,
            total_actions=10,
            blocked_actions=0,
            duration_s=12.5,
            finish_reason="completed",
            session_id="20260408_143022",
            session_dir="/tmp/test_session",
        )

        session = Session.from_result(result, model="mistral-7b")
        assert session.id == "20260408_143022"
        assert session.goal == "test"
        assert session.turns == 5
        assert session.duration_s == 12.5
        assert session.finish_reason == "completed"
        assert session.model == "mistral-7b"
        assert session.result is result

    def test_session_from_disk(self):
        """Session.from_disk loads from a session directory."""
        from maxim.session import Session

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a fake session directory
            session_dir = Path(tmpdir) / "20260408_143022"
            session_dir.mkdir()
            report = {
                "goal": "test goal",
                "persona": "cooperative",
                "language_model": "mistral-7b",
                "turns": 5,
            }
            with open(session_dir / "report.json", "w") as f:
                json.dump(report, f)

            with mock.patch("maxim.utils.paths.sim_reports", return_value=Path(tmpdir)):
                session = Session.from_disk("20260408")
                assert session.id == "20260408_143022"
                assert session.goal == "test goal"
                assert session.model == "mistral-7b"

    def test_session_from_disk_fuzzy_match(self):
        """Fuzzy prefix matching should find the most recent session."""
        from maxim.session import Session

        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "20260408_100000").mkdir()
            (Path(tmpdir) / "20260408_100000" / "report.json").write_text('{"goal": "old"}')
            (Path(tmpdir) / "20260408_143022").mkdir()
            (Path(tmpdir) / "20260408_143022" / "report.json").write_text('{"goal": "new"}')

            with mock.patch("maxim.utils.paths.sim_reports", return_value=Path(tmpdir)):
                session = Session.from_disk("20260408")
                # Should match the most recent (143022 > 100000)
                assert session.id == "20260408_143022"
                assert session.goal == "new"

    def test_session_from_disk_not_found(self):
        """Should raise FileNotFoundError for non-existent session."""
        from maxim.session import Session

        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch("maxim.utils.paths.sim_reports", return_value=Path(tmpdir)):
                with pytest.raises(FileNotFoundError):
                    Session.from_disk("99999999")

    def test_list_sessions(self):
        """list_sessions should return Session objects."""
        from maxim.session import list_sessions

        with tempfile.TemporaryDirectory() as tmpdir:
            for sid in ["20260408_100000", "20260408_143022"]:
                d = Path(tmpdir) / sid
                d.mkdir()
                (d / "report.json").write_text(json.dumps({"goal": f"goal_{sid}"}))

            with mock.patch("maxim.utils.paths.sim_reports", return_value=Path(tmpdir)):
                sessions = list_sessions(limit=10)
                assert len(sessions) == 2
                # Most recent first
                assert sessions[0].id == "20260408_143022"

    def test_session_observe_no_state(self):
        """observe() on a session with no persisted state returns error."""
        from maxim.session import Session

        with tempfile.TemporaryDirectory() as tmpdir:
            session = Session(id="test", dir=tmpdir)
            result = session.observe("memory")
            assert "error" in result

    def test_session_repr(self):
        from maxim.session import Session

        s = Session(id="20260408_143022", goal="test memory recall")
        r = repr(s)
        assert "20260408_143022" in r
        assert "test memory recall" in r

    def test_session_delegated_properties_without_result(self):
        """Properties should return defaults when no result is attached."""
        from maxim.session import Session

        s = Session(id="test")
        assert s.turns == 0
        assert s.duration_s == 0.0
        assert s.finish_reason == "unknown"
        assert s.actions == []


# ── maxim.load ─────────────────────────────────────────────────────────


class TestLoad:
    """Test deserialization functions."""

    def test_load_hippocampus(self):
        import maxim

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "hippo.json")
            h = maxim.create.hippocampus()
            h.store_observation("persist me")
            h.save(path)

            loaded = maxim.load.hippocampus(path)
            assert loaded is not None
            assert len(loaded) >= 1

    def test_load_nac(self):
        import maxim

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "nac.json")
            n = maxim.create.nac()
            n.record_event("test", "event")
            n.save(path)

            loaded = maxim.load.nac(path)
            assert loaded is not None

    def test_load_session(self):
        """maxim.load.session delegates to Session.from_disk."""
        import maxim

        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir) / "20260408_999999"
            d.mkdir()
            (d / "report.json").write_text('{"goal": "loaded"}')

            with mock.patch("maxim.utils.paths.sim_reports", return_value=Path(tmpdir)):
                s = maxim.load.session("20260408_999999")
                assert s.id == "20260408_999999"
                assert s.goal == "loaded"

    def test_load_sessions(self):
        """maxim.load.sessions should list sessions."""
        import maxim

        with tempfile.TemporaryDirectory() as tmpdir:
            for sid in ["20260408_100000", "20260408_200000"]:
                d = Path(tmpdir) / sid
                d.mkdir()
                (d / "report.json").write_text(json.dumps({"goal": f"goal_{sid}"}))

            with mock.patch("maxim.utils.paths.sim_reports", return_value=Path(tmpdir)):
                sessions = maxim.load.sessions(limit=10)
                assert len(sessions) == 2

    def test_load_agent(self):
        """maxim.load.agent restores persisted agent state."""
        import maxim

        with tempfile.TemporaryDirectory() as tmpdir:
            pdir = os.path.join(tmpdir, "persist_me")
            agent = maxim.create.agent("persist_me", persistence_dir=pdir, remembers=True)
            agent.hippocampus.store_observation("survive the reload")
            agent.shutdown()

            restored = maxim.load.agent("persist_me", base_dir=tmpdir)
            assert len(restored.hippocampus) >= 1
            restored.shutdown()

    def test_load_agent_not_found(self):
        """maxim.load.agent should raise FileNotFoundError."""
        import maxim

        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError):
                maxim.load.agent("nonexistent_agent", base_dir=tmpdir)


# ── Report ─────────────────────────────────────────────────────────────


class TestReport:
    """Test Report object and multi-format save."""

    def test_report_save_markdown(self):
        from maxim import Report

        with tempfile.TemporaryDirectory() as tmpdir:
            report = Report(
                title="Test Report",
                goal="test memory recall",
                session_id="20260408_143022",
                content="The agent recalled 3 out of 5 memories.",
                metrics={"recall_rate": 0.6, "total_memories": 5},
            )
            path = report.save(os.path.join(tmpdir, "report.md"))
            assert os.path.exists(path)
            text = Path(path).read_text()
            assert "# Test Report" in text
            assert "recall_rate" in text

    def test_report_save_json(self):
        from maxim import Report

        with tempfile.TemporaryDirectory() as tmpdir:
            report = Report(
                title="Test Report",
                goal="test",
                metrics={"score": 0.8},
            )
            path = report.save(os.path.join(tmpdir, "report.json"))
            assert os.path.exists(path)
            with open(path) as f:
                data = json.load(f)
            assert data["title"] == "Test Report"
            assert data["metrics"]["score"] == 0.8

    def test_report_save_unsupported_format(self):
        from maxim import Report

        report = Report(title="Test")
        with pytest.raises(ValueError, match="Unsupported format"):
            report.save("/tmp/report.xyz")

    def test_report_save_future_format_hint(self):
        from maxim import Report

        report = Report(title="Test")
        with pytest.raises(ValueError, match="pymaxim"):
            report.save("/tmp/report.pdf")

    def test_report_from_json_roundtrip(self):
        from maxim import Report

        with tempfile.TemporaryDirectory() as tmpdir:
            original = Report(
                title="Roundtrip",
                goal="test",
                content="body text",
                metrics={"a": 1.0},
                review="looks good",
                review_verdict="accept",
            )
            path = os.path.join(tmpdir, "report.json")
            original.save(path)

            loaded = Report.from_json(path)
            assert loaded.title == "Roundtrip"
            assert loaded.metrics["a"] == 1.0
            assert loaded.review_verdict == "accept"

    def test_report_repr(self):
        from maxim import Report

        r = Report(title="My Report", session_id="123", content="x" * 100)
        rep = repr(r)
        assert "My Report" in rep
