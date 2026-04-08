"""Tests for persistence fixes across all subsystems.

Covers:
- NAc atomic writes + AgentFactory persistence path + save on shutdown
- SCN persistence path in MemoryHub + save on session end
- Angular Gyrus atomic writes
- Entity to_dict/from_dict round-trip + save/load
- Session.save() method
- Agent shutdown saves all subsystems
- Cross-subsystem persistence isolation
"""

from __future__ import annotations

import json
import os
import tempfile



# ── NAc atomic writes ─────────────────────────────────────────────────


class TestNAcPersistence:
    """NAc should use atomic writes and persist via AgentFactory."""

    def test_nac_save_creates_file(self):
        import maxim
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "nac.json")
            nac = maxim.create.nac()
            nac.record_event("test", "event_1")
            nac.save(path)
            assert os.path.exists(path)
            with open(path) as f:
                data = json.load(f)
            assert data["version"] == "1.0"

    def test_nac_save_load_roundtrip(self):
        import maxim
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "nac.json")
            nac = maxim.create.nac()
            nac.record_event("action", "ate_mushroom")
            nac.save(path)

            nac2 = maxim.load.nac(path)
            stats = nac2.stats()
            assert stats is not None

    def test_nac_atomic_write(self):
        """NAc.save() should use atomic_write_json (not raw json.dump)."""
        import maxim
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "nac.json")
            nac = maxim.create.nac()
            nac.record_event("test", "event")
            nac.save(path)
            # Atomic write should not leave temp files
            files = os.listdir(tmpdir)
            assert len(files) == 1
            assert files[0] == "nac.json"


# ── SCN persistence ───────────────────────────────────────────────────


class TestSCNPersistence:
    """SCN should be persistable."""

    def test_scn_save_load_roundtrip(self):
        import maxim
        from maxim.time.temporal_signature import TemporalSignature

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "scn.json")
            scn = maxim.create.scn()
            sig = TemporalSignature.from_timestamp(1712592000.0)  # 2024-04-08 14:00 UTC
            scn.register("mem_001", sig)
            scn.save(path)
            assert os.path.exists(path)

            scn2 = maxim.create.scn()
            scn2.load(path)


# ── Angular Gyrus atomic writes ───────────────────────────────────────


class TestAngularGyrusPersistence:
    """Angular Gyrus should use atomic writes."""

    def test_angular_gyrus_save_creates_file(self):
        import maxim
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "ag.json")
            ag = maxim.create.angular_gyrus(persistence_path=path)
            ag.save(path)
            assert os.path.exists(path)


# ── Entity serialization ──────────────────────────────────────────────


class TestEntitySerialization:
    """Entity.to_dict()/from_dict() round-trip."""

    def test_entity_to_dict_basic(self):
        from maxim import Entity
        e = Entity(name="arm", entity_type="limb")
        d = e.to_dict()
        assert d["name"] == "arm"
        assert d["entity_type"] == "limb"

    def test_entity_to_dict_with_metadata(self):
        from maxim import Entity
        e = Entity(name="guard", entity_type="npc", metadata={"faction": "rebels"})
        e.vital_metrics["health"] = 0.8
        d = e.to_dict()
        assert d["metadata"]["faction"] == "rebels"
        assert d["vital_metrics"]["health"] == 0.8

    def test_entity_to_dict_with_children(self):
        from maxim import Entity
        parent = Entity(name="body", entity_type="body")
        Entity(name="arm", entity_type="limb", parent=parent)
        Entity(name="leg", entity_type="limb", parent=parent)
        d = parent.to_dict()
        assert len(d["children"]) == 2
        assert d["children"][0]["name"] == "arm"
        assert d["children"][1]["name"] == "leg"

    def test_entity_from_dict_basic(self):
        from maxim import Entity
        d = {"name": "arm", "entity_type": "limb"}
        e = Entity.from_dict(d)
        assert e.name == "arm"
        assert e.entity_type == "limb"

    def test_entity_from_dict_with_metadata(self):
        from maxim import Entity
        d = {
            "name": "guard",
            "entity_type": "npc",
            "metadata": {"faction": "rebels"},
            "vital_metrics": {"health": 0.8},
        }
        e = Entity.from_dict(d)
        assert e.metadata["faction"] == "rebels"
        assert e.vital_metrics["health"] == 0.8

    def test_entity_from_dict_with_children(self):
        from maxim import Entity
        d = {
            "name": "body",
            "entity_type": "body",
            "children": [
                {"name": "arm", "entity_type": "limb"},
                {"name": "leg", "entity_type": "limb"},
            ],
        }
        e = Entity.from_dict(d)
        assert len(e.children) == 2
        assert e.children[0].name == "arm"
        assert e.children[0].parent is e

    def test_entity_roundtrip(self):
        """to_dict → from_dict should preserve structure."""
        from maxim import Entity
        parent = Entity(name="robot", entity_type="body")
        arm = Entity(name="arm", entity_type="limb", parent=parent)
        arm.vital_metrics["position"] = 45.0
        arm.metadata["material"] = "aluminum"

        d = parent.to_dict()
        restored = Entity.from_dict(d)

        assert restored.name == "robot"
        assert len(restored.children) == 1
        restored_arm = restored.children[0]
        assert restored_arm.name == "arm"
        assert restored_arm.vital_metrics["position"] == 45.0
        assert restored_arm.metadata["material"] == "aluminum"
        assert restored_arm.parent is restored

    def test_entity_save_load_json(self):
        """Entity.save() / Entity.load() via JSON file."""
        from maxim import Entity
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "entity.json")
            e = Entity(name="test_bot", entity_type="robot")
            e.vital_metrics["battery"] = 0.95
            e.metadata["version"] = "2.0"
            Entity(name="sensor_array", entity_type="sensor_module", parent=e)
            e.save(path)

            loaded = Entity.load(path)
            assert loaded.name == "test_bot"
            assert loaded.vital_metrics["battery"] == 0.95
            assert len(loaded.children) == 1
            assert loaded.children[0].name == "sensor_array"

    def test_load_entity_via_maxim_load(self):
        """maxim.load.entity() should handle JSON files."""
        import maxim
        from maxim import Entity
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "guard.json")
            e = Entity(name="guard", entity_type="npc")
            e.save(path)

            loaded = maxim.load.entity(path)
            assert loaded.name == "guard"


# ── Session.save() ────────────────────────────────────────────────────


class TestSessionSave:
    """Session.save() should persist metadata."""

    def test_session_save_creates_file(self):
        from maxim.session import Session
        with tempfile.TemporaryDirectory() as tmpdir:
            session = Session(
                id="20260408_test",
                dir=os.path.join(tmpdir, "20260408_test"),
                goal="test persistence",
                model="mistral-7b",
            )
            result_dir = session.save()
            assert os.path.exists(result_dir)
            meta_path = os.path.join(result_dir, "session.json")
            assert os.path.exists(meta_path)
            with open(meta_path) as f:
                data = json.load(f)
            assert data["session_id"] == "20260408_test"
            assert data["goal"] == "test persistence"
            assert data["model"] == "mistral-7b"


# ── Agent shutdown saves all subsystems ───────────────────────────────


class TestAgentShutdownPersistence:
    """Agent.shutdown() should save hippocampus, NAc, and ATL."""

    def test_agent_shutdown_saves_hippocampus(self):
        import maxim
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = maxim.create.agent("persist_test", persistence_dir=tmpdir)
            if agent.hippocampus:
                agent.hippocampus.store_observation("remember me")
            agent.shutdown()

            hippo_path = os.path.join(tmpdir, "hippocampus.json")
            assert os.path.exists(hippo_path), "Hippocampus should be saved on shutdown"

    def test_agent_shutdown_saves_nac(self):
        import maxim
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = maxim.create.agent("persist_nac", persistence_dir=tmpdir, learns=True)
            if agent.nac:
                agent.nac.record_event("test", "shutdown_event")
            agent.shutdown()

            nac_path = os.path.join(tmpdir, "nac.json")
            assert os.path.exists(nac_path), "NAc should be saved on shutdown"

    def test_agent_shutdown_roundtrip(self):
        """Agent data should survive shutdown + load.agent()."""
        import maxim
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create agent and add data
            import os
            pdir = os.path.join(tmpdir, "roundtrip")
            agent = maxim.create.agent("roundtrip", persistence_dir=pdir, remembers=True, learns=True)
            agent.hippocampus.store_observation("important memory")
            agent.nac.record_event("action", "learned_event")
            agent.shutdown()

            # Load agent from same directory — should restore state
            agent2 = maxim.load.agent("roundtrip", base_dir=tmpdir)
            assert len(agent2.hippocampus) >= 1, "Hippocampus should reload memories"
            agent2.shutdown()


# ── Cross-subsystem persistence isolation ─────────────────────────────


class TestPersistenceIsolation:
    """Two agents with different persistence dirs should not interfere."""

    def test_two_agents_isolated_persistence(self):
        import maxim
        with tempfile.TemporaryDirectory() as tmpdir:
            dir_a = os.path.join(tmpdir, "agent_a")
            dir_b = os.path.join(tmpdir, "agent_b")
            os.makedirs(dir_a)
            os.makedirs(dir_b)

            a = maxim.create.agent("agent_a", persistence_dir=dir_a)
            b = maxim.create.agent("agent_b", persistence_dir=dir_b)

            a.hippocampus.store_observation("only A knows this")
            a.shutdown()
            b.shutdown()

            # Verify A's data is in A's dir, not B's
            assert os.path.exists(os.path.join(dir_a, "hippocampus.json"))
            # B's hippocampus should exist but have no memories
            b_hippo_path = os.path.join(dir_b, "hippocampus.json")
            if os.path.exists(b_hippo_path):
                with open(b_hippo_path) as f:
                    data = json.load(f)
                # B should have 0 memories
                memories = data.get("memories", {})
                assert len(memories) == 0
