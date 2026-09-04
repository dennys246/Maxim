"""Guards for 1.1.4 PR 4 — the two-AUT-one-world harness and the ship gate.

The gate, executable and non-vacuous (D64's lesson): the reduced end-to-end
smoke runs the REAL `run_agentic_loop` (substrate-primary, no LLM in the
action path) for two full AUTs against one `FakeBridgeServer`, then asserts
world-modality EC nodes exist on the LIVE store AND in the ec.json the FULL
close persisted. Plus: the consolidation/aut-mode kwargs pin against the
pure `_loop_kwargs` (the exact kwargs the harness passes — not a
hand-composed sequence), the staleness gate, and the D77 acquisition fix.
"""

from __future__ import annotations

import threading
import time


from maxim.simulation.minecraft_harness import (
    FakeBridgeServer,
    MinecraftSyncPump,
    _loop_kwargs,
    build_minecraft_aut,
    run_minecraft_aut,
    smoke_verdict,
    verdict_is_green,
)


class TestLoopKwargsPins:
    def test_consolidation_full_and_substrate_primary_are_pinned(self, tmp_path):
        """The ship gate's second half: the HARNESS passes consolidation=full
        — pinned on the exact kwargs function the harness uses."""

        class _Stub:
            pass

        aut = _Stub()
        aut.percept_source = object()
        aut.bio = _Stub()
        aut.bio.pain_bus = aut.bio.memory_hub = aut.bio.hippocampus = None
        kwargs = _loop_kwargs(aut, max_steps=5, stop_event=threading.Event(), target_hz=2.0)
        assert kwargs["consolidation"] == "full"
        assert kwargs["aut_mode"] == "substrate-primary"


class TestStalenessGate:
    def test_pump_refuses_stale_snapshots(self):
        class _StaleClient:
            def state_age_s(self):
                return 99.0

        class _Backend:
            def __init__(self):
                self.calls = 0

            def sync_world_sensors(self):
                self.calls += 1
                return 1

        class _Aut:
            agent_id = "stale_test"
            client = _StaleClient()
            backend = _Backend()

        pump = MinecraftSyncPump(_Aut(), interval_s=0.01, max_state_age_s=5.0)
        pump.start()
        time.sleep(0.15)
        pump.stop()
        assert pump.stale_skips > 0
        assert _Aut.backend.calls == 0, "a stale snapshot must never reach the substrate"


class TestD77AcquisitionRegeneration:
    def test_picked_up_breads_eat_still_applies_self_effect(self, tmp_path):
        """D77 guard: after acquisition regenerates the item's tools, the
        replacement eat tool's self_effect must still write to the agent
        body (it was dead — regeneration omitted embodiment=)."""
        import yaml

        from maxim.embodiment.body import Embodiment
        from maxim.embodiment.entity_map import EntityMap
        from maxim.embodiment.spec import _parse_entity
        from maxim.tools.registry import ToolRegistry
        from maxim.utils.paths import bundled_data

        body = _parse_entity(
            {
                "name": "agent_body",
                "sensors": {"food": {"range": [0, 20], "initial": 10}},
            }
        )
        embodiment = Embodiment(body, agent_id="d77_test")

        bread_spec = yaml.safe_load((bundled_data() / "components" / "items" / "minecraft_bread.yaml").read_text())
        bread = _parse_entity(dict(bread_spec["entity"]))

        registry = ToolRegistry()
        entity_map = EntityMap()
        entity_map.register(bread)

        class _Ex:
            pass

        ex = _Ex()
        ex.registry = registry
        ex._entity_map = entity_map
        ex.embodiment = embodiment
        from maxim.runtime.executor import Executor

        Executor._handle_entity_acquisition(ex, {"entity_acquired": "minecraft_bread"})
        eat = registry.get("minecraft_bread_eat_bread")
        assert eat is not None, "acquisition must register the item's tools"
        before = body.vital_metrics["food"]
        out = eat.execute()
        assert out.success
        assert body.vital_metrics["food"] > before, (
            "the regenerated eat tool's self_effect is dead — D77 (regeneration without embodiment=)"
        )
        assert bread.vital_metrics["portions"] == 4.0


class TestReducedEndToEndSmoke:
    # Deliberately NOT slow-marked: ~5s, and this IS the 1.1.4 ship gate —
    # a gate in the nightly-only lane is a gate that mostly does not run
    # (D65's shape).

    """THE 1.1.4 ship gate, reduced: two full AUTs, one fake world, real
    agent loops, substrate-primary, full close — then the non-vacuous
    verdict on live AND persisted world nodes."""

    def test_two_aut_smoke_is_green_and_non_vacuous(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)  # the loop writes CWD-relative diagnostics
        server = FakeBridgeServer(seed=42, state_interval_s=0.05)
        auts = []
        pumps = []
        close_calls: dict[str, dict[str, int]] = {}
        try:
            for name in ("aut_a", "aut_b"):
                aut = build_minecraft_aut(
                    agent_id=name,
                    bridge_port=server.port,
                    persistence_dir=str(tmp_path / name),
                    action_timeout_s=2.0,
                )
                # RUNTIME close-flavor discriminator (architecture-lens
                # review): on_session_end_lightweight ALSO writes ec.json,
                # so the persisted-nodes assertion alone cannot tell the
                # flavors apart — this observes which closer actually ran,
                # pinning the loop->_end_bio_session seam no other test
                # covers.
                counts = {"full": 0, "lightweight": 0}
                close_calls[name] = counts
                hub = aut.bio.memory_hub
                real_full = hub.on_session_end
                real_light = hub.on_session_end_lightweight

                def _full(*a, _c=counts, _f=real_full, **k):
                    _c["full"] += 1
                    return _f(*a, **k)

                def _light(*a, _c=counts, _f=real_light, **k):
                    _c["lightweight"] += 1
                    return _f(*a, **k)

                hub.on_session_end = _full
                hub.on_session_end_lightweight = _light
                auts.append(aut)
                pump = MinecraftSyncPump(aut, interval_s=0.05, max_state_age_s=5.0)
                pump.start()
                pumps.append(pump)

            threads = [
                threading.Thread(target=run_minecraft_aut, kwargs={"aut": a, "max_steps": 30, "target_hz": 8.0})
                for a in auts
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=60.0)
                assert not t.is_alive(), "an AUT loop failed to finish"
        finally:
            for p in pumps:
                p.stop()
            for a in auts:
                a.client.close()
            server.close()

        for aut, pump in zip(auts, pumps):
            verdict = smoke_verdict(aut, pump)
            # THE gate, via the one shared function the CLI also uses. Its
            # liveness half exists because the executor lens DEMONSTRATED
            # the node counts alone go green with the pumps never started
            # (the body's non-neutral initials mint one static encode).
            assert verdict_is_green(verdict), f"{verdict}: the smoke gate is RED"
            assert verdict["ec_json_exists"], f"{verdict}: the close never persisted the EC"
        for name, counts in close_calls.items():
            assert counts["full"] >= 1, f"{name}: the FULL close never ran ({counts})"
            assert counts["lightweight"] == 0, (
                f"{name}: the session took the LIGHTWEIGHT close ({counts}) — "
                "the harness's consolidation='full' did not reach _end_bio_session"
            )

    def test_dead_world_feed_is_red(self, tmp_path, monkeypatch):
        """The executor lens's vacuity demonstration, pinned as a negative
        control: pumps never started -> the gate must be RED even though a
        static-initials encode mints a node or two."""
        monkeypatch.chdir(tmp_path)
        server = FakeBridgeServer(seed=7, state_interval_s=0.05)
        aut = None
        try:
            aut = build_minecraft_aut(
                agent_id="dead_feed",
                bridge_port=server.port,
                persistence_dir=str(tmp_path / "dead_feed"),
                action_timeout_s=2.0,
            )
            pump = MinecraftSyncPump(aut, interval_s=0.05)  # NEVER started
            run_minecraft_aut(aut, max_steps=15, target_hz=8.0)
            verdict = smoke_verdict(aut, pump)
            assert not verdict_is_green(verdict), (
                f"{verdict}: a dead world feed must not smoke GREEN (the vacuity the review demonstrated)"
            )
        finally:
            if aut is not None:
                aut.client.close()
            server.close()
