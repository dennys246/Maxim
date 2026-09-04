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
                threading.Thread(
                    # 60 steps (not 30): the gained N=16 channel legitimately
                    # CONCENTRATES (~4 clusters vs the 6-sensor body's ~12),
                    # so a contended loop needs more proposal ticks to clear
                    # the >=2 change-driven bar — widen the run, never the gate.
                    target=run_minecraft_aut,
                    kwargs={"aut": a, "max_steps": 60, "target_hz": 8.0},
                )
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


class TestBodyFakeLockstep:
    def test_fake_snapshot_covers_every_declared_world_sensor(self):
        """The L11 lockstep: the fake's snapshot keys must cover the body's
        declared modality:world set exactly — a fake that drifts under the
        body silently starves sensors in the CI smoke, and one that grows
        past it emits keys the backend ignores."""
        import yaml

        from maxim.utils.paths import bundled_data

        spec = yaml.safe_load((bundled_data() / "components" / "bodies" / "minecraft_player.yaml").read_text())
        declared = {
            name
            for name, sd in spec["entity"]["sensors"].items()
            if isinstance(sd, dict) and sd.get("modality") == "world"
        }
        assert len(declared) >= 13, "the L11 re-measure needs the channel above the ~12 safe band"
        fake = FakeBridgeServer(seed=1)
        try:
            snap = fake._snapshot()
        finally:
            fake.close()
        assert set(snap) == declared, (
            f"fake/body drift: fake-only={set(snap) - declared} body-only={declared - set(snap)}"
        )


class TestL11AnalyzerOnSyntheticTrace:
    def test_analyzer_produces_both_arms_and_a_verdict(self, tmp_path):
        """The frozen analyzer end to end on a synthetic mini-trace with
        REALISTIC capture timing (the review round caught the first draft
        encoding ideal timing the real capture never produces): the event
        line carries the ts of the PRE-event snapshot, and the changed
        state lands in the NEXT snapshot — exactly how capture() writes.
        Numbers here are not evidence; the shape and the timing are."""
        import importlib.util
        import json as _json
        import random
        from datetime import datetime, timedelta, timezone
        from pathlib import Path as _P

        spec = importlib.util.spec_from_file_location(
            "l11r", str(_P(__file__).resolve().parents[2] / "scripts" / "l11_real_trace_remeasure.py")
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        rng = random.Random(3)
        t0 = datetime.now(timezone.utc)
        base = {
            "health": 20.0,
            "food": 20.0,
            "saturation": 5.0,
            "oxygen": 20.0,
            "light_level": 7.0,
            "y_altitude": 64.0,
            "nearest_hostile_dist": 64.0,
            "hostile_count": 0.0,
            "nearest_player_dist": 64.0,
            "distance_from_spawn": 2.0,
            "speed": 0.0,
            "on_ground": 1.0,
            "is_raining": 0.0,
            "xp_level": 0.0,
            "look_pitch": 0.0,
            "time_of_day": 0.6,
        }
        event_at = {
            100: "damage",
            200: "spawn",
            300: "damage",
            400: "spawn",
            470: "damage",
            500: "spawn",
            530: "damage",
            560: "spawn",
            590: "damage",
        }
        trace = tmp_path / "trace.jsonl"
        with trace.open("w") as f:
            f.write(_json.dumps({"kind": "header", "ts": t0.isoformat()}) + "\n")
            excited = 0
            for i in range(650):
                ts = (t0 + timedelta(seconds=0.5 * i)).isoformat()
                state = {k: v + rng.gauss(0, 0.005 * (abs(v) + 1)) for k, v in base.items()}
                if excited > 0:
                    # post-event world: the changed values PERSIST a few frames
                    state["health"] = 6.0 + rng.gauss(0, 0.1)
                    state["nearest_hostile_dist"] = 2.0 + rng.gauss(0, 0.1)
                    state["hostile_count"] = 4.0
                    excited -= 1
                f.write(_json.dumps({"kind": "snapshot", "ts": ts, "state": state}) + "\n")
                if i in event_at:
                    # capture() stamps the drain with the SAME clock as the
                    # snapshot it just wrote — the pre-event push
                    f.write(_json.dumps({"kind": "event", "ts": ts, "event_kind": event_at[i], "text": "x"}) + "\n")
                    excited = 4
        out = tmp_path / "verdict.json"

        class _A:
            pass

        a = _A()
        a.trace = str(trace)
        a.json = str(out)
        a.allow_dirty = True  # synthetic shape test, not evidence
        rc = mod.analyze(a)
        assert rc == 0
        verdict = _json.loads(out.read_text())
        for arm in ("A4", "A0"):
            for key in ("stability", "separation", "discrimination", "primary_min", "clusters"):
                assert key in verdict["result"][arm]
        assert verdict["result"]["decision"]["verdict"] in (
            "retired-eligible",
            "mitigation-confirmed",
            "not-confirmed",
            "refuted-blind",
        )
        assert verdict["result"]["apparatus"]["resolved_onsets"] >= 8

    def test_analyzer_refuses_a_trace_with_no_resolvable_onsets(self, tmp_path):
        """The review round's demonstrated gap, pinned: events past the last
        snapshot resolve to no onset and must REFUSE (exit 4), never
        zero-fill into a verdict."""
        import importlib.util
        import json as _json
        from datetime import datetime, timedelta, timezone
        from pathlib import Path as _P

        spec = importlib.util.spec_from_file_location(
            "l11r2", str(_P(__file__).resolve().parents[2] / "scripts" / "l11_real_trace_remeasure.py")
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        t0 = datetime.now(timezone.utc)
        base = {
            "health": 20.0,
            "food": 20.0,
            "saturation": 5.0,
            "oxygen": 20.0,
            "light_level": 7.0,
            "y_altitude": 64.0,
            "nearest_hostile_dist": 64.0,
            "hostile_count": 0.0,
            "nearest_player_dist": 64.0,
            "distance_from_spawn": 2.0,
            "speed": 0.0,
            "on_ground": 1.0,
            "is_raining": 0.0,
            "xp_level": 0.0,
            "look_pitch": 0.0,
            "time_of_day": 0.6,
        }
        trace = tmp_path / "trace.jsonl"
        with trace.open("w") as f:
            for i in range(620):
                ts = (t0 + timedelta(seconds=0.5 * i)).isoformat()
                f.write(_json.dumps({"kind": "snapshot", "ts": ts, "state": dict(base)}) + "\n")
            for j in range(10):  # all AFTER the last snapshot: zero onsets
                ts = (t0 + timedelta(seconds=400 + j)).isoformat()
                f.write(
                    _json.dumps({"kind": "event", "ts": ts, "event_kind": "damage" if j % 2 else "spawn", "text": "x"})
                    + "\n"
                )

        class _A:
            pass

        a = _A()
        a.trace = str(trace)
        a.json = str(tmp_path / "v.json")
        a.allow_dirty = True
        import pytest as _pytest

        with _pytest.raises(SystemExit) as exc:
            mod.analyze(a)
        assert exc.value.code == 4
