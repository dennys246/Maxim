"""Guards for 1.1.4 PR 3 — the Minecraft world seam's Python half.

Everything here runs against injected-transport fakes (a socketpair feeding
scripted NDJSON) — the JS bridge is dev-side and CI has no Minecraft server;
the PR 4 smoke benchmark is the end-to-end arm. Pins: the frozen wire
protocol (state/event/action_result), the PerceptSource contract with
text-shaped percepts, the backend's honesty contract (unconfirmed = failure)
and world-owned measured write-back, the body YAML parsing + feeding the
world channel (the A4 caller chain, end to end at unit level), the finite
minecraft_bread mechanic (D67), and the designed-rest/WARNING distinction
(the plan's PR 3 seam obligation).
"""

from __future__ import annotations

import json
import socket
import time


from maxim.embodiment.backends.minecraft import (
    declared_world_sensor_names,
    minecraft_modulator_factory,
)
from maxim.embodiment.spec import _parse_entity
from maxim.simulation.minecraft import MinecraftClient, MinecraftPerceptSource


def _paired_client(action_timeout_s: float = 2.0):
    """A connected MinecraftClient whose far end is ours to script."""
    ours, theirs = socket.socketpair()
    client = MinecraftClient(connection_factory=lambda: theirs, action_timeout_s=action_timeout_s)
    client.connect()
    return client, ours


def _push(sock: socket.socket, obj) -> None:
    sock.sendall(json.dumps(obj).encode() + b"\n")


def _wait(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


class TestClientProtocol:
    def test_state_snapshots_and_events_flow(self):
        client, far = _paired_client()
        try:
            _push(far, {"type": "state", "data": {"health": 17.0, "food": "12", "junk": "x"}})
            _push(far, {"type": "event", "kind": "damage", "text": "took damage"})
            assert _wait(lambda: client.latest_state().get("health") == 17.0)
            assert client.latest_state() == {"health": 17.0, "food": 12.0}  # junk dropped
            assert _wait(client.has_events)
            assert client.state_age_s() < 2.0
        finally:
            client.close()

    def test_malformed_and_unknown_lines_are_skipped_not_fatal(self):
        client, far = _paired_client()
        try:
            far.sendall(b"not json\n")
            _push(far, {"type": "mystery", "data": 1})
            _push(far, {"type": "state", "data": {"health": 5}})
            assert _wait(lambda: client.latest_state().get("health") == 5.0)
        finally:
            client.close()

    def test_action_round_trip_and_timeout(self):
        client, far = _paired_client(action_timeout_s=0.5)
        try:
            import threading

            def responder():
                buf = b""
                while b"\n" not in buf:
                    buf += far.recv(4096)
                msg = json.loads(buf.split(b"\n")[0])
                _push(
                    far,
                    {
                        "type": "action_result",
                        "id": msg["id"],
                        "ok": True,
                        "detail": "arrived",
                        "state": {"health": 20},
                    },
                )

            t = threading.Thread(target=responder, daemon=True)
            t.start()
            result = client.call_action("move_to", {"x": 1, "z": 2})
            assert result["ok"] and result["detail"] == "arrived"
            # no responder this time -> UNKNOWN, not confirmed failure
            result = client.call_action("move_to", {"x": 3, "z": 4})
            assert not result["ok"] and result["unknown"] and "unconfirmed" in result["detail"]
        finally:
            client.close()

    def test_action_result_state_is_absorbed_before_routing(self):
        """THE BLOCKER regression (executor lens): the embedded snapshot must
        reach latest_state so the post-action sync reads POST-action truth —
        and a LATE result's snapshot is still absorbed while its routing
        entry is dropped (no _pending leak)."""
        client, far = _paired_client(action_timeout_s=0.3)
        try:
            _push(far, {"type": "state", "data": {"health": 20.0}})
            assert _wait(lambda: client.latest_state().get("health") == 20.0)
            # a late/unsolicited result: never requested id
            _push(far, {"type": "action_result", "id": 999, "ok": True, "detail": "late", "state": {"health": 7.0}})
            assert _wait(lambda: client.latest_state().get("health") == 7.0)
            assert client._pending == {}, "late/unsolicited results must not grow _pending"
        finally:
            client.close()

    def test_close_wakes_a_blocked_call_action_promptly(self):
        import threading

        client, _far = _paired_client(action_timeout_s=30.0)
        out = {}

        def caller():
            out["r"] = client.call_action("move_to", {"x": 1, "z": 2})

        t = threading.Thread(target=caller, daemon=True)
        t.start()
        time.sleep(0.1)
        start = time.monotonic()
        client.close()
        t.join(timeout=2.0)
        assert not t.is_alive(), "close() must wake a blocked call_action"
        assert time.monotonic() - start < 2.0
        assert out["r"]["unknown"] is True

    def test_reconnect_after_close_is_not_stillborn(self):
        client, far = _paired_client()
        client.close()
        ours2, theirs2 = socket.socketpair()
        client._factory = lambda: theirs2
        client.connect()
        try:
            _push(ours2, {"type": "state", "data": {"health": 3.0}})
            assert _wait(lambda: client.latest_state().get("health") == 3.0), (
                "a reconnected client's reader must not be stillborn"
            )
        finally:
            client.close()

    def test_lines_fragmented_across_recv_boundaries_parse(self):
        client, far = _paired_client()
        try:
            payload = json.dumps({"type": "state", "data": {"health": 11.0}}).encode() + b"\n"
            far.sendall(payload[:7])
            time.sleep(0.05)
            far.sendall(payload[7:])
            assert _wait(lambda: client.latest_state().get("health") == 11.0)
        finally:
            client.close()


class TestPerceptSource:
    def test_protocol_shape_and_text_percepts(self):
        from maxim.simulation.sources import PerceptSource

        client, far = _paired_client()
        try:
            source = MinecraftPerceptSource(client)
            assert isinstance(source, PerceptSource)
            assert source.name == "minecraft"
            assert not source.has_pending() and not source.is_exhausted()
            assert source.next_percept() is None
            _push(far, {"type": "event", "kind": "spawn", "text": "a zombie appeared nearby"})
            assert _wait(source.has_pending)
            percept = source.next_percept()
            # text-shaped by construction: MemoryHub drops non-text percepts
            text = getattr(percept, "transcript_chunk", None) or getattr(percept, "content", None)
            assert text and "zombie" in text
        finally:
            client.close()


class _FakeClient:
    """Scripted in-process client for backend tests (no sockets)."""

    def __init__(self, ok=True, detail="done", state=None):
        self.ok = ok
        self.detail = detail
        self.state = state or {}
        self.calls = []

    def call_action(self, name, params=None):
        self.calls.append((name, params))
        return {"ok": self.ok, "detail": self.detail, "state": dict(self.state)}

    def latest_state(self):
        return dict(self.state)


def _player_executor(client):
    """Parsed real body + Embodiment + attached backend, no bridge process."""
    import yaml

    from maxim.embodiment.body import Embodiment
    from maxim.embodiment.spec import attach_backends
    from maxim.utils.paths import bundled_data

    spec = yaml.safe_load((bundled_data() / "components" / "bodies" / "minecraft_player.yaml").read_text())
    entity = _parse_entity(dict(spec["entity"]))
    world = declared_world_sensor_names(entity)
    attach_backends(entity, modulator_factory=minecraft_modulator_factory(client, world_sensors=world))
    embodiment = Embodiment(entity, agent_id="mc_test")
    backend = entity.modulators["avatar"]._backend
    backend.bind_embodiment(embodiment)

    class _Ex:
        pass

    ex = _Ex()
    ex.embodiment = embodiment
    return ex, entity, backend


class TestWorldBackend:
    def test_declared_world_sensors_derive_from_the_yaml(self):
        client = _FakeClient()
        _ex, entity, backend = _player_executor(client)
        assert set(backend.world_owned_sensors) == {
            "health",
            "food",
            "light_level",
            "y_altitude",
            "nearest_hostile_dist",
            "time_of_day",
        }

    def test_confirmed_action_succeeds_and_syncs_measured_state(self):
        client = _FakeClient(ok=True, state={"health": 13.0, "food": 9.0, "unknown_key": 1.0})
        _ex, entity, backend = _player_executor(client)
        result = entity.modulators["avatar"].execute("move_to", {"x": 1.0, "z": 2.0})
        assert result.success
        assert client.calls == [("move_to", {"x": 1.0, "z": 2.0})]
        assert entity.vital_metrics["health"] == 13.0
        assert entity.vital_metrics["food"] == 9.0
        assert "unknown_key" not in entity.vital_metrics  # body YAML is the contract

    def test_refused_action_is_a_real_failure(self):
        client = _FakeClient(ok=False, detail="no path")
        _ex, entity, backend = _player_executor(client)
        result = entity.modulators["avatar"].execute("move_to", {"x": 1.0, "z": 2.0})
        assert not result.success and "no path" in (result.error or "")

    def test_timeout_books_neutral_not_failure(self):
        """The Reachy convention, faithfully (review fold): dispatch accepted,
        completion unverifiable -> mechanically optimistic + neutral tier."""
        client = _FakeClient(ok=False, detail="unconfirmed after 15s")
        client_result = {"ok": False, "unknown": True, "detail": "unconfirmed after 15s"}
        client.call_action = lambda name, params=None: client_result
        _ex, entity, backend = _player_executor(client)
        result = entity.modulators["avatar"].execute("move_to", {"x": 1.0, "z": 2.0})
        assert result.success, "unknown is not failure"
        assert result.metadata.get("outcome_valence") == "neutral"

    def test_world_channel_reads_the_synced_state_and_encodes_gained(self):
        """The A4 caller chain end to end at unit level: bridge state →
        world-owned sensors → the declared world channel → a GAINED encode
        producing a world-modality EC node with the gain in its geometry."""
        from maxim.runtime.agent_loop import _read_world_ranges, _read_world_states
        from maxim.similarity.ec import ECConfig, EntorhinalCortex
        from maxim.similarity.encoder import SensorEncoder

        client = _FakeClient(state={"health": 4.0, "food": 3.0, "light_level": 1.0, "nearest_hostile_dist": 2.0})
        ex, entity, backend = _player_executor(client)
        assert backend.sync_world_sensors() == 4
        vals = _read_world_states(ex)
        assert vals["health"] == 4.0 and vals["nearest_hostile_dist"] == 2.0
        ec = EntorhinalCortex(ECConfig())
        encoder = SensorEncoder(ec=ec)
        node = encoder.encode_sensors(agent_id="mc_test", sensors=vals, modality="world", ranges=_read_world_ranges(ex))
        assert node is not None
        assert ec._substrate_nodes[node][1] == "world"
        assert ec.encoder_provenance["sensor:world"]["gain_exponent"] == 3.0


class TestMinecraftBread:
    """D67's mechanic, where scarcity is the point (plan decision D5)."""

    def _bread_tools(self):
        import yaml

        from maxim.embodiment.entity_map import EntityMap
        from maxim.embodiment.tool_bridge import generate_tools_for_entity
        from maxim.tools.registry import ToolRegistry
        from maxim.utils.paths import bundled_data

        spec = yaml.safe_load((bundled_data() / "components" / "items" / "minecraft_bread.yaml").read_text())
        bread = _parse_entity(dict(spec["entity"]))
        registry = ToolRegistry()
        entity_map = EntityMap()
        entity_map.register(bread)
        tools = generate_tools_for_entity(bread, registry, entity_map=entity_map)
        eat = next(t for t in tools if "eat_bread" in t.name)
        return bread, eat

    def test_eating_decrements_portions(self):
        bread, eat = self._bread_tools()
        assert bread.vital_metrics["portions"] == 5.0
        out = eat.execute()
        assert out.success
        assert bread.vital_metrics["portions"] == 4.0

    def test_target_null_still_takes_the_default(self):
        """Review fold: an LLM passing target: null must not mint unlimited
        bread — None/empty take the declared default."""
        bread, eat = self._bread_tools()
        out = eat.execute(target=None)
        assert out.success
        assert bread.vital_metrics["portions"] == 4.0

    def test_empty_loaf_refuses(self):
        bread, eat = self._bread_tools()
        bread.vital_metrics["portions"] = 0.0
        out = eat.execute()
        assert not out.success, "an empty loaf must refuse the eat (requires portions >= 1)"
        assert bread.vital_metrics["portions"] == 0.0


class TestDesignedRestVsWarning:
    """The plan's PR 3 seam obligation: a gained body at neutral encodes
    nothing BY DESIGN — the loop must log that at debug, keeping the WARNING
    for genuine no-cluster outcomes."""

    def test_designed_rest_flag_tracks_the_last_encode(self):
        from maxim.similarity.ec import ECConfig, EntorhinalCortex
        from maxim.similarity.encoder import SensorEncoder

        encoder = SensorEncoder(ec=EntorhinalCortex(ECConfig()))
        at_rest = {f"s{i}": 0.5 for i in range(4)}
        assert encoder.encode_sensors(agent_id="a", sensors=at_rest, modality="world") is None
        assert encoder.last_encode_was_designed_rest(agent_id="a", modality="world")
        # repeat rest ticks ride the delta gate; the flag persists
        assert encoder.encode_sensors(agent_id="a", sensors=at_rest, modality="world") is None
        assert encoder.last_encode_was_designed_rest(agent_id="a", modality="world")
        moved = dict(at_rest, s0=1.0)
        assert encoder.encode_sensors(agent_id="a", sensors=moved, modality="world") is not None
        assert not encoder.last_encode_was_designed_rest(agent_id="a", modality="world")

    def test_loop_logs_designed_rest_at_debug_not_warning(self, caplog):
        from maxim.runtime.agent_loop import _encode_was_designed_rest
        from maxim.similarity.ec import ECConfig, EntorhinalCortex
        from maxim.similarity.encoder import SensorEncoder

        encoder = SensorEncoder(ec=EntorhinalCortex(ECConfig()))
        at_rest = {f"s{i}": 0.5 for i in range(4)}
        encoder.encode_sensors(agent_id="a", sensors=at_rest, modality="world")
        assert _encode_was_designed_rest(encoder, "a", "world")
        assert not _encode_was_designed_rest(object(), "a", "world"), (
            "a fake without the probe must keep the WARNING path"
        )
