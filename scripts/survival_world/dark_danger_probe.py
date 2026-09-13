#!/usr/bin/env python3
"""Dark=danger OFFLINE wiring probe (1.3 Step 2, R2-mold): where does damage-fear land?

The Phase-1 prereg needs to know, BEFORE it is designed, which learning channels fire when
the agent takes damage while the dark world-cluster is active. Three channels are possible
(docs/wiring/substrate-learning-channels.md), and which of them actually carries the fear
decides what "the agent learns dark = danger" can honestly claim:

  A. state-blind negative CAUSAL LINK on the in-flight tool ("moving is bad" — everywhere);
  B. cluster-keyed negative reward bias via ``update_cluster_reward`` (reward<0 is accepted
     and clamped to [-cap, +cap]; sole live caller is ``tool_dispatch.record_outcome``) —
     and if so, keyed to the WORLD cluster (dark) or the INTEROCEPTION cluster (the health
     channel itself);
  C. neither — the fear-write is missing and becomes a NAMED Phase-1 build item.

Fully OFFLINE: a scripted NDJSON bridge (same frozen protocol as the real
``scripts/minecraft_bridge/index.js``) stages a dark+threatened state and a health drop
co-timed with an executed action; everything maxim-side is the REAL path
(``build_minecraft_aut`` -> real executor -> ``read_learning_side_effects`` ->
``record_outcome`` -> NAc), no hand-composed credit (D43). Print-only wiring probe — no
gated record, no claim; the finding lands in docs/wiring/. Runtime: seconds.

    python scripts/survival_world/dark_danger_probe.py
"""

from __future__ import annotations

import json
import shutil
import socket
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS_DIR))

from survival_world.common import make_fresh_encoder, settle_until  # noqa: E402

AGENT_ID = "dark_danger_probe"
EPISODES = 8
DAMAGE_PER_EPISODE = 6.0  # health 20 -> 14 during the in-flight action

# Full 17-field state (the shape scripts/minecraft_bridge/index.js::snapshot emits).
LIT_SAFE = {
    "health": 20,
    "food": 20,
    "saturation": 5,
    "oxygen": 20,
    "light_level": 15,
    "y_altitude": 69,
    "nearest_hostile_dist": 64,
    "hostile_count": 0,
    "nearest_player_dist": 64,
    "distance_from_spawn": 10,
    "offset_x": 5.5,
    "offset_z": 10.5,
    "speed": 0.0,
    "on_ground": 1,
    "is_raining": 0,
    "xp_level": 0,
    "look_pitch": 0,
    "time_of_day": 0.04,
}
DARK_THREAT = {
    **LIT_SAFE,
    "light_level": 0,
    "y_altitude": 40,
    "nearest_hostile_dist": 4,
    "hostile_count": 3,
}


class ScriptedSurvivalBridge:
    """A deterministic NDJSON bridge serving OPERATOR-SET states.

    Unlike exp56's anchor-following ``ScriptedBridgeServer`` (teleport flows), this one
    hands full control of the sensor vector to the probe: ``set_state`` changes what the
    periodic pushes carry; ``queue_result`` scripts the post-action state of the next
    ``action_result`` (echoing the request id per the frozen protocol). Dev instrument
    only — never a confirmatory record.
    """

    def __init__(self, initial: dict[str, Any], *, interval_s: float = 0.02) -> None:
        self._state = dict(initial)
        self._results: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._srv.bind(("127.0.0.1", 0))
        self._srv.listen(1)
        self.port: int = self._srv.getsockname()[1]
        self._interval = interval_s
        self._stop = threading.Event()
        threading.Thread(target=self._serve, daemon=True).start()

    def set_state(self, state: dict[str, Any]) -> None:
        with self._lock:
            self._state = dict(state)

    def queue_result(self, *, ok: bool, detail: str, post_state: dict[str, Any]) -> None:
        with self._lock:
            self._results.append({"ok": ok, "detail": detail, "state": dict(post_state)})

    def close(self) -> None:
        self._stop.set()
        try:
            self._srv.close()
        except OSError:
            pass

    def _serve(self) -> None:
        try:
            conn, _ = self._srv.accept()
        except OSError:
            return
        conn.settimeout(0.05)
        buf = b""
        last_push = 0.0
        while not self._stop.is_set():
            now = time.monotonic()
            if now - last_push >= self._interval:
                with self._lock:
                    line = json.dumps({"type": "state", "data": self._state})
                try:
                    conn.sendall(line.encode() + b"\n")
                except OSError:
                    return
                last_push = now
            try:
                buf += conn.recv(4096)
            except socket.timeout:
                continue
            except OSError:
                return
            while b"\n" in buf:
                raw, buf = buf.split(b"\n", 1)
                if not raw.strip():
                    continue
                msg = json.loads(raw)
                if msg.get("type") != "action":
                    continue
                with self._lock:
                    if self._results:
                        res = self._results.pop(0)
                        self._state = dict(res["state"])  # action outcome becomes the world
                    else:
                        res = {"ok": True, "detail": "scripted default", "state": dict(self._state)}
                    reply = {"type": "action_result", "id": msg.get("id"), **res}
                try:
                    conn.sendall((json.dumps(reply) + "\n").encode())
                except OSError:
                    return


def main() -> int:
    from maxim.agents.context_pool import ContextPool
    from maxim.runtime.agent_loop import _encode_current_clusters, _read_drive_states
    from maxim.runtime.tool_dispatch import (
        build_tool_signature,
        read_learning_side_effects,
        record_outcome,
    )
    from maxim.simulation.minecraft_harness import build_minecraft_aut
    from maxim.tools.introspection import INTROSPECTION_TOOL_NAMES

    bridge = ScriptedSurvivalBridge(LIT_SAFE)
    persistence_dir = tempfile.mkdtemp(prefix="dark_danger_probe_")
    aut = build_minecraft_aut(
        agent_id=AGENT_ID,
        bridge_port=bridge.port,
        bridge_host="127.0.0.1",
        persistence_dir=persistence_dir,
        entity_ref="bodies/minecraft_player",
    )
    encoder = make_fresh_encoder(aut)
    pool: Any = ContextPool()
    recent: list[dict] = []
    findings: dict[str, Any] = {}
    try:
        available = [t for t in aut.executor.registry.list() if t not in INTROSPECTION_TOOL_NAMES]
        probe_tool = next((t for t in available if t.endswith("_eat")), None)
        if probe_tool is None:
            print(f"INSTRUMENT ERROR: no *_eat tool in roster {available!r}")
            return 4
        sig = build_tool_signature(probe_tool, {})

        if settle_until(aut, lambda vm: vm.get("light_level") == 15, timeout_s=5.0) is None:
            print("INSTRUMENT ERROR: scripted bridge never delivered the lit state")
            return 4
        lit_clusters = _encode_current_clusters(encoder, AGENT_ID, aut.executor)

        bridge.set_state(DARK_THREAT)
        if settle_until(aut, lambda vm: vm.get("light_level") == 0, timeout_s=5.0) is None:
            print("INSTRUMENT ERROR: scripted bridge never delivered the dark state")
            return 4
        dark_clusters = _encode_current_clusters(encoder, AGENT_ID, aut.executor)

        lit_world, dark_world = lit_clusters.get("world"), dark_clusters.get("world")
        print(f"lit clusters:  {lit_clusters}")
        print(f"dark clusters: {dark_clusters}")
        if not lit_world or not dark_world or lit_world == dark_world:
            print("INSTRUMENT ERROR: scripted lit/dark states did not encode to distinct world clusters")
            return 4

        # ── Training: EPISODES of (dark cluster active) + (health drops during the action) ──
        for ep in range(EPISODES):
            hurt = {**DARK_THREAT, "health": DARK_THREAT["health"] - DAMAGE_PER_EPISODE}
            bridge.queue_result(ok=True, detail="ate bread while a zombie hit you", post_state=hurt)
            clusters = _encode_current_clusters(encoder, AGENT_ID, aut.executor)
            out = aut.executor.execute({"tool_name": probe_tool, "params": {}})
            side = read_learning_side_effects(out)
            record_outcome(
                agent_id=AGENT_ID,
                tool_name=probe_tool,
                success=bool(getattr(out, "success", False)),
                result_summary=str(getattr(out, "output", ""))[:80] or None,
                error=getattr(out, "error", None),
                reasoning="dark-danger probe: damage during action in the dark",
                recent_outcomes=recent,
                max_recent=20,
                llm_worker=None,
                context_pool=pool,
                nac=aut.bio.nac,
                tool_params={},
                cluster_id=clusters.get("interoception"),
                clusters=clusters,
                embodiment_failed=side.embodiment_failed,
                drive_potential_diff=side.drive_potential_diff,
                drive_credit_withheld=side.drive_credit_withheld,
                drive_relief_channel=side.drive_relief_channel,
                outcome_valence=side.outcome_valence,
            )
            if ep == 0:
                findings["episode0_side_effects"] = {
                    "drive_potential_diff": side.drive_potential_diff,
                    "drive_relief_channel": side.drive_relief_channel,
                    "drive_credit_withheld": side.drive_credit_withheld,
                    "outcome_valence": str(side.outcome_valence),
                    "success": bool(getattr(out, "success", False)),
                }
            # The real loop ticks evaluate_failures() once per iteration (agent_loop.py
            # ~1450-1461: "Ticking evaluate_failures() here" — the pain intake). Without
            # this tick the health drop never reaches the PainBus and the probe would
            # measure the absence of its OWN harness, not of the wiring. Settle on the
            # dropped health first so the pain evaluator sees the post-damage body.
            if settle_until(aut, lambda vm: vm.get("health") == hurt["health"], timeout_s=5.0) is None:
                print(f"INSTRUMENT ERROR: episode {ep}: damaged health never reached vital_metrics")
                return 4
            try:
                fired = aut.executor.embodiment.evaluate_failures()
            except Exception as exc:
                print(f"INSTRUMENT ERROR: evaluate_failures raised: {exc!r}")
                return 4
            findings.setdefault("failure_events_per_episode", []).append(len(fired or []))
            bridge.set_state(DARK_THREAT)  # heal between episodes (fresh 20 next round)
            settle_until(aut, lambda vm: vm.get("health") == 20, timeout_s=5.0)

        # ── Readout 0: did PAIN fire at all? (the layer beneath both credit channels) ──
        findings["pain_bus"] = {
            "stats": aut.bio.pain_bus.get_stats(),
            "recent_signals": len(aut.bio.pain_bus.recent),
        }

        # ── Readout A: state-blind negative causal link on the tool ──
        neg_links = aut.bio.nac.get_negative_outcomes(sig)
        findings["A_negative_causal_links"] = [
            {"confidence": round(getattr(link, "confidence", 0.0), 3)} for link in neg_links
        ]

        # ── Readout B: cluster-keyed bias, per candidate key ──
        nac = aut.bio.nac
        findings["B_cluster_bias"] = {
            "dark_world": nac.cluster_reward_bias(AGENT_ID, dark_world, sig),
            "lit_world": nac.cluster_reward_bias(AGENT_ID, lit_world, sig),
            "dark_interoception": nac.cluster_reward_bias(AGENT_ID, dark_clusters.get("interoception"), sig),
        }

        # ── Readout C: behaviour — is the tool avoided STATE-CONTINGENTLY? ──
        drives = _read_drive_states(aut.executor)
        picks = {}
        for label, cl in (("dark", dark_clusters), ("lit", lit_clusters), ("none", None)):
            rec = nac.recommend_action(
                agent_id=AGENT_ID,
                available_tools=available,
                current_drives=drives,
                current_clusters=cl,
                min_confidence=0.0,
            )
            picks[label] = (
                None
                if rec is None
                else {
                    "tool": rec.get("tool_name"),
                    "reasoning": str(rec.get("reasoning", ""))[:160],
                }
            )
        findings["C_recommendation_by_context"] = picks
    finally:
        try:
            aut.bio.on_session_end()
        except Exception as exc:
            print(f"WARNING: bio teardown raised: {exc!r}")
        try:
            aut.client.close()
        except (OSError, ConnectionError) as exc:
            print(f"WARNING: client close raised: {exc!r}")
        bridge.close()
        shutil.rmtree(persistence_dir, ignore_errors=True)

    print("\n--- wiring map ---")
    print(json.dumps(findings, indent=2, default=str))
    a = bool(findings["A_negative_causal_links"])
    b_world = findings["B_cluster_bias"]["dark_world"]
    b_intero = findings["B_cluster_bias"]["dark_interoception"]
    print("\n--- verdict ---")
    print(f"A  state-blind negative causal link formed:        {a}")
    print(f"B  world-cluster (dark) bias:                      {b_world:+.4f}")
    print(f"B  interoception-cluster bias:                     {b_intero:+.4f}")
    print(f"B  lit-world bias (specificity control, expect 0): {findings['B_cluster_bias']['lit_world']:+.4f}")
    if b_world < 0:
        print("=> cluster-keyed fear DOES form on the dark WORLD cluster — Phase 1 can claim it.")
    elif b_intero < 0:
        print("=> damage books negatively to INTEROCEPTION, not the world cluster — dark-keyed")
        print("   fear needs a wiring extension (named Phase-1 build item).")
    elif a:
        print("=> only the STATE-BLIND channel fired: avoidance would punish the action everywhere,")
        print("   not in the dark specifically — the fear-write is a named Phase-1 build item.")
    elif findings.get("pain_bus", {}).get("stats", {}).get("total_published", -1) == 0:
        print("=> NO PAIN WAS EVER PUBLISHED: evaluate_failures fired zero events because the")
        print("   body declares NO failure modes (minecraft_player.yaml has drives but no pain")
        print("   triggers), so damage never becomes a PainSignal and every negative-learning")
        print("   channel starves upstream — while the pain->NAc subscribers sit wired and idle")
        print("   (see pain_bus.stats.subscriber_count). Named Phase-1 build item: declare the")
        print("   body's failure modes (health-damage nociception), then RE-RUN this probe to")
        print("   measure which channels the pain actually reaches.")
    else:
        print("=> NO negative learning fired despite pain publishing — the break is in the")
        print("   pain->NAc attribution window/traces; inspect pain_bus stats vs A/B readouts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
