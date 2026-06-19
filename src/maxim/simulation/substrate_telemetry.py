"""Per-tick substrate telemetry — Phase 0 of grounded_language_acquisition.md.

Snapshots EC node count, NAc reward_bias, drive states, and recently
proposed actions to a JSONL file. The cradle-prelinguistic harness
reads this offline to answer Phase 0's measurement questions:

  * EC node count over time — do clusters form?
  * NAc reward_bias distribution — do biases form around drive
    resolution?
  * Sensor readings — what does the substrate see while it learns?

The writer is fail-soft: any exception during snapshot or write is
swallowed (logged at DEBUG). Telemetry must never crash the AUT loop.

Read alongside:
- docs/plans/grounded_language_acquisition.md Phase 0
- src/maxim/runtime/agent_loop.py::propose_via_substrate (the call site)
- src/maxim/simulation/orchestrator.py (the wire-up)
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class SubstrateTelemetry:
    """Per-tick JSONL writer for substrate-primary mode.

    Constructed in the orchestrator when ``--research`` is set with
    ``--aut-mode substrate-primary``. The agent loop calls
    ``snapshot()`` each iteration after the substrate proposal step;
    output goes to ``{sim_workspace}/substrate_telemetry.jsonl``.
    """

    def __init__(self, *, log_path: str | Path, agent_id: str) -> None:
        self._log_path = Path(log_path)
        self._agent_id = agent_id
        self._lock = threading.Lock()
        # Touch the file so callers can `tail -f` immediately.
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_path.touch()
        self._tick_count = 0

    @property
    def log_path(self) -> Path:
        return self._log_path

    @property
    def tick_count(self) -> int:
        return self._tick_count

    def snapshot(
        self,
        *,
        step: int,
        nac: Any | None,
        ec: Any | None,
        executor: Any | None,
        proposal: Any | None,
    ) -> None:
        """Write one tick of substrate state to the JSONL file.

        Args:
            step: Loop step counter (monotonic per session).
            nac: NAc instance — ``None`` skips the NAc snapshot.
            ec: EC instance (typically ``memory_hub.ec``) — ``None``
                skips the EC snapshot.
            executor: Executor with ``embodiment``. ``None`` or no
                embodiment skips drive/sensor snapshots.
            proposal: ``LLMProposal`` from ``propose_via_substrate`` —
                ``None`` for IDLE ticks (substrate had no opinion).
        """
        try:
            row = {
                "ts": time.time(),
                "step": step,
                "agent_id": self._agent_id,
                "ec": _ec_snapshot(ec),
                "nac": _nac_snapshot(nac, self._agent_id),
                "drives": _drive_snapshot(executor),
                "proposal": _proposal_snapshot(proposal),
            }
        except Exception as e:  # never fail the loop on telemetry
            logger.debug("substrate telemetry snapshot failed: %s", e)
            return

        try:
            with self._lock:
                with self._log_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row, sort_keys=True))
                    f.write("\n")
                self._tick_count += 1
        except Exception as e:
            logger.debug("substrate telemetry write failed: %s", e)


# ---------------------------------------------------------------------------
# Snapshot helpers — each returns a JSON-serialisable shape; never raise.
# ---------------------------------------------------------------------------


def _ec_snapshot(ec: Any | None) -> dict[str, Any]:
    if ec is None:
        return {"node_count": 0, "available": False}
    try:
        node_count = getattr(ec, "substrate_node_count", 0)
        # Don't dump full embeddings — that would balloon the JSONL.
        # The node count + modality histogram is enough for Phase 0
        # cluster-formation analysis; embeddings are persisted via
        # standard EC save() at session end.
        modalities: dict[str, int] = {}
        nodes = getattr(ec, "_substrate_nodes", None)
        if isinstance(nodes, dict):
            for _emb_mod in nodes.values():
                # Stored as ``(embedding, modality)`` per EC.register_substrate_node
                if isinstance(_emb_mod, tuple) and len(_emb_mod) >= 2:
                    mod = str(_emb_mod[1])
                    modalities[mod] = modalities.get(mod, 0) + 1
        return {
            "node_count": int(node_count),
            "modalities": modalities,
            "available": True,
        }
    except Exception:
        return {"node_count": 0, "available": False}


def _nac_snapshot(nac: Any | None, agent_id: str) -> dict[str, Any]:
    if nac is None:
        return {"available": False}
    try:
        # _reward_bias is keyed (agent_id, node_id) -> float in [0, max]
        bias_map = getattr(nac, "_reward_bias", {})
        if not isinstance(bias_map, dict):
            return {"available": False}
        agent_biases: dict[str, float] = {}
        for key, val in bias_map.items():
            if isinstance(key, tuple) and len(key) == 2 and key[0] == agent_id and isinstance(val, (int, float)):
                agent_biases[str(key[1])] = float(val)
        # Causal link count is also useful for tracking learning rate.
        link_count = 0
        links = getattr(nac, "_links", {})
        # Per-event positive/negative max-confidence map. This is the field that
        # exposes whether the substrate's LEARNING SIGNAL differentiates harmful
        # from safe actions — `reward_bias` is clamped >=0 and stays ~0 for
        # harmful actions, so aversion lives only in negative-valence CausalLinks
        # (which recommend_action subtracts at 0.5 weight). Without this, Exp 41's
        # within-session-learning hypothesis (H2) is unmeasurable from telemetry.
        # Shape: {event_signature: {"pos": max_pos_conf, "neg": max_neg_conf,
        # "net": pos - 0.5*neg}}. Mirrors recommend_action's scoring arithmetic.
        causal: dict[str, dict[str, float]] = {}
        if isinstance(links, dict):
            for event_sig, link_list in links.items():
                if not isinstance(link_list, list):
                    continue
                link_count += len(link_list)
                pos_max = 0.0
                neg_max = 0.0
                for link in link_list:
                    valence = getattr(getattr(link, "outcome_valence", None), "value", None)
                    conf = getattr(link, "confidence", None)
                    if not isinstance(conf, (int, float)):
                        continue
                    if valence == "positive":
                        pos_max = max(pos_max, float(conf))
                    elif valence == "negative":
                        neg_max = max(neg_max, float(conf))
                if pos_max or neg_max:
                    causal[str(event_sig)] = {
                        "pos": round(pos_max, 4),
                        "neg": round(neg_max, 4),
                        "net": round(pos_max - 0.5 * neg_max, 4),
                    }
        return {
            "available": True,
            "reward_bias_count": len(agent_biases),
            "reward_bias": agent_biases,
            "causal_link_count": link_count,
            "causal_links": causal,
        }
    except Exception:
        return {"available": False}


def _drive_snapshot(executor: Any | None) -> dict[str, Any]:
    if executor is None:
        return {"available": False}
    embodiment = getattr(executor, "embodiment", None)
    if embodiment is None or getattr(embodiment, "root", None) is None:
        return {"available": False}
    drives: dict[str, float] = {}
    sensors: dict[str, float] = {}
    try:
        for ent in embodiment.root.walk():
            for ds_name in getattr(ent, "drive_specs", {}):
                if "." in ds_name:
                    mod_name, sensor_name = ds_name.split(".", 1)
                    mod = ent.modulators.get(mod_name)
                    if mod is None or not hasattr(mod, "vital_metrics"):
                        continue
                    val = mod.vital_metrics.get(sensor_name)
                else:
                    val = ent.vital_metrics.get(ds_name)
                if isinstance(val, (int, float)):
                    drives[ds_name] = float(val)
            # Capture vital_metrics not already classified as drives —
            # gives us non-drive sensors (durability, integrity, etc.)
            for vname, vval in getattr(ent, "vital_metrics", {}).items():
                if vname in drives:
                    continue
                if isinstance(vval, (int, float)):
                    sensors[f"{ent.name}.{vname}"] = float(vval)
    except Exception:
        return {"available": False}
    return {"available": True, "drives": drives, "sensors": sensors}


def _proposal_snapshot(proposal: Any | None) -> dict[str, Any] | None:
    if proposal is None:
        return None
    try:
        action = getattr(proposal, "action", None) or {}
        return {
            "tool_name": action.get("tool_name"),
            "params": dict(action.get("params") or {}),
            "confidence": float(getattr(proposal, "confidence", 0.0)),
            "reasoning": str(getattr(proposal, "reasoning", ""))[:200],
            "strategy_used": getattr(proposal, "strategy_used", None),
        }
    except Exception:
        return None
