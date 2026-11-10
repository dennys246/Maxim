"""Exp 63c freeze, fidelity check (judge-free half; delta confounding F3).

Every simcore.simulate() call the replay makes (with and without the carried memories) is ALSO run through the
SHIPPED recall: the exact pool loaded into a real Hippocampus (load_state), the cue enriched through the shipped
BioEnrichmentPipeline with the turn's own capture excluded, and the logged enrichment_trace's (memory_ids,
memory_paths) compared with the simulator's. The judge half (re-deriving R3d through the shipped 63c judge port) waits
on the build session's port.

python fidelity.py <script_module> <intents_module> [policies...]   (env variants as replay.py)
"""

import json
import logging
import runpy
import sys

import simcore
from maxim.integration.bio_enrichment import BioEnrichmentPipeline, EnrichmentContext
from maxim.memory.hippocampus import Hippocampus, HippocampusConfig

traces: list[dict] = []
handler = logging.Handler()
handler.emit = lambda r: traces.append(r.data) if getattr(r, "event", "") == "enrichment_trace" else None
log = logging.getLogger("maxim.integration.bio_enrichment")
log.addHandler(handler)
log.setLevel(logging.INFO)

STAT = {"calls": 0, "mismatch": 0, "examples": [], "paths": {}}
_orig = simcore.simulate


def shipped(pool, cue, goal, own_ids):
    hc = Hippocampus(HippocampusConfig(auto_save_after_sleep=False))
    hc.load_state({"memories": [json.loads(json.dumps(m.to_dict(), default=str)) for m in pool]})
    own = next(iter(own_ids), None)
    n = len(traces)
    BioEnrichmentPipeline(hippocampus=hc, trace_agent_id="sim_aut").enrich(
        cue, context=EnrichmentContext(active_goal=goal, user_input=cue, own_capture_id=own), bypass_gate=True
    )
    if len(traces) == n:
        return None
    tr = traces[-1]
    return list(zip(tr.get("memory_ids", []), tr.get("memory_paths", [])))


def checked(pool, cue, goal=simcore.GOAL, own_ids=()):
    sim = _orig(pool, cue, goal, own_ids)
    real = shipped(pool, cue, goal, own_ids)
    STAT["calls"] += 1
    for _, p in real or []:
        STAT["paths"][p] = STAT["paths"].get(p, 0) + 1
    if real is None or [tuple(x) for x in real] != [tuple(x) for x in sim]:
        STAT["mismatch"] += 1
        if len(STAT["examples"]) < 5:
            STAT["examples"].append({"cue": cue, "sim": sim, "shipped": real})
    return sim


simcore.simulate = checked
sys.argv = ["replay.py", *sys.argv[1:]]
runpy.run_path("replay.py", run_name="__main__")
print(f"\nFIDELITY: simulate() calls={STAT['calls']} mismatches vs shipped recall={STAT['mismatch']} "
      f"shipped paths={STAT['paths']}")
for e in STAT["examples"]:
    print("  MISMATCH", json.dumps(e, default=str)[:600])
