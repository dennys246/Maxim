"""Two-session round-trip: NAc causal learning survives a process boundary.

The acceptance test for docs/plans/archive/nac_cross_session_persistence.md
(the 1.0 research claim: cross-session learning without fine-tuning).

Session 1 learns (causal link, Pavlovian valence, cluster bias, an
episodic memory, an EC signature) and ends. Session 2 — a DIFFERENT
process with a DIFFERENT PYTHONHASHSEED — loads via the same
``build_bio_stack`` entry point every production agent uses, and must
RECALL THE CONTENT before doing any new learning.

Why the shape is load-bearing:

- Two processes with differing PYTHONHASHSEED: without this, the Step-0
  randomized-hash bug reads as "persistence works, recall is just noisy"
  (see tests/unit/test_stable_hash_two_process.py).
- Assertions are on RECALLED CONTENT, not file existence: the reverted
  save-only attempt shipped a test asserting nac.json appears — the half
  that already worked — and would have passed over an implementation
  that TRUNCATES the previous session.
- Session 2 also learns (tool_b) and ends: the final on-disk state must
  contain BOTH sessions' learning, which is exactly what the truncating
  implementation destroyed.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_SRC = str(Path(__file__).resolve().parents[2] / "src")

# Fixed timestamp so the EC temporal_hash dimension is identical across
# sessions — the test isolates process-boundary effects, not time-of-day.
_TS = 1753800000.0


def _run_session(code: str, hashseed: str) -> dict:
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = hashseed
    env["PYTHONPATH"] = REPO_SRC + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )
    assert proc.returncode == 0, f"session subprocess failed (seed={hashseed}):\n{proc.stderr[-4000:]}"
    return json.loads(proc.stdout.strip().splitlines()[-1])


_EPISODE_SNIPPET = f"""
from maxim.memory.types import Action, Context, Decision, EpisodicMemory, Outcome, Perception

def make_episode():
    return EpisodicMemory(
        id="mem_s1_stove",
        run_id="s1",
        timestamp={_TS},
        perception=Perception(detected_objects=["stove"]),
        context=Context(active_goal="avoid the hot stove"),
        decision=Decision(intent={{"goal": "avoid the hot stove"}}, reasoning="it burned me before"),
        action=Action(tool_name="touch_stove"),
        outcome=Outcome(success=False),
    )
"""


_SESSION_1 = """
import json

from maxim.decisions.nac import Valence
from maxim.runtime.bio_stack import build_bio_stack

{episode_snippet}

bio = build_bio_stack(persistence_dir={dir!r}, agent_id="rt_agent")
bio.memory_hub.on_session_start()

# Causal learning: tool_a succeeded.
bio.nac.observe(
    "tool_call", "tool_a", "result", "success", Valence.POSITIVE, 0.5,
    context={{"agent_id": "rt_agent"}},
)
# Pavlovian percept valence (slow cross-session surface).
bio.nac.record_percept_valence("dragon", "burn", -0.5, agent_id="rt_agent")
# Cluster-keyed bias (fast surface, 1-day wall half-life).
bio.nac.update_cluster_reward("rt_agent", "cluster_1", "tool:warm", 1.0)
# Episodic memory + EC signature (the NAc min_similarity=0.5 gate rides
# on EC signature matching surviving the process boundary).
mem = make_episode()
bio.hippocampus.store(mem)
bio.ec.register(mem.id, memory=mem)

results = bio.memory_hub.on_session_end()
print(json.dumps({{
    "memory_id": mem.id,
    "cluster_bias": bio.nac.cluster_reward_bias("rt_agent", "cluster_1", "tool:warm"),
    "percept_valence": bio.nac.get_percept_valence("dragon", "burn", agent_id="rt_agent"),
}}))
"""


_SESSION_2 = """
import json

from maxim.decisions.nac import Valence
from maxim.runtime.bio_stack import build_bio_stack
from maxim.similarity.signature import SituationSignature

{episode_snippet}

bio = build_bio_stack(persistence_dir={dir!r}, agent_id="rt_agent")
bio.memory_hub.on_session_start()

# ---- RECALL BEFORE ANY NEW LEARNING ----
links_a = bio.nac.get_links_for_event("tool_a")
restored = {{
    "tool_a_links": len(links_a),
    "tool_a_outcome": links_a[0].outcome_signature if links_a else None,
    "tool_a_valence": links_a[0].outcome_valence.value if links_a else None,
    "percept_valence": bio.nac.get_percept_valence("dragon", "burn", agent_id="rt_agent"),
    "cluster_bias": bio.nac.cluster_reward_bias("rt_agent", "cluster_1", "tool:warm"),
}}

# Episodic recall by content.
recalled = bio.hippocampus.recall(tool="touch_stove")
restored["recalled_goals"] = [
    getattr(getattr(m, "context", None), "active_goal", None) for m in recalled
]

# EC: a fresh signature for the IDENTICAL situation, hashed under THIS
# process's seed, must match the persisted one above the 0.5 gate that
# decisions/nac.py:~1335 queries with.
fresh_sig = SituationSignature.from_memory(make_episode())
matches = bio.ec.find_similar(fresh_sig, k=5, min_similarity=0.5)
restored["ec_matches"] = [[mid, round(score, 4)] for mid, score in matches]

# ---- NEW LEARNING (session 2 must ADD, not truncate) ----
bio.nac.observe(
    "tool_call", "tool_b", "result", "success", Valence.POSITIVE, 0.5,
    context={{"agent_id": "rt_agent"}},
)
bio.memory_hub.on_session_end()
print(json.dumps(restored))
"""


class TestTwoSessionRoundTrip:
    def test_session_2_recalls_session_1_content_and_does_not_truncate(self, tmp_path):
        d = str(tmp_path / "agent_home")

        s1 = _run_session(
            _SESSION_1.format(dir=d, episode_snippet=_EPISODE_SNIPPET),
            hashseed="1",
        )
        agent_home = Path(d)
        for fname in ("nac.json", "ec.json", "hippocampus.json"):
            assert (agent_home / fname).exists(), f"session 1 did not persist {fname}"
        assert s1["cluster_bias"] > 0

        nac_after_s1 = json.loads((agent_home / "nac.json").read_text())
        assert "tool_a" in nac_after_s1["links"]
        assert "saved_at" in nac_after_s1

        s2 = _run_session(
            _SESSION_2.format(dir=d, episode_snippet=_EPISODE_SNIPPET),
            hashseed="2",
        )

        # -- Recalled CONTENT, not file existence --------------------------
        assert s2["tool_a_links"] == 1, "session 2 did not restore session 1's causal link"
        assert s2["tool_a_outcome"] == "success"
        assert s2["tool_a_valence"] == "positive"
        # Pavlovian valence survives (7-day half-life; gap is seconds) —
        # session 2 restores what session 1 actually STORED (the alpha-
        # scaled accumulation), not a fresh default.
        assert s1["percept_valence"] < 0
        assert s2["percept_valence"] == pytest.approx(s1["percept_valence"], rel=0.02)
        # Cluster bias survives a seconds-long gap near-fresh (1-day half-life).
        assert s2["cluster_bias"] > 0.9 * s1["cluster_bias"]
        # Episodes survive: recall by tool returns session 1's goal content.
        assert "avoid the hot stove" in s2["recalled_goals"], (
            f"episodic recall lost session 1 content: {s2['recalled_goals']}"
        )
        # EC signature match across the process boundary at ~identity —
        # pre-Step-0 this scored ~0.425, under NAc's 0.5 gate.
        ec_match_ids = {mid for mid, _ in s2["ec_matches"]}
        assert s1["memory_id"] in ec_match_ids, f"EC lost cross-process signature matching: {s2['ec_matches']}"
        top_score = dict(s2["ec_matches"])[s1["memory_id"]]
        assert top_score >= 0.99

        # -- Non-truncation: disk after session 2 holds BOTH sessions ------
        nac_after_s2 = json.loads((agent_home / "nac.json").read_text())
        assert "tool_a" in nac_after_s2["links"], (
            "session 2 TRUNCATED session 1's learning — the exact failure mode the reverted save-only patch shipped"
        )
        assert "tool_b" in nac_after_s2["links"]
        # Episodes: session 1's memory is still on disk after session 2's
        # sleep consolidation.
        hippo_after_s2 = (agent_home / "hippocampus.json").read_text()
        assert "avoid the hot stove" in hippo_after_s2


class TestLoadPersistedOptOut:
    def test_load_persisted_false_starts_fresh_but_keeps_write_paths(self, tmp_path):
        """The sim orchestrator NPC pattern (review fold, Arch #2):
        write-but-don't-read. An agent home with prior state must NOT be
        restored when load_persisted=False, while persistence paths stay
        set so session-end saves keep working."""
        from maxim.decisions.nac import Valence
        from maxim.runtime.bio_stack import build_bio_stack

        d = str(tmp_path / "agent_home")
        bio = build_bio_stack(persistence_dir=d, agent_id="fresh_agent")
        bio.nac.observe(
            "tool_call",
            "tool_a",
            "result",
            "success",
            Valence.POSITIVE,
            0.5,
            context={"agent_id": "fresh_agent"},
        )
        bio.memory_hub.on_session_start()
        bio.memory_hub.on_session_end()
        assert (Path(d) / "nac.json").exists()

        bio2 = build_bio_stack(persistence_dir=d, agent_id="fresh_agent", load_persisted=False)
        assert bio2.nac.get_links_for_event("tool_a") == []
        assert len(bio2.hippocampus) == 0
        # Write paths still armed — saves keep accumulating.
        assert bio2.nac.config.persistence_path == str(Path(d) / "nac.json")
        assert bio2.ec.config.persistence_path == str(Path(d) / "ec.json")
