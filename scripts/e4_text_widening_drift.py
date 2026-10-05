#!/usr/bin/env python3
"""E4 text-widening drift measurement (#911): does reward widening pull foreign strings into a rewarded text node?

Implements docs/experiments/protocols/e4_text_widening_drift_preregistration.md, which is the authority on any
divergence (a change to either after first data needs an amendment header there). In short:

- A latent-hazard UPPER BOUND: no live path gives a text node positive reward today, so this asks what happens the
  day one is wired. Offline, in-process, CPU, no LLM, no rig.
- Every string of the Exp 24 paraphrase fixture (``data/roy_paraphrase_pairs.json``, SHA-256 pinned) is a
  ``make_text_percept(text, agent_id=AGENT)`` encoded by the production ``LinguisticEncoder.encode`` path
  (``decomposer=None``); the harness never calls ``ec.pattern_complete_or_separate`` itself (a spy only observes
  it).
- Arms, for bias b in {0, 0.1, 0.2}: R1 SEQUENTIAL (only the seed node ``"you sense food nearby."`` rewarded; decides),
  R1 REPLAY-ISOLATED (each string against what the EC's own centroid rule would hold from the b=0 run's members
  that precede it; decides), R1 BARE-ISOLATED (reported; decides only the drift positive control), RA SEQUENTIAL
  (every node rewarded; bound only).
- Instrument checks, each a refusal (exit 4, no verdict): replay known answer, drift positive control (threshold
  0.40 vs bare-isolated), frozen-centroid negative control, R1/RA identity at b=0, determinism (two runs).
- Outcome: COLLAPSE (some foreign s in A(0.2), not in A(0.0), not in I(0.2)), NO HEADROOM, or NO COLLAPSE.

Exit 0 = an outcome (``status: ok``); 3 = provenance refusal; 4 = apparatus/instrument refusal (the ``refusal``
reason is in the record). The committed record is written only under ``--write-experiment-results`` from a clean
tree; otherwise it goes to a temp directory. Only a ``status: ok``, ``mock: false`` record from a clean tree at a
commit on ``main`` closes #911.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import struct
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any

# Every `maxim` / `sentence_transformers` / `huggingface_hub` import in this module is inside a function, so
# `pin_environment()` (the first thing `main` does) runs before any of them (prereg: Environment). Importing the
# module (the tests do) changes no environment.
ALLOWED_MAXIM_ENV = frozenset({"MAXIM_DATA_HOME"})


def pin_environment() -> list[str]:
    """Pin the offline, temp-home environment; return the unexpected MAXIM_* variables (a refusal: e.g.
    MAXIM_NAC_REWARD_BIAS_DISABLED would silently zero every arm), acted on once the record path is known."""
    unexpected = sorted(k for k in os.environ if k.startswith("MAXIM_") and k not in ALLOWED_MAXIM_ENV)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["MAXIM_DATA_HOME"] = tempfile.mkdtemp(prefix="maxim-e4-home-")
    return unexpected


REPO = Path(__file__).resolve().parents[1]

PREREG = "docs/experiments/protocols/e4_text_widening_drift_preregistration.md"
RECORD = REPO / "docs" / "experiments" / "data" / "e4_text_widening_drift" / "diagnosis.json"
FIXTURE = REPO / "data" / "roy_paraphrase_pairs.json"
FIXTURE_SHA256 = "9b83311986a4a17ba7815d2e389aab755d436be3bcfcc663b9ecb4fddce345a6"
MODEL = "paraphrase-mpnet-base-v2"
MODEL_REPO = f"sentence-transformers/{MODEL}"
MODEL_REVISION = "6cc9279c672dc57f94445ef259b28a1b736fec8f"
AGENT = "e4"
SEED = "you sense food nearby."
SEED_PAIR = "pair_01_food_detect"
BIASES = (0.0, 0.1, 0.2)
CAP = 0.2
CONTROL_THRESHOLD = 0.40  # Exp 24's collapsing setting (drift positive control)
TOL_BIAS = 1e-12
TOL_REPLAY = 1e-9
MARGINAL = 0.01
DIM = 768
SCOPE = "latent_hazard_upper_bound"
STAMPED_ENV = (
    "MAXIM_DATA_HOME",
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE",
    "HF_HOME",
    "HF_HUB_CACHE",
    "SENTENCE_TRANSFORMERS_HOME",
    "OMP_NUM_THREADS",
)  # no live path gives a text node positive reward today (prereg: Question)


class Refusal(Exception):
    """An apparatus or instrument refusal: exit 4, a ``failed`` record, no verdict."""

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(f"{reason}: {detail}")
        self.reason = reason
        self.detail = detail


# ---------------------------------------------------------------------------
# Fixture and walk
# ---------------------------------------------------------------------------


def load_walk() -> list[dict[str, Any]]:
    """Unique strings in first-seen order (every pair's a then b, pairs before distractors). Each string's class is
    the class of the first fixture entry containing it; ``entry`` is that entry's id."""
    return load_fixture()[0]


def load_fixture() -> tuple[list[dict[str, Any]], dict[str, list[dict[str, str]]]]:
    """The walk and the fixture's entries, from ONE read behind the SHA pin."""
    raw = FIXTURE.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != FIXTURE_SHA256:
        raise Refusal("fixture_sha_mismatch", f"{FIXTURE.name} is {digest}, the prereg pins {FIXTURE_SHA256}")
    data = json.loads(raw)
    walk: list[dict[str, Any]] = []
    seen: set[str] = set()
    for kind in ("pairs", "distractors"):
        for entry in data[kind]:
            for half in ("a", "b"):
                text = entry[half]
                if text not in seen:
                    seen.add(text)
                    walk.append({"text": text, "cls": entry["class"], "entry": entry["id"], "kind": kind})
    if walk[0]["text"] != SEED or walk[0]["entry"] != SEED_PAIR:
        raise Refusal("fixture_walk", f"the walk does not start at the seed {SEED!r}")
    return walk, {k: data[k] for k in ("pairs", "distractors")}


# ---------------------------------------------------------------------------
# Apparatus
# ---------------------------------------------------------------------------


def cos(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def load_encoder() -> dict[str, Any]:
    """The production load path, then the one recorded deviation: the shared singleton moves to CPU."""
    from maxim.exceptions import ModelLoadError
    from maxim.similarity.encoder import _get_encoder, require_semantic_encoder

    try:
        require_semantic_encoder(MODEL, context="E4 text-widening drift measurement")
    except ModelLoadError as exc:
        raise Refusal("encoder_fallback", f"the real semantic encoder did not load: {exc}") from exc
    model = _get_encoder(MODEL)
    # The LOADED model's own revision, not what the cache's refs/main resolves to (a local path or another
    # SENTENCE_TRANSFORMERS_HOME would diverge).
    loaded = getattr(getattr(model[0].auto_model, "config", None), "_commit_hash", None)
    if loaded != MODEL_REVISION:
        raise Refusal(
            "model_revision_mismatch", f"the loaded model is revision {loaded!r}, the prereg pins {MODEL_REVISION}"
        )
    production_device = str(model.device)
    model.to("cpu")
    import sentence_transformers
    import torch
    import transformers

    return {
        "model": MODEL,
        "snapshot_revision": loaded,
        "production_device": production_device,
        "measurement_device": str(model.device),
        "sentence_transformers": sentence_transformers.__version__,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "torch_num_threads": torch.get_num_threads(),
    }


class Stack:
    """A fresh EC, NAc, ATL and encoder, with a spy on the EC's encode call (observes, never drives)."""

    def __init__(self, *, threshold: float | None = None, freeze_text: bool = False) -> None:
        from maxim.decisions.nac import NAc, NACConfig
        from maxim.memory.atl import ATL, ATLConfig
        from maxim.similarity.ec import ECConfig, EntorhinalCortex
        from maxim.similarity.encoder import EncoderConfig, LinguisticEncoder

        config = ECConfig()
        if threshold is not None:
            config = replace(config, pattern_complete_threshold=threshold)
        if freeze_text:
            config = replace(config, frozen_centroid_modalities=frozenset(config.frozen_centroid_modalities) | {"text"})
        self.ec = EntorhinalCortex(config=config)
        self.nac = NAc(NACConfig())
        self.atl = ATL(ATLConfig())
        self.encoder = LinguisticEncoder(
            ec=self.ec, atl=self.atl, config=EncoderConfig(model_name=MODEL), nac=self.nac, decomposer=None
        )
        self.base = self.ec.config.pattern_complete_threshold
        self.alpha = self.nac.config.reward_bias_alpha
        self.rewarded: str | None = None
        self.credited: dict[str, float] = {}
        self.calls: list[dict[str, Any]] = []
        self._install_spy()

    def _install_spy(self) -> None:
        original = self.ec.pattern_complete_or_separate

        def spy(*args: Any, **kwargs: Any) -> Any:
            embedding = kwargs["embedding"]
            override = kwargs.get("threshold_override")
            call: dict[str, Any] = {
                "modality": kwargs.get("modality"),
                "override": dict(override) if override else None,
                "geometry": kwargs.get("geometry"),
                "geometry_expected": self.encoder.geometry_for(embedding, "text"),
            }
            if self.rewarded is not None:
                meta = self.ec.substrate_node_metadata(self.rewarded)
                call["cos_rewarded"] = cos(embedding, meta["embedding"])
                call["rewarded_threshold"] = (override or {}).get(self.rewarded, self.base)
                call["rewarded_members"] = meta["member_count"]
            result = original(*args, **kwargs)
            call["node"] = result.node_id
            call["is_new"] = result.is_new
            call["winner_similarity"] = result.similarity
            call["best_similarity"] = result.best_similarity
            self.calls.append(call)
            return result

        self.ec.pattern_complete_or_separate = spy  # type: ignore[method-assign]

    def credit(self, node_id: str, bias: float) -> None:
        """Credit ``node_id`` to exactly ``bias`` (prereg: credit_node(AGENT, node, b / alpha), at every b; a zero
        credit stores nothing, so no override follows it)."""
        self.nac.credit_node(AGENT, node_id, bias / self.alpha)
        got = self.nac.reward_bias(AGENT, node_id)
        if abs(got - bias) > TOL_BIAS:
            raise Refusal("bias_known_answer", f"credited {node_id} to {bias}, NAc holds {got!r}")
        if bias > 0.0:
            self.credited[node_id] = bias

    def encode(self, text: str) -> tuple[str, list[float]]:
        from maxim.agents.percept_factory import make_text_percept

        percept = make_text_percept(text, agent_id=AGENT)
        self.encoder.encode(percept)
        prov = self.ec.encoder_provenance.get("linguistic", {})
        if (
            prov.get("using_fallback") is not False
            or prov.get("model_name") != MODEL
            or prov.get("embedding_dim") != DIM
        ):
            raise Refusal("encoder_fallback", f"EC provenance reports {self.ec.encoder_provenance!r}")
        return percept.substrate_node_id, list(percept.embedding)

    def check_wiring(self, call: dict[str, Any]) -> None:
        """Every encode after a credit carries exactly the credited overrides; none at b = 0 (prereg: Wiring)."""
        expected = {n: self.base - b for n, b in self.credited.items()}
        got = call["override"] or {}
        if set(got) != set(expected) or any(abs(got[n] - v) > TOL_BIAS for n, v in expected.items()):
            raise Refusal("wiring", f"threshold_override {got!r}, expected {expected!r}")
        if call["geometry"] != call["geometry_expected"]:
            raise Refusal("wiring", f"geometry {call['geometry']!r} != {call['geometry_expected']!r}")
        if call.get("modality") != "text":
            raise Refusal("wiring", f"modality {call.get('modality')!r}, not 'text'")


def realised_override(bias: float, base: float) -> float:
    """The override value NAc actually hands the EC at ``bias`` (prereg: the realised, not nominal, value)."""
    from maxim.decisions.nac import NAc, NACConfig

    if bias <= 0.0:
        return base
    nac = NAc(NACConfig())
    nac.credit_node(AGENT, "probe", bias / nac.config.reward_bias_alpha)
    return nac.get_threshold_overrides(AGENT, base_threshold=base)["probe"]


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------


def run_sequential(walk: list[dict[str, Any]], bias: float, *, reward_all: bool, **stack_kw: Any) -> dict[str, Any]:
    """One EC over the whole walk. R1 credits only the seed node; RA credits every node as it forms."""
    stack = Stack(**stack_kw)
    rows: list[dict[str, Any]] = []
    embeddings: dict[str, list[float]] = {}
    first_member: dict[str, str] = {}
    for i, item in enumerate(walk):
        node, emb = stack.encode(item["text"])
        call = stack.calls[-1]
        stack.check_wiring(call)
        embeddings[item["text"]] = emb
        first_member.setdefault(node, item["text"])
        if i == 0:
            stack.rewarded = node
            stack.credit(node, bias)
        elif reward_all and call["is_new"]:
            stack.credit(node, bias)
        rows.append(
            {
                "text": item["text"],
                "cls": item["cls"],
                "node": first_member[node],  # canonical: the node's first-formed member
                "is_new": call["is_new"],
                "winner_similarity": call["winner_similarity"],
                "best_similarity": call["best_similarity"],
                # Keyed by each node's canonical first member (uuids differ run to run).
                "override_passed": (
                    {first_member[n]: v for n, v in call["override"].items()} if call["override"] else None
                ),
                "cos_rewarded": call.get("cos_rewarded"),
                "rewarded_threshold": call.get("rewarded_threshold"),
                "rewarded_members_before": call.get("rewarded_members"),
            }
        )
    meta = stack.ec.substrate_node_metadata(stack.rewarded)
    return {
        "bias": bias,
        "reward": "all" if reward_all else "seed",
        "base": stack.base,
        "rows": rows,
        "embeddings": embeddings,
        "rewarded_final_centroid": meta["embedding"],
        "rewarded_final_members": meta["member_count"],
        "stored_reward_bias": stack.nac.reward_bias(AGENT, stack.rewarded),
    }


def in_rewarded(run: dict[str, Any]) -> set[str]:
    return {r["text"] for r in run["rows"] if r["node"] == SEED and r["text"] != SEED}


def replay_references(base_run: dict[str, Any], *, frozen: bool) -> dict[str, list[float]]:
    """For each string after the seed: what the EC's centroid rule would hold from the b = 0 run's rewarded-node
    members that precede it (the running mean for text; the seed for a frozen modality)."""
    refs: dict[str, list[float]] = {}
    emb = base_run["embeddings"]
    members: list[list[float]] = []
    for row in base_run["rows"]:
        if members:
            if frozen:
                refs[row["text"]] = list(members[0])
            else:
                refs[row["text"]] = [sum(col) / len(members) for col in zip(*members)]
        if row["node"] == SEED:
            members.append(emb[row["text"]])
    return refs


def replay_admits(
    base_run: dict[str, Any], threshold: float, *, frozen: bool, walk: list[dict[str, Any]]
) -> dict[str, dict[str, float | bool]]:
    """``I(b)``: for each string outside pair_01, does its replay reference admit it at ``threshold``?"""
    refs = replay_references(base_run, frozen=frozen)
    outside = {w["text"] for w in walk if w["entry"] != SEED_PAIR}
    out: dict[str, dict[str, float | bool]] = {}
    for text, ref in refs.items():
        if text not in outside:
            continue
        c = cos(base_run["embeddings"][text], ref)
        out[text] = {"cos": c, "admitted": c >= threshold}
    return out


def run_bare_isolated(walk: list[dict[str, Any]], bias: float, **stack_kw: Any) -> dict[str, bool]:
    """A fresh stack per string: encode the seed, credit it to ``bias``, encode the string; did it join the seed?"""
    out: dict[str, bool] = {}
    for item in walk[1:]:
        stack = Stack(**stack_kw)
        seed_node, _ = stack.encode(SEED)
        stack.rewarded = seed_node
        stack.credit(seed_node, bias)
        node, _ = stack.encode(item["text"])
        stack.check_wiring(stack.calls[-1])
        out[item["text"]] = node == seed_node
    return out


# ---------------------------------------------------------------------------
# Metrics and verdict
# ---------------------------------------------------------------------------


def foreign(walk: list[dict[str, Any]]) -> set[str]:
    seed_cls = walk[0]["cls"]
    return {w["text"] for w in walk if w["cls"] != seed_cls}


def pair_purity(run: dict[str, Any], entries: dict[str, list[dict[str, str]]], walk: list[dict[str, Any]]) -> int:
    owner = {w["text"]: w["entry"] for w in walk}
    node_of = {r["text"]: r["node"] for r in run["rows"]}
    pure = 0
    for pair in entries["pairs"]:
        a, b = pair["a"], pair["b"]
        if node_of[a] != node_of[b]:
            continue
        holders = {owner[t] for t, n in node_of.items() if n == node_of[a]}
        if holders == {pair["id"]}:
            pure += 1
    return pure


def distractor_collapse(run: dict[str, Any], entries: dict[str, list[dict[str, str]]]) -> int:
    node_of = {r["text"]: r["node"] for r in run["rows"]}
    return sum(1 for d in entries["distractors"] if node_of[d["a"]] == node_of[d["b"]])


def eligible_but_lost(run: dict[str, Any]) -> list[str]:
    return [
        r["text"]
        for r in run["rows"]
        if r["cos_rewarded"] is not None and r["cos_rewarded"] >= r["rewarded_threshold"] and r["node"] != SEED
    ]


def centroid_drift(run: dict[str, Any]) -> float:
    return cos(run["embeddings"][SEED], run["rewarded_final_centroid"])


def decide(
    walk: list[dict[str, Any]],
    seq: dict[float, dict[str, Any]],
    replay: dict[float, dict[str, dict[str, float | bool]]],
    base: float,
) -> dict[str, Any]:
    """The frozen rule: COLLAPSE, NO HEADROOM or NO COLLAPSE, checked in that order."""
    far = foreign(walk)
    a0 = in_rewarded(seq[0.0]) & far
    a2 = in_rewarded(seq[CAP]) & far
    i2 = {t for t, v in replay[CAP].items() if v["admitted"]} & far
    i0 = {t for t, v in replay[0.0].items() if v["admitted"]} & far
    collapse = sorted(s for s in a2 if s not in a0 and s not in i2)
    rows0 = {r["text"]: r for r in seq[0.0]["rows"]}
    rows2 = {r["text"]: r for r in seq[CAP]["rows"]}
    override_cap = realised_override(CAP, base)
    margins: dict[str, dict[str, float]] = {}
    for s in collapse:
        r0 = rows0[s]
        not_a0 = (
            (base - r0["cos_rewarded"]) if r0["cos_rewarded"] < base else (r0["winner_similarity"] - r0["cos_rewarded"])
        )
        margins[s] = {
            "in_A(0.2)": rows2[s]["cos_rewarded"] - override_cap,
            "not_in_I(0.2)": override_cap - float(replay[CAP][s]["cos"]),
            "not_in_A(0.0)": not_a0,
        }
        if any(v < 0 for v in margins[s].values()):  # each clause's margin is >= 0 by construction
            raise Refusal("margin_inconsistent", f"{s!r}: a clause margin is negative: {margins[s]!r}")
    headroom = sorted(s for s in far if s not in a0 and s not in i2)
    if collapse:
        marginal = all(min(m.values()) < MARGINAL for m in margins.values())
        outcome = "COLLAPSE (marginal)" if marginal else "COLLAPSE"
    elif not headroom:
        outcome = "NO HEADROOM"
    else:
        outcome = "NO COLLAPSE"
    return {
        "outcome": outcome,
        "collapse_strings": collapse,
        "collapse_margins": margins,
        "headroom_strings": headroom,
        "headroom_count": len(headroom),
        "A(0.0)": sorted(a0),
        "A(0.2)": sorted(a2),
        "I(0.2)": sorted(i2),
        # Overreach the reward caused: admitted at the cap, not already admitted at the base threshold.
        "widening_overreach": sorted(i2 - i0),
        "E(0.2)": eligible_but_lost(seq[CAP]),
    }


# ---------------------------------------------------------------------------
# The matrix, the instrument checks, the record
# ---------------------------------------------------------------------------


def run_matrix(walk: list[dict[str, Any]], entries: dict[str, list[dict[str, str]]]) -> dict[str, Any]:
    base = Stack().base
    r1 = {b: run_sequential(walk, b, reward_all=False) for b in BIASES}
    ra = {b: run_sequential(walk, b, reward_all=True) for b in BIASES}
    replay = {b: replay_admits(r1[0.0], realised_override(b, base), frozen=False, walk=walk) for b in BIASES}
    bare = {b: run_bare_isolated(walk, b) for b in BIASES}

    # Instrument: replay known answer (at b = 0 the replay reference IS the EC's centroid at encode time).
    replay_delta = 0.0
    for row in r1[0.0]["rows"][1:]:
        if row["text"] not in replay[0.0]:
            continue
        r = replay[0.0][row["text"]]["cos"]
        replay_delta = max(replay_delta, abs(float(r) - row["cos_rewarded"]))
        if abs(float(r) - row["cos_rewarded"]) > TOL_REPLAY:
            raise Refusal("replay_known_answer", f"{row['text']!r}: replay {r!r} vs EC {row['cos_rewarded']!r}")
    # And the harness's cosine is the EC's own: a string that joined the rewarded node has winner == cos_rewarded.
    for row in r1[0.0]["rows"][1:]:
        if row["node"] == SEED and abs(row["winner_similarity"] - row["cos_rewarded"]) > TOL_REPLAY:
            raise Refusal("replay_known_answer", f"{row['text']!r}: EC winner {row['winner_similarity']!r} vs harness")

    # Instrument: identity (R1 and RA at b = 0).
    if [x["node"] for x in r1[0.0]["rows"]] != [x["node"] for x in ra[0.0]["rows"]]:
        raise Refusal("identity", "R1 and RA assignments differ at b = 0")

    # Instrument: drift positive control at threshold 0.40 (sequential vs bare-isolated, no reward).
    ctrl_seq = run_sequential(walk, 0.0, reward_all=False, threshold=CONTROL_THRESHOLD)
    ctrl_bare = run_bare_isolated(walk, 0.0, threshold=CONTROL_THRESHOLD)
    drift_seen = sorted(s for s in in_rewarded(ctrl_seq) & foreign(walk) if not ctrl_bare[s])
    if not drift_seen:
        raise Refusal("positive_control", "at threshold 0.40 no foreign string drifted into the seed node")

    # Instrument: negative control (frozen text centroid; COLLAPSE impossible by construction).
    frozen_seq = {b: run_sequential(walk, b, reward_all=False, freeze_text=True) for b in (0.0, CAP)}
    frozen_replay = {
        b: replay_admits(frozen_seq[0.0], realised_override(b, base), frozen=True, walk=walk) for b in (0.0, CAP)
    }
    frozen_verdict = decide(walk, frozen_seq, frozen_replay, base)
    if frozen_verdict["collapse_strings"]:
        raise Refusal("negative_control", f"COLLAPSE under a frozen centroid: {frozen_verdict['collapse_strings']}")

    verdict = decide(walk, r1, replay, base)
    pre_ec = {w["text"]: cos(r1[0.0]["embeddings"][w["text"]], r1[0.0]["embeddings"][SEED]) for w in walk}
    purity0, collapse0 = pair_purity(r1[0.0], entries, walk), distractor_collapse(r1[0.0], entries)
    drift0 = centroid_drift(r1[0.0])

    def summary(run: dict[str, Any]) -> dict[str, Any]:
        return {
            "bias": run["bias"],
            "reward": run["reward"],
            "stored_reward_bias": run["stored_reward_bias"],
            "realised_override": realised_override(run["bias"], base),
            "override_passed_after_credit": next(
                (r["override_passed"] for r in run["rows"][1:] if r["override_passed"] is not None), None
            ),
            "rows": run["rows"],
            "absorbed_by_class": _by_class(in_rewarded(run), walk),
            "eligible_but_lost": eligible_but_lost(run),
            "pair_purity_delta": pair_purity(run, entries, walk) - purity0,
            "distractor_collapse_delta": distractor_collapse(run, entries) - collapse0,
            "rewarded_drift_delta": centroid_drift(run) - drift0,
            "rewarded_final_members": run["rewarded_final_members"],
        }

    return {
        "base_threshold": base,
        "verdict": verdict,
        "R1_sequential": [summary(r1[b]) for b in BIASES],
        "RA_sequential": [summary(ra[b]) for b in BIASES],
        "R1_replay_isolated": {str(b): replay[b] for b in BIASES},
        "R1_bare_isolated": {str(b): bare[b] for b in BIASES},
        "foreign_A(0.1)_subset_of_A(0.2)": (in_rewarded(r1[0.1]) & foreign(walk))
        <= (in_rewarded(r1[CAP]) & foreign(walk)),
        "baseline": {"pair_purity": purity0, "distractor_collapse": collapse0, "rewarded_drift": drift0},
        # The replay's final reference is the b = 0 final centroid, so this equals baseline.rewarded_drift by
        # construction (reported because the prereg names it, not as a separate measurement).
        "replay_centroid_drift": drift0,
        "pre_ec_cos_to_seed": pre_ec,
        "controls": {
            "replay_known_answer": {"max_abs_delta": replay_delta, "tolerance": TOL_REPLAY},
            "identity": "pass (near-tautological: a zero credit stores nothing)",
            "drift_positive_control": {"threshold": CONTROL_THRESHOLD, "drifted": drift_seen},
            "negative_control": {
                "frozen_text": True,
                "collapse_strings": [],
                "A(0.0)": frozen_verdict["A(0.0)"],
                "A(0.2)": frozen_verdict["A(0.2)"],
                "I(0.2)": frozen_verdict["I(0.2)"],
            },
        },
        "_embeddings": r1[0.0]["embeddings"],
    }


def _by_class(texts: set[str], walk: list[dict[str, Any]]) -> dict[str, list[str]]:
    cls = {w["text"]: w["cls"] for w in walk}
    out: dict[str, list[str]] = {}
    for t in sorted(texts):
        out.setdefault(cls[t], []).append(t)
    return out


def _digest(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def _pack_embeddings(walk: list[dict[str, Any]], embeddings: dict[str, list[float]]) -> dict[str, Any]:
    blob = b"".join(struct.pack(f"<{len(embeddings[w['text']])}f", *embeddings[w["text"]]) for w in walk)
    return {
        "dtype": "float32-le",
        "dim": len(embeddings[walk[0]["text"]]),
        "order": [w["text"] for w in walk],
        "sha256": hashlib.sha256(blob).hexdigest(),
        "base64": base64.b64encode(blob).decode(),
    }


def measure() -> dict[str, Any]:
    walk, entries = load_fixture()
    encoder = load_encoder()
    first = run_matrix(walk, entries)
    second = run_matrix(walk, entries)
    emb1, emb2 = first.pop("_embeddings"), second.pop("_embeddings")
    if (
        _digest(first) != _digest(second)
        or _pack_embeddings(walk, emb1)["sha256"] != _pack_embeddings(walk, emb2)["sha256"]
    ):
        raise Refusal("determinism", "two runs of the matrix differ")
    from maxim.decisions.nac import NACConfig
    from maxim.similarity.ec import ECConfig
    from maxim.similarity.encoder import _get_encoder

    encoder["measurement_device_after"] = str(_get_encoder(MODEL).device)
    first["encoder"] = encoder
    first["configs"] = {"ECConfig": repr(ECConfig()), "NACConfig": repr(NACConfig())}
    first["embeddings"] = _pack_embeddings(walk, emb1)
    first["walk"] = walk
    first["determinism"] = {"runs": 2, "matrix_digest": _digest(second)}
    return first


def _write(path: Path, report: dict[str, Any]) -> None:
    from maxim.utils.atomic_io import atomic_write_json
    from maxim.utils.format_version import with_format_version

    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, with_format_version(report))
    print(f"[e4] record written: {path}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--write-experiment-results",
        action="store_true",
        help="write the committed record under docs/experiments/data/ (clean tree enforced); default: a temp dir",
    )
    args = ap.parse_args(argv)
    unexpected_env = pin_environment()
    sys.path.insert(0, str(REPO / "scripts"))

    from _provenance import (
        DirtyTreeError,
        ProvenanceError,
        evidence_out_paths_or_exit,
        in_process_code_provenance,
        preflight_gated_record_or_exit,
        stamp_diagnosis,
    )

    # No allow_dirty, deliberately: a committed record only from a clean tree (prereg: Record).
    [out_path] = evidence_out_paths_or_exit(REPO, [RECORD], write_experiment_results=args.write_experiment_results)
    preflight_gated_record_or_exit(REPO, out_path)
    import maxim

    try:
        provenance = in_process_code_provenance(
            REPO, maxim.__file__, out_path=out_path if args.write_experiment_results else None
        )
    except (ProvenanceError, DirtyTreeError) as exc:
        print(f"[FAIL] gated-record preflight: {exc}", file=sys.stderr)
        return 3

    report: dict[str, Any] = {
        "experiment": "e4_text_widening_drift",
        "issue": 911,
        "prereg": PREREG,
        "fixture": {"path": str(FIXTURE.relative_to(REPO)), "sha256": FIXTURE_SHA256},
        "scope": SCOPE,
        # An explicit allowlist, never a prefix match: a prefix would copy an exported HF_TOKEN into the record.
        "environment": {k: os.environ[k] for k in STAMPED_ENV if k in os.environ},
    }
    try:
        if unexpected_env:
            raise Refusal("env_toggle_set", f"unexpected MAXIM_* variables set: {unexpected_env}")
        report.update(measure())
    except Refusal as exc:
        report["refusal"] = exc.reason
        report["refusal_detail"] = exc.detail
        stamp_diagnosis(report, mock=False, code_provenance=provenance)
        _write(out_path, report)
        print(f"[REFUSED — {exc.reason}] {exc.detail}", file=sys.stderr)
        return 4
    stamp_diagnosis(report, mock=False, code_provenance=provenance)
    _write(out_path, report)
    verdict = report["verdict"]
    print(f"[e4] outcome: {verdict['outcome']} (headroom {verdict['headroom_count']}; scope {SCOPE})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
