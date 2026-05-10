"""Substrate divergence analyzer — compare two Roy session directories.

Roy harness session R2 of 5. Reads ``aut_hippocampus.json`` /
``aut_nac.json`` (and, when present, ``ec.json`` / ``atl.json``) from
two ``~/.maxim/sim_reports/<id>/`` directories and emits the per-axis
metrics named in
``docs/plans/persona_convergence_crucible.md`` § "What we record per
Roy iteration":

- NAc ``_percept_valences`` L2 distance for shared entity classes
  (reserved field — surfaces as ``available: false`` until 1.1 wires
  the dump path)
- NAc ``_reward_bias`` per-key distribution divergence (L2 + per-key
  top-N breakdown), causal-link count delta
- EC node-count delta, modality histogram delta, top-N closest pairs
  by cosine similarity
- Hippocampus episode count delta, valence-distribution KS statistic
  (+p-value), salience-distribution KS statistic
- ATL concept count delta + name-set overlap

The library is **read-only**: it loads each side's JSON files, computes
metrics, and returns frozen dataclasses with ``__str__`` methods that
print a one-screen report block. Nothing is persisted. Missing files
yield ``available=False`` markers, never raise.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

# ─────────────────────────────────────────────────────────────────────────
# JSON loading (fail-soft)
# ─────────────────────────────────────────────────────────────────────────


def _load_json(path: Path) -> dict[str, Any] | None:
    """Return parsed JSON dict, or None on missing/corrupt file.

    The analyzer is a diagnostic tool — corrupt files are logged
    implicitly via the ``available: false`` field in the returned diff
    dataclass rather than raised.
    """
    if not path.is_file():
        return None
    try:
        with path.open(encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


# ─────────────────────────────────────────────────────────────────────────
# Math helpers (no scipy dep beyond what core deps already provide)
# ─────────────────────────────────────────────────────────────────────────


def _l2(values: Iterable[float]) -> float:
    return math.sqrt(sum(v * v for v in values))


def _l2_dict_diff(a: dict[str, float], b: dict[str, float]) -> tuple[float, list[tuple[str, float]]]:
    """L2 distance over the union of keys, treating missing keys as 0.0.

    Returns ``(l2, per_key_deltas_sorted_desc_by_abs_delta)``.
    """
    keys = set(a) | set(b)
    deltas: list[tuple[str, float]] = []
    for k in keys:
        d = float(a.get(k, 0.0)) - float(b.get(k, 0.0))
        if d != 0.0:
            deltas.append((k, d))
    l2 = _l2(d for _, d in deltas)
    deltas.sort(key=lambda kv: abs(kv[1]), reverse=True)
    return l2, deltas


def _cosine(u: list[float], v: list[float]) -> float:
    if len(u) != len(v) or not u:
        return 0.0
    dot = sum(a * b for a, b in zip(u, v))
    nu = math.sqrt(sum(a * a for a in u))
    nv = math.sqrt(sum(b * b for b in v))
    if nu == 0.0 or nv == 0.0:
        return 0.0
    return dot / (nu * nv)


def _ks_2samp(sample_a: list[float], sample_b: list[float]) -> tuple[float, float]:
    """Two-sample Kolmogorov-Smirnov test.

    Uses scipy when available (gives a real p-value); falls back to a
    pure-Python statistic + asymptotic p-value otherwise. Returns
    ``(statistic, p_value)``. Empty sample on either side returns
    ``(0.0, 1.0)`` — no evidence of divergence.
    """
    if not sample_a or not sample_b:
        return 0.0, 1.0
    try:
        from scipy.stats import ks_2samp  # type: ignore[import-not-found]

        result = ks_2samp(sample_a, sample_b)
        return float(result.statistic), float(result.pvalue)
    except ImportError:
        pass

    # Pure-Python fallback: empirical CDF supremum + Kolmogorov asymptotic
    sa = sorted(sample_a)
    sb = sorted(sample_b)
    n, m = len(sa), len(sb)
    i = j = 0
    cdf_a = cdf_b = 0.0
    d = 0.0
    while i < n and j < m:
        if sa[i] <= sb[j]:
            i += 1
            cdf_a = i / n
        else:
            j += 1
            cdf_b = j / m
        d = max(d, abs(cdf_a - cdf_b))
    en = math.sqrt(n * m / (n + m))
    # Kolmogorov asymptotic two-sided p-value approximation
    arg = (en + 0.12 + 0.11 / en) * d
    p = 2.0 * sum((-1) ** (k - 1) * math.exp(-2.0 * k * k * arg * arg) for k in range(1, 101))
    p = max(0.0, min(1.0, p))
    return d, p


# ─────────────────────────────────────────────────────────────────────────
# Dataclasses
# ─────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class NacDiff:
    available: bool
    reason: str = ""
    reward_bias_l2: float = 0.0
    reward_bias_top_deltas: tuple[tuple[str, float], ...] = ()
    goal_reward_bias_l2: float = 0.0
    goal_reward_bias_top_deltas: tuple[tuple[str, float], ...] = ()
    percept_valence_available: bool = False
    percept_valence_l2: float = 0.0
    percept_valence_top_deltas: tuple[tuple[str, float], ...] = ()
    causal_link_count_a: int = 0
    causal_link_count_b: int = 0

    @property
    def causal_link_count_delta(self) -> int:
        return self.causal_link_count_a - self.causal_link_count_b

    def __str__(self) -> str:
        if not self.available:
            return f"NAc diff: available=false ({self.reason})"
        lines = [
            "NAc diff:",
            f"  reward_bias L2: {self.reward_bias_l2:.4f}  ({len(self.reward_bias_top_deltas)} keys differ)",
        ]
        for key, delta in self.reward_bias_top_deltas[:5]:
            lines.append(f"    {key}: Δ {delta:+.4f}")
        lines.append(
            f"  goal_reward_bias L2: {self.goal_reward_bias_l2:.4f}  "
            f"({len(self.goal_reward_bias_top_deltas)} keys differ)"
        )
        for key, delta in self.goal_reward_bias_top_deltas[:5]:
            lines.append(f"    {key}: Δ {delta:+.4f}")
        if self.percept_valence_available:
            lines.append(
                f"  percept_valences L2: {self.percept_valence_l2:.4f}  "
                f"({len(self.percept_valence_top_deltas)} keys differ)"
            )
            for key, delta in self.percept_valence_top_deltas[:5]:
                lines.append(f"    {key}: Δ {delta:+.4f}")
        else:
            lines.append("  percept_valences: not persisted (reserved for 1.1)")
        lines.append(
            f"  causal links: a={self.causal_link_count_a}  b={self.causal_link_count_b}  "
            f"Δ={self.causal_link_count_delta:+d}"
        )
        return "\n".join(lines)


@dataclass(frozen=True)
class EcDiff:
    available: bool
    reason: str = ""
    node_count_a: int = 0
    node_count_b: int = 0
    modality_histogram_a: dict[str, int] = field(default_factory=dict)
    modality_histogram_b: dict[str, int] = field(default_factory=dict)
    closest_node_pairs: tuple[tuple[str, str, float], ...] = ()

    @property
    def node_count_delta(self) -> int:
        return self.node_count_a - self.node_count_b

    @property
    def modality_histogram_delta(self) -> dict[str, int]:
        keys = set(self.modality_histogram_a) | set(self.modality_histogram_b)
        return {k: self.modality_histogram_a.get(k, 0) - self.modality_histogram_b.get(k, 0) for k in keys}

    def __str__(self) -> str:
        if not self.available:
            return f"EC diff: available=false ({self.reason})"
        lines = [
            "EC diff:",
            f"  substrate nodes: a={self.node_count_a}  b={self.node_count_b}  Δ={self.node_count_delta:+d}",
        ]
        hist_delta = self.modality_histogram_delta
        if hist_delta:
            parts = ", ".join(
                f"{m}: a={self.modality_histogram_a.get(m, 0)} b={self.modality_histogram_b.get(m, 0)} "
                f"(Δ {hist_delta[m]:+d})"
                for m in sorted(hist_delta)
            )
            lines.append(f"  modality histogram: {parts}")
        if self.closest_node_pairs:
            lines.append(f"  top {len(self.closest_node_pairs)} closest cross-side node pairs (cosine):")
            for nid_a, nid_b, cos in self.closest_node_pairs:
                lines.append(f"    {nid_a} ↔ {nid_b}: {cos:.3f}")
        return "\n".join(lines)


@dataclass(frozen=True)
class HippocampusDiff:
    available: bool
    reason: str = ""
    episode_count_a: int = 0
    episode_count_b: int = 0
    memory_count_a: int = 0
    memory_count_b: int = 0
    valence_ks_statistic: float = 0.0
    valence_ks_pvalue: float = 1.0
    valence_mean_a: float = 0.0
    valence_mean_b: float = 0.0
    salience_ks_statistic: float = 0.0
    salience_ks_pvalue: float = 1.0
    salience_mean_a: float = 0.0
    salience_mean_b: float = 0.0

    @property
    def episode_count_delta(self) -> int:
        return self.episode_count_a - self.episode_count_b

    @property
    def memory_count_delta(self) -> int:
        return self.memory_count_a - self.memory_count_b

    def __str__(self) -> str:
        if not self.available:
            return f"Hippocampus diff: available=false ({self.reason})"
        return "\n".join(
            [
                "Hippocampus diff:",
                f"  episodes: a={self.episode_count_a}  b={self.episode_count_b}  Δ={self.episode_count_delta:+d}",
                f"  memories: a={self.memory_count_a}  b={self.memory_count_b}  Δ={self.memory_count_delta:+d}",
                f"  valence:  mean a={self.valence_mean_a:+.3f}  mean b={self.valence_mean_b:+.3f}  "
                f"KS={self.valence_ks_statistic:.3f}  p={self.valence_ks_pvalue:.3f}",
                f"  salience: mean a={self.salience_mean_a:+.3f}  mean b={self.salience_mean_b:+.3f}  "
                f"KS={self.salience_ks_statistic:.3f}  p={self.salience_ks_pvalue:.3f}",
            ]
        )


@dataclass(frozen=True)
class AtlDiff:
    available: bool
    reason: str = ""
    concept_count_a: int = 0
    concept_count_b: int = 0
    shared_names: tuple[str, ...] = ()
    only_in_a: tuple[str, ...] = ()
    only_in_b: tuple[str, ...] = ()

    @property
    def concept_count_delta(self) -> int:
        return self.concept_count_a - self.concept_count_b

    @property
    def jaccard(self) -> float:
        union = len(self.shared_names) + len(self.only_in_a) + len(self.only_in_b)
        return len(self.shared_names) / union if union else 0.0

    def __str__(self) -> str:
        if not self.available:
            return f"ATL diff: available=false ({self.reason})"
        lines = [
            "ATL diff:",
            f"  concepts: a={self.concept_count_a}  b={self.concept_count_b}  Δ={self.concept_count_delta:+d}",
            f"  name overlap: shared={len(self.shared_names)}  "
            f"only_in_a={len(self.only_in_a)}  only_in_b={len(self.only_in_b)}  "
            f"jaccard={self.jaccard:.3f}",
        ]
        if self.only_in_a:
            preview = ", ".join(self.only_in_a[:5])
            more = "" if len(self.only_in_a) <= 5 else f", … (+{len(self.only_in_a) - 5})"
            lines.append(f"  only_in_a sample: {preview}{more}")
        if self.only_in_b:
            preview = ", ".join(self.only_in_b[:5])
            more = "" if len(self.only_in_b) <= 5 else f", … (+{len(self.only_in_b) - 5})"
            lines.append(f"  only_in_b sample: {preview}{more}")
        return "\n".join(lines)


@dataclass(frozen=True)
class SubstrateDiff:
    session_a: Path
    session_b: Path
    nac: NacDiff
    ec: EcDiff
    hippocampus: HippocampusDiff
    atl: AtlDiff

    def __str__(self) -> str:
        return "\n\n".join(
            [
                f"Substrate divergence: {self.session_a} ↔ {self.session_b}",
                str(self.nac),
                str(self.ec),
                str(self.hippocampus),
                str(self.atl),
            ]
        )


# ─────────────────────────────────────────────────────────────────────────
# Per-axis differs
# ─────────────────────────────────────────────────────────────────────────


def _candidate_paths(session_dir: Path, names: Iterable[str]) -> Path | None:
    """Return the first existing file under ``session_dir`` matching one of ``names``."""
    for name in names:
        p = session_dir / name
        if p.is_file():
            return p
    return None


def nac_diff(dir_a: str | Path, dir_b: str | Path) -> NacDiff:
    """Compute NAc divergence between two sim_reports session directories."""
    pa = _candidate_paths(Path(dir_a), ("aut_nac.json", "nac.json"))
    pb = _candidate_paths(Path(dir_b), ("aut_nac.json", "nac.json"))
    if pa is None or pb is None:
        missing = [name for name, p in (("a", pa), ("b", pb)) if p is None]
        return NacDiff(available=False, reason=f"missing NAc snapshot in side(s): {','.join(missing)}")

    state_a = _load_json(pa)
    state_b = _load_json(pb)
    if state_a is None or state_b is None:
        return NacDiff(available=False, reason="corrupt or unreadable NAc snapshot")

    rb_a: dict[str, float] = {k: float(v) for k, v in state_a.get("reward_bias", {}).items()}
    rb_b: dict[str, float] = {k: float(v) for k, v in state_b.get("reward_bias", {}).items()}
    rb_l2, rb_top = _l2_dict_diff(rb_a, rb_b)

    grb_a: dict[str, float] = {k: float(v) for k, v in state_a.get("goal_reward_bias", {}).items()}
    grb_b: dict[str, float] = {k: float(v) for k, v in state_b.get("goal_reward_bias", {}).items()}
    grb_l2, grb_top = _l2_dict_diff(grb_a, grb_b)

    # _percept_valences — reserved for 1.1; only surface metrics if both
    # sides actually persisted the field.
    pv_a_raw = state_a.get("percept_valences")
    pv_b_raw = state_b.get("percept_valences")
    pv_available = isinstance(pv_a_raw, dict) and isinstance(pv_b_raw, dict)
    pv_l2 = 0.0
    pv_top: list[tuple[str, float]] = []
    if pv_available:
        pv_a = {k: float(v) for k, v in pv_a_raw.items()}
        pv_b = {k: float(v) for k, v in pv_b_raw.items()}
        pv_l2, pv_top = _l2_dict_diff(pv_a, pv_b)

    links_a = state_a.get("links", {}) or {}
    links_b = state_b.get("links", {}) or {}
    link_count_a = sum(len(v) for v in links_a.values() if isinstance(v, list))
    link_count_b = sum(len(v) for v in links_b.values() if isinstance(v, list))

    return NacDiff(
        available=True,
        reward_bias_l2=rb_l2,
        reward_bias_top_deltas=tuple(rb_top[:20]),
        goal_reward_bias_l2=grb_l2,
        goal_reward_bias_top_deltas=tuple(grb_top[:20]),
        percept_valence_available=pv_available,
        percept_valence_l2=pv_l2,
        percept_valence_top_deltas=tuple(pv_top[:20]),
        causal_link_count_a=link_count_a,
        causal_link_count_b=link_count_b,
    )


def ec_diff(dir_a: str | Path, dir_b: str | Path, *, top_n: int = 10) -> EcDiff:
    """Compute EC divergence (substrate-node count, modality, top-N pairs)."""
    pa = _candidate_paths(Path(dir_a), ("ec.json", "aut_ec.json"))
    pb = _candidate_paths(Path(dir_b), ("ec.json", "aut_ec.json"))
    if pa is None or pb is None:
        missing = [name for name, p in (("a", pa), ("b", pb)) if p is None]
        return EcDiff(available=False, reason=f"missing EC snapshot in side(s): {','.join(missing)}")

    state_a = _load_json(pa)
    state_b = _load_json(pb)
    if state_a is None or state_b is None:
        return EcDiff(available=False, reason="corrupt or unreadable EC snapshot")

    nodes_a = state_a.get("substrate_nodes", {}) or {}
    nodes_b = state_b.get("substrate_nodes", {}) or {}

    hist_a: dict[str, int] = {}
    hist_b: dict[str, int] = {}
    for nid, ndata in nodes_a.items():
        if not isinstance(ndata, dict):
            continue
        mod = str(ndata.get("modality", "unknown"))
        hist_a[mod] = hist_a.get(mod, 0) + 1
    for nid, ndata in nodes_b.items():
        if not isinstance(ndata, dict):
            continue
        mod = str(ndata.get("modality", "unknown"))
        hist_b[mod] = hist_b.get(mod, 0) + 1

    # Top-N closest cross-side pairs by cosine similarity. O(|A|·|B|) —
    # bounded by the substrate node counts which are typically O(10²-10³)
    # per side for Roy iterations. If the sweep grows, swap in a Faiss /
    # Annoy index and keep the contract the same.
    pairs: list[tuple[str, str, float]] = []
    if nodes_a and nodes_b:
        emb_a: list[tuple[str, list[float]]] = []
        for nid, ndata in nodes_a.items():
            if isinstance(ndata, dict) and isinstance(ndata.get("embedding"), list):
                emb_a.append((nid, [float(x) for x in ndata["embedding"]]))
        emb_b: list[tuple[str, list[float]]] = []
        for nid, ndata in nodes_b.items():
            if isinstance(ndata, dict) and isinstance(ndata.get("embedding"), list):
                emb_b.append((nid, [float(x) for x in ndata["embedding"]]))
        for nid_a, vec_a in emb_a:
            for nid_b, vec_b in emb_b:
                cos = _cosine(vec_a, vec_b)
                pairs.append((nid_a, nid_b, cos))
        pairs.sort(key=lambda t: t[2], reverse=True)
        pairs = pairs[:top_n]

    return EcDiff(
        available=True,
        node_count_a=len(nodes_a),
        node_count_b=len(nodes_b),
        modality_histogram_a=hist_a,
        modality_histogram_b=hist_b,
        closest_node_pairs=tuple(pairs),
    )


def hippocampus_diff(dir_a: str | Path, dir_b: str | Path) -> HippocampusDiff:
    """Compute Hippocampus divergence (episode/memory counts, KS on valence + salience)."""
    pa = _candidate_paths(Path(dir_a), ("aut_hippocampus.json", "hippocampus.json"))
    pb = _candidate_paths(Path(dir_b), ("aut_hippocampus.json", "hippocampus.json"))
    if pa is None or pb is None:
        missing = [name for name, p in (("a", pa), ("b", pb)) if p is None]
        return HippocampusDiff(
            available=False,
            reason=f"missing hippocampus snapshot in side(s): {','.join(missing)}",
        )

    state_a = _load_json(pa)
    state_b = _load_json(pb)
    if state_a is None or state_b is None:
        return HippocampusDiff(available=False, reason="corrupt or unreadable hippocampus snapshot")

    episodes_a = state_a.get("episodes", []) or []
    episodes_b = state_b.get("episodes", []) or []
    memories_a = state_a.get("memories", []) or []
    memories_b = state_b.get("memories", []) or []

    valences_a = [float(e.get("valence", 0.0)) for e in episodes_a if isinstance(e, dict)]
    valences_b = [float(e.get("valence", 0.0)) for e in episodes_b if isinstance(e, dict)]

    saliences_a = [
        float(m.get("perception", {}).get("salience", 0.5))
        for m in memories_a
        if isinstance(m, dict) and isinstance(m.get("perception"), dict)
    ]
    saliences_b = [
        float(m.get("perception", {}).get("salience", 0.5))
        for m in memories_b
        if isinstance(m, dict) and isinstance(m.get("perception"), dict)
    ]

    val_ks, val_p = _ks_2samp(valences_a, valences_b)
    sal_ks, sal_p = _ks_2samp(saliences_a, saliences_b)

    return HippocampusDiff(
        available=True,
        episode_count_a=len(episodes_a),
        episode_count_b=len(episodes_b),
        memory_count_a=len(memories_a),
        memory_count_b=len(memories_b),
        valence_ks_statistic=val_ks,
        valence_ks_pvalue=val_p,
        valence_mean_a=(sum(valences_a) / len(valences_a)) if valences_a else 0.0,
        valence_mean_b=(sum(valences_b) / len(valences_b)) if valences_b else 0.0,
        salience_ks_statistic=sal_ks,
        salience_ks_pvalue=sal_p,
        salience_mean_a=(sum(saliences_a) / len(saliences_a)) if saliences_a else 0.0,
        salience_mean_b=(sum(saliences_b) / len(saliences_b)) if saliences_b else 0.0,
    )


def atl_diff(dir_a: str | Path, dir_b: str | Path) -> AtlDiff:
    """Compute ATL divergence (concept count + name-set overlap)."""
    pa = _candidate_paths(Path(dir_a), ("atl.json", "aut_atl.json"))
    pb = _candidate_paths(Path(dir_b), ("atl.json", "aut_atl.json"))
    if pa is None or pb is None:
        missing = [name for name, p in (("a", pa), ("b", pb)) if p is None]
        return AtlDiff(available=False, reason=f"missing ATL snapshot in side(s): {','.join(missing)}")

    state_a = _load_json(pa)
    state_b = _load_json(pb)
    if state_a is None or state_b is None:
        return AtlDiff(available=False, reason="corrupt or unreadable ATL snapshot")

    concepts_a = state_a.get("concepts", {}) or {}
    concepts_b = state_b.get("concepts", {}) or {}

    names_a = {str(c.get("name", "")).strip() for c in concepts_a.values() if isinstance(c, dict) and c.get("name")}
    names_b = {str(c.get("name", "")).strip() for c in concepts_b.values() if isinstance(c, dict) and c.get("name")}
    names_a.discard("")
    names_b.discard("")

    shared = sorted(names_a & names_b)
    only_a = sorted(names_a - names_b)
    only_b = sorted(names_b - names_a)

    return AtlDiff(
        available=True,
        concept_count_a=len(concepts_a),
        concept_count_b=len(concepts_b),
        shared_names=tuple(shared),
        only_in_a=tuple(only_a),
        only_in_b=tuple(only_b),
    )


def substrate_diff(dir_a: str | Path, dir_b: str | Path) -> SubstrateDiff:
    """Aggregate per-axis diffs into a single ``SubstrateDiff``."""
    pa = Path(dir_a)
    pb = Path(dir_b)
    return SubstrateDiff(
        session_a=pa,
        session_b=pb,
        nac=nac_diff(pa, pb),
        ec=ec_diff(pa, pb),
        hippocampus=hippocampus_diff(pa, pb),
        atl=atl_diff(pa, pb),
    )


# ─────────────────────────────────────────────────────────────────────────
# JSON serialization (for `--json` CLI flag)
# ─────────────────────────────────────────────────────────────────────────


def _nac_to_json(d: NacDiff) -> dict[str, Any]:
    payload: dict[str, Any] = {"available": d.available}
    if not d.available:
        payload["reason"] = d.reason
        return payload
    payload.update(
        {
            "reward_bias_l2": d.reward_bias_l2,
            "reward_bias_top_deltas": [{"key": k, "delta": v} for k, v in d.reward_bias_top_deltas],
            "goal_reward_bias_l2": d.goal_reward_bias_l2,
            "goal_reward_bias_top_deltas": [{"key": k, "delta": v} for k, v in d.goal_reward_bias_top_deltas],
            "percept_valence": {
                "available": d.percept_valence_available,
                "l2": d.percept_valence_l2,
                "top_deltas": [{"key": k, "delta": v} for k, v in d.percept_valence_top_deltas],
            },
            "causal_link_count_a": d.causal_link_count_a,
            "causal_link_count_b": d.causal_link_count_b,
            "causal_link_count_delta": d.causal_link_count_delta,
        }
    )
    return payload


def _ec_to_json(d: EcDiff) -> dict[str, Any]:
    payload: dict[str, Any] = {"available": d.available}
    if not d.available:
        payload["reason"] = d.reason
        return payload
    payload.update(
        {
            "node_count_a": d.node_count_a,
            "node_count_b": d.node_count_b,
            "node_count_delta": d.node_count_delta,
            "modality_histogram_a": d.modality_histogram_a,
            "modality_histogram_b": d.modality_histogram_b,
            "modality_histogram_delta": d.modality_histogram_delta,
            "closest_node_pairs": [{"a": a, "b": b, "cosine": c} for a, b, c in d.closest_node_pairs],
        }
    )
    return payload


def _hippocampus_to_json(d: HippocampusDiff) -> dict[str, Any]:
    payload: dict[str, Any] = {"available": d.available}
    if not d.available:
        payload["reason"] = d.reason
        return payload
    payload.update(
        {
            "episode_count_a": d.episode_count_a,
            "episode_count_b": d.episode_count_b,
            "episode_count_delta": d.episode_count_delta,
            "memory_count_a": d.memory_count_a,
            "memory_count_b": d.memory_count_b,
            "memory_count_delta": d.memory_count_delta,
            "valence": {
                "mean_a": d.valence_mean_a,
                "mean_b": d.valence_mean_b,
                "ks_statistic": d.valence_ks_statistic,
                "ks_pvalue": d.valence_ks_pvalue,
            },
            "salience": {
                "mean_a": d.salience_mean_a,
                "mean_b": d.salience_mean_b,
                "ks_statistic": d.salience_ks_statistic,
                "ks_pvalue": d.salience_ks_pvalue,
            },
        }
    )
    return payload


def _atl_to_json(d: AtlDiff) -> dict[str, Any]:
    payload: dict[str, Any] = {"available": d.available}
    if not d.available:
        payload["reason"] = d.reason
        return payload
    payload.update(
        {
            "concept_count_a": d.concept_count_a,
            "concept_count_b": d.concept_count_b,
            "concept_count_delta": d.concept_count_delta,
            "jaccard": d.jaccard,
            "shared_names": list(d.shared_names),
            "only_in_a": list(d.only_in_a),
            "only_in_b": list(d.only_in_b),
        }
    )
    return payload


def substrate_diff_to_json(d: SubstrateDiff) -> dict[str, Any]:
    """Serialize a :class:`SubstrateDiff` to a plain JSON-ready dict."""
    return {
        "session_a": str(d.session_a),
        "session_b": str(d.session_b),
        "nac": _nac_to_json(d.nac),
        "ec": _ec_to_json(d.ec),
        "hippocampus": _hippocampus_to_json(d.hippocampus),
        "atl": _atl_to_json(d.atl),
    }


__all__ = [
    "AtlDiff",
    "EcDiff",
    "HippocampusDiff",
    "NacDiff",
    "SubstrateDiff",
    "atl_diff",
    "ec_diff",
    "hippocampus_diff",
    "nac_diff",
    "substrate_diff",
    "substrate_diff_to_json",
]
