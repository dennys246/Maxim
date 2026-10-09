#!/usr/bin/env python3
"""GL1 name-vs-consequence collision census: where do shipped affordance NAMES share a text node while their
declared body CONSEQUENCES disagree?

Grounding line GL1 (docs/plans/grounding.md, the GL1 row; state page docs/wiring/body-and-word-worlds.md §5). An
OFFLINE MEASUREMENT of the innate word prior, not a behavioural claim: no ledger row, no gate, no prereg. The
record is a ``diagnosis`` (``scripts/_provenance.py::stamp_diagnosis``): it informs a design and never counts as
support.

What it does:

1. Walks every shipped component under ``src/maxim/_data/components`` through ``ComponentRegistry`` (so
   ``extends`` inheritance is honoured; the registry's default user/legacy search paths are excluded) and collects
   every affordance instance: component ref, entity, modulator, affordance name, declared ``self_effect`` /
   ``target_effect`` (sensor -> delta) and ``requires``. ``archetypes/*.yaml`` are vocabulary templates, not
   entities (the registry does not index them); they are listed as excluded.
2. Encodes each affordance NAME through the production affordance path
   (``imagination/trigger.py::_make_aff_encoder(...).encode_decomposed(name, "text", ...)``) into ONE fresh EC at
   the production threshold (``ECConfig()``), recording the node of every chunk (the compound and each word).
   EC node ids are uuids, so nodes are relabelled ``N<k>`` in creation order (deterministic). Pairwise
   compound-name cosines are computed from the same encoder's ``embed``.
3. Reports (a) counts; (b) SAME-NODE collisions: two consequence variants whose compound lands on one node while
   their declared effects differ in sign on a shared (channel, sensor) or in harm class (one harmful, one safe);
   (c) same-consequence variant pairs on DIFFERENT nodes; (d) shared-word links (two compounds on different
   nodes sharing a word node); (e) order dependence: the text centroid is a running mean, so the walk runs in
   three deterministic orders (sorted, reverse-sorted, seeded shuffle) and the collision sets are compared.

Harm class (a single application from rest, against DECLARED drives only): a delta on a homeostatic sensor is
harmful when ``|delta| > comfort_band`` (rest = ``set_point``); on an entropic sensor when it moves in the drift
direction AND ``rest + delta`` is at or past ``deprivation_threshold`` (``>=`` for an up-drift, ``<=`` for a
down-drift), where rest is the sensor's declared ``rest:`` on that body (the component YAML, ``extends`` resolved),
else its declared ``initial`` state (``0.0`` for an up-drift sensor that declares neither; a down-drift sensor with
neither is unclassified on that body; an explicit ``rest: null`` means "no rest" and does not fall back). The
receiving body is the owning entity when it declares a drive
on that sensor; otherwise every shipped ``bodies/*`` component that declares one (an item's ``self_effect`` lands
on whoever invokes it). A sensor no body drives is unclassified. An affordance is harmful if any classified
sensor is, safe if at least one is classified and none is harmful.

Instrument checks (exit 4, a ``failed`` record, no result): the encoder must be the real model (never the hash
fallback), two runs of the sorted walk must assign identical nodes, and the known answer must hold (``touch`` on
``items/cradle_blanket`` and on ``items/cradle_fire_pit`` share a node and are a harm-class collision).

Usage (offline; the model must already be in the HF cache)::

    export PYTHONPATH="$PWD/src"
    HF_HOME=~/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \\
        python scripts/grounding_census.py [--write-experiment-results [--allow-dirty]]

Without ``--write-experiment-results`` the JSON and markdown go to a temp directory (both paths are printed);
with it they replace ``docs/experiments/data/census_name_consequence/census.json`` and
``docs/experiments/grounding_census.md``, from a clean ``src/`` + ``scripts/`` tree unless ``--allow-dirty``
(stamped into the record). An instrument refusal NEVER replaces a result: its ``failed`` record and summary go to
the siblings ``census.failed.json`` / ``census.failed.md`` next to the record path (in either mode), so a refused
re-run cannot overwrite the committed passing record; a passing ``--write-experiment-results`` run removes any stale
siblings. Exit 0 = a result; 3 = provenance refusal; 4 = instrument
refusal.

The data directory token is ``census_name_consequence`` (not ``grounding_*``) so a future ``grounding_*`` prereg
cannot govern this record by its data-dir token; a future prereg whose token is ``census`` WOULD, so GL3.B0 must
name its prereg and data directory accordingly.
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import math
import os
import random
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

# Every `maxim` / torch / HF import is inside a function, so `pin_environment()` (the first thing `main` does) runs
# before any of them. Importing this module (the unit test does) changes no environment.

REPO = Path(__file__).resolve().parents[1]
COMPONENTS = REPO / "src" / "maxim" / "_data" / "components"
RECORD = REPO / "docs" / "experiments" / "data" / "census_name_consequence" / "census.json"
SUMMARY = REPO / "docs" / "experiments" / "grounding_census.md"
MODEL = "paraphrase-mpnet-base-v2"
AGENT = "grounding_census"
SHUFFLE_SEED = 1120  # the #1120 audit that named the collisions
KNOWN_ANSWER = (("items/cradle_blanket", "touch"), ("items/cradle_fire_pit", "touch"))
# The state page §5 cosine table, re-measured here so the census and the page can be compared.
STATE_PAGE_PAIRS = (
    ("fire breath", "flame jet"),
    ("fire breath", "water jet"),
    ("flame jet", "water jet"),
    ("fire", "flame"),
    ("touch blanket", "touch fire pit"),
    ("turn left", "turn right"),
    ("escape water", "flee"),
)
TOP_N = 25
COSINE_STORE_FLOOR = 0.44  # compound pairs at or above the production threshold are stored individually
STAMPED_ENV = ("HF_HOME", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "MAXIM_DATA_HOME", "OMP_NUM_THREADS")


def pin_environment() -> None:
    """Offline model load and a throwaway MAXIM_DATA_HOME (nothing under ~/.maxim is read or written)."""
    os.environ.setdefault("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["MAXIM_DATA_HOME"] = tempfile.mkdtemp(prefix="maxim-grounding-census-home-")


class Refusal(Exception):
    """An instrument refusal: exit 4, a ``failed`` record, no result."""

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(f"{reason}: {detail}")
        self.reason = reason
        self.detail = detail


# ---------------------------------------------------------------------------
# Inventory
# ---------------------------------------------------------------------------


@dataclass
class AffordanceRecord:
    """One affordance instance on one shipped component (inherited ones included)."""

    ref: str
    entity: str
    category: str
    modulator: str
    name: str
    self_effect: dict[str, float] = field(default_factory=dict)
    target_effect: dict[str, float] = field(default_factory=dict)
    requires: dict[str, float] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.ref}:{self.modulator}.{self.name}"

    @property
    def effects(self) -> dict[str, float]:
        """Declared consequence as ``{"self:<sensor>": d, "target:<sensor>": d}`` (zero deltas dropped)."""
        out = {f"self:{k}": float(v) for k, v in self.self_effect.items() if v}
        out.update({f"target:{k}": float(v) for k, v in self.target_effect.items() if v})
        return out

    @property
    def variant(self) -> str:
        """A consequence variant: the name plus its exact declared effects (inheritance duplicates collapse)."""
        eff = ",".join(f"{k}={v:+g}" for k, v in sorted(self.effects.items()))
        return f"{self.name}[{eff}]"


def collect_affordances(registry: Any) -> tuple[list[AffordanceRecord], dict[str, Any]]:
    """Every affordance on every component the registry resolves (entity tree walked, children included)."""
    records: list[AffordanceRecord] = []
    failed: list[dict[str, str]] = []
    child_affordances = 0  # affordances on a child entity: the census walks them, production does not
    refs = sorted(registry.list_refs())
    for ref in refs:
        try:
            entity = registry.instantiate(ref)
        except Exception as exc:  # a component that does not parse is reported, never silently skipped
            failed.append({"ref": ref, "error": repr(exc)})
            continue
        info = registry.get_info(ref)
        category = info.category if info is not None else ref.split("/", 1)[0]
        stack = [entity]
        while stack:
            ent = stack.pop()
            if ent is not entity:
                child_affordances += sum(len(m.affordances) for m in ent.modulators.values())
            for mod_name in sorted(ent.modulators):
                mod = ent.modulators[mod_name]
                for aff_name in sorted(mod.affordances):
                    schema = mod.affordances[aff_name]
                    records.append(
                        AffordanceRecord(
                            ref=ref,
                            entity=ent.name,
                            category=category,
                            modulator=mod_name,
                            name=aff_name,
                            self_effect=dict(schema.self_effect or {}),
                            target_effect=dict(schema.target_effect or {}),
                            requires=dict(schema.requires or {}),
                        )
                    )
            stack.extend(reversed(list(getattr(ent, "children", None) or [])))
    return records, {"refs": len(refs), "failed": failed, "child_affordances": child_affordances}


def _number(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def declared_sensor_spec(spec: dict[str, Any] | None, key: str) -> dict[str, Any] | None:
    """The sensor's block in a resolved component spec (``registry.get(ref)``): ``entity.sensors.<key>`` or, for a
    modulator sub-sensor ``mod.sub``, ``entity.modulators.<mod>.sensors.<sub>``. None when absent."""
    entity = (spec or {}).get("entity") or {}
    if "." in key:
        mod_name, sub = key.split(".", 1)
        block = (((entity.get("modulators") or {}).get(mod_name) or {}).get("sensors") or {}).get(sub)
    else:
        block = (entity.get("sensors") or {}).get(key)
    return block if isinstance(block, dict) else None


def sensor_rest(entity: Any, key: str, spec: dict[str, Any] | None = None) -> float | None:
    """The rest of the sensor a drive is keyed on: the component YAML's declared ``rest:`` (``spec`` = the resolved
    component spec) when it declares one, else the declared ``initial`` state of the instantiated entity's sensor
    (an entity-level sensor such as ``hunger``, or a modulator sub-sensor such as ``arms.thermal``, seeded into the
    modulator's ``vital_metrics``). An explicit ``rest: null`` is the YAML saying the sensor HAS no rest (cyclic or
    place-dependent): None, never a fallback to ``initial``. None when neither is declared."""
    block = declared_sensor_spec(spec, key)
    if block is not None and "rest" in block:
        return _number(block["rest"])
    if "." in key:
        mod_name, sub = key.split(".", 1)
        mod = (getattr(entity, "modulators", None) or {}).get(mod_name)
        value = (getattr(mod, "vital_metrics", None) or {}).get(sub) if mod is not None else None
    else:
        sensor = (getattr(entity, "sensors", None) or {}).get(key)
        value = getattr(sensor, "_initial", None) if sensor is not None else None
    return _number(value)


def collect_drive_table(
    registry: Any, skip: set[str]
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, float | None]]]:
    """``({ref: {sensor_key: drive_spec}}, {ref: {sensor_key: rest}})`` for every component that declares a drive
    (``bodies/*`` are the receivers; the owner rule needs the rest). Rest is the sensor's declared ``rest:``, else
    its declared ``initial`` (:func:`sensor_rest`). ``skip`` = refs ``collect_affordances`` reported unparseable."""
    table: dict[str, dict[str, Any]] = {}
    rests: dict[str, dict[str, float | None]] = {}
    for ref in sorted(set(registry.list_refs()) - skip):
        entity = registry.instantiate(ref)
        specs = dict(getattr(entity, "drive_specs", {}) or {})
        if specs:
            table[ref] = specs
            resolved = registry.get(ref)
            rests[ref] = {key: sensor_rest(entity, key, resolved) for key in sorted(specs)}
    return table, rests


def sensor_harm(delta: float, spec: Any, rest: float | None = None) -> bool | None:
    """True when ONE application of ``delta`` from rest breaches the drive (see the module docstring).

    Homeostatic: rest is the set point, so the test is ``|delta| > comfort_band``. Entropic: the move must be in the
    drift direction and land at or past ``deprivation_threshold`` from ``rest`` (the declared rest, else the declared
    initial state; ``0.0`` when an up-drift sensor declares neither). None = unclassifiable (a down-drift sensor with no rest)."""
    if hasattr(spec, "comfort_band"):  # homeostatic
        return abs(delta) > float(spec.comfort_band)
    direction = getattr(spec, "drift_direction", "up")
    threshold = float(getattr(spec, "deprivation_threshold", math.inf))
    if rest is None:
        if direction != "up":
            return None
        rest = 0.0
    if direction == "up":
        return delta > 0 and rest + delta >= threshold
    return delta < 0 and rest + delta <= threshold


def _drive_mode(spec: Any) -> str:
    return "homeostatic" if hasattr(spec, "comfort_band") else "entropic"


def classify_harm(
    rec: AffordanceRecord,
    drive_table: dict[str, dict[str, Any]],
    rests: dict[str, dict[str, float | None]] | None = None,
) -> dict[str, Any]:
    """Harm class of one affordance instance, with the per-sensor reasoning kept for the record. ``rests``
    (``{ref: {sensor: rest}}``) sets an entropic sensor's rest per receiving body (absent = undeclared)."""
    rests = rests or {}
    bodies = {ref: specs for ref, specs in drive_table.items() if ref.startswith("bodies/")}
    sensors: dict[str, Any] = {}
    for key, delta in sorted(rec.effects.items()):
        channel, sensor = key.split(":", 1)
        owner_specs = drive_table.get(rec.ref, {})
        if channel == "self" and sensor in owner_specs:
            receivers = {rec.ref: owner_specs[sensor]}
            rule = "owner"
        else:
            receivers = {b: specs[sensor] for b, specs in bodies.items() if sensor in specs}
            rule = "any_body"
        verdicts = {b: sensor_harm(delta, spec, rests.get(b, {}).get(sensor)) for b, spec in receivers.items()}
        receivers = {b: spec for b, spec in receivers.items() if verdicts[b] is not None}
        if not receivers:
            sensors[key] = {"class": "unclassified"}
            continue
        harmful_in = sorted(b for b in receivers if verdicts[b])
        sensors[key] = {
            "class": "harmful" if harmful_in else "safe",
            "rule": rule,
            "harmful_in": harmful_in,
            "receivers": len(receivers),
            # Which test made it harmful: homeostatic = |delta| > comfort_band; entropic = deprivation reached.
            "harmful_drive_modes": sorted({_drive_mode(receivers[b]) for b in harmful_in}),
        }
    classes = {s["class"] for s in sensors.values()}
    if "harmful" in classes:
        harm = "harmful"
    elif "safe" in classes:
        harm = "safe"
    else:
        harm = "unclassified" if sensors else "no_effect"
    return {"class": harm, "sensors": sensors}


# ---------------------------------------------------------------------------
# Encoding (the production affordance path)
# ---------------------------------------------------------------------------


def build_aff_encoder() -> Any:
    """A fresh EC (production ``ECConfig()``), ATL and NAc, and the production affordance encoder over them."""
    from maxim.decisions.nac import NAc, NACConfig
    from maxim.imagination.trigger import _make_aff_encoder
    from maxim.memory.atl import ATL, ATLConfig
    from maxim.similarity.ec import ECConfig, EntorhinalCortex
    from maxim.similarity.encoder import EncoderConfig, LinguisticEncoder

    ec = EntorhinalCortex(config=ECConfig())
    base = LinguisticEncoder(ec=ec, atl=ATL(ATLConfig()), config=EncoderConfig(model_name=MODEL), nac=NAc(NACConfig()))
    aff = _make_aff_encoder(base)
    if aff is None:
        raise Refusal("aff_encoder", "_make_aff_encoder returned None (decomposer setup failed)")
    return aff


def assert_real_encoder(aff: Any) -> None:
    """Refuse a hash-fallback encoder, checked on the realized state the encoder stamps into its EC."""
    prov = aff.ec.encoder_provenance.get("linguistic", {})
    if aff.using_fallback or prov.get("using_fallback") is not False or prov.get("model_name") != MODEL:
        raise Refusal(
            "encoder_fallback",
            f"the affordance encoder is not the real {MODEL} (using_fallback={aff.using_fallback}, "
            f"EC provenance {aff.ec.encoder_provenance!r}); the census never computes on the hash fallback",
        )


def encode_walk(records: list[AffordanceRecord], order: list[int], make_encoder: Callable[[], Any]) -> dict[str, Any]:
    """Encode every record's NAME, in ``order``, into one fresh EC. Returns per-record chunk -> node labels."""
    from maxim.similarity.decomposer import AFFORDANCE_STRATEGY

    aff = make_encoder()
    labels: dict[str, str] = {}  # uuid -> N<k>, creation order
    first_text: dict[str, str] = {}
    per_record: dict[int, list[tuple[str, str]]] = {}
    for idx in order:
        name = records[idx].name
        chunks = [c.text for c in AFFORDANCE_STRATEGY.extract(name)]
        node_ids = aff.encode_decomposed(name, "text", AGENT)
        assert_real_encoder(aff)
        if len(node_ids) != len(chunks):
            raise Refusal("chunk_alignment", f"{name!r}: {len(chunks)} chunks but {len(node_ids)} node ids")
        out = []
        for text, nid in zip(chunks, node_ids):
            if nid not in labels:
                labels[nid] = f"N{len(labels)}"
                first_text[labels[nid]] = text
            out.append((text, labels[nid]))
        per_record[idx] = out
    return {"per_record": per_record, "node_first_text": first_text, "n_nodes": len(labels)}


def compound_cosines(names: list[str], embed: Callable[[str], list[float]]) -> dict[tuple[str, str], float]:
    """Pairwise cosine of the compound texts (``fire_breath`` -> ``fire breath``), sorted-key pairs."""
    vecs = {}
    for n in names:
        v = embed(n.replace("_", " "))
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        vecs[n] = [x / norm for x in v]
    out = {}
    for a, b in itertools.combinations(sorted(vecs), 2):
        out[(a, b)] = sum(x * y for x, y in zip(vecs[a], vecs[b]))
    return out


# ---------------------------------------------------------------------------
# Analysis (pure: no encoder, unit-tested on synthetic assignments)
# ---------------------------------------------------------------------------


def _sign(x: float) -> int:
    return (x > 0) - (x < 0)


def opposite_sensors(a: dict[str, float], b: dict[str, float]) -> list[str]:
    """Shared (channel, sensor) keys whose declared deltas have opposite signs."""
    return sorted(k for k in set(a) & set(b) if _sign(a[k]) * _sign(b[k]) < 0)


def sign_pattern(effects: dict[str, float]) -> tuple[tuple[str, int], ...]:
    return tuple(sorted((k, _sign(v)) for k, v in effects.items()))


def analyse(
    records: list[AffordanceRecord],
    walk: dict[str, Any],
    harm: list[dict[str, Any]],
    cosines: dict[tuple[str, str], float] | None = None,
) -> dict[str, Any]:
    """Collisions, cross-node same-consequence pairs and shared-word links for one walk."""
    per = walk["per_record"]
    compound_node = {i: per[i][0][1] for i in per}

    # Consequence variants (name + exact effects); each keeps its instances and the node(s) its compound took.
    variants: dict[str, dict[str, Any]] = {}
    for i, rec in enumerate(records):
        if i not in per:
            continue
        v = variants.setdefault(
            rec.variant,
            {"name": rec.name, "effects": rec.effects, "harm": harm[i]["class"], "refs": [], "nodes": set()},
        )
        v["refs"].append(f"{rec.ref}:{rec.modulator}")
        v["nodes"].add(compound_node[i])
        if harm[i]["class"] == "harmful":
            v["harm"] = "harmful"  # an identical effect classifies identically unless the owner rule differs
    split_names = sorted(k for k, v in variants.items() if len(v["nodes"]) > 1)

    def cos(a: str, b: str) -> float | None:
        if cosines is None or a == b:
            return 1.0 if a == b else None
        return cosines.get((min(a, b), max(a, b)))

    collisions: list[dict[str, Any]] = []
    same_cons_diff_node: list[dict[str, Any]] = []
    keys = sorted(variants)
    for ka, kb in itertools.combinations(keys, 2):
        va, vb = variants[ka], variants[kb]
        if not va["effects"] or not vb["effects"]:
            continue
        shared_nodes = va["nodes"] & vb["nodes"]
        if shared_nodes:
            opp = opposite_sensors(va["effects"], vb["effects"])
            harm_flip = {va["harm"], vb["harm"]} == {"harmful", "safe"}
            if opp or harm_flip:
                magnitude = sum(abs(va["effects"][k] - vb["effects"][k]) for k in opp)
                collisions.append(
                    {
                        "a": ka,
                        "b": kb,
                        "node": sorted(shared_nodes)[0],
                        "same_name": va["name"] == vb["name"],
                        "opposite_sensors": opp,
                        "harm": [va["harm"], vb["harm"]],
                        "harm_flip": harm_flip,
                        "opposite_magnitude": round(magnitude, 6),
                        "cosine": cos(va["name"], vb["name"]),
                        "refs_a": sorted(va["refs"]),
                        "refs_b": sorted(vb["refs"]),
                    }
                )
        elif (
            va["name"] != vb["name"]
            and sign_pattern(va["effects"]) == sign_pattern(vb["effects"])
            and va["harm"] == vb["harm"]
        ):
            same_cons_diff_node.append(
                {
                    "a": ka,
                    "b": kb,
                    "nodes": [sorted(va["nodes"]), sorted(vb["nodes"])],
                    "exact_same_deltas": va["effects"] == vb["effects"],
                    "harm": va["harm"],
                    "cosine": cos(va["name"], vb["name"]),
                }
            )
    collisions.sort(key=lambda c: (-int(c["harm_flip"]), int(c["same_name"]), -c["opposite_magnitude"], c["a"], c["b"]))

    # Shared-word links and component placement (multi-word names only; one entry per distinct name).
    name_chunks: dict[str, list[tuple[str, str]]] = {}
    for i in sorted(per):
        name_chunks.setdefault(records[i].name, per[i])
    word_nodes: dict[str, dict[str, set[str]]] = {}  # word node -> {compound name -> {words}}
    absorbed, own_node, onto_other = [], [], []
    compound_nodes_all = {chunks[0][1] for chunks in name_chunks.values()}
    for name, chunks in sorted(name_chunks.items()):
        compound = chunks[0][1]
        for word, node in chunks[1:]:
            word_nodes.setdefault(node, {}).setdefault(name, set()).add(word)
            if node == compound:
                absorbed.append(f"{name}:{word}")
            elif node in compound_nodes_all:
                onto_other.append(f"{name}:{word}->{node}")
            else:
                own_node.append(f"{name}:{word}")
    effect_names = {r.name for r in records if r.effects}
    links = []
    for node, members in sorted(word_nodes.items()):
        for a, b in itertools.combinations(sorted(members), 2):
            if name_chunks[a][0][1] == name_chunks[b][0][1]:
                continue  # already on one compound node: a collision question, not a word link
            links.append(
                {
                    "a": a,
                    "b": b,
                    "word_node": node,
                    "words": sorted(members[a] | members[b]),
                    "cosine": cos(a, b),
                }
            )

    distinct_names = sorted({r.name for r in records})
    co_noded = set()
    by_node: dict[str, set[str]] = {}
    for name in distinct_names:
        if name in name_chunks:
            by_node.setdefault(name_chunks[name][0][1], set()).add(name)
    for members in by_node.values():
        co_noded.update(itertools.combinations(sorted(members), 2))

    drift = {}
    if cosines is not None:
        thr = walk.get("threshold")
        if thr is not None:
            absorbed_below = [
                {"a": a, "b": b, "cosine": round(cosines[(a, b)], 4)}
                for (a, b) in sorted(co_noded)
                if (a, b) in cosines and cosines[(a, b)] < thr
            ]
            separated_above = [
                {"a": a, "b": b, "cosine": round(c, 4)}
                for (a, b), c in sorted(cosines.items())
                if c >= thr and (a, b) not in co_noded and a in name_chunks and b in name_chunks
            ]
            drift = {
                "co_noded_below_threshold": absorbed_below,
                "separated_at_or_above_threshold": separated_above,
            }

    return {
        "n_nodes": walk["n_nodes"],
        "compound_nodes": len(compound_nodes_all),
        "variants": len(variants),
        "variants_with_effect": sum(1 for v in variants.values() if v["effects"]),
        "names_split_across_nodes": split_names,
        "collisions": collisions,
        "collision_keys": sorted(f"{c['a']} || {c['b']}" for c in collisions),
        "harm_flip_collisions": sum(1 for c in collisions if c["harm_flip"]),
        "opposite_sign_collisions": sum(1 for c in collisions if c["opposite_sensors"]),
        "cross_name_collisions": sum(1 for c in collisions if not c["same_name"]),
        "same_consequence_different_node": same_cons_diff_node,
        "shared_word_links": links,
        "shared_word_links_between_effect_names": [
            link for link in links if link["a"] in effect_names and link["b"] in effect_names
        ],
        "components": {
            "absorbed_into_own_compound": absorbed,
            "landed_on_another_compound": onto_other,
            "own_node": own_node,
        },
        "co_noded_name_pairs": sorted(f"{a} || {b}" for a, b in co_noded),
        "drift": drift,
        "nodes": [
            {"node": n, "first_text": walk["node_first_text"].get(n), "names": sorted(m)}
            for n, m in sorted(by_node.items(), key=lambda kv: (-len(kv[1]), kv[0]))
        ],
    }


def own_yaml_blocks() -> dict[str, int]:
    """Affordances counted in each file's OWN ``affordances:`` blocks, no inheritance, archetypes excluded: only to
    reconcile with the state page §5's walk ("42 of 405"), never used by the analysis."""
    import yaml

    total = with_effect = 0

    def visit(node: Any) -> None:
        nonlocal total, with_effect
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "affordances" and isinstance(value, dict):
                    for spec in value.values():
                        total += 1
                        if isinstance(spec, dict) and (spec.get("self_effect") or spec.get("target_effect")):
                            with_effect += 1
                else:
                    visit(value)
        elif isinstance(node, list):
            for item in node:
                visit(item)

    for path in sorted(COMPONENTS.rglob("*.yaml")):
        if "archetypes" not in path.relative_to(COMPONENTS).parts:
            visit(yaml.safe_load(path.read_text()))
    return {"affordances": total, "with_declared_effect": with_effect}


def compact_links(result: dict[str, Any], node_first_text: dict[str, str], top: int = 20) -> dict[str, Any]:
    """Replace the full shared-word link list (thousands of pairs, mostly drift) with its count and the word
    nodes that fan out widest; links between effect-bearing names stay in full."""
    links = result.pop("shared_word_links")
    fan: dict[str, dict[str, Any]] = {}
    for link in links:
        f = fan.setdefault(link["word_node"], {"names": set(), "words": set()})
        f["names"].update((link["a"], link["b"]))
        f["words"].update(link["words"])
    result["shared_word_links_count"] = len(links)
    result["shared_word_fanout"] = [
        {
            "word_node": n,
            "first_text": node_first_text.get(n),
            "compounds": len(f["names"]),
            "words": sorted(f["words"]),
        }
        for n, f in sorted(fan.items(), key=lambda kv: (-len(kv[1]["names"]), kv[0]))[:top]
    ]
    return result


def walk_orders(n: int, records: list[AffordanceRecord], seed: int) -> dict[str, list[int]]:
    """Three deterministic orders over the records: sorted, reverse-sorted and a seeded shuffle."""
    srt = sorted(range(n), key=lambda i: (records[i].name, records[i].ref, records[i].modulator))
    shuffled = list(srt)
    random.Random(seed).shuffle(shuffled)
    return {"sorted": srt, "reverse_sorted": list(reversed(srt)), f"shuffle_seed_{seed}": shuffled}


def jaccard(a: set[str], b: set[str]) -> float:
    return 1.0 if not a and not b else len(a & b) / len(a | b)


def order_dependence(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """How the collision set and the name partition move across walk orders."""
    sets = {k: set(v["collision_keys"]) for k, v in results.items()}
    parts = {k: set(v["co_noded_name_pairs"]) for k, v in results.items()}
    inter = set.intersection(*sets.values()) if sets else set()
    union = set.union(*sets.values()) if sets else set()
    pairwise = {
        f"{a} vs {b}": {
            "collision_jaccard": round(jaccard(sets[a], sets[b]), 4),
            "co_noded_names_jaccard": round(jaccard(parts[a], parts[b]), 4),
        }
        for a, b in itertools.combinations(sorted(sets), 2)
    }
    return {
        "per_order": {
            k: {
                "nodes": v["n_nodes"],
                "compound_nodes": v["compound_nodes"],
                "collisions": len(v["collisions"]),
                "harm_flip": v["harm_flip_collisions"],
                "opposite_sign": v["opposite_sign_collisions"],
                "co_noded_name_pairs": len(v["co_noded_name_pairs"]),
            }
            for k, v in results.items()
        },
        "collisions_in_every_order": len(inter),
        "collisions_in_any_order": len(union),
        "order_sensitive_collisions": sorted(union - inter),
        "pairwise": pairwise,
    }


def known_answer(records: list[AffordanceRecord], walk: dict[str, Any], analysis: dict[str, Any]) -> dict[str, Any]:
    """``touch`` on the blanket and on the fire pit must share a node and be a harm-class collision."""
    per = walk["per_record"]
    nodes = {}
    variants = {}
    for i, rec in enumerate(records):
        for ref, name in KNOWN_ANSWER:
            if rec.ref == ref and rec.name == name and i in per:
                nodes[ref] = per[i][0][1]
                variants[ref] = rec.variant
    a, b = (variants.get(r) for r, _ in KNOWN_ANSWER)
    hit = next((c for c in analysis["collisions"] if {c["a"], c["b"]} == {a, b}), None)
    ok = len(set(nodes.values())) == 1 and len(nodes) == 2 and hit is not None and hit["harm_flip"]
    return {"ok": ok, "nodes": nodes, "variants": [a, b], "collision": hit}


# ---------------------------------------------------------------------------
# Run, record, summary
# ---------------------------------------------------------------------------


def ec_config_record(config: Any) -> dict[str, Any]:
    """A deterministic JSON form of an ``ECConfig``: ``dataclasses.asdict`` with every set/frozenset sorted into a
    list (``repr`` of a frozenset follows string-hash order, which ``PYTHONHASHSEED`` randomizes per process)."""

    def norm(value: Any) -> Any:
        if isinstance(value, (set, frozenset)):
            return sorted(norm(v) for v in value)
        if isinstance(value, dict):
            return {str(k): norm(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
        if isinstance(value, (list, tuple)):
            return [norm(v) for v in value]
        if isinstance(value, Path):
            return str(value)
        return value

    return norm(dataclasses.asdict(config))


def measure(seed: int) -> dict[str, Any]:
    from maxim.embodiment.component_registry import ComponentRegistry
    from maxim.similarity.ec import ECConfig

    registry = ComponentRegistry(search_paths=[COMPONENTS], include_defaults=False)
    records, inventory = collect_affordances(registry)
    drive_table, rests = collect_drive_table(registry, {f["ref"] for f in inventory["failed"]})
    harm = [classify_harm(r, drive_table, rests) for r in records]
    threshold = ECConfig().pattern_complete_threshold
    archetypes = sorted(str(p.relative_to(REPO)) for p in (COMPONENTS / "archetypes").glob("*.yaml"))
    indexed = {Path(info.source_path).resolve() for info in (registry.get_info(r) for r in registry.list_refs())}
    unindexed = sorted(str(p.relative_to(REPO)) for p in COMPONENTS.rglob("*.yaml") if p.resolve() not in indexed)

    encoder_info = load_encoder()
    orders = walk_orders(len(records), records, seed)
    results: dict[str, dict[str, Any]] = {}
    walks: dict[str, dict[str, Any]] = {}
    for label, order in orders.items():
        walks[label] = encode_walk(records, order, build_aff_encoder)
        walks[label]["threshold"] = threshold

    # Determinism: a second sorted walk must assign the same nodes.
    again = encode_walk(records, orders["sorted"], build_aff_encoder)
    if again["per_record"] != walks["sorted"]["per_record"]:
        raise Refusal("determinism", "two sorted walks assigned different nodes")

    names = sorted({r.name for r in records})
    embedder = build_aff_encoder()
    cosines = compound_cosines(names, embedder.embed)
    assert_real_encoder(embedder)
    for label in orders:
        results[label] = compact_links(analyse(records, walks[label], harm, cosines), walks[label]["node_first_text"])

    ka = known_answer(records, walks["sorted"], results["sorted"])
    if not ka["ok"]:
        raise Refusal("known_answer", f"blanket/fire-pit touch is not a same-node harm collision: {ka!r}")

    probe = build_aff_encoder()
    page_pairs = []
    for a, b in STATE_PAGE_PAIRS:
        va, vb = probe.embed(a), probe.embed(b)
        na = math.sqrt(sum(x * x for x in va))
        nb = math.sqrt(sum(x * x for x in vb))
        page_pairs.append({"a": a, "b": b, "cosine": round(sum(x * y for x, y in zip(va, vb)) / (na * nb), 4)})

    cos_values = sorted(cosines.values())
    hist_edges = [-1.0, 0.2, 0.3, 0.4, 0.44, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
    hist = {f"[{lo},{hi})": sum(1 for c in cos_values if lo <= c < hi) for lo, hi in itertools.pairwise(hist_edges)}
    with_effect = [i for i, r in enumerate(records) if r.effects]
    harm_counts: dict[str, int] = {}
    for h in harm:
        harm_counts[h["class"]] = harm_counts.get(h["class"], 0) + 1
    harmful_by_sensor: dict[str, int] = {}
    for h in harm:
        for key, sens in h["sensors"].items():
            if sens["class"] == "harmful":
                harmful_by_sensor[key] = harmful_by_sensor.get(key, 0) + 1
    name_counts: dict[str, int] = {}
    for r in records:
        name_counts[r.name] = name_counts.get(r.name, 0) + 1

    primary = results["sorted"]
    return {
        "encoder": encoder_info,
        "threshold": threshold,
        "ec_config": ec_config_record(ECConfig()),
        "shuffle_seed": seed,
        "inventory": {
            "components_indexed": inventory["refs"],
            "components_failed_to_parse": inventory["failed"],
            "child_entity_affordances": inventory["child_affordances"],
            "excluded_not_entities": archetypes,
            "unindexed_yaml": unindexed,
            "drive_bearing_components": sorted(drive_table),
            "own_yaml_blocks_no_inheritance": own_yaml_blocks(),
        },
        "counts": {
            "affordances": len(records),
            "with_declared_effect": len(with_effect),
            "distinct_names": len(names),
            "distinct_names_with_effect": len({records[i].name for i in with_effect}),
            "consequence_variants_with_effect": primary["variants_with_effect"],
            "harm_class": harm_counts,
            "harmful_instances_by_sensor": dict(sorted(harmful_by_sensor.items())),
            "most_repeated_names": sorted(name_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:15],
        },
        "affordances": [
            {
                "ref": r.ref,
                "entity": r.entity,
                "modulator": r.modulator,
                "name": r.name,
                "self_effect": r.self_effect,
                "target_effect": r.target_effect,
                "requires": r.requires,
                "harm": harm[i],
                "chunks_sorted_walk": walks["sorted"]["per_record"][i],
            }
            for i, r in enumerate(records)
        ],
        "results": {"primary_order": "sorted", **results},
        "order_dependence": order_dependence(results),
        "known_answer": ka,
        "cosines": {
            "pairs": len(cos_values),
            "histogram": hist,
            "at_or_above_threshold": [
                {"a": a, "b": b, "cosine": round(c, 4)}
                for (a, b), c in sorted(cosines.items(), key=lambda kv: -kv[1])
                if c >= COSINE_STORE_FLOOR
            ],
            "state_page_pairs": page_pairs,
        },
    }


def load_encoder() -> dict[str, Any]:
    """Load the real model through the production loader (refusing the fallback) and pin it to CPU."""
    from maxim.exceptions import ModelLoadError
    from maxim.similarity.encoder import _get_encoder, require_semantic_encoder

    try:
        require_semantic_encoder(MODEL, context="GL1 grounding census")
    except ModelLoadError as exc:
        raise Refusal("encoder_fallback", f"the real semantic encoder did not load: {exc}") from exc
    model = _get_encoder(MODEL)
    revision = getattr(getattr(model[0].auto_model, "config", None), "_commit_hash", None)
    production_device = str(model.device)
    model.to("cpu")  # deterministic across machines; the production device is recorded
    import sentence_transformers
    import torch
    import transformers

    return {
        "model": MODEL,
        "model_id": f"sentence-transformers/{MODEL}",
        "snapshot_revision": revision,
        "production_device": production_device,
        "measurement_device": str(model.device),
        "sentence_transformers": sentence_transformers.__version__,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }


def _fmt_collision(c: dict[str, Any]) -> str:
    why = []
    if c["harm_flip"]:
        why.append(f"harm {c['harm'][0]} vs {c['harm'][1]}")
    if c["opposite_sensors"]:
        why.append("opposite " + ", ".join(c["opposite_sensors"]))
    cos = "" if c["cosine"] is None else f" (cos {c['cosine']:.3f})"
    ra = ", ".join(c["refs_a"][:3]) + (" …" if len(c["refs_a"]) > 3 else "")
    rb = ", ".join(c["refs_b"][:3]) + (" …" if len(c["refs_b"]) > 3 else "")
    return f"| `{c['a']}` | `{c['b']}` | {c['node']}{cos} | {'; '.join(why)} | {ra} / {rb} |"


def _variant_name(variant: str) -> str:
    return variant.split("[", 1)[0]


def headline_facts(report: dict[str, Any]) -> dict[str, Any]:
    """The direct reading of the collision lists the headline states (all computed from the record): which names
    the harm flips are between, the cross-name collisions per order, and how many harmful instances are turns."""
    primary = report["results"]["sorted"]
    flips = [c for c in primary["collisions"] if c["harm_flip"]]
    same_flips = [c for c in flips if c["same_name"]]
    cross = [c for c in primary["collisions"] if not c["same_name"]]
    other_cross_flips: dict[str, dict[str, int]] = {}
    for label, res in report["results"].items():
        if label in ("primary_order", "sorted") or not isinstance(res, dict):
            continue
        for c in res["collisions"]:
            if c["harm_flip"] and not c["same_name"]:
                pair = " ↔ ".join(sorted((_variant_name(c["a"]), _variant_name(c["b"]))))
                per_order = other_cross_flips.setdefault(label, {})
                per_order[pair] = per_order.get(pair, 0) + 1
    harmful = [a for a in report["affordances"] if a["harm"]["class"] == "harmful"]
    turn_harm = [a for a in harmful if (a["harm"]["sensors"].get("self:azimuth") or {}).get("class") == "harmful"]
    cos_values = sorted({round(c["cosine"], 3) for c in same_flips if c["cosine"] is not None})
    return {
        # The qualitative claims the headline makes, each a boolean over the lists (the text branches on them).
        "all_harm_flips_same_name": bool(flips) and len(same_flips) == len(flips),
        "cross_name_all_opposite_and_both_harmful": bool(cross)
        and all(c.get("opposite_sensors") and list(c.get("harm") or []) == ["harmful", "harmful"] for c in cross),
        "cross_name_all_azimuth": bool(cross)
        and all("self:azimuth" in (c.get("opposite_sensors") or []) for c in cross),
        "azimuth_harm_all_comfort_band": bool(turn_harm)
        and all(a["harm"]["sensors"]["self:azimuth"].get("harmful_drive_modes") == ["homeostatic"] for a in turn_harm),
        "harm_flips": len(flips),
        "same_name_harm_flips": len(same_flips),
        "same_name_harm_flip_names": sorted({_variant_name(c["a"]) for c in same_flips}),
        "same_name_harm_flip_cosines": cos_values,
        "cross_name_harm_flips": len(flips) - len(same_flips),
        "cross_name_collisions": len(cross),
        "cross_name_collision_pairs": sorted(
            {" ↔ ".join(sorted((_variant_name(c["a"]), _variant_name(c["b"])))) for c in cross}
        ),
        "cross_name_harm_flips_other_orders": {
            k: dict(sorted(v.items())) for k, v in sorted(other_cross_flips.items())
        },
        "harmful_instances": len(harmful),
        "harmful_instances_azimuth": len(turn_harm),
        "harmful_azimuth_names": sorted({a["name"] for a in turn_harm}),
    }


def _ticks(items: list[str]) -> str:
    return ", ".join(f"`{x}`" for x in items)


def headline_bullet(hf: dict[str, Any], other_flip_text: str) -> str:
    """The headline's qualitative sentences, each stated only when its computed boolean holds (a re-run can never
    print a claim its numbers contradict)."""
    flips, same = hf["harm_flips"], hf["same_name_harm_flips"]
    names = ", ".join(f"`{n}`↔`{n}`" for n in hf["same_name_harm_flip_names"])
    cosines = ", ".join(f"{x:.3f}" for x in hf["same_name_harm_flip_cosines"])
    if hf["all_harm_flips_same_name"]:
        out = (
            f"- **Every primary-walk harm flip is a SAME-name variant pair**: {same} of {flips} ({names}, cosine "
            f"{cosines}): one name, one node, harmful and safe consequences."
        )
    elif flips:
        out = (
            f"- **{same} of {flips}** primary-walk harm flips are SAME-name variant pairs"
            + (f" ({names}, cosine {cosines})" if same else "")
            + "; the rest are between DIFFERENT names."
        )
    else:
        out = "- **No** harm flips in the primary walk."
    out += f" **{hf['cross_name_harm_flips']}** cross-name harm flips in the primary walk{other_flip_text}."
    n_cross = hf["cross_name_collisions"]
    pairs = _ticks(hf["cross_name_collision_pairs"])
    if not n_cross:
        out += " **No** cross-name collisions."
    else:
        kind = "all orient (azimuth) pairs" if hf["cross_name_all_azimuth"] else "not all orient (azimuth) pairs"
        signs = (
            "opposite signs, both harmful"
            if hf["cross_name_all_opposite_and_both_harmful"]
            else "NOT all opposite-signed with both sides harmful"
        )
        out += f" The **{n_cross}** cross-name collisions are {kind} ({pairs}): {signs}."
    out += (
        f' **{hf["harmful_instances_azimuth"]} of {hf["harmful_instances"]}** "harmful" instances are azimuth turns'
        + (f" ({_ticks(hf['harmful_azimuth_names'])})" if hf["harmful_azimuth_names"] else "")
    )
    if hf["harmful_instances_azimuth"]:
        out += (
            ", classed harmful because one turn exceeds the orienting drive's comfort band."
            if hf["azimuth_harm_all_comfort_band"]
            else ", not all of them through the orienting drive's comfort band."
        )
    else:
        out += "."
    return out + " A measurement of the word prior, nothing more."


def render_summary(report: dict[str, Any], json_rel: str) -> str:
    """The human-readable summary; every number is read from the record it accompanies."""
    if report.get("refusal"):
        return (
            "# GL1 name-vs-consequence collision census\n\n"
            f"**REFUSED — {report['refusal']}**: {report['refusal_detail']}\n\nRecord: `{json_rel}`.\n"
        )
    c = report["counts"]
    p = report["results"]["sorted"]
    od = report["order_dependence"]
    prov = report["code_provenance"]
    hf = headline_facts(report)
    other_flips = hf["cross_name_harm_flips_other_orders"]
    other_flip_text = (
        "; in the other orders: "
        + "; ".join(
            f"{k} {sum(v.values())} ({', '.join(f'`{pair}` ×{n}' for pair, n in v.items())})"
            for k, v in other_flips.items()
        )
        + " (order-sensitive)"
        if other_flips
        else "; none in the other orders either"
    )
    lines = [
        "# GL1 name-vs-consequence collision census",
        "",
        "> **An offline measurement, not a behavioural claim.** No ledger row, no gate, no prereg: the record is a",
        "> `diagnosis` (it informs the grounding line's design and never counts as support). Plan:",
        "> [docs/plans/grounding.md](../plans/grounding.md) (GL1 row); state page:",
        "> [docs/wiring/body-and-word-worlds.md](../wiring/body-and-word-worlds.md) §5.",
        "",
        f"Script: [`scripts/grounding_census.py`](../../scripts/grounding_census.py). Record: `{json_rel}`. "
        f"Encoder `{report['encoder']['model_id']}` (revision `{report['encoder']['snapshot_revision']}`, "
        f"{report['encoder']['measurement_device']}), EC text threshold **{report['threshold']}** "
        f"(`ECConfig()`), commit `{prov.get('executed_git_hash', '?')[:12]}`, "
        f"tree dirty: **{prov.get('working_tree_dirty_src_scripts')}**"
        + (", `allow_dirty: true`" if prov.get("allow_dirty") else "")
        + ".",
        "",
        "## Headline",
        "",
        headline_bullet(hf, other_flip_text),
        f"- **{c['affordances']}** affordance instances on {report['inventory']['components_indexed']} shipped "
        f"components (`extends` honoured), **{c['with_declared_effect']}** with a declared effect, "
        f"**{c['distinct_names']}** distinct names ({c['distinct_names_with_effect']} with an effect; "
        f"{c['consequence_variants_with_effect']} distinct consequence variants). Counted in each file's own "
        f"`affordances:` blocks without inheritance (the state page §5 walk): "
        f"{report['inventory']['own_yaml_blocks_no_inheritance']}.",
        f"- Harm classes per instance: {c['harm_class']}; harmful instances by sensor: "
        f"{c['harmful_instances_by_sensor']}.",
        f"- Sorted walk: **{p['n_nodes']}** text nodes ({p['compound_nodes']} holding a compound); "
        f"names whose instances split across nodes: {p['names_split_across_nodes'] or 'none'}.",
        f"- **{len(p['collisions'])}** same-node collisions between consequence variants: "
        f"**{p['harm_flip_collisions']}** harm-class flips (harmful vs safe), "
        f"**{p['opposite_sign_collisions']}** with opposite signs on a shared sensor; "
        f"{p['cross_name_collisions']} are between DIFFERENT names.",
        f"- **{len(p['same_consequence_different_node'])}** same-consequence pairs (same sign pattern and harm "
        "class, different names) landing on DIFFERENT nodes.",
        f"- **{p['shared_word_links_count']}** shared-word links (compounds on different nodes sharing a word node). "
        f"Components: {len(p['components']['absorbed_into_own_compound'])} absorbed into their own compound's "
        f"node, {len(p['components']['landed_on_another_compound'])} onto another compound's node, "
        f"{len(p['components']['own_node'])} on a node of their own.",
        f"- Known answer (`touch` on blanket vs fire pit): **{'PASS' if report['known_answer']['ok'] else 'FAIL'}** "
        f"(node {sorted(set(report['known_answer']['nodes'].values()))}).",
        "",
        "## Order dependence (the text centroid is a running mean)",
        "",
        "| Order | Nodes | Collisions | Harm flips | Opposite sign | Co-noded name pairs |",
        "|---|---|---|---|---|---|",
    ]
    for k, v in od["per_order"].items():
        lines.append(
            f"| {k} | {v['nodes']} | {v['collisions']} | {v['harm_flip']} | {v['opposite_sign']} | "
            f"{v['co_noded_name_pairs']} |"
        )
    lines += [
        "",
        f"Collisions in every order: **{od['collisions_in_every_order']}**; in any order: "
        f"**{od['collisions_in_any_order']}** ({len(od['order_sensitive_collisions'])} order-sensitive).",
        "",
    ]
    for k, v in od["pairwise"].items():
        lines.append(
            f"- {k}: collision Jaccard {v['collision_jaccard']}, co-noded-name Jaccard {v['co_noded_names_jaccard']}"
        )
    lines += [
        "",
        f"## Top collisions (sorted walk; harm flips first, then cross-name, then opposite-sign magnitude; "
        f"first {min(TOP_N, len(p['collisions']))} of {len(p['collisions'])})",
        "",
        "| Variant A | Variant B | Node (cosine) | Why | Instances A / B |",
        "|---|---|---|---|---|",
    ]
    lines += [_fmt_collision(col) for col in p["collisions"][:TOP_N]]
    lines += [
        "",
        "## Same consequence, different node (cross-name)",
        "",
    ]
    lines += [
        f"- `{x['a']}` ({x['nodes'][0]}) vs `{x['b']}` ({x['nodes'][1]}), harm {x['harm']}, "
        f"cos {x['cosine'] if x['cosine'] is None else round(x['cosine'], 3)}"
        + (", identical deltas" if x["exact_same_deltas"] else "")
        for x in p["same_consequence_different_node"]
    ] or ["- none"]
    ewl = p["shared_word_links_between_effect_names"]
    lines += [
        "",
        f"## Shared-word links between effect-bearing names ({len(ewl)} of {p['shared_word_links_count']})",
        "",
    ]
    lines += [f"- `{x['a']}` ~ `{x['b']}` via {x['words']} on {x['word_node']}" for x in ewl[:TOP_N]] or ["- none"]
    lines += ["", "Widest word nodes (compounds linked through one word node):", ""]
    lines += [
        f"- {x['word_node']} (first text `{x['first_text']}`): {x['compounds']} compounds via {x['words'][:8]}"
        for x in p["shared_word_fanout"][:5]
    ]
    lines += [
        "",
        "## Order-sensitive collisions (present in some walk orders, not all)",
        "",
    ]
    lines += [f"- `{k}`" for k in od["order_sensitive_collisions"]] or ["- none"]
    lines += [
        "",
        "## State-page cosines, re-measured",
        "",
        "| Pair | Cosine |",
        "|---|---|",
    ]
    lines += [f"| {x['a']} ↔ {x['b']} | {x['cosine']} |" for x in report["cosines"]["state_page_pairs"]]
    drift = p.get("drift") or {}
    lines += [
        "",
        "## Method notes and caveats",
        "",
        "- Harm class is a single application from rest against DECLARED drives (homeostatic: rest = `set_point`, "
        "harmful when `|delta| > comfort_band`; entropic: rest = the sensor's declared `rest:` on that body, else "
        "its declared `initial` state, harmful when the move is in the drift direction and `rest + delta` reaches "
        "`deprivation_threshold`; an up-drift sensor declaring neither rests at 0, a down-drift one is "
        "unclassified), on the owning "
        "entity if it drives that sensor, else on every shipped body that does. Undriven sensors (e.g. `hp`) are "
        "unclassified, so harm flips are a LOWER bound on consequence disagreement. Orienting drives (`azimuth`) "
        "count as drives.",
        f"- Compound pairs co-noded below the threshold (absorbed by running-mean drift): "
        f"{len(drift.get('co_noded_below_threshold', []))}; pairs at or above it on different nodes: "
        f"{len(drift.get('separated_at_or_above_threshold', []))}.",
        "- The walk encodes every instance (inheritance duplicates included, as production re-encodes a name per "
        "entity), into ONE EC holding every shipped component's names: a population-level prior, not any single "
        "scenario's EC.",
        "- `archetypes/*.yaml` are vocabulary templates, not entities, and are excluded.",
        "- The census walks every entity's `children` as well as its top-level modulators; production encodes "
        "top-level modulators only. "
        + (
            f"Affordances on child entities in this inventory: **{report['inventory']['child_entity_affordances']}**"
            + (
                ", so the two agree here (a future child affordance would enter the census before production)."
                if report["inventory"]["child_entity_affordances"] == 0
                else "; those are in the census but not in production's encoding."
            )
            if "child_entity_affordances" in report["inventory"]
            else "This record predates the child-affordance count."
        ),
        f"- Measured on `{report['encoder']['measurement_device']}`; production runs the encoder on "
        f"`{report['encoder']['production_device']}`. Node assignment near the threshold can differ by device.",
        "- The data directory token is `census_name_consequence` (not `grounding_*`) so a future `grounding_*` "
        "prereg cannot govern this record by its token. A future prereg whose token is `census` WOULD: GL3.B0 "
        "must name its prereg and data directory so it does not.",
        "- An instrument refusal never replaces this record: a refused run writes `census.failed.json` / "
        "`census.failed.md` beside it instead.",
        "",
    ]
    return "\n".join(lines)


def failed_paths(json_path: Path) -> tuple[Path, Path]:
    """Where a refusal is written: siblings of the record, so a refused run never overwrites a result."""
    return (
        json_path.with_name(f"{json_path.stem}.failed.json"),
        json_path.with_name(f"{json_path.stem}.failed.md"),
    )


def _write(json_path: Path, md_path: Path, report: dict[str, Any]) -> None:
    from maxim.utils.atomic_io import atomic_write_json, atomic_write_text
    from maxim.utils.format_version import with_format_version

    json_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(json_path, with_format_version(report))
    try:
        json_rel = str(json_path.resolve().relative_to(REPO))
    except ValueError:
        json_rel = str(json_path)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(md_path, render_summary(report, json_rel))
    print(f"[census] record written: {json_path}\n[census] summary written: {md_path}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument(
        "--write-experiment-results",
        action="store_true",
        help="replace the committed record + summary under docs/experiments/ (clean tree enforced); default: temp",
    )
    ap.add_argument(
        "--allow-dirty", action="store_true", help="with --write-experiment-results: allow a dirty tree (stamped)"
    )
    ap.add_argument("--shuffle-seed", type=int, default=SHUFFLE_SEED, help="seed of the third walk order")
    args = ap.parse_args(argv)
    pin_environment()
    sys.path.insert(0, str(REPO / "scripts"))

    from _provenance import (
        DirtyTreeError,
        ProvenanceError,
        evidence_out_paths_or_exit,
        in_process_code_provenance,
        stamp_diagnosis,
    )

    json_path, md_path = evidence_out_paths_or_exit(
        REPO, [RECORD, SUMMARY], write_experiment_results=args.write_experiment_results, allow_dirty=args.allow_dirty
    )
    import maxim

    try:
        provenance = in_process_code_provenance(
            REPO,
            maxim.__file__,
            out_path=json_path if args.write_experiment_results else None,
            allow_dirty=args.allow_dirty,
        )
    except (ProvenanceError, DirtyTreeError) as exc:
        print(f"[FAIL] provenance preflight: {exc}", file=sys.stderr)
        return 3

    report: dict[str, Any] = {
        "experiment": "grounding_census",
        "stage": "GL1",
        "plan": "docs/plans/grounding.md",
        "scope": "offline_measurement_not_a_behavioural_claim",
        "git_commit": provenance.get("executed_git_hash"),
        "git_dirty": provenance.get("working_tree_dirty_src_scripts"),
        "environment": {k: os.environ[k] for k in STAMPED_ENV if k in os.environ},
    }
    try:
        report.update(measure(args.shuffle_seed))
    except Refusal as exc:
        report["refusal"] = exc.reason
        report["refusal_detail"] = exc.detail
        stamp_diagnosis(report, mock=False, code_provenance=provenance)
        _write(*failed_paths(json_path), report)
        print(f"[REFUSED — {exc.reason}] {exc.detail}", file=sys.stderr)
        return 4
    stamp_diagnosis(report, mock=False, code_provenance=provenance)
    _write(json_path, md_path, report)
    if args.write_experiment_results:
        # A passing record supersedes any earlier refusal beside it: a stale census.failed.* must not outlive it.
        for stale in failed_paths(json_path):
            if stale.exists():
                stale.unlink()
                print(f"[census] removed stale refusal: {stale}")
    p = report["results"]["sorted"]
    print(
        f"[census] {report['counts']['affordances']} affordances, {len(p['collisions'])} same-node collisions "
        f"({p['harm_flip_collisions']} harm flips, {p['opposite_sign_collisions']} opposite-sign); "
        f"known answer {'PASS' if report['known_answer']['ok'] else 'FAIL'}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
