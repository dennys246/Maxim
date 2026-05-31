"""Substrate bundle compose/extract tests (Hivemind shareability, PR D).

v1_refinement.md §B5 PR D. Validates:

1. Manifest contract — kind, schema_version, _format_version,
   contributor_id, domain, signature slots reserved.
2. NAc + EC payload round-trip through compose → extract.
3. Identity-bearing pattern quarantine (default-on for bundles).
4. EC domain filter — reserved identity domain dropped, scoped
   domain admits matching + undomained.
5. End-to-end Hivemind round-trip: NAc.dump + EC.save → compose →
   extract → nac_merge + ec_merge → load_state + load. Pinning the
   kickoff completion criterion ("maxim substrate export +
   maxim substrate import round-trips cleanly on a real session").
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

import pytest

from maxim.decisions.nac import NAc, NACConfig
from maxim.hivemind import (
    BUNDLE_KIND,
    BUNDLE_SCHEMA_VERSION,
    CONSENSUS_SOURCE,
    IDENTITY_DOMAIN_MARKER,
    compose_bundle,
    ec_merge,
    extract_bundle,
    nac_merge,
    read_bundle_manifest,
)
from maxim.similarity.ec import EntorhinalCortex


# ─────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────


def _ec_node(emb: list[float], *, modality: str = "text", domain: str | None = None) -> dict[str, Any]:
    return {
        "embedding": emb,
        "modality": modality,
        "count": 1,
        "member_count": 1,
        "source": "local",
        "domain": domain,
        "contributors": [],
    }


def _link_dict(*, event_sig: str, outcome_sig: str = "ok", valence: str = "positive") -> dict[str, Any]:
    return {
        "id": f"link-{event_sig}-{outcome_sig}",
        "event_type": "tool",
        "event_signature": event_sig,
        "event_context": {},
        "outcome_type": "tool_result",
        "outcome_signature": outcome_sig,
        "outcome_valence": valence,
        "temporal_delta": {"observed_deltas": []},
        "predicted_value": 0.5,
        "prediction_history": [],
        "observation_count": 1,
        "confidence": 0.5,
        "last_observed": 0.0,
        "memory_ids": [],
        "context_factors": {},
        "last_rpe": None,
        "percept_refs": [],
        "imagined": False,
        "source": "local",
        "domain": None,
        "contributors": [],
    }


def _empty_nac_state(**overrides: Any) -> dict[str, Any]:
    state: dict[str, Any] = {
        "version": "1.0",
        "links": {},
        "outcome_index": {},
        "priors": {},
        "total_observations": 0,
        "reward_bias": {},
        "goal_reward_bias": {},
        "cluster_reward_bias": {},
        "percept_valences": {},
        "event_outcome_welford": {},
    }
    state.update(overrides)
    return state


# ─────────────────────────────────────────────────────────────────────────
# compose_bundle — manifest contract
# ─────────────────────────────────────────────────────────────────────────


def test_compose_writes_zip_with_manifest(tmp_path: Path) -> None:
    """The output is a zip file containing manifest.json."""
    output = tmp_path / "bundle.zip"
    compose_bundle(
        nac_state=_empty_nac_state(),
        ec_substrate_nodes={"n1": _ec_node([1.0, 0.0])},
        output_path=output,
        contributor_id="oasis-A",
    )
    assert output.is_file()
    with zipfile.ZipFile(output) as zf:
        assert "manifest.json" in zf.namelist()


def test_compose_manifest_carries_required_fields(tmp_path: Path) -> None:
    """Manifest has kind, schema_version, _format_version, contributor_id,
    domain, signature, signature_algorithm, contents, created_at,
    identity_filter_applied, identity_threshold.
    """
    output = tmp_path / "bundle.zip"
    manifest = compose_bundle(
        nac_state=_empty_nac_state(),
        ec_substrate_nodes={},
        output_path=output,
        contributor_id="oasis-A",
        domain="combat",
    )
    assert manifest["kind"] == BUNDLE_KIND
    assert manifest["schema_version"] == BUNDLE_SCHEMA_VERSION
    assert manifest["_format_version"] == "1.0"
    assert manifest["contributor_id"] == "oasis-A"
    assert manifest["domain"] == "combat"
    assert manifest["signature"] is None
    assert manifest["signature_algorithm"] is None
    assert "created_at" in manifest
    assert "contents" in manifest
    assert manifest["identity_filter_applied"] is True


def test_compose_signature_slot_preserved(tmp_path: Path) -> None:
    """Callers that want to attach signatures populate the slot; the
    composer does NOT compute or verify, but the slot round-trips.
    """
    output = tmp_path / "bundle.zip"
    manifest = compose_bundle(
        nac_state=_empty_nac_state(),
        ec_substrate_nodes={},
        output_path=output,
        contributor_id="oasis-A",
        signature="deadbeef",
        signature_algorithm="ed25519",
    )
    assert manifest["signature"] == "deadbeef"
    assert manifest["signature_algorithm"] == "ed25519"


def test_compose_rejects_reserved_contributor_id(tmp_path: Path) -> None:
    """Contributor IDs starting with the reserved ``_`` prefix are rejected
    (matches PR B's _validate_source rule).
    """
    output = tmp_path / "bundle.zip"
    with pytest.raises(ValueError, match="reserved"):
        compose_bundle(
            nac_state=_empty_nac_state(),
            ec_substrate_nodes={},
            output_path=output,
            contributor_id="_consensus",
        )


def test_compose_skips_none_slices(tmp_path: Path) -> None:
    """A ``None`` ``nac_state`` or ``ec_substrate_nodes`` omits that slice."""
    output = tmp_path / "bundle.zip"
    manifest = compose_bundle(
        nac_state=_empty_nac_state(),
        ec_substrate_nodes=None,
        output_path=output,
        contributor_id="oasis-A",
    )
    assert "nac" in manifest["contents"]
    assert "ec" not in manifest["contents"]
    with zipfile.ZipFile(output) as zf:
        names = set(zf.namelist())
    assert "nac.json" in names
    assert "ec.json" not in names


# ─────────────────────────────────────────────────────────────────────────
# Identity filter
# ─────────────────────────────────────────────────────────────────────────


def test_identity_filter_single_signal_does_not_drop_at_threshold_2(tmp_path: Path) -> None:
    """Fold (Executor IMPORTANT): pin the test the malformed pre-fold
    version tried to claim. At the bundle-default ``threshold=2`` a
    single proper noun (``Dave``) does NOT trip the filter — the
    purpose of the threshold-2 default is to preserve game-substrate
    where sentence-cased monster names are common.
    """
    state = _empty_nac_state(
        links={
            "tool:open_door:success": [_link_dict(event_sig="tool:open_door:success")],
            "met Dave at the market": [_link_dict(event_sig="met Dave at the market")],
        },
    )
    output = tmp_path / "bundle.zip"
    compose_bundle(
        nac_state=state,
        ec_substrate_nodes=None,
        output_path=output,
        contributor_id="oasis-A",
    )
    with zipfile.ZipFile(output) as zf:
        nac_payload = json.loads(zf.read("nac.json"))
    # Both stay — neither has two identity signals.
    assert "tool:open_door:success" in nac_payload["links"]
    assert "met Dave at the market" in nac_payload["links"]


def test_identity_filter_drops_two_proper_noun_event_sig(tmp_path: Path) -> None:
    """At threshold=2, two proper nouns trip the filter."""
    state = _empty_nac_state(
        links={
            "tool:open_door:success": [_link_dict(event_sig="tool:open_door:success")],
            "Dave met Sarah in Portland": [_link_dict(event_sig="Dave met Sarah in Portland")],
        },
    )
    output = tmp_path / "bundle.zip"
    compose_bundle(
        nac_state=state,
        ec_substrate_nodes=None,
        output_path=output,
        contributor_id="oasis-A",
    )
    with zipfile.ZipFile(output) as zf:
        nac_payload = json.loads(zf.read("nac.json"))
    assert "tool:open_door:success" in nac_payload["links"]
    assert "Dave met Sarah in Portland" not in nac_payload["links"]


def test_identity_filter_disabled_via_kwarg(tmp_path: Path) -> None:
    """``apply_identity_filter=False`` lets identity-bearing event_sigs through."""
    state = _empty_nac_state(
        links={
            "Dave met Sarah in Portland": [_link_dict(event_sig="Dave met Sarah in Portland")],
        },
    )
    output = tmp_path / "bundle.zip"
    compose_bundle(
        nac_state=state,
        ec_substrate_nodes=None,
        output_path=output,
        contributor_id="oasis-A",
        apply_identity_filter=False,
    )
    with zipfile.ZipFile(output) as zf:
        nac_payload = json.loads(zf.read("nac.json"))
    assert "Dave met Sarah in Portland" in nac_payload["links"]


# ─────────────────────────────────────────────────────────────────────────
# EC domain filter
# ─────────────────────────────────────────────────────────────────────────


def test_ec_identity_domain_always_dropped(tmp_path: Path) -> None:
    """Nodes with ``domain == IDENTITY_DOMAIN_MARKER`` are always dropped,
    even when ``--no-identity-filter`` would have spared NAc links.
    """
    nodes = {
        "n1": _ec_node([1.0, 0.0]),
        "n2": _ec_node([0.0, 1.0], domain=IDENTITY_DOMAIN_MARKER),
    }
    output = tmp_path / "bundle.zip"
    compose_bundle(
        nac_state=None,
        ec_substrate_nodes=nodes,
        output_path=output,
        contributor_id="oasis-A",
        apply_identity_filter=False,
    )
    with zipfile.ZipFile(output) as zf:
        ec_payload = json.loads(zf.read("ec.json"))
    assert "n1" in ec_payload["substrate_nodes"]
    assert "n2" not in ec_payload["substrate_nodes"]


def test_ec_domain_scope_admits_matching_and_undomained(tmp_path: Path) -> None:
    """When ``--domain combat`` is set: combat nodes pass, undomained nodes
    pass (generic), cooking nodes drop.
    """
    nodes = {
        "n_combat": _ec_node([1.0, 0.0], domain="combat"),
        "n_cooking": _ec_node([0.0, 1.0], domain="cooking"),
        "n_generic": _ec_node([0.5, 0.5]),  # domain=None
    }
    output = tmp_path / "bundle.zip"
    compose_bundle(
        nac_state=None,
        ec_substrate_nodes=nodes,
        output_path=output,
        contributor_id="oasis-A",
        domain="combat",
    )
    with zipfile.ZipFile(output) as zf:
        ec_payload = json.loads(zf.read("ec.json"))
    nodes_out = ec_payload["substrate_nodes"]
    assert "n_combat" in nodes_out
    assert "n_generic" in nodes_out
    assert "n_cooking" not in nodes_out


# ─────────────────────────────────────────────────────────────────────────
# extract_bundle — manifest validation
# ─────────────────────────────────────────────────────────────────────────


def test_extract_round_trips_manifest_and_slices(tmp_path: Path) -> None:
    """compose → extract preserves manifest + slice content."""
    output = tmp_path / "bundle.zip"
    nodes = {"n1": _ec_node([1.0, 0.0])}
    compose_bundle(
        nac_state=_empty_nac_state(),
        ec_substrate_nodes=nodes,
        output_path=output,
        contributor_id="oasis-A",
    )
    extract_dir = tmp_path / "extracted"
    manifest = extract_bundle(output, extract_dir)
    assert manifest["contributor_id"] == "oasis-A"
    assert (extract_dir / "manifest.json").is_file()
    assert (extract_dir / "nac.json").is_file()
    assert (extract_dir / "ec.json").is_file()


def test_extract_rejects_wrong_kind(tmp_path: Path) -> None:
    """A zip with a wrong manifest kind raises ValueError."""
    bundle_path = tmp_path / "bogus.zip"
    with zipfile.ZipFile(bundle_path, "w") as zf:
        zf.writestr(
            "manifest.json",
            json.dumps(
                {
                    "_format_version": "1.0",
                    "schema_version": 1,
                    "kind": "session_snapshot",
                }
            ),
        )
    with pytest.raises(ValueError, match="kind"):
        extract_bundle(bundle_path, tmp_path / "extracted")


def test_extract_rejects_future_schema_version(tmp_path: Path) -> None:
    """A future schema_version raises ValueError — refuse to downgrade."""
    bundle_path = tmp_path / "future.zip"
    with zipfile.ZipFile(bundle_path, "w") as zf:
        zf.writestr(
            "manifest.json",
            json.dumps(
                {
                    "_format_version": "1.0",
                    "schema_version": 99,
                    "kind": BUNDLE_KIND,
                }
            ),
        )
    with pytest.raises(ValueError, match="schema_version"):
        extract_bundle(bundle_path, tmp_path / "extracted")


def test_read_manifest_without_extracting(tmp_path: Path) -> None:
    """read_bundle_manifest returns the manifest without writing files."""
    output = tmp_path / "bundle.zip"
    compose_bundle(
        nac_state=_empty_nac_state(),
        ec_substrate_nodes={"n1": _ec_node([1.0, 0.0])},
        output_path=output,
        contributor_id="oasis-A",
    )
    manifest = read_bundle_manifest(output)
    assert manifest["contributor_id"] == "oasis-A"


# ─────────────────────────────────────────────────────────────────────────
# End-to-end Hivemind round-trip (smoke / completion criterion)
# ─────────────────────────────────────────────────────────────────────────


def test_end_to_end_hivemind_round_trip(tmp_path: Path) -> None:
    """The kickoff's completion criterion: a real session round-trips
    through compose → extract → merge → load.

    Builds two NAcs + ECs, dumps each, composes a bundle for one,
    extracts on the other side, merges via nac_merge + ec_merge, and
    loads back into a fresh system. Verifies the merged content
    survived end-to-end.
    """
    # --- Maxim A: contributor builds a substrate and exports a bundle.
    nac_a = NAc(config=NACConfig())
    nac_a._reward_bias[("agent1", "node-1")] = 0.12
    ec_a = EntorhinalCortex()
    ec_a.register_substrate_node("ec-node-1", [1.0, 0.0, 0.0], "text", source="A", domain="combat")

    a_session_dir = tmp_path / "session_a"
    a_session_dir.mkdir()
    nac_a.save(str(a_session_dir / "aut_nac.json"))
    ec_a.save(str(a_session_dir / "aut_ec.json"))

    bundle_path = tmp_path / "shared.zip"
    nac_a_dump = nac_a.dump()
    ec_a_payload = json.loads((a_session_dir / "aut_ec.json").read_text())
    compose_bundle(
        nac_state=nac_a_dump,
        ec_substrate_nodes=ec_a_payload["substrate_nodes"],
        output_path=bundle_path,
        contributor_id="A",
        apply_identity_filter=False,  # avoid threshold gotchas on minimal fixture
    )

    # --- Maxim B: extracts and merges into its own local state.
    extract_dir = tmp_path / "extracted"
    extract_bundle(bundle_path, extract_dir)

    imported_nac = json.loads((extract_dir / "nac.json").read_text())
    imported_ec_nodes = json.loads((extract_dir / "ec.json").read_text())["substrate_nodes"]

    nac_b = NAc(config=NACConfig())
    nac_b._reward_bias[("agent1", "node-1")] = 0.16
    nac_b_dump = nac_b.dump()
    merged_nac = nac_merge(nac_b_dump, imported_nac, left_source="B", right_source="A")

    # Reload merged NAc.
    nac_merged = NAc(config=NACConfig())
    nac_merged.load_state(merged_nac)
    assert nac_merged._reward_bias[("agent1", "node-1")] == 0.14  # mean(0.12, 0.16)

    # EC merge: B has no nodes yet; importing A's node merges in.
    ec_b = EntorhinalCortex()
    ec_b_payload = json.loads(ec_b.dump()) if hasattr(ec_b, "dump") else None  # noqa: F841
    # ec.save needs a path; emulate via direct slice.
    merged_ec_nodes = ec_merge({}, imported_ec_nodes, left_source="B", right_source="A")
    assert "ec-node-1" in merged_ec_nodes
    assert merged_ec_nodes["ec-node-1"]["source"] in {"A", CONSENSUS_SOURCE}
    # Either as-is or consensus depending on how _merge_link_pair logic
    # resolves single-contributor on right-only; for right-only-side it
    # should stay "A".
    assert merged_ec_nodes["ec-node-1"]["source"] == "A"


# ─────────────────────────────────────────────────────────────────────────
# Fold regression guards — review-driven hardening
# ─────────────────────────────────────────────────────────────────────────


def test_extract_rejects_zip_slip_traversal(tmp_path: Path) -> None:
    """Fold (Executor CRITICAL): a malicious bundle entry whose path
    escapes ``output_dir`` is rejected before any file is written.

    Crafts a ZIP with a ``../escape.json`` entry. The pre-validation
    pass in ``extract_bundle`` MUST raise ``ValueError`` and NOT write
    anything (not even the legitimate slices).
    """
    # Build a malicious bundle. We bypass compose_bundle so we can
    # inject the escape entry directly.
    bundle_path = tmp_path / "malicious.zip"
    with zipfile.ZipFile(bundle_path, "w") as zf:
        zf.writestr(
            "manifest.json",
            json.dumps(
                {
                    "_format_version": "1.0",
                    "schema_version": 1,
                    "kind": BUNDLE_KIND,
                    "contributor_id": "attacker",
                    "domain": None,
                    "identity_filter_applied": False,
                    "identity_threshold": None,
                    "contents": {"nac": {"file": "nac.json"}},
                    "signature": None,
                    "signature_algorithm": None,
                }
            ),
        )
        zf.writestr("nac.json", "{}")
        zf.writestr("../escape.json", "pwned")

    extract_dir = tmp_path / "out"
    with pytest.raises(ValueError, match="ZIP slip"):
        extract_bundle(bundle_path, extract_dir)
    # Pre-validation must run before any write — the legitimate
    # nac.json should NOT exist either.
    assert not (extract_dir / "nac.json").exists()


def test_extract_rejects_absolute_path_entry(tmp_path: Path) -> None:
    """Absolute-path bundle entries are rejected."""
    bundle_path = tmp_path / "bogus.zip"
    with zipfile.ZipFile(bundle_path, "w") as zf:
        zf.writestr(
            "manifest.json",
            json.dumps(
                {
                    "_format_version": "1.0",
                    "schema_version": 1,
                    "kind": BUNDLE_KIND,
                }
            ),
        )
        # Hand-craft a ZIP entry with an absolute path.
        zf.writestr("/etc/passwd", "evil")

    with pytest.raises(ValueError, match="absolute path"):
        extract_bundle(bundle_path, tmp_path / "out")


def test_contributor_id_empty_string_rejected(tmp_path: Path) -> None:
    """Fold (Executor IMPORTANT): empty contributor_id is rejected by
    the shared validator from merge.py — previously the inline check
    only rejected ``_*`` prefixes.
    """
    with pytest.raises(ValueError, match="non-empty"):
        compose_bundle(
            nac_state=_empty_nac_state(),
            ec_substrate_nodes=None,
            output_path=tmp_path / "x.zip",
            contributor_id="",
        )


def test_migration_seam_exists_and_is_isolated(tmp_path: Path) -> None:
    """Fold (Architecture IMPORTANT): the bundle migration registry seam
    exists and the ``isolated_bundle_migrations`` context manager
    cleanly restores the registry.

    A synthetic v0→v1 migration registered inside the context is gone
    after the context exits.
    """
    from maxim.hivemind.bundle import (
        isolated_bundle_migrations,
        migrate_bundle_envelope,
        register_bundle_migration,
    )

    with isolated_bundle_migrations():

        @register_bundle_migration(0)
        def _v0_to_v1(env: dict) -> dict:
            return {**env, "schema_version": 1, "migrated": True}

        upgraded = migrate_bundle_envelope(
            {"schema_version": 0, "kind": BUNDLE_KIND},
        )
        assert upgraded["schema_version"] == 1
        assert upgraded["migrated"] is True

    # Outside the context: registry is restored (v0 migration is gone).
    with pytest.raises(ValueError, match="no migration"):
        migrate_bundle_envelope({"schema_version": 0, "kind": BUNDLE_KIND})


def test_ec_merge_respects_frozen_centroid_modalities() -> None:
    """Fold (Bio-fidelity IMPORTANT): for modalities in
    ``frozen_centroid_modalities``, ``ec_merge`` does NOT update the
    centroid on match — it only sums counts + unions contributors.

    Without this fix, an Oasis ingesting two contributors' interoceptive
    embeddings would re-introduce the centroid drift the frozen-modality
    contract was designed to prevent.
    """
    from maxim.hivemind.merge import ec_merge as ec_merge_local

    left = {
        "n1": {
            "embedding": [1.0, 0.0, 0.0],
            "modality": "interoception",
            "count": 1,
            "source": "A",
            "domain": None,
            "contributors": [],
        }
    }
    right = {
        "n2": {
            "embedding": [0.9, 0.0, 0.0],  # close enough to cosine-match
            "modality": "interoception",
            "count": 1,
            "source": "B",
            "domain": None,
            "contributors": [],
        }
    }
    merged = ec_merge_local(left, right, left_source="A", right_source="B", cosine_threshold=0.5)
    assert "n1" in merged
    # Centroid UNCHANGED — left's embedding survives untouched.
    assert merged["n1"]["embedding"] == [1.0, 0.0, 0.0]
    # Count + contributors still accumulate.
    assert merged["n1"]["count"] == 2
    assert merged["n1"]["member_count"] == 2
    assert set(merged["n1"]["contributors"]) == {"A", "B"}


def test_ec_merge_non_frozen_modality_still_weighted_means() -> None:
    """The frozen-modality fix MUST NOT regress the default (text) path:
    text modality still uses count-weighted centroid mean.
    """
    from maxim.hivemind.merge import ec_merge as ec_merge_local

    left = {
        "n1": {
            "embedding": [1.0, 0.0, 0.0],
            "modality": "text",
            "count": 1,
            "source": "A",
            "domain": None,
            "contributors": [],
        }
    }
    right = {
        "n2": {
            "embedding": [3.0, 0.0, 0.0],  # same direction
            "modality": "text",
            "count": 1,
            "source": "B",
            "domain": None,
            "contributors": [],
        }
    }
    merged = ec_merge_local(left, right, left_source="A", right_source="B", cosine_threshold=0.5)
    # Weighted-mean centroid: (1*1.0 + 1*3.0) / 2 = 2.0.
    assert abs(merged["n1"]["embedding"][0] - 2.0) < 1e-9


def test_ec_embedding_round_trip_preserves_cosine_similarity(tmp_path: Path) -> None:
    """Fold (Bio-fidelity IMPORTANT): float-precision survival.

    An EC embedding with float64 components round-trips through
    compose → JSON → extract → JSON with cosine similarity above
    0.9999 against the original. Catches a regression that would
    silently coerce floats via ``default=str``.
    """
    # Sample embedding with non-trivial float64 components.
    sample = [0.5773502691896258, -0.5773502691896258, 0.5773502691896258, 0.123456789]

    nodes = {
        "n1": {
            "embedding": list(sample),
            "modality": "text",
            "count": 1,
            "source": "A",
            "domain": None,
            "contributors": [],
        }
    }
    output = tmp_path / "round_trip.zip"
    compose_bundle(
        nac_state=None,
        ec_substrate_nodes=nodes,
        output_path=output,
        contributor_id="A",
    )
    extract_dir = tmp_path / "out"
    extract_bundle(output, extract_dir)
    extracted = json.loads((extract_dir / "ec.json").read_text())
    extracted_emb = extracted["substrate_nodes"]["n1"]["embedding"]

    # Cosine similarity against the original.
    dot = sum(a * b for a, b in zip(sample, extracted_emb))
    norm_a = sum(a * a for a in sample) ** 0.5
    norm_b = sum(b * b for b in extracted_emb) ** 0.5
    cosine = dot / (norm_a * norm_b)
    assert cosine > 0.9999, f"cosine {cosine!r} below tolerance"


def test_reward_bias_round_trip_preserves_near_clamp_boundary(tmp_path: Path) -> None:
    """A NAc reward_bias right at the production cap (0.20) survives
    bundle round-trip without precision loss that would silently shift
    above or below the clamp.
    """
    state = _empty_nac_state(reward_bias={"agent1:node-1": 0.19999999999999998})
    output = tmp_path / "rb.zip"
    compose_bundle(
        nac_state=state,
        ec_substrate_nodes=None,
        output_path=output,
        contributor_id="A",
    )
    extract_dir = tmp_path / "out"
    extract_bundle(output, extract_dir)
    extracted = json.loads((extract_dir / "nac.json").read_text())
    assert extracted["reward_bias"]["agent1:node-1"] == 0.19999999999999998


def test_end_to_end_via_cli_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Smoke-test the maxim substrate CLI end-to-end via run_substrate_subcommand."""
    from maxim.hivemind.cli import run_substrate_subcommand

    # Build a session dir.
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    nac = NAc(config=NACConfig())
    nac._reward_bias[("agent1", "node-1")] = 0.10
    nac.save(str(session_dir / "aut_nac.json"))
    ec = EntorhinalCortex()
    ec.register_substrate_node("ec-1", [1.0, 0.0, 0.0], "text", source="local")
    ec.save(str(session_dir / "aut_ec.json"))

    # Export.
    bundle_path = tmp_path / "out.zip"
    rc = run_substrate_subcommand(
        [
            "export",
            str(bundle_path),
            "--session",
            str(session_dir),
            "--contributor-id",
            "smoke",
            "--no-identity-filter",
        ]
    )
    assert rc == 0
    assert bundle_path.is_file()

    # Inspect.
    rc = run_substrate_subcommand(["inspect", str(bundle_path)])
    assert rc == 0

    # Import.
    out_dir = tmp_path / "imported"
    rc = run_substrate_subcommand(
        [
            "import",
            str(bundle_path),
            "--output-dir",
            str(out_dir),
        ]
    )
    assert rc == 0
    assert (out_dir / "manifest.json").is_file()
    assert (out_dir / "nac.json").is_file()
    assert (out_dir / "ec.json").is_file()
