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

import pathlib

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
    assert manifest["signer_identity"] is None
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


def test_compose_signer_identity_defaults_null(tmp_path: Path) -> None:
    """CC13: the reserved ``signer_identity`` slot is ``None`` at 1.0 when
    the caller does not populate it.
    """
    output = tmp_path / "bundle.zip"
    manifest = compose_bundle(
        nac_state=_empty_nac_state(),
        ec_substrate_nodes={},
        output_path=output,
        contributor_id="oasis-A",
    )
    assert manifest["signer_identity"] is None


def test_compose_signer_identity_slot_round_trips(tmp_path: Path) -> None:
    """CC13: a caller-populated ``signer_identity`` survives compose →
    extract → read_bundle_manifest unchanged. The composer does NOT
    validate it (no verification at 1.0).
    """
    output = tmp_path / "bundle.zip"
    compose_bundle(
        nac_state=_empty_nac_state(),
        ec_substrate_nodes={},
        output_path=output,
        contributor_id="oasis-A",
        signature="deadbeef",
        signature_algorithm="ed25519",
        signer_identity="did:key:z6Mk-example",
    )
    extracted = extract_bundle(output, tmp_path / "out")
    assert extracted["signer_identity"] == "did:key:z6Mk-example"
    inspected = read_bundle_manifest(output)
    assert inspected["signer_identity"] == "did:key:z6Mk-example"


def test_extract_tolerates_manifest_without_signer_identity(tmp_path: Path) -> None:
    """CC13 backward-compat: a hand-built bundle whose manifest predates the
    ``signer_identity`` field (i.e. omits it entirely) extracts cleanly —
    the field is optional and read via ``.get(...)``.
    """
    output = tmp_path / "legacy.zip"
    manifest = {
        "_format_version": "1.0",
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "kind": BUNDLE_KIND,
        "contributor_id": "oasis-A",
        "domain": None,
        "created_at": "2026-01-01T00:00:00+00:00",
        "identity_filter_applied": True,
        "identity_threshold": 2,
        "contents": {},
        "signature": None,
        "signature_algorithm": None,
        # NOTE: no "signer_identity" key — predates CC13.
    }
    with zipfile.ZipFile(output, "w") as zf:
        zf.writestr("manifest.json", json.dumps(manifest, indent=2, sort_keys=True))
    extracted = extract_bundle(output, tmp_path / "out")
    assert extracted.get("signer_identity") is None
    assert extracted["contributor_id"] == "oasis-A"


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

        # target_version is pinned rather than defaulted: this test is about
        # the SEAM and its isolation, not about the current schema version.
        # It previously relied on BUNDLE_SCHEMA_VERSION being 1, so the gate-7
        # bump to 2 broke it — the chain ran on to 1→2, which the isolated
        # registry deliberately does not contain.
        upgraded = migrate_bundle_envelope(
            {"schema_version": 0, "kind": BUNDLE_KIND},
            target_version=1,
        )
        assert upgraded["schema_version"] == 1
        assert upgraded["migrated"] is True

    # Outside the context: registry is restored (v0 migration is gone).
    with pytest.raises(ValueError, match="no migration"):
        migrate_bundle_envelope({"schema_version": 0, "kind": BUNDLE_KIND}, target_version=1)


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


# ─────────────────────────────────────────────────────────────────────────
# Bundle content scrub — model-generated text must not ship
# ─────────────────────────────────────────────────────────────────────────


class TestBundleScrubsModelGeneratedText:
    """The bundle composer must scrub LLM-generated / raw-tool-output text
    from NAc links before serialization.

    Three leak paths (all real producer paths on current HEAD):

    1. ``event_context["goal"]`` — tool_dispatch.py sets it to the LLM's
       own ``reasoning[:100]``.
    2. ``outcome_signature`` — ``f"{{success|failure}}:{{outcome_summary}}"``
       where outcome_summary is raw tool output / error text (paths,
       hostnames, credentials). Also appears as ``outcome_index`` keys.
    3. ``memory_ids`` — hippocampus episode IDs (episodes stay local by
       the load-bearing privacy invariant; their IDs shouldn't ship
       either).
    """

    def _leaky_nac(self) -> NAc:
        from maxim.decisions.causal_link import Valence

        nac = NAc(config=NACConfig())
        nac.observe(
            event_type="tool",
            event_signature="tool:probe_scan",
            outcome_type="tool_result",
            outcome_signature="failure:SENTINEL_OUTCOME cred=hunter2 /Users/x/id_rsa",
            outcome_valence=Valence.NEGATIVE,
            delta_seconds=1.0,
            context={
                "agent_id": "agent1",
                "goal": "SENTINEL_REASONING_XYZZY plan to open the vault",
            },
            memory_id="SENTINEL_EPISODE_MEM_42",
        )
        return nac

    def test_sentinels_do_not_reach_serialized_nac_json(self, tmp_path: Path) -> None:
        nac = self._leaky_nac()
        output = tmp_path / "bundle.zip"
        compose_bundle(
            nac_state=nac.dump(),
            ec_substrate_nodes=None,
            output_path=output,
            contributor_id="A",
        )

        with zipfile.ZipFile(output) as zf:
            nac_json = zf.read("nac.json").decode("utf-8")

        # 1. LLM reasoning via event_context["goal"]
        assert "SENTINEL_REASONING_XYZZY" not in nac_json
        # 2. raw tool output / credentials via outcome_signature
        assert "SENTINEL_OUTCOME" not in nac_json
        assert "hunter2" not in nac_json
        # 3. hippocampus episode IDs via memory_ids
        assert "SENTINEL_EPISODE_MEM_42" not in nac_json

    def test_link_survives_scrub_with_allowlisted_fields(self, tmp_path: Path) -> None:
        """Scrub, don't drop: the link ships with agent_id kept, the
        outcome_signature replaced by the canonical valence-preserving
        ``{outcome_type}:{valence}`` form, and memory_ids emptied."""
        nac = self._leaky_nac()
        output = tmp_path / "bundle.zip"
        compose_bundle(
            nac_state=nac.dump(),
            ec_substrate_nodes=None,
            output_path=output,
            contributor_id="A",
        )

        with zipfile.ZipFile(output) as zf:
            data = json.loads(zf.read("nac.json").decode("utf-8"))

        assert "tool:probe_scan" in data["links"]
        (link,) = data["links"]["tool:probe_scan"]
        assert link["event_context"] == {"agent_id": "agent1"}
        assert link["outcome_signature"] == "tool_result:negative"
        assert link["memory_ids"] == []

    def test_local_nac_state_is_not_mutated_by_composition(self, tmp_path: Path) -> None:
        """compose_bundle is a pure function: the scrub applies to the
        bundle only, the local dump keeps full debugging context."""
        nac = self._leaky_nac()
        state = nac.dump()
        compose_bundle(
            nac_state=state,
            ec_substrate_nodes=None,
            output_path=tmp_path / "bundle.zip",
            contributor_id="A",
        )
        (link,) = state["links"]["tool:probe_scan"]
        assert link["event_context"]["goal"].startswith("SENTINEL_REASONING_XYZZY")
        assert link["outcome_signature"].startswith("failure:SENTINEL_OUTCOME")
        assert link["memory_ids"] == ["SENTINEL_EPISODE_MEM_42"]

    def test_ast_guard_every_nac_json_assignment_routes_through_scrubber(self) -> None:
        """AST architectural check (CI): no dict may reach
        ``bundle_contents["nac.json"]`` without passing through
        ``scrub_nac_state_for_bundle`` inline in the ``json.dumps`` call.

        Same style as the existing AST checks (test_phase0_fixes.py):
        parse the source, walk, assert structure. A future edit that
        serializes NAc state via a different path fails here loudly.
        """
        import ast

        import maxim.hivemind.bundle as bundle_mod

        source = Path(bundle_mod.__file__).read_text()
        tree = ast.parse(source)

        nac_json_assignments = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if (
                    isinstance(target, ast.Subscript)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "bundle_contents"
                    and isinstance(target.slice, ast.Constant)
                    and target.slice.value == "nac.json"
                ):
                    nac_json_assignments.append(node)

        assert nac_json_assignments, "no bundle_contents['nac.json'] assignment found — guard is stale, update it"

        for assign in nac_json_assignments:
            value = assign.value
            assert isinstance(value, ast.Call), (
                f"line {assign.lineno}: nac.json must be assigned from a json.dumps call"
            )
            func = value.func
            is_json_dumps = (
                isinstance(func, ast.Attribute)
                and func.attr == "dumps"
                and isinstance(func.value, ast.Name)
                and func.value.id == "json"
            )
            assert is_json_dumps, f"line {assign.lineno}: nac.json must be serialized via json.dumps"
            assert value.args, f"line {assign.lineno}: json.dumps must take the payload positionally"
            payload = value.args[0]
            assert (
                isinstance(payload, ast.Call)
                and isinstance(payload.func, ast.Name)
                and payload.func.id == "scrub_nac_state_for_bundle"
            ), (
                f"line {assign.lineno}: the dict serialized into nac.json must be produced by "
                f"scrub_nac_state_for_bundle(...) inline in the json.dumps call — "
                f"model-generated text (LLM reasoning in event_context, raw tool output in "
                f"outcome_signature, hippocampus episode IDs in memory_ids) must not ship"
            )


# ─────────────────────────────────────────────────────────────────────────
# Scrub extensions — 2026-08-12 five-lens privacy audit
# ─────────────────────────────────────────────────────────────────────────


class TestBundleScrubsNacStateSurfaces:
    """The audit's cross-confirmed findings beyond the CausalLink fields:
    goal_reward_bias keys are raw goal text; tool:use:<action> signatures
    embed verbatim LLM tool params across four surfaces; percept_refs are
    the same reference class as memory_ids; priors is an unguarded
    pass-through."""

    def _compose(self, tmp_path: Path, state: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        output = tmp_path / "bundle.zip"
        compose_bundle(
            nac_state=state,
            ec_substrate_nodes=None,
            output_path=output,
            contributor_id="A",
            **kwargs,
        )
        with zipfile.ZipFile(output) as zf:
            return json.loads(zf.read("nac.json").decode("utf-8"))

    def test_goal_reward_bias_dropped(self, tmp_path: Path) -> None:
        state = _empty_nac_state(goal_reward_bias={"help debug the auth tokens in ~/Scripts/.env SENTINEL_GOAL": 0.4})
        data = self._compose(tmp_path, state)
        assert data["goal_reward_bias"] == {}
        assert "SENTINEL_GOAL" not in json.dumps(data)

    def test_priors_dropped(self, tmp_path: Path) -> None:
        state = _empty_nac_state(priors={"tool:SENTINEL_PRIOR": [0.5, 0.3]})
        data = self._compose(tmp_path, state)
        assert data["priors"] == {}

    def test_use_action_free_text_truncated_across_all_surfaces(self, tmp_path: Path) -> None:
        sig = "tool:use:pry open the rusted lock at /Users/x SENTINEL_ACTION"
        link = _link_dict(event_sig=sig)
        link["id"] = "1b6ffc729b3141cc"  # production ids are sha256[:16], not sig-derived
        state = _empty_nac_state(
            links={sig: [link]},
            event_outcome_welford={f"aut\x1f{sig}": {"mean": 0.5, "m2": 0.1, "n": 2.0}},
            cluster_reward_bias={f"aut\x1fcid-1\x1f{sig}": 0.5},
            cluster_reward_source={f"aut\x1fcid-1\x1f{sig}": "causal"},
        )
        data = self._compose(tmp_path, state)
        text = json.dumps(data)
        assert "SENTINEL_ACTION" not in text
        assert "tool:use" in data["links"]
        (link,) = data["links"]["tool:use"]
        assert link["event_signature"] == "tool:use"
        assert "aut\x1ftool:use" in data["event_outcome_welford"]
        assert "aut\x1fcid-1\x1ftool:use" in data["cluster_reward_bias"]
        assert "aut\x1fcid-1\x1ftool:use" in data["cluster_reward_source"]

    def test_use_action_identifier_kept(self, tmp_path: Path) -> None:
        """tool:use:dodge is the documented transfer vocabulary — kept."""
        sig = "tool:use:dodge"
        state = _empty_nac_state(
            links={sig: [_link_dict(event_sig=sig)]},
            cluster_reward_bias={f"aut\x1fcid-1\x1f{sig}": 0.5},
        )
        data = self._compose(tmp_path, state)
        assert sig in data["links"]
        assert f"aut\x1fcid-1\x1f{sig}" in data["cluster_reward_bias"]

    def test_truncation_collisions_merge(self, tmp_path: Path) -> None:
        sig_a = "tool:use:pry the lock open"
        sig_b = "tool:use:smash the window in"
        state = _empty_nac_state(
            links={
                sig_a: [_link_dict(event_sig=sig_a, outcome_sig="ok-a")],
                sig_b: [_link_dict(event_sig=sig_b, outcome_sig="ok-b")],
            },
            event_outcome_welford={
                f"aut\x1f{sig_a}": {"mean": 1.0, "m2": 0.0, "n": 2.0},
                f"aut\x1f{sig_b}": {"mean": 0.0, "m2": 0.0, "n": 2.0},
            },
            cluster_reward_bias={
                f"aut\x1fcid-1\x1f{sig_a}": 0.4,
                f"aut\x1fcid-1\x1f{sig_b}": 0.2,
            },
            cluster_reward_source={
                f"aut\x1fcid-1\x1f{sig_a}": "causal",
                f"aut\x1fcid-1\x1f{sig_b}": "reward_bias",
            },
        )
        data = self._compose(tmp_path, state)
        # links: both collapse to the canonical (event_sig, outcome_sig)
        # pair and FOLD into one merged link — shipping duplicates would
        # let nac_merge's by-outcome pairing clobber all but one on the
        # receiving side (review-round BLOCKING finding).
        (merged_link,) = data["links"]["tool:use"]
        assert merged_link["observation_count"] == 2
        assert merged_link["outcome_signature"] == "tool_result:positive"
        # outcome_index: rebuilt on canonical keys from the shipped links
        assert data["outcome_index"] == {"tool_result:positive": [merged_link["id"]]}
        # welford: parallel merge — n sums, mean averages (equal n)
        merged = data["event_outcome_welford"]["aut\x1ftool:use"]
        assert merged["n"] == 4.0
        assert merged["mean"] == pytest.approx(0.5)
        # bias: mean on collision
        assert data["cluster_reward_bias"]["aut\x1fcid-1\x1ftool:use"] == pytest.approx(0.3)
        # source: disagreement promotes to "mixed" (NAc's own semantics)
        assert data["cluster_reward_source"]["aut\x1fcid-1\x1ftool:use"] == "mixed"

    def test_scrubbed_bundle_survives_nac_merge_without_link_loss(self, tmp_path: Path) -> None:
        """Review-round BLOCKING repro: two failure links with distinct
        summaries under one event signature must not be clobbered when
        the scrubbed bundle is merged on the receiving side. Pre-fold,
        first-token truncation made both outcome signatures 'failure'
        and nac_merge silently discarded one link's observations."""
        link_a = _link_dict(event_sig="tool:probe", outcome_sig="failure:timeout", valence="negative")
        link_a["id"] = "aaaa000011112222"  # production ids are sha256[:16], not sig-derived
        link_b = _link_dict(event_sig="tool:probe", outcome_sig="failure:connection refused", valence="negative")
        link_b["id"] = "bbbb000011112222"
        state = _empty_nac_state(links={"tool:probe": [link_a, link_b]})
        data = self._compose(tmp_path, state)
        merged = nac_merge(data, _empty_nac_state(), left_source="A", right_source="B")
        (link,) = merged["links"]["tool:probe"]
        # Both observations survive the scrub-time fold + the merge.
        assert link["observation_count"] == 2
        assert "timeout" not in json.dumps(merged)

    def test_percept_valences_entity_class_gate(self, tmp_path: Path) -> None:
        """Identifier-shaped entity classes ship (transfer vocabulary);
        LLM-coined multi-word imagined-entity names do not — and the
        gate holds regardless of the identity filter flag."""
        state = _empty_nac_state(
            percept_valences={
                "aut\x1frusty_sword\x1fsharp_edge": -0.4,
                "aut\x1fdave the blacksmith SENTINEL_ENTITY\x1fSHARP": -0.2,
            }
        )
        data = self._compose(tmp_path, state, apply_identity_filter=False)
        assert "aut\x1frusty_sword\x1fsharp_edge" in data["percept_valences"]
        assert "SENTINEL_ENTITY" not in json.dumps(data)

    def test_percept_refs_zeroed(self, tmp_path: Path) -> None:
        link = _link_dict(event_sig="tool:probe")
        link["percept_refs"] = [{"percept_id": "SENTINEL_PERCEPT", "content_hash": "abc"}]
        state = _empty_nac_state(links={"tool:probe": [link]})
        data = self._compose(tmp_path, state)
        (out_link,) = data["links"]["tool:probe"]
        assert out_link["percept_refs"] == []
        assert "SENTINEL_PERCEPT" not in json.dumps(data)

    def test_identity_filter_extends_to_welford_keys(self, tmp_path: Path) -> None:
        """An identity-quarantined event signature must not ship through
        its Welford twin (the links filter alone leaves that gap)."""
        idsig = "met Dave Smithers at the market"
        state = _empty_nac_state(
            links={idsig: [_link_dict(event_sig=idsig)]},
            event_outcome_welford={
                f"aut\x1f{idsig}": {"mean": 0.5, "m2": 0.0, "n": 1.0},
                "aut\x1ftool:probe": {"mean": 0.5, "m2": 0.0, "n": 1.0},
            },
        )
        data = self._compose(tmp_path, state, identity_threshold=1)
        assert "Dave" not in json.dumps(data)
        assert "aut\x1ftool:probe" in data["event_outcome_welford"]
        # opt-out keeps them (trusted-internal backup semantics)
        data_off = self._compose(tmp_path, state, apply_identity_filter=False)
        assert f"aut\x1f{idsig}" in data_off["event_outcome_welford"]


class TestManifestProvenancePathRedaction:
    def _manifest(self, tmp_path: Path, provenance: dict[str, Any] | None) -> dict[str, Any]:
        return compose_bundle(
            nac_state=_empty_nac_state(),
            ec_substrate_nodes=None,
            output_path=tmp_path / "bundle.zip",
            contributor_id="A",
            encoder_provenance=provenance,
        )

    def test_local_model_path_redacted(self, tmp_path: Path) -> None:
        manifest = self._manifest(
            tmp_path,
            {"linguistic": {"model_name": "/Users/denny/models/mpnet-finetuned", "embedding_dim": 384}},
        )
        recorded = manifest["encoder_provenance"]["recorded"]
        assert recorded["linguistic"]["model_name"] == "[REDACTED_PATH]"
        assert recorded["linguistic"]["embedding_dim"] == 384

    def test_hub_model_name_kept_and_none_stays_none(self, tmp_path: Path) -> None:
        manifest = self._manifest(tmp_path, {"linguistic": {"model_name": "paraphrase-mpnet-base-v2"}})
        assert manifest["encoder_provenance"]["recorded"]["linguistic"]["model_name"] == "paraphrase-mpnet-base-v2"
        assert self._manifest(tmp_path, None)["encoder_provenance"]["recorded"] is None


class TestEcSliceBoundary:
    """The EC payload's text-bearing keys (signatures carry goal_keywords
    = the first words of LLM reasoning) must never ship. The slice
    boundary is load-bearing and was previously unpinned."""

    def test_ec_json_ships_only_substrate_nodes(self, tmp_path: Path) -> None:
        output = tmp_path / "bundle.zip"
        compose_bundle(
            nac_state=None,
            ec_substrate_nodes={"n1": _ec_node([1.0, 0.0])},
            output_path=output,
            contributor_id="A",
        )
        with zipfile.ZipFile(output) as zf:
            data = json.loads(zf.read("ec.json").decode("utf-8"))
        assert set(data.keys()) == {"substrate_nodes"}

    def test_session_export_never_ships_ec_signatures(self, tmp_path: Path) -> None:
        """CLI export reads only the substrate_nodes + encoder_provenance
        slices of aut_ec.json — a signatures key carrying reasoning
        fragments must not reach the bundle."""
        from maxim.hivemind.cli import run_substrate_subcommand

        session_dir = tmp_path / "session"
        session_dir.mkdir()
        (session_dir / "aut_nac.json").write_text(json.dumps(_empty_nac_state()))
        (session_dir / "aut_ec.json").write_text(
            json.dumps(
                {
                    "substrate_nodes": {"n1": _ec_node([1.0, 0.0])},
                    "signatures": {"sig-1": {"goal_keywords": ["SENTINEL_REASONING_WORD"]}},
                    "lsh": {"planes": "SENTINEL_LSH"},
                    "inverted": {"SENTINEL_INVERTED": ["n1"]},
                }
            )
        )
        bundle_path = tmp_path / "out.zip"
        rc = run_substrate_subcommand(
            ["export", str(bundle_path), "--session", str(session_dir), "--contributor-id", "smoke"]
        )
        assert rc == 0
        with zipfile.ZipFile(bundle_path) as zf:
            all_content = b"".join(zf.read(n) for n in zf.namelist()).decode("utf-8")
        assert "SENTINEL_REASONING_WORD" not in all_content
        assert "SENTINEL_LSH" not in all_content
        assert "SENTINEL_INVERTED" not in all_content


class TestGate7TypedBundles:
    """Gate 7 — a bundle declares the body it was learned on, and a receiver refuses a mismatch.

    Why this exists: tool signatures are entity-prefixed
    (`tool_bridge.py::generate_tools_for_entity` builds `f"{ent.name}_{aff}"`),
    so a bundle learned on one body merges into another "successfully",
    contributes exactly 0.0, and reads out as "this agent has learned nothing
    yet" — D43 barrier 3. Gate 7 does not make cross-body sharing WORK; it
    makes its absence LOUD, which is this codebase's rule for a silent no-op.

    The design note is `docs/plans/d43_merge_correctness.md` §5a, including why
    the capability key is emitted alongside from day one.
    """

    def test_matching_body_is_accepted(self):
        from maxim.hivemind.bundle import assert_bundle_body_compatible

        assert_bundle_body_compatible({"body_ref": "reachy_mini"}, receiver_body="reachy_mini")

    def test_mismatched_body_is_refused(self):
        from maxim.hivemind.bundle import BundleBodyMismatch, assert_bundle_body_compatible

        with pytest.raises(BundleBodyMismatch) as e:
            assert_bundle_body_compatible({"body_ref": "infant_operant"}, receiver_body="reachy_mini")
        assert e.value.bundle_body == "infant_operant"
        assert e.value.receiver_body == "reachy_mini"
        # the message must say WHY, not just that
        assert "0.0" in str(e.value)

    def test_absence_is_unverifiable_not_compatible(self):
        """The load-bearing case: a pre-gate-7 bundle must not PASS a check it
        was never subject to. Same reasoning as the format-version "0.x" sentinel."""
        from maxim.hivemind.bundle import BundleBodyUnverifiable, assert_bundle_body_compatible

        with pytest.raises(BundleBodyUnverifiable):
            assert_bundle_body_compatible({"body_ref": None}, receiver_body="reachy_mini")
        with pytest.raises(BundleBodyUnverifiable):
            assert_bundle_body_compatible({}, receiver_body="reachy_mini")

    def test_unverified_can_be_accepted_explicitly(self):
        from maxim.hivemind.bundle import assert_bundle_body_compatible

        assert_bundle_body_compatible({"body_ref": None}, receiver_body="reachy_mini", allow_unverified=True)

    def test_v1_bundles_migrate_and_then_refuse(self):
        """End to end: a v1 manifest upgrades to v2, is stamped with an UNKNOWN
        body, and is then refused by default rather than silently accepted."""
        from maxim.hivemind.bundle import (
            BUNDLE_SCHEMA_VERSION,
            BundleBodyUnverifiable,
            assert_bundle_body_compatible,
            migrate_bundle_envelope,
        )

        migrated = migrate_bundle_envelope({"schema_version": 1, "kind": "maxim.substrate.bundle"})
        assert migrated["schema_version"] == BUNDLE_SCHEMA_VERSION == 2
        assert migrated["body_ref"] is None
        assert migrated["affordance_namespace"] is None
        assert migrated["capability_map"] == {}
        with pytest.raises(BundleBodyUnverifiable):
            assert_bundle_body_compatible(migrated, receiver_body="reachy_mini")

    def test_compose_writes_the_gate7_fields_and_the_capability_map(self, tmp_path):
        """The capability map is forward insurance: bundles carry BOTH the
        body-prefixed signature and the body-agnostic `(modulator, affordance)`
        key, so adopting a capability namespace later is a reader-side change
        with no migration — the half `register_bundle_migration` cannot cover."""
        from maxim.hivemind.bundle import compose_bundle, read_bundle_manifest

        out = tmp_path / "b.zip"
        compose_bundle(
            nac_state={"links": {}, "version": "1.0"},
            ec_substrate_nodes={},
            output_path=out,
            contributor_id="c1",
            body_ref="reachy_mini",
            affordance_namespace="reachy_mini.v1",
            capability_map={"tool:reachy_mini_turn_left": "orient/turn_left"},
        )
        m = read_bundle_manifest(out)
        assert m["schema_version"] == 2
        assert m["body_ref"] == "reachy_mini"
        assert m["affordance_namespace"] == "reachy_mini.v1"
        assert m["capability_map"]["tool:reachy_mini_turn_left"] == "orient/turn_left"

    def test_compose_defaults_are_none_not_a_guess(self):
        """A caller that does not declare a body must not have one inferred."""
        import tempfile

        from maxim.hivemind.bundle import compose_bundle, read_bundle_manifest

        with tempfile.TemporaryDirectory() as d:
            out = pathlib.Path(d) / "b.zip"
            compose_bundle(
                nac_state={"links": {}, "version": "1.0"},
                ec_substrate_nodes={},
                output_path=out,
                contributor_id="c1",
            )
            m = read_bundle_manifest(out)
            assert m["body_ref"] is None
            assert m["capability_map"] == {}
