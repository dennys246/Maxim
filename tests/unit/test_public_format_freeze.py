"""PUBLIC FORMAT 1 -- the freeze guard (docs/plans/public_format_freeze.md; public_oasis Phase 0 item 2).

Once a stranger downloads a release, its format is a public compatibility promise. These tests hold the
checked-in fixtures (``tests/fixtures/public_format_1/``, built deterministically by its ``build.py``) to
it in both directions:

- every future build still VERIFIES and INGESTS the published bytes (readers keep their promise), and
- every future build still COMPOSES them byte-for-byte (producers cannot drift the format by accident).

A deliberate format change is allowed -- it is a recorded decision: a freeze-log line in the plan, a
dated amendment to sharing_threat_model.md §5 where a receiver duty changes, and a regenerated fixture.
"""

from __future__ import annotations

import importlib.util
import json
import zipfile
from pathlib import Path

import pytest

from maxim.utils.optional_deps import optional_dependency_available

pytestmark = pytest.mark.skipif(
    not (optional_dependency_available("cryptography") and optional_dependency_available("rfc8785")),
    reason="the public format is signed: the [sign] extra",
)

S = "\x1f"
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "public_format_1"
GOLDEN = json.loads((FIXTURES / "golden.json").read_text())
CHANGE_RULE = (
    "PUBLIC FORMAT 1 changed. If deliberate: record it in docs/plans/public_format_freeze.md §Freeze log, add a "
    "dated sharing_threat_model.md §5 amendment if a receiver duty changed, and regenerate with "
    "`python tests/fixtures/public_format_1/build.py`."
)


def _build():
    # By FILE, under its own name: a plain ``import build`` can resolve to the PyPA ``build`` package
    # (or leave the fixture shadowing it in sys.modules for the rest of the session).
    spec = importlib.util.spec_from_file_location("public_format_1_build", FIXTURES / "build.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _keys():
    return {GOLDEN["signer_identity"]: GOLDEN["public_key"]}


def test_the_published_release_still_verifies_as_published():
    from maxim.hivemind.bundle import verify_bundle_zip

    golden = GOLDEN["release_v2.zip"]
    with zipfile.ZipFile(FIXTURES / "release_v2.zip") as zf:
        result = verify_bundle_zip(zf, trusted_keys=_keys(), accept_v1=False)
    assert result.ok, result.reason
    assert (result.scheme, result.release_sequence, result.license) == (2, 1, "CDLA-Permissive-2.0")
    assert result.payload_digest == golden["payload_identity"], CHANGE_RULE
    assert dict(sorted(result.entry_digests.items())) == golden["entry_digests"], CHANGE_RULE


def test_the_published_bundles_still_ingest(tmp_path):
    from maxim.hivemind.ingest import IngestionJournal, ingest_bundle

    build = _build()
    common = dict(
        receiver_nac=None,
        receiver_ec_nodes=None,
        trusted_sources=frozenset({build.CONTRIBUTOR}),
        receiver_body=build.BODY,
        receiver_agent_id="receiver",
        inherent_trusted_sources=frozenset({build.CONTRIBUTOR}),
    )
    release = ingest_bundle(
        FIXTURES / "release_v2.zip",
        journal=IngestionJournal(tmp_path / "j1.json"),
        require_signed=True,
        trusted_keys=_keys(),
        accept_v1=False,
        **common,
    )
    assert release.verification is not None and release.verification.ok
    unsigned = ingest_bundle(
        FIXTURES / "unsigned_schema2.zip", journal=IngestionJournal(tmp_path / "j2.json"), **common
    )
    key = f"receiver{S}n1{S}"
    for report in (release, unsigned):
        # Every frozen row kind is READ, not merely tolerated. (Merge semantics -- how links fold into a
        # store, #913 -- are deliberately not pinned here: the format is what a reader accepts.)
        assert set(report.ec_nodes) == {"n1"}
        assert report.nac["cluster_reward_bias"] == {key + "tool:flee": pytest.approx(0.4)}
        assert report.nac["cluster_reward_source"] == {key + "tool:flee": "drive_relief"}
        assert report.nac["cluster_fear"] == {key + "drive:oxygen": pytest.approx(-0.5 * 0.75)}  # vicarious
        assert report.nac["inherent_bias_keys"] == [key + "tool:flee"] and report.inherent_keys_admitted == 1
        assert report.biases_dropped == 1  # the node-less "orient" entry has no situation to land on
        assert list(report.nac["links"]) == ["tool:probe"]
    assert release.nac["links"]["tool:probe"][0]["event_context"] == {"agent_id": "receiver"}  # `_agent` re-keyed


@pytest.mark.parametrize("name", ["release_v2.zip", "unsigned_schema2.zip"])
def test_the_producer_still_composes_the_frozen_format(tmp_path, name):
    """Byte-for-byte: every member of a recomposed fixture is identical to the published one."""
    build = _build()
    out = tmp_path / name
    build.compose(out, signed=name == "release_v2.zip")
    assert build.fingerprint(out) == GOLDEN[name], CHANGE_RULE


def test_the_published_files_are_the_ones_golden_describes():
    """The checked-in zips and golden.json agree (neither was edited without the other)."""
    build = _build()
    for name in ("release_v2.zip", "unsigned_schema2.zip"):
        assert build.fingerprint(FIXTURES / name) == GOLDEN[name]


def test_the_frozen_constants():
    """Names and values a stranger's client or bundle depends on. Changing one is a format change."""
    from maxim.hivemind import bundle, entry_index, ingest, merge, signing, store

    frozen = {
        "BUNDLE_KIND": (bundle.BUNDLE_KIND, "substrate_bundle"),
        "BUNDLE_SCHEMA_VERSION": (bundle.BUNDLE_SCHEMA_VERSION, 3),
        "UNSIGNED_BUNDLE_SCHEMA_VERSION": (bundle.UNSIGNED_BUNDLE_SCHEMA_VERSION, 2),
        "SIGNATURE_MEMBER": (signing.SIGNATURE_MEMBER, "signature.json"),
        "SIGNATURE_SCHEME_V2": (signing.SIGNATURE_SCHEME_V2, 2),
        "SIGNATURE_ALGORITHM": (signing.SIGNATURE_ALGORITHM, "ed25519"),
        "BUNDLE_V2_TAG": (signing.BUNDLE_V2_TAG, b"maxim-bundle-v2"),
        "MAX_RELEASE_SEQUENCE": (signing.MAX_RELEASE_SEQUENCE, 2**53 - 1),
        "INDEX_VERSION": (entry_index.INDEX_VERSION, 1),
        "AGENT_TOKEN": (entry_index.AGENT_TOKEN, "_agent"),
        "CLUSTER_FIELDS": (
            entry_index.CLUSTER_FIELDS,
            ("cluster_fear", "cluster_reward_bias", "cluster_reward_source"),
        ),
        "EC_NODE_FIELDS": (entry_index.EC_NODE_FIELDS, ("modality", "embedding", "geometry", "count", "domain")),
        "NAC_KEY_SEP": (merge.NAC_KEY_SEP, "\x1f"),
        "NODE_ID_CHARSET": (merge.NODE_ID_CHARSET.pattern, r"^[A-Za-z0-9_.\-]{1,128}$"),
        "RELEASE_ID": (store._RELEASE_ID_RE.pattern, r"^[0-9a-f]{64}$"),
        # the slice allowlists: what a published slice may carry
        "NAC_FIELDS": (
            sorted(bundle._BUNDLE_NAC_FIELDS),
            sorted(
                "cluster_fear cluster_reward_bias cluster_reward_source event_outcome_welford goal_reward_bias "
                "inherent_bias_keys links outcome_index percept_valences priors reward_bias total_observations "
                "version".split()
            ),
        ),
        "LINK_FIELDS": (
            sorted(bundle._BUNDLE_LINK_FIELDS),
            sorted(
                "confidence context_factors contributors domain event_context event_signature event_type id "
                "imagined last_observed last_rpe memory_ids observation_count outcome_signature outcome_type "
                "outcome_valence percept_refs predicted_value prediction_history source temporal_delta".split()
            ),
        ),
        "EC_NODE_FIELDS_SHIPPED": (
            sorted(bundle._BUNDLE_EC_NODE_FIELDS),
            sorted("contributors count domain embedding geometry member_count modality source".split()),
        ),
        "EVENT_CONTEXT_FIELDS": (sorted(bundle._BUNDLE_EVENT_CONTEXT_ALLOWLIST), ["agent_id"]),
        # the archive limits a reader enforces (sharing_threat_model.md §5)
        "MAX_BUNDLE_ENTRIES": (bundle.MAX_BUNDLE_ENTRIES, 16),
        "MAX_ENTRY_UNCOMPRESSED_BYTES": (bundle.MAX_ENTRY_UNCOMPRESSED_BYTES, 64 * 1024 * 1024),
        "MAX_TOTAL_UNCOMPRESSED_BYTES": (bundle.MAX_TOTAL_UNCOMPRESSED_BYTES, 128 * 1024 * 1024),
        # the §5 adapter bounds on admitted values
        "MAX_NODES_PER_SLICE": (ingest.MAX_NODES_PER_SLICE, 50_000),
        "MAX_FOREIGN_COUNT": (ingest.MAX_FOREIGN_COUNT, 1_000),
        "MAX_FOREIGN_TOTAL_OBSERVATIONS": (ingest.MAX_FOREIGN_TOTAL_OBSERVATIONS, 1_000_000),
        "CAP_FOREIGN_CONFIDENCE": (ingest.CAP_FOREIGN_CONFIDENCE, 0.9),
        "FOREIGN_FEAR_DISCOUNT": (ingest.FOREIGN_FEAR_DISCOUNT, 0.75),
        "MAX_FOREIGN_DELTAS": (ingest.MAX_FOREIGN_DELTAS, 50),
        "MAX_FOREIGN_EMBEDDING_NORM": (ingest.MAX_FOREIGN_EMBEDDING_NORM, 1000.0),
    }
    drifted = {k: actual for k, (actual, expected) in frozen.items() if actual != expected}
    assert drifted == {}, CHANGE_RULE
