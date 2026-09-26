"""Build the PUBLIC FORMAT 1 fixtures (docs/plans/public_format_freeze.md).

Deterministic: a fixed test key (NOT a real Queen key -- the seed is public, below), a fixed ``created_at``
and a literal learned state. ``tests/unit/test_public_format_freeze.py`` checks two things against the
files this writes:

- every future build VERIFIES and INGESTS the checked-in bundles (a stranger's downloaded release keeps
  working), and
- every future build still COMPOSES them identically (the signed-payload and entry digests in
  ``golden.json``) -- so a format change is a decision recorded in the freeze log, never a side effect.

Regenerate ONLY for a deliberate, recorded format change:  ``python tests/fixtures/public_format_1/build.py``
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
S = "\x1f"
SEED = bytes(range(32))  # public, test-only
SIGNER_IDENTITY = "fixture-queen"
CONTRIBUTOR = "fixture-oasis"
BODY = "fixture_body"
CREATED_AT = "2026-09-26T00:00:00+00:00"
LICENSE = "CDLA-Permissive-2.0"


def signer() -> Any:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    from maxim.hivemind.signing import BundleSigner

    return BundleSigner(Ed25519PrivateKey.from_private_bytes(SEED), signer_identity=SIGNER_IDENTITY)


def nac_state() -> dict[str, Any]:
    link = {
        "id": "l1",
        "event_type": "tool_execution",
        "event_signature": "tool:probe",
        "event_context": {"agent_id": "aut"},
        "outcome_type": "tool_result",
        "outcome_signature": "tool_result:positive",
        "outcome_valence": "positive",
        "temporal_delta": {"observed_deltas": [1.0]},
        "predicted_value": 0.5,
        "prediction_history": [],
        "observation_count": 3,
        "confidence": 0.5,
        "last_observed": 100.0,
        "memory_ids": [],
        "context_factors": {},
        "last_rpe": None,
        "percept_refs": [],
        "imagined": False,
        "source": "local",
        "domain": None,
        "contributors": [],
    }
    return {
        "version": "1.0",
        "links": {"tool:probe": [link]},
        "outcome_index": {},
        "priors": {},
        "total_observations": 3,
        "reward_bias": {},
        "goal_reward_bias": {},
        "cluster_reward_bias": {f"aut{S}n1{S}tool:flee": 0.4, f"aut{S}orient{S}tool:turn": 0.2},
        "cluster_reward_source": {f"aut{S}n1{S}tool:flee": "drive_relief"},
        "inherent_bias_keys": [f"aut{S}n1{S}tool:flee"],
        "percept_valences": {},
        "event_outcome_welford": {},
        "cluster_fear": {f"aut{S}n1{S}drive:oxygen": -0.5},
    }


def ec_nodes() -> dict[str, Any]:
    return {
        "n1": {
            "embedding": [1.0, 0.0, 0.0, 0.0],
            "modality": "world",
            "count": 10,
            "source": "local",
            "domain": None,
            "geometry": "g1",
        }
    }


def compose(out: Path, *, signed: bool) -> dict[str, Any]:
    """Compose one fixture bundle with the fixed clock."""
    from unittest import mock

    import maxim.hivemind.bundle as bundle_mod
    from maxim.hivemind.signing import UNCOUNTED, SignedRelease

    release = SignedRelease(signer=signer(), release_sequence=1, license=LICENSE, counter=UNCOUNTED) if signed else None
    with mock.patch.object(bundle_mod, "_utc_now_iso", lambda: CREATED_AT):
        return bundle_mod.compose_bundle(
            nac_state=nac_state(),
            ec_substrate_nodes=ec_nodes(),
            output_path=out,
            contributor_id=CONTRIBUTOR,
            body_ref=BODY,
            apply_identity_filter=False,
            release=release,
            license=None if signed else LICENSE,
        )


def fingerprint(path: Path) -> dict[str, Any]:
    """What the freeze pins about one bundle: its payload identity, its entries, its member contents."""
    from maxim.hivemind.bundle import content_payload_digest, verify_bundle_zip

    with zipfile.ZipFile(path) as zf:
        members = {n: hashlib.sha256(zf.read(n)).hexdigest() for n in sorted(zf.namelist())}
        out: dict[str, Any] = {"payload_identity": content_payload_digest(zf), "member_sha256": members}
        if "signature.json" in members:
            v = verify_bundle_zip(zf, trusted_keys={SIGNER_IDENTITY: signer().public_key_b64}, accept_v1=False)
            out["entry_digests"] = dict(sorted(v.entry_digests.items()))
    return out


def main() -> None:
    golden: dict[str, Any] = {"signer_identity": SIGNER_IDENTITY, "public_key": signer().public_key_b64}
    for name, signed in (("release_v2.zip", True), ("unsigned_schema2.zip", False)):
        compose(HERE / name, signed=signed)
        golden[name] = fingerprint(HERE / name)
    (HERE / "golden.json").write_text(json.dumps(golden, indent=2, sort_keys=True) + "\n")
    print(json.dumps(golden, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
