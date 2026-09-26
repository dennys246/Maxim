"""public_oasis Phase 0 item 6: ``signer_identity`` is bound to the signature.

Release format v2 (docs/plans/oasis_entry_index_v2.md) signs the whole manifest, ``signer_identity``
included, so relabelling a v2 release breaks its signature. The one-key-under-two-identities refusal
stays (at verification and at registration): it guards legacy v1 bundles, where the identity was not
signed and only selected the key.
"""

from __future__ import annotations

import zipfile

import pytest

from maxim.utils.optional_deps import optional_dependency_available

_needs_crypto = pytest.mark.skipif(
    not optional_dependency_available("cryptography"), reason="signed bundles need the [sign] extra (cryptography)"
)


def _respelled(key_b64: str) -> str:
    """Another valid base64 spelling of the SAME 32 bytes: flip the last data character's unused bits."""
    import base64

    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
    last = key_b64[-2]  # 44 chars, one '=' pad: index -2 is the last data character
    other = alphabet[alphabet.index(last) ^ 1]
    alt = key_b64[:-2] + other + key_b64[-1]
    assert alt != key_b64 and base64.b64decode(alt) == base64.b64decode(key_b64)
    return alt


def _release_zip(tmp_path, signer):
    from maxim.hivemind.bundle import compose_bundle
    from tests.unit._signed_bundle_helpers import release

    path = tmp_path / "b.zip"
    compose_bundle(
        nac_state=None,
        ec_substrate_nodes={},
        output_path=path,
        contributor_id="x",
        body_ref="minecraft_bench",
        release=release(signer),
    )
    return path


def _verify(path, keys):
    from maxim.hivemind.bundle import verify_bundle_zip

    with zipfile.ZipFile(path) as zf:
        return verify_bundle_zip(zf, trusted_keys=keys)


@_needs_crypto
def test_a_v2_release_relabelled_to_another_trusted_identity_fails_verification(tmp_path):
    from maxim.hivemind.signing import BundleSigner
    from tests.unit._signed_bundle_helpers import rewrite_manifest

    queen = BundleSigner.generate(signer_identity="queen-a")
    other = BundleSigner.generate(signer_identity="experimental-x")
    path = _release_zip(tmp_path, other)
    keys = {"queen-a": queen.public_key_b64, "experimental-x": other.public_key_b64}
    assert _verify(path, keys).ok  # as signed
    rewrite_manifest(path, lambda m: m.__setitem__("signer_identity", "queen-a"))
    result = _verify(path, keys)
    assert not result.ok and "does not verify" in result.reason  # the identity is inside the signature


@_needs_crypto
@pytest.mark.parametrize("respell", [False, True])
def test_one_key_trusted_under_two_identities_is_refused(tmp_path, respell):
    """Both labels the same key -- including DIFFERENT base64 spellings of it. Refused before the
    signature is checked, so it holds for a relabelled copy too."""
    from maxim.hivemind.signing import BundleSigner
    from tests.unit._signed_bundle_helpers import rewrite_manifest

    signer = BundleSigner.generate(signer_identity="experimental-x")
    path = _release_zip(tmp_path, signer)
    other_spelling = _respelled(signer.public_key_b64) if respell else signer.public_key_b64
    aliased = {"queen-a": other_spelling, "experimental-x": signer.public_key_b64}
    for label in ("experimental-x", "queen-a"):
        rewrite_manifest(path, lambda m, label=label: m.__setitem__("signer_identity", label))
        result = _verify(path, aliased)
        assert not result.ok and "one key under two identities" in result.reason


@pytest.mark.parametrize("respell", [False, True])
def test_the_registry_refuses_one_key_under_two_identities(tmp_path, respell):
    """No crypto needed -- so this guard is tested on every lane, not only the extras lane."""
    import base64

    from maxim.hivemind.registry import HiveRegistry, HiveRegistryError

    key = base64.b64encode(bytes(range(32))).decode()
    other = _respelled(key) if respell else key
    with pytest.raises(HiveRegistryError, match="register each key once"):
        HiveRegistry(tmp_path / "hive.json").add(
            "o", "https://oasis.example", queen_keys={"queen-a": key, "queen-b": other}
        )
