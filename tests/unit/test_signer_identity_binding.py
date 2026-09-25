"""public_oasis Phase 0 item 6, resolved as "not exploitable by construction; guarded".

``signer_identity`` is not in the signed payload (it is written after signing). It is still safe
because verification uses the key trusted FOR the claimed identity, so relabelling a bundle makes the
wrong key check it. The one residual case -- one key trusted under two identities -- is refused at
verification and at registration. Binding the identity into the payload is deferred to the next
release-format change (docs/plans/deferred/signed_signer_identity.md).
"""

from __future__ import annotations

import json
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


def _signed_parts(tmp_path, signer):
    from maxim.hivemind.bundle import compose_bundle

    path = tmp_path / "b.zip"
    compose_bundle(
        nac_state=None,
        ec_substrate_nodes={},
        output_path=path,
        contributor_id="x",
        body_ref="minecraft_bench",
        signer=signer,
    )
    with zipfile.ZipFile(path) as z:
        manifest = json.loads(z.read("manifest.json"))
        slices = {n: z.read(n).decode() for n in z.namelist() if n != "manifest.json"}
    return manifest, slices


@_needs_crypto
def test_a_bundle_relabelled_to_another_trusted_identity_fails_verification(tmp_path):
    from maxim.hivemind.bundle import verify_bundle_signature_parts
    from maxim.hivemind.signing import BundleSigner

    queen = BundleSigner.generate(signer_identity="queen-a")
    other = BundleSigner.generate(signer_identity="experimental-x")
    manifest, slices = _signed_parts(tmp_path, other)
    keys = {"queen-a": queen.public_key_b64, "experimental-x": other.public_key_b64}
    assert verify_bundle_signature_parts(manifest, slices, trusted_keys=keys)[0] is True  # as signed
    relabelled = {**manifest, "signer_identity": "queen-a"}
    ok, reason = verify_bundle_signature_parts(relabelled, slices, trusted_keys=keys)
    assert ok is False and "does not verify" in reason


@_needs_crypto
@pytest.mark.parametrize("respell", [False, True])
def test_one_key_trusted_under_two_identities_is_refused(tmp_path, respell):
    """The only case where a relabel would verify: both labels are the same key -- including when the
    two entries are DIFFERENT base64 spellings of it (the review's bypass of a string comparison)."""
    from maxim.hivemind.bundle import verify_bundle_signature_parts
    from maxim.hivemind.signing import BundleSigner

    signer = BundleSigner.generate(signer_identity="experimental-x")
    manifest, slices = _signed_parts(tmp_path, signer)
    other_spelling = _respelled(signer.public_key_b64) if respell else signer.public_key_b64
    aliased = {"queen-a": other_spelling, "experimental-x": signer.public_key_b64}
    for label in ("experimental-x", "queen-a"):
        ok, reason = verify_bundle_signature_parts({**manifest, "signer_identity": label}, slices, trusted_keys=aliased)
        assert ok is False and "one key under two identities" in reason


@pytest.mark.parametrize(("respell", "why"), [(False, "register each key once"), (True, "non-canonical")])
def test_the_registry_refuses_one_key_under_two_identities(tmp_path, respell, why):
    """No crypto needed -- so this guard is tested on every lane, not only the extras lane. A
    RE-SPELLED alias is now refused one step earlier, as a non-canonical key."""
    import base64

    from maxim.hivemind.registry import HiveRegistry, HiveRegistryError

    key = base64.b64encode(bytes(range(32))).decode()
    other = _respelled(key) if respell else key
    with pytest.raises(HiveRegistryError, match=why):
        HiveRegistry(tmp_path / "hive.json").add(
            "o", "https://oasis.example", queen_keys={"queen-a": key, "queen-b": other}
        )


@pytest.mark.parametrize(
    ("pubkey", "why"),
    [
        ("PUB", "not valid base64"),
        ("not base64!", "not valid base64"),
        ("AAECAwQ=", "decodes to 5 bytes"),
        ("AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8gIQ==", "decodes to 34 bytes"),
    ],
)
def test_the_registry_refuses_a_key_verification_could_never_use(tmp_path, pubkey, why):
    """Refused where the operator types it, not stored to fail at the first pull."""
    from maxim.hivemind.registry import HiveRegistry, HiveRegistryError

    with pytest.raises(HiveRegistryError, match=why):
        HiveRegistry(tmp_path / "hive.json").add("o", "https://oasis.example", queen_keys={"q": pubkey})


def test_hive_add_refuses_a_bad_key_with_exit_2_and_names_whitespace(tmp_path, capsys):
    import base64

    from maxim.hivemind.hive_cli import run_hive_subcommand

    reg = str(tmp_path / "hive.json")
    assert run_hive_subcommand(["--registry", reg, "add", "o", "https://o.example", "--queen-key", "q=PUB"]) == 2
    assert "not valid base64" in capsys.readouterr().err
    key_file_line = base64.b64encode(bytes(range(32))).decode() + "\n"  # as load_or_create_signer writes it
    assert (
        run_hive_subcommand(["--registry", reg, "add", "o", "https://o.example", "--queen-key", f"q={key_file_line}"])
        == 2
    )
    assert "whitespace" in capsys.readouterr().err


def test_the_registry_names_the_canonical_spelling_of_a_respelled_key(tmp_path):
    import base64

    from maxim.hivemind.registry import HiveRegistry, HiveRegistryError

    key = base64.b64encode(bytes(range(32))).decode()
    with pytest.raises(HiveRegistryError, match=key):
        HiveRegistry(tmp_path / "hive.json").add("o", "https://oasis.example", queen_keys={"q": _respelled(key)})
    assert HiveRegistry(tmp_path / "hive.json").add("o", "https://oasis.example", queen_keys={"q": key})
