"""Guard tests for the maxim oasis / maxim hive CLIs + the Oasis registry (1.2 P2P Slice C).

Registry semantics + arg-parsing rc contract (0 success / 2 operator error); the
live fetch→ingest pull is exercised in tests/integration/test_hive_pull_e2e.py.
"""

from __future__ import annotations

import pytest

from maxim.hivemind.oasis_cli import run_oasis_subcommand
from maxim.hivemind.hive_cli import run_hive_subcommand
from maxim.hivemind.registry import HiveRegistry, HiveRegistryError
from maxim.hivemind.store import OasisStore
from maxim.hivemind.bundle import compose_bundle
from maxim.utils.optional_deps import optional_dependency_available

_HAS_CRYPTO = optional_dependency_available("cryptography")
_needs_crypto = pytest.mark.skipif(not _HAS_CRYPTO, reason="signed bundles need the [sign] extra (cryptography)")

_EC_NODES = {"node-1": {"modality": "world", "embedding": [0.1, 0.2, 0.3], "domain": None}}


class TestRegistry:
    def test_add_get_remove_roundtrip(self, tmp_path):
        reg = HiveRegistry(tmp_path / "hive.json")
        reg.add("alpha", "https://a.example", queen_keys={"queen-a": "PUB"}, domains=("combat",))
        entry = reg.get("alpha")
        assert entry is not None
        assert entry["url"] == "https://a.example"
        assert entry["queen_keys"] == {"queen-a": "PUB"}
        assert entry["domains"] == ["combat"]
        assert reg.remove("alpha") is True
        assert reg.get("alpha") is None
        assert reg.remove("alpha") is False

    def test_add_replaces_by_name(self, tmp_path):
        reg = HiveRegistry(tmp_path / "hive.json")
        reg.add("alpha", "https://a.example")
        reg.add("alpha", "https://b.example")
        assert [o["url"] for o in reg.list_oases()] == ["https://b.example"]

    def test_bad_url_refused(self, tmp_path):
        reg = HiveRegistry(tmp_path / "hive.json")
        with pytest.raises(HiveRegistryError, match="http"):
            reg.add("alpha", "ftp://a.example")

    def test_corrupt_registry_fails_loud(self, tmp_path):
        path = tmp_path / "hive.json"
        path.write_text("{ not valid json", encoding="utf-8")
        reg = HiveRegistry(path)
        with pytest.raises(HiveRegistryError, match="present but unreadable"):
            reg.list_oases()


class TestHiveCliArgs:
    def test_add_then_list_then_remove(self, tmp_path, capsys):
        reg = str(tmp_path / "hive.json")
        assert (
            run_hive_subcommand(["--registry", reg, "add", "alpha", "https://a.example", "--queen-key", "q=PUB"]) == 0
        )
        assert run_hive_subcommand(["--registry", reg, "list"]) == 0
        assert "alpha" in capsys.readouterr().out
        assert run_hive_subcommand(["--registry", reg, "remove", "alpha"]) == 0

    def test_add_bad_queen_key_spec_rc2(self, tmp_path):
        reg = str(tmp_path / "hive.json")
        assert run_hive_subcommand(["--registry", reg, "add", "a", "https://a.example", "--queen-key", "noeq"]) == 2

    def test_remove_missing_rc2(self, tmp_path):
        assert run_hive_subcommand(["--registry", str(tmp_path / "hive.json"), "remove", "ghost"]) == 2

    def test_list_empty(self, tmp_path, capsys):
        assert run_hive_subcommand(["--registry", str(tmp_path / "hive.json"), "list"]) == 0
        assert "no registered" in capsys.readouterr().out

    def test_pull_unknown_oasis_rc2(self, tmp_path):
        reg = str(tmp_path / "hive.json")
        rc = run_hive_subcommand(
            ["--registry", reg, "pull", "--from", "ghost", "--session", str(tmp_path), "--receiver-body", "b"]
        )
        assert rc == 2

    def test_pull_oasis_without_queen_keys_rc2(self, tmp_path):
        reg = str(tmp_path / "hive.json")
        run_hive_subcommand(["--registry", reg, "add", "alpha", "https://a.example"])
        rc = run_hive_subcommand(
            ["--registry", reg, "pull", "--from", "alpha", "--session", str(tmp_path), "--receiver-body", "b"]
        )
        assert rc == 2


class TestTrustPolicy:
    """1.2 Slice D — per-Oasis consumer trust policy over the shipped ingest hooks."""

    def test_defaults_are_queen_only(self, tmp_path):
        from maxim.hivemind.registry import trust_policy

        reg = HiveRegistry(tmp_path / "hive.json")
        entry = reg.add("alpha", "https://a.example", queen_keys={"q": "PUB"})
        policy = trust_policy(entry)
        assert policy == {"allow_unsigned": False, "inherent_trust": False, "trusted_sources": []}

    def test_policy_defaults_apply_to_legacy_entries(self):
        """A registry file written before Slice D has no policy keys — defaults hold."""
        from maxim.hivemind.registry import trust_policy

        assert trust_policy({"name": "old", "url": "https://o.example"}) == {
            "allow_unsigned": False,
            "inherent_trust": False,
            "trusted_sources": [],
        }

    def test_set_trust_roundtrips_and_persists(self, tmp_path):
        from maxim.hivemind.registry import trust_policy

        path = tmp_path / "hive.json"
        HiveRegistry(path).add("alpha", "https://a.example")
        HiveRegistry(path).set_trust("alpha", allow_unsigned=True, inherent_trust=True, trusted_sources=["peer-7"])
        policy = trust_policy(HiveRegistry(path).get("alpha"))
        assert policy["allow_unsigned"] is True
        assert policy["inherent_trust"] is True
        assert policy["trusted_sources"] == ["peer-7"]

    def test_re_add_preserves_trust_policy(self, tmp_path):
        """Correcting a URL must not silently reset a trust grant."""
        from maxim.hivemind.registry import trust_policy

        path = tmp_path / "hive.json"
        reg = HiveRegistry(path)
        reg.add("alpha", "https://a.example")
        reg.set_trust("alpha", allow_unsigned=True, trusted_sources=["peer-7"])
        reg.add("alpha", "https://moved.example")
        policy = trust_policy(reg.get("alpha"))
        assert reg.get("alpha")["url"] == "https://moved.example"
        assert policy["allow_unsigned"] is True
        assert policy["trusted_sources"] == ["peer-7"]

    def test_re_add_preserves_queen_keys_and_unknown_fields(self, tmp_path):
        """A URL correction must not silently degrade the posture.

        Wiping the Queen keys while an `allow_unsigned` grant survives would turn
        "Queen-only + escape hatch" into "admit anything" from a URL edit.
        """
        from maxim.hivemind.registry import trust_policy

        path = tmp_path / "hive.json"
        reg = HiveRegistry(path)
        reg.add("alpha", "https://a.example", queen_keys={"queen-a": "PUB"})
        reg.set_trust("alpha", allow_unsigned=True)
        reg.add("alpha", "https://moved.example")  # no --queen-key passed
        entry = reg.get("alpha")
        assert entry["url"] == "https://moved.example"
        assert entry["queen_keys"] == {"queen-a": "PUB"}  # NOT wiped
        assert trust_policy(entry)["allow_unsigned"] is True

    def test_set_trust_unknown_oasis_raises(self, tmp_path):
        with pytest.raises(HiveRegistryError, match="no registered oasis"):
            HiveRegistry(tmp_path / "hive.json").set_trust("ghost", allow_unsigned=True)

    def test_trust_cli_rc_contract(self, tmp_path):
        reg = str(tmp_path / "hive.json")
        run_hive_subcommand(["--registry", reg, "add", "alpha", "https://a.example"])
        assert run_hive_subcommand(["--registry", reg, "trust", "alpha", "--allow-unsigned"]) == 0
        # nothing to set → rc 2 (a no-op trust command is an operator error)
        assert run_hive_subcommand(["--registry", reg, "trust", "alpha"]) == 2
        assert run_hive_subcommand(["--registry", reg, "trust", "ghost", "--allow-unsigned"]) == 2

    def test_no_queen_keys_refused_at_policy_gate_unless_allow_unsigned(self, tmp_path, monkeypatch):
        """Queen-only default refuses BEFORE any network; the opt-in gets past the gate."""
        from maxim.hivemind import substrate_client as sc

        reg = str(tmp_path / "hive.json")
        run_hive_subcommand(["--registry", reg, "add", "alpha", "https://a.example"])
        listed: list = []
        monkeypatch.setattr(sc, "list_releases", lambda *a, **k: listed.append(a) or [])
        pull = [
            "--registry",
            reg,
            "pull",
            "--from",
            "alpha",
            "--session",
            str(tmp_path),
            "--receiver-body",
            "b",
            "--api-key",
            "k",
        ]
        # queen-only default + no keys → refused at the policy gate, no network touched
        assert run_hive_subcommand(pull) == 2
        assert listed == []

        # disabling verification for this Oasis gets past the gate and reaches it
        run_hive_subcommand(["--registry", reg, "trust", "alpha", "--allow-unsigned"])
        assert run_hive_subcommand(pull) == 0  # no releases offered
        assert len(listed) == 1


class TestIngestArgvConstruction:
    """The security-critical decisions of `hive pull`, tested directly.

    These are the lines that decide what foreign substrate reaches the merge:
    which trusted_sources V1 gets, whether signature verification runs, and
    whether the decay-exempt safety floor is opened.
    """

    @staticmethod
    def _policy(**over):
        base = {"allow_unsigned": False, "inherent_trust": False, "trusted_sources": []}
        base.update(over)
        return base

    def _argv(self, **kw):
        from maxim.hivemind.hive_cli import _build_ingest_argv
        from pathlib import Path as _P

        defaults = dict(
            session="/s",
            receiver_body="body",
            contributor="peer-7",
            queen_verified=True,
            queen_keys={"queen-a": "PUB"},
            policy=self._policy(),
        )
        defaults.update(kw)
        return _build_ingest_argv(_P("/tmp/b.zip"), **defaults)

    def test_queen_verified_requires_signature_and_passes_keys(self):
        argv = self._argv()
        assert "--require-signed" in argv
        assert "--trust-key" in argv and "queen-a=PUB" in argv
        assert argv[argv.index("--trust") + 1] == "peer-7"
        assert "--inherent-trust" not in argv  # default: safety floor refused

    def test_unverified_path_omits_signature_enforcement_entirely(self):
        argv = self._argv(queen_verified=False, policy=self._policy(allow_unsigned=True, inherent_trust=True))
        assert "--require-signed" not in argv
        assert "--trust-key" not in argv
        # the safety floor NEVER opens for unverified content, even when opted in
        assert "--inherent-trust" not in argv

    def test_allow_list_becomes_the_v1_trusted_sources_not_the_manifest_id(self):
        """The BLOCKER fix: V1 must receive the OPERATOR list, not the bundle's own id.

        Passing the self-declared contributor_id would make V1 a tautology and
        leave the allow-list enforceable only in the CLI — bypassable on the
        unverified path where the id is attacker-chosen.
        """
        argv = self._argv(
            contributor="attacker-claimed",
            policy=self._policy(trusted_sources=["peer-7", "peer-8"]),
        )
        trusted = [argv[i + 1] for i, a in enumerate(argv) if a == "--trust"]
        assert trusted == ["peer-7", "peer-8"]
        assert "attacker-claimed" not in argv

    def test_inherent_trust_is_scoped_to_the_allow_list(self):
        argv = self._argv(
            contributor="attacker-claimed",
            policy=self._policy(inherent_trust=True, trusted_sources=["peer-7"]),
        )
        inherent = [argv[i + 1] for i, a in enumerate(argv) if a == "--inherent-trust"]
        assert inherent == ["peer-7"]

    def test_passthrough_flags(self):
        argv = self._argv(apply=True, allow_unstamped_geometry=True)
        assert "--apply" in argv and "--allow-unstamped-geometry" in argv
        assert "--apply" not in self._argv()


class TestMalformedPolicyFailsLoud:
    def test_string_false_does_not_silently_opt_in(self):
        """bool("false") is True — a coerced policy would invert a safety default."""
        from maxim.hivemind.registry import trust_policy

        with pytest.raises(HiveRegistryError, match="must be a JSON boolean"):
            trust_policy({"allow_unsigned": "false"})

    def test_non_list_trusted_sources_refused(self):
        from maxim.hivemind.registry import trust_policy

        with pytest.raises(HiveRegistryError, match="list of non-empty strings"):
            trust_policy({"trusted_sources": "peer-7"})
        with pytest.raises(HiveRegistryError, match="list of non-empty strings"):
            trust_policy({"trusted_sources": 5})

    def test_cli_surfaces_malformed_policy_as_rc2_not_traceback(self, tmp_path):
        import json

        path = tmp_path / "hive.json"
        path.write_text(
            json.dumps(
                {
                    "_format_version": "1.0",
                    "oases": [{"name": "a", "url": "https://a.example", "allow_unsigned": "yes"}],
                }
            ),
            encoding="utf-8",
        )
        assert run_hive_subcommand(["--registry", str(path), "list"]) == 2


class TestConflictingTrustFlags:
    def test_opposing_flags_are_rejected_not_silently_loosened(self, tmp_path):
        reg = str(tmp_path / "hive.json")
        run_hive_subcommand(["--registry", reg, "add", "alpha", "https://a.example"])
        # argparse mutually-exclusive groups exit(2) rather than granting the looser value
        with pytest.raises(SystemExit):
            run_hive_subcommand(["--registry", reg, "trust", "alpha", "--allow-unsigned", "--require-signed"])
        with pytest.raises(SystemExit):
            run_hive_subcommand(["--registry", reg, "trust", "alpha", "--inherent", "--no-inherent"])


class TestCliDispatch:
    """The substrate-family dispatch helper extracted from _main_impl (item 16.4)."""

    def test_family_verbs_route_and_others_fall_through(self, tmp_path):
        from maxim.cli import _dispatch_hivemind_cli

        reg = str(tmp_path / "hive.json")
        # a family verb routes and returns its rc (0 for an empty `hive list`)
        assert _dispatch_hivemind_cli(["hive", "--registry", reg, "list"]) == 0
        # non-family verbs and an empty argv fall through (None) so _main_impl's
        # if-chain continues to config/peer/etc exactly as before the extraction
        assert _dispatch_hivemind_cli(["config", "get"]) is None
        assert _dispatch_hivemind_cli([]) is None


class TestReleaseIdHardening:
    """Regression guards for the cross-confirmed path-traversal BLOCKER."""

    def test_is_valid_release_id_rejects_traversal(self):
        from maxim.hivemind.store import is_valid_release_id

        assert is_valid_release_id("a" * 64)
        for bad in ("../../evil", "/etc/passwd", "a" * 63, "A" * 64, "", "..", "a/b"):
            assert not is_valid_release_id(bad), bad

    def test_fetch_bundle_refuses_malformed_id(self, tmp_path):
        from maxim.hivemind import substrate_client as sc

        dest = tmp_path / "out.zip"
        with pytest.raises(sc.SubstrateExchangeError, match="malformed release id"):
            sc.fetch_bundle("http://oasis.example", "../../../evil", dest, api_key="k")
        assert not dest.exists()

    def test_pull_skips_malformed_release_id_without_fetching(self, tmp_path, monkeypatch):
        from maxim.hivemind import substrate_client as sc

        reg = str(tmp_path / "hive.json")
        run_hive_subcommand(["--registry", reg, "add", "alpha", "https://a.example", "--queen-key", "q=PUB"])
        # A malicious/MITM Oasis returns a traversal id; pull must skip it before
        # ever building a path or fetching.
        monkeypatch.setattr(sc, "list_releases", lambda *a, **k: [{"id": "../../../evil", "domain": None}])
        fetched: list = []
        monkeypatch.setattr(sc, "fetch_bundle", lambda *a, **k: fetched.append(a))
        rc = run_hive_subcommand(
            [
                "--registry",
                reg,
                "pull",
                "--from",
                "alpha",
                "--session",
                str(tmp_path),
                "--receiver-body",
                "b",
                "--api-key",
                "k",
            ]
        )
        assert rc == 2
        assert fetched == []  # never attempted a fetch with the malformed id


class TestOasisCli:
    def test_publish_unsigned_rc2_with_hint(self, tmp_path, capsys):
        out = tmp_path / "b.zip"
        compose_bundle(
            nac_state=None,
            ec_substrate_nodes=_EC_NODES,
            output_path=out,
            contributor_id="oasis-alpha",
            body_ref="minecraft_bench",
        )
        rc = run_oasis_subcommand(["publish", str(out), "--root", str(tmp_path / "store")])
        assert rc == 2
        assert "sign" in capsys.readouterr().err

    def test_publish_missing_file_rc2(self, tmp_path):
        assert run_oasis_subcommand(["publish", str(tmp_path / "nope.zip"), "--root", str(tmp_path / "store")]) == 2

    def test_publish_non_zip_rc2_not_traceback(self, tmp_path, capsys):
        junk = tmp_path / "notabundle.zip"
        junk.write_text("this is not a zip", encoding="utf-8")
        rc = run_oasis_subcommand(["publish", str(junk), "--root", str(tmp_path / "store")])
        assert rc == 2
        assert "error:" in capsys.readouterr().err

    def test_serve_refuses_insecure_public_bind(self, tmp_path, monkeypatch):
        monkeypatch.setattr("maxim.tunnel.keys.read_key", lambda *a, **k: None)
        started: list = []
        monkeypatch.setattr("maxim.runtime.leader_proxy.start_leader_proxy", lambda **k: started.append(k) or None)
        rc = run_oasis_subcommand(["serve", "--root", str(tmp_path / "s"), "--bind-host", "0.0.0.0"])
        assert rc == 2
        assert started == []  # refused before starting a server

    def test_status_empty_store(self, tmp_path, capsys):
        rc = run_oasis_subcommand(["status", "--root", str(tmp_path / "store")])
        assert rc == 0
        assert "releases (Queen tier):        0" in capsys.readouterr().out

    @_needs_crypto
    def test_publish_signed_then_status(self, tmp_path, capsys):
        from maxim.hivemind.signing import BundleSigner

        out = tmp_path / "b.zip"
        compose_bundle(
            nac_state=None,
            ec_substrate_nodes=_EC_NODES,
            output_path=out,
            contributor_id="oasis-alpha",
            body_ref="minecraft_bench",
            signer=BundleSigner.generate(signer_identity="queen-a"),
        )
        root = str(tmp_path / "store")
        assert run_oasis_subcommand(["publish", str(out), "--root", root]) == 0
        assert run_oasis_subcommand(["status", "--root", root]) == 0
        assert "releases (Queen tier):        1" in capsys.readouterr().out
        # the release is the content digest of the published bytes
        assert len(OasisStore(tmp_path / "store").list_releases()) == 1
