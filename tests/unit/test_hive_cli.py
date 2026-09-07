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
