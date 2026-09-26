"""public_oasis Phase 0 item 5: a pull or contribution never sends the local leader key to a remote Oasis.

The leader key also grants inference; ``hive pull`` / ``hive contribute`` used to fall back to it for
ANY registered Oasis, handing that credential to whoever runs it. Now the fallback applies only to a
loopback Oasis (this machine); anything else needs an explicit ``--api-key``.
"""

from __future__ import annotations

import pytest

from maxim.hivemind import hive_cli
from maxim.hivemind.hive_cli import _oasis_api_key, run_hive_subcommand
from maxim.utils.net import is_loopback_url

LEADER_KEY = "leader-secret-also-grants-inference"


@pytest.mark.parametrize(
    ("url", "loopback"),
    [
        ("http://127.0.0.1:8000", True),
        ("http://localhost:8000", True),
        ("http://LOCALHOST", True),
        ("http://[::1]:8000", True),
        ("https://oasis.pymaxim.bio", False),
        ("http://10.0.0.5:8000", False),  # a LAN Oasis is still another machine
        ("http://127.0.0.1.evil.example", False),  # a name, not a loopback literal -- no DNS
        ("not a url", False),
        ("", False),
    ],
)
def test_is_loopback_url(url, loopback):
    assert is_loopback_url(url) is loopback


@pytest.fixture
def leader_key(monkeypatch):
    calls: list[int] = []

    def _read_key(*a, **k):
        calls.append(1)
        return LEADER_KEY

    monkeypatch.setattr("maxim.tunnel.keys.read_key", _read_key)
    return calls


def test_the_leader_key_is_used_only_for_a_loopback_oasis(leader_key):
    assert _oasis_api_key(None, "http://127.0.0.1:8000") == LEADER_KEY
    assert _oasis_api_key(None, "https://oasis.pymaxim.bio") is None
    assert _oasis_api_key("issued-by-that-oasis", "https://oasis.pymaxim.bio") == "issued-by-that-oasis"


@pytest.mark.parametrize("url", ["https://oasis.pymaxim.bio", "http://127.0.0.1:8000"])
def test_hive_pull_sends_the_leader_key_only_to_this_machine(tmp_path, monkeypatch, leader_key, capsys, url):
    from maxim.hivemind import substrate_client as sc

    sent: list = []

    def _list_releases(u, *, api_key):
        sent.append(api_key)
        from maxim.utils import http

        raise http.HTTPAuthError(u, status=401)

    monkeypatch.setattr(sc, "list_releases", _list_releases)
    reg = str(tmp_path / "hive.json")
    assert (
        run_hive_subcommand(
            ["--registry", reg, "add", "o", url, "--queen-key", "q=AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8="]
        )
        == 0
    )
    rc = run_hive_subcommand(
        ["--registry", reg, "pull", "--from", "o", "--session", str(tmp_path), "--receiver-body", "b"]
    )
    assert rc == 2
    if url.startswith("http://127.0.0.1"):
        assert sent == [LEADER_KEY]
    else:
        assert sent == [None]  # the leader key never left this machine
        assert "local leader key was not sent" in capsys.readouterr().err


def test_hive_contribute_sends_the_leader_key_only_to_this_machine(tmp_path, monkeypatch, leader_key, capsys):
    from maxim.hivemind import substrate_client as sc

    sent: list = []

    def _contribute(u, path, *, api_key):
        sent.append(api_key)
        from maxim.utils import http

        raise http.HTTPAuthError(u, status=401)

    monkeypatch.setattr(sc, "contribute", _contribute)
    bundle = tmp_path / "b.zip"
    bundle.write_bytes(b"x")
    reg = str(tmp_path / "hive.json")
    run_hive_subcommand(["--registry", reg, "add", "remote", "https://oasis.pymaxim.bio"])
    assert run_hive_subcommand(["--registry", reg, "contribute", str(bundle), "--to", "remote"]) == 2
    assert sent == [None]
    assert "local leader key was not sent" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("exc_name", "explicit", "url", "hinted"),
    [
        ("HTTPAuthError", None, "https://oasis.pymaxim.bio", True),
        ("HTTPTimeout", None, "https://oasis.pymaxim.bio", False),  # not an auth failure
        ("HTTPServerError", None, "https://oasis.pymaxim.bio", False),
        ("HTTPAuthError", "given", "https://oasis.pymaxim.bio", False),  # a key WAS sent; it was wrong
        ("HTTPAuthError", None, "http://127.0.0.1:8000", False),  # loopback: the leader key was offered
    ],
)
def test_the_hint_appears_only_when_a_missing_key_is_the_cause(exc_name, explicit, url, hinted):
    from maxim.utils import http

    exc = getattr(http, exc_name)(url, status=401)
    assert bool(hive_cli._no_key_hint(exc, explicit, url)) is hinted
