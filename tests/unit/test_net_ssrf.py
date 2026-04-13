"""Tests for maxim.utils.net.validate_base_url (Plan 2 R2d)."""

from __future__ import annotations

from unittest.mock import patch

from maxim.utils.net import _is_private_ip, validate_base_url


def _fake_getaddrinfo(ip: str):
    def _inner(host, port, proto=None):
        return [(0, 0, 0, "", (ip, port))]

    return _inner


# ── is_private_ip ────────────────────────────────────────────────────────


def test_is_private_ip_loopback():
    assert _is_private_ip("127.0.0.1")


def test_is_private_ip_rfc1918():
    assert _is_private_ip("10.0.0.1")
    assert _is_private_ip("192.168.1.1")
    assert _is_private_ip("172.16.0.1")


def test_is_private_ip_public():
    assert not _is_private_ip("8.8.8.8")


def test_is_private_ip_malformed_fails_closed():
    assert _is_private_ip("not-an-ip")


# ── validate_base_url ────────────────────────────────────────────────────


def test_http_public_rejected():
    with patch("maxim.utils.net.socket.getaddrinfo", side_effect=_fake_getaddrinfo("8.8.8.8")):
        assert validate_base_url("http://evil.com", allow_local=False) is None


def test_http_public_rejected_even_with_allow_local():
    with patch("maxim.utils.net.socket.getaddrinfo", side_effect=_fake_getaddrinfo("8.8.8.8")):
        assert validate_base_url("http://evil.com", allow_local=True) is None


def test_http_private_ok_when_allow_local():
    with patch("maxim.utils.net.socket.getaddrinfo", side_effect=_fake_getaddrinfo("10.0.0.1")):
        assert validate_base_url("http://10.0.0.1:8080/v1", allow_local=True) == "http://10.0.0.1:8080/v1"


def test_http_private_rejected_when_not_allow_local():
    with patch("maxim.utils.net.socket.getaddrinfo", side_effect=_fake_getaddrinfo("10.0.0.1")):
        assert validate_base_url("http://10.0.0.1:8080/v1", allow_local=False) is None


def test_https_public_allowed():
    with patch("maxim.utils.net.socket.getaddrinfo", side_effect=_fake_getaddrinfo("8.8.8.8")):
        assert validate_base_url("https://api.openai.com/v1", allow_local=False)


def test_https_private_allowed_when_allow_local():
    with patch("maxim.utils.net.socket.getaddrinfo", side_effect=_fake_getaddrinfo("10.0.0.1")):
        assert validate_base_url("https://leader.lan/v1", allow_local=True)


def test_https_private_rejected_when_not_allow_local():
    with patch("maxim.utils.net.socket.getaddrinfo", side_effect=_fake_getaddrinfo("10.0.0.1")):
        assert validate_base_url("https://leader.lan/v1", allow_local=False) is None


def test_malformed_url_rejected():
    assert validate_base_url("", allow_local=False) is None
    assert validate_base_url("not-a-url", allow_local=False) is None


def test_dns_failure_rejected():
    with patch("maxim.utils.net.socket.getaddrinfo", side_effect=OSError("dns")):
        assert validate_base_url("https://does-not-resolve.example", allow_local=False) is None


def test_ftp_scheme_rejected():
    assert validate_base_url("ftp://evil.com", allow_local=True) is None


# ── Backward-compat: _OpenAIBackend still imports via the alias ──────────


def test_openai_backend_reexports_alias():
    from maxim.models.language.openai_backend import _validate_base_url
    from maxim.utils.net import validate_base_url as _canonical

    assert _validate_base_url is _canonical
