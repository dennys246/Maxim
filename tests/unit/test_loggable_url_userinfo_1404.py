"""#1404 item 5: the URL an HTTP log event carries keeps no credentials.

``utils/http.py::_loggable_url`` stripped the query and fragment (the 2026-08-12 privacy audit) but kept
``parts.netloc``, so a URL's ``user:pass@`` reached the default-on WARNING ``http_request_failed``, which lands in
``MAXIM_LOG_FILE`` and the leader proxy's remotely readable log buffer.
"""

from __future__ import annotations

import logging

import pytest


@pytest.mark.parametrize(
    "url, expected",
    [
        ("https://user:s3cret@host.example/path?q=1", "https://host.example/path?…"),
        ("https://s3cret@host.example:8443/v1", "https://host.example:8443/v1"),
        ("http://u:p%40ss@[::1]:8100/v1/models", "http://[::1]:8100/v1/models"),
        ("ws://user:s3cret@relay.example/socket", "ws://relay.example/socket"),
    ],
    ids=["user-and-password", "token-as-user", "encoded-at-ipv6", "non-http-scheme"],
)
def test_the_logged_url_drops_userinfo(url: str, expected: str) -> None:
    from maxim.utils.http import _loggable_url

    logged = _loggable_url(url)
    assert logged == expected
    assert "s3cret" not in logged and "p%40ss" not in logged


@pytest.mark.parametrize("url", ["http://[::1/x", "https://u:s3cret@[bad/x"], ids=["unbalanced", "unbalanced-userinfo"])
def test_a_url_urlsplit_refuses_is_logged_as_a_placeholder_not_raised(url: str) -> None:
    from maxim.utils.http import _loggable_url

    logged = _loggable_url(url)
    assert logged == "<url>"


def test_the_failed_request_warning_carries_no_password(caplog) -> None:
    """A real refused loopback dial (port 9, discard: nothing listens), so the event is the shipped one."""
    from maxim.utils import http

    with caplog.at_level(logging.WARNING):
        with pytest.raises(http.HTTPError):
            http.fetch_url("http://user:s3cret@127.0.0.1:9/v1/models", timeout=2.0)
    failed = [r for r in caplog.records if "http_request_failed" in r.getMessage()]
    assert failed, "expected the http_request_failed WARNING"
    assert all("s3cret" not in str(vars(r)) for r in failed)


def test_control_query_and_fragment_are_still_stripped() -> None:
    from maxim.utils.http import _loggable_url

    assert (
        _loggable_url("https://api.duckduckgo.com/?q=secret+llm+query&format=json") == "https://api.duckduckgo.com/?…"
    )
    assert _loggable_url("https://example.com/docs/page") == "https://example.com/docs/page"


def test_the_host_is_what_follows_the_last_at() -> None:
    from maxim.utils.http import _loggable_url

    assert _loggable_url("https://u@v:s3cret@host.example/x") == "https://host.example/x"


@pytest.mark.parametrize(
    "url",
    ["user:s3cret@host:80/x", "u:s3cret@h", "https:u:s3cret@h/x", "mailto:u:s3cret@h", r"https:\\u:s3cret@h\x"],
    ids=["no-scheme-slashes", "bare", "https-no-slashes", "mailto", "backslashes"],
)
def test_userinfo_without_a_netloc_is_logged_as_a_placeholder(url: str) -> None:
    from maxim.utils.http import _loggable_url

    assert _loggable_url(url) == "<url>"


def test_control_an_at_in_the_path_of_a_real_url_is_kept() -> None:
    from maxim.utils.http import _loggable_url

    assert _loggable_url("https://h.example/p@x:y") == "https://h.example/p@x:y"
    assert _loggable_url("file:///tmp/x") == "file:///tmp/x"
