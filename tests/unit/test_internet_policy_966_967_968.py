"""The internet policy's batch-1 review findings, fixed in 1.3.2 (owner decisions 2026-10-14).

- #966: the private-address PRE-check classifies through the one shared classifier
  (``utils/net.py::is_public_address``), so it agrees with the connect-time check (CGNAT), and a DNS failure gets
  its own reason instead of "private/internal IPs"; ``block_private_ips=False`` warns at load.
- #967: robots.txt FAILS CLOSED (owner): a robots.txt that cannot be read (5xx, timeout, network) refuses the fetch
  and is not cached; a missing one (4xx) means no restrictions (RFC 9309); the summary says so.
- #968: domain entries normalise with IDNA 2008 (UTS 46, non-transitional), so ``straße.de`` blocks the
  ``xn--strae-oqa.de`` host a URL carries; ``idna`` is a declared dependency.
"""

from __future__ import annotations

import json
import logging
import socket
from pathlib import Path

import pytest

from maxim.tools import http_fetch
from maxim.utils import http as _http
from maxim.utils import internet_access as ia


@pytest.fixture(autouse=True)
def _fresh_caches():
    http_fetch._robots_cache.clear()
    yield
    http_fetch._robots_cache.clear()


# ── #966 ────────────────────────────────────────────────────────────────────────────────────────────


def test_the_precheck_refuses_cgnat_like_the_connect_check():
    from maxim.utils.net import is_public_address

    assert not is_public_address("100.64.1.2")  # the connect-time classifier
    reason = ia.InternetAccessPolicy().url_refusal("http://100.64.1.2/")
    assert reason is not None and "private" in reason.lower()


def test_a_dns_failure_gets_its_own_reason(monkeypatch):
    def nx(*args, **kwargs):
        raise socket.gaierror(socket.EAI_NONAME, "name not known")

    monkeypatch.setattr(ia, "_cached_getaddrinfo", nx)
    reason = ia.InternetAccessPolicy().url_refusal("https://nx.invalid/")
    assert reason is not None and "resolve" in reason.lower() and "private" not in reason.lower()


def test_turning_off_private_ip_blocking_warns_at_load(tmp_path: Path, caplog):
    path = tmp_path / "internet_policy.json"
    path.write_text(json.dumps({"_format_version": "1.0", "block_private_ips": False}))
    with caplog.at_level(logging.WARNING, logger="maxim.utils.internet_access"):
        policy = ia._operator_policy(path)
    assert policy is not None and policy.block_private_ips is False
    assert any("block_private_ips" in r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING)


# ── #967 ────────────────────────────────────────────────────────────────────────────────────────────


class _Resp:
    def __init__(self, text: str) -> None:
        self.text, self.status, self.headers = text, 200, {}


def _fetch_raising(exc: Exception):
    def fetch(*args, **kwargs):
        raise exc

    return fetch


@pytest.mark.parametrize(
    "exc",
    [
        _http.HTTPServerError("robots", status=503),
        _http.HTTPTimeout("robots"),
        _http.HTTPConnectionError("robots"),
    ],
    ids=["5xx", "timeout", "connection"],
)
def test_an_unreadable_robots_txt_refuses_the_fetch_and_is_not_cached(monkeypatch, exc):
    monkeypatch.setattr(_http, "fetch_url", _fetch_raising(exc))
    allowed, reason = http_fetch.check_robots_txt("https://example.com/page")
    assert allowed is False and reason and "robots.txt" in reason
    monkeypatch.setattr(_http, "fetch_url", lambda *a, **k: _Resp("User-agent: *\nDisallow: /private\n"))
    assert http_fetch.check_robots_txt("https://example.com/page") == (True, None)  # not cached: retried


def test_a_missing_robots_txt_means_no_restrictions(monkeypatch):
    """RFC 9309: 4xx = no robots.txt = no rules. Not an error, so not refused."""
    monkeypatch.setattr(_http, "fetch_url", _fetch_raising(_http.HTTPClientError("robots", status=404)))
    assert http_fetch.check_robots_txt("https://example.com/page") == (True, None)


def test_a_disallow_rule_still_refuses(monkeypatch):
    monkeypatch.setattr(_http, "fetch_url", lambda *a, **k: _Resp("User-agent: *\nDisallow: /private\n"))
    allowed, reason = http_fetch.check_robots_txt("https://example.com/private/x")
    assert allowed is False and "disallowed" in (reason or "")


def test_the_policy_summary_says_an_unreadable_robots_txt_refuses():
    assert "cannot be read" in ia.InternetAccessPolicy().summary()


# ── #968 ────────────────────────────────────────────────────────────────────────────────────────────


def test_a_deviation_character_entry_blocks_the_idna_2008_host():
    assert ia.normalize_host("straße.de") == "xn--strae-oqa.de"
    policy = ia.InternetAccessPolicy(block_domains=frozenset({"straße.de"}))
    assert policy.domain_refusal("xn--strae-oqa.de") is not None


def test_idna_is_a_declared_dependency():
    import tomllib

    deps = tomllib.loads((Path(__file__).resolve().parents[2] / "pyproject.toml").read_text())["project"][
        "dependencies"
    ]
    assert any(d.split(">")[0].split("=")[0].split("<")[0].strip() == "idna" for d in deps)


# ── #967 review folds (owner decisions 2026-10-14: follow redirects; 401/403 allow) ─────────────────


class _R:
    def __init__(self, status: int = 200, text: str = "", headers: dict | None = None) -> None:
        self.status, self.text, self.headers = status, text, headers or {}


def _serve(monkeypatch, routes: dict):
    """A fake fetch_url serving ``routes[url]``: a _R, or an exception to raise. Records the URLs asked."""
    asked: list[str] = []

    def fetch(url, **kwargs):
        asked.append(url)
        r = routes[url]
        if isinstance(r, Exception):
            raise r
        return r

    monkeypatch.setattr(_http, "fetch_url", fetch)
    return asked


def test_a_429_is_unreadable_not_missing(monkeypatch):
    _serve(monkeypatch, {"https://example.com/robots.txt": _http.HTTPRateLimited("robots", status=429)})
    allowed, reason = http_fetch.check_robots_txt("https://example.com/page")
    assert allowed is False and "could not be read" in (reason or "")


@pytest.mark.parametrize("status", [401, 403, 404, 410])
def test_any_other_4xx_means_no_restrictions(monkeypatch, status):
    _serve(monkeypatch, {"https://example.com/robots.txt": _http.HTTPClientError("robots", status=status)})
    assert http_fetch.check_robots_txt("https://example.com/page") == (True, None)


def test_the_robots_txt_is_the_urls_own_origin(monkeypatch):
    asked = _serve(
        monkeypatch,
        {
            "http://example.com:8080/robots.txt": _R(text="User-agent: *\nDisallow: /x\n"),
            "http://[2001:db8::1]/robots.txt": _R(text=""),
        },
    )
    assert http_fetch.check_robots_txt("http://user:pw@example.com:8080/x/y")[0] is False
    assert http_fetch.check_robots_txt("http://[2001:db8::1]/page") == (True, None)
    assert asked == ["http://example.com:8080/robots.txt", "http://[2001:db8::1]/robots.txt"]


def test_a_redirect_is_followed_and_its_target_rules_apply(monkeypatch):
    _serve(
        monkeypatch,
        {
            "https://example.com/robots.txt": _R(
                status=301, headers={"Location": "https://www.example.com/robots.txt"}
            ),
            "https://www.example.com/robots.txt": _R(text="User-agent: *\nDisallow: /private\n"),
        },
    )
    allowed, reason = http_fetch.check_robots_txt("https://example.com/private/x")
    assert allowed is False and "disallowed" in (reason or "")


def test_too_many_redirects_or_no_location_is_unreadable(monkeypatch):
    loop = {f"https://a.example/{i}": _R(status=302, headers={"location": f"/{i + 1}"}) for i in range(10)}
    loop["https://a.example/robots.txt"] = _R(status=302, headers={"location": "/0"})
    loop["https://b.example/robots.txt"] = _R(status=301)
    _serve(monkeypatch, loop)
    assert http_fetch.check_robots_txt("https://a.example/page")[0] is False
    assert http_fetch.check_robots_txt("https://b.example/page")[0] is False


def test_an_idna_2003_only_entry_names_the_punycode_form():
    with pytest.raises(ValueError, match="punycode"):
        ia.normalize_host("☃.net")
