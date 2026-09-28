"""#832 -- the internet policy reaches search, the builder owns the getter, the recorded state is
true, and the shared policy is frozen.

Follow-ups from the #822 review (items 1, 3, 4, 5; item 2 waits for #922):

1. `internet_search` read only `policy.enabled` and the timeout: its result filter used the tool's own
   list and its rate limit a constructor default, so the operator's `allow_domains` / `block_domains`
   and `max_pages_per_minute` never applied to search.
3. `cli.py` recorded the launch CAP as `state.data["internet_access"]`, not the effective state, and
   the robot runtime recorded nothing.
4. `build_tool_registry(internet_policy_getter=...)` let a caller hand in any callable. The builder
   now takes a required `internet_launch_enabled` and builds the live getter itself.
5. The cached `InternetAccessPolicy` was one mutable instance shared by every tool. It is frozen, holds
   only the operator's rules (on/off is the toggle's, composed at read time by
   `EffectiveInternetPolicy`), and its shape was reviewed first (owner decisions 2026-09-28): three
   fields nothing read are retired, the stale private domain copies are gone, and an unknown key in
   the file fails closed.

Every test uses explicit tmp policy/access files, never the user's state.
"""

from __future__ import annotations

import inspect
import socket
from pathlib import Path

import pytest

from maxim.tools import internet_search as search_mod
from maxim.tools.internet_search import InternetSearchTool
from maxim.utils import internet_access as ia

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _fresh_policy_cache():
    ia._reset_policy_cache()
    yield
    ia._reset_policy_cache()


@pytest.fixture
def paths(tmp_path: Path, monkeypatch) -> tuple[Path, Path]:
    """Point the DEFAULT policy/access files at tmp, so the builder's own getter reads them."""
    policy_path, access_path = tmp_path / "policy.json", tmp_path / "access.json"
    monkeypatch.setattr(ia, "_default_policy_path", lambda: policy_path)
    monkeypatch.setattr(ia, "_default_internet_access_path", lambda: access_path)
    return policy_path, access_path


def _write_policy(path: Path, **fields) -> None:
    ia.save_internet_policy(ia.InternetAccessPolicy(**fields), path)


def _results(*urls: str) -> list[dict[str, str]]:
    return [search_mod._make_search_result(f"title {u}", u, "snippet") for u in urls]


@pytest.fixture
def fake_search(monkeypatch):
    """The provider returns fixed results; DNS is forbidden, so a filter that resolved would fail."""
    returned: dict[str, list] = {"results": []}

    def _search(query, max_results, timeout_s):
        return list(returned["results"]), None

    def _no_dns(*args, **kwargs):
        raise AssertionError("the search result filter must not resolve DNS")

    monkeypatch.setattr(search_mod, "_search_duckduckgo", _search)
    monkeypatch.setattr(socket, "getaddrinfo", _no_dns)
    monkeypatch.setattr(ia, "_cached_getaddrinfo", _no_dns)
    return returned


def _urls(result) -> list[str]:
    return [r["url"] for r in (result.output or [])]


# ── item 1: search obeys the policy ─────────────────────────────────────────────────────────────


def test_search_drops_results_from_the_policys_block_list(paths, fake_search) -> None:
    policy_path, access_path = paths
    _write_policy(policy_path, block_domains={"blocked.example"})
    getter = ia.live_internet_policy_getter(True, policy_path=policy_path, access_path=access_path)
    fake_search["results"] = _results(
        "https://blocked.example/a", "https://sub.blocked.example/b", "https://ok.example/c"
    )

    result = InternetSearchTool(get_internet_policy=getter).execute(query="q")

    assert result.success is True
    assert _urls(result) == ["https://ok.example/c"]


def test_search_keeps_only_the_policys_allow_list(paths, fake_search) -> None:
    policy_path, access_path = paths
    _write_policy(policy_path, allow_domains={"good.example"})
    getter = ia.live_internet_policy_getter(True, policy_path=policy_path, access_path=access_path)
    fake_search["results"] = _results(
        "https://good.example/a", "https://docs.good.example/b", "https://other.example/c"
    )

    result = InternetSearchTool(get_internet_policy=getter).execute(query="q")

    assert _urls(result) == ["https://good.example/a", "https://docs.good.example/b"]


def test_control_without_lists_every_result_survives(paths, fake_search) -> None:
    """Anti-vacuity arm: the drops above are caused by the lists."""
    policy_path, access_path = paths
    _write_policy(policy_path)
    getter = ia.live_internet_policy_getter(True, policy_path=policy_path, access_path=access_path)
    fake_search["results"] = _results("https://blocked.example/a", "https://other.example/c")

    assert _urls(InternetSearchTool(get_internet_policy=getter).execute(query="q")) == [
        "https://blocked.example/a",
        "https://other.example/c",
    ]


def test_search_uses_the_policys_page_limit_and_follows_a_change(paths, fake_search) -> None:
    policy_path, access_path = paths
    _write_policy(policy_path, max_pages_per_minute=2)
    getter = ia.live_internet_policy_getter(True, policy_path=policy_path, access_path=access_path)
    fake_search["results"] = _results("https://ok.example/a")
    tool = InternetSearchTool(get_internet_policy=getter)

    assert tool.execute(query="q").success is True
    assert tool.execute(query="q").success is True
    third = tool.execute(query="q")
    assert third.success is False and third.metadata.get("rate_limited") is True
    assert "2 requests/minute" in (third.error or "")

    # The operator raises it: the next call sees it, even within one mtime tick (the cache is keyed
    # on the file's identity, and every save replaces the file).
    _write_policy(policy_path, max_pages_per_minute=5)
    assert tool.execute(query="q").success is True


def test_the_search_tool_takes_no_limit_of_its_own() -> None:
    """One source for the page limit: the policy. A constructor default could only disagree with it."""
    assert "rate_limit_per_minute" not in inspect.signature(InternetSearchTool.__init__).parameters


def test_the_tools_own_default_block_actually_matches_a_host(paths, fake_search) -> None:
    """`BLOCKED_DOMAINS` named `grokipedia`, which no hostname equals or ends with `.grokipedia`."""
    policy_path, access_path = paths
    _write_policy(policy_path)
    getter = ia.live_internet_policy_getter(True, policy_path=policy_path, access_path=access_path)
    fake_search["results"] = _results("https://grokipedia.com/page/X", "https://ok.example/c")

    assert _urls(InternetSearchTool(get_internet_policy=getter).execute(query="q")) == ["https://ok.example/c"]


def test_the_policys_domain_check_is_shared_and_needs_no_dns(monkeypatch) -> None:
    """`can_access` and the search filter read the same list logic."""

    def _no_dns(*args, **kwargs):
        raise AssertionError("domain_refusal must not resolve DNS")

    monkeypatch.setattr(socket, "getaddrinfo", _no_dns)
    monkeypatch.setattr(ia, "_cached_getaddrinfo", _no_dns)
    policy = ia.InternetAccessPolicy(block_domains={"Bad.Example"}, allow_domains={"bad.example", "ok.example"})
    assert "blocked" in (policy.domain_refusal("sub.bad.example") or "")
    assert "not in allow list" in (policy.domain_refusal("else.example") or "")
    assert policy.domain_refusal("OK.example") is None
    assert policy.domain_refusal("notbad.example") is not None  # not a suffix match on a label boundary


# ── item 4: the builder owns the getter ─────────────────────────────────────────────────────────


def test_the_builder_requires_an_explicit_internet_decision() -> None:
    from maxim.runtime.bootstrap import build_tool_registry

    params = inspect.signature(build_tool_registry).parameters
    assert "internet_policy_getter" not in params  # no caller can hand in a bare callable
    flag = params["internet_launch_enabled"]
    assert flag.kind is inspect.Parameter.KEYWORD_ONLY and flag.default is inspect.Parameter.empty
    with pytest.raises(TypeError):
        build_tool_registry()  # forgetting the decision is an error, not "no internet"


def test_without_the_launch_grant_no_internet_tool_is_registered(paths) -> None:
    from maxim.runtime.bootstrap import build_tool_registry

    names = set(build_tool_registry(internet_launch_enabled=False).list())
    assert not names & {"internet_search", "http_fetch", "internet_access_toggle"}


def test_the_builders_tools_read_the_live_persisted_policy(paths, fake_search) -> None:
    """The getter the builder builds is the live one: the persisted toggle reaches the tool."""
    from maxim.runtime.bootstrap import build_tool_registry

    policy_path, access_path = paths
    _write_policy(policy_path, block_domains={"blocked.example"})
    fake_search["results"] = _results("https://blocked.example/a", "https://ok.example/c")
    search = build_tool_registry(internet_launch_enabled=True).get("internet_search")

    assert _urls(search.execute(query="q")) == ["https://ok.example/c"]  # the operator's list applies
    ia.set_internet_access(False, source="tool", path=access_path)
    refused = search.execute(query="q")
    assert refused.success is False and refused.metadata.get("policy_blocked") is True


# ── item 3: the recorded state is the effective state ──────────────────────────────────────────


def test_the_effective_state_is_the_cap_and_the_live_policy(paths) -> None:
    policy_path, access_path = paths
    _write_policy(policy_path)
    assert ia.effective_internet_enabled(True) is True
    assert ia.effective_internet_enabled(False) is False  # the launch cap
    ia.set_internet_access(False, source="tool", path=access_path)
    assert ia.effective_internet_enabled(True) is False  # the persisted toggle
    ia.set_internet_access(True, source="tool", path=access_path)
    policy_path.write_text("{not json")
    assert ia.effective_internet_enabled(True) is False  # an unreadable policy fails closed


@pytest.mark.parametrize("rel", ["src/maxim/cli.py", "src/maxim/embodied_runtime/agentic_runtime.py"])
def test_both_runtimes_record_the_effective_state(rel: str) -> None:
    source = (REPO / rel).read_text()
    assert 'state.data["internet_access"] = effective_internet_enabled(' in source
    assert "internet_launch_enabled=" in source


def test_the_loop_reads_the_recorded_state_with_one_default() -> None:
    """The reads defaulted to True in two places and False in a third."""
    source = (REPO / "src/maxim/runtime/agent_loop.py").read_text()
    assert 'state.data.get("internet_access", True)' not in source
    assert source.count('state.data.get("internet_access", False)') == 3


# ── item 5: the shared policy is frozen, operator-only and strict ─────────────────────────────


def test_the_policy_is_frozen_and_hashable() -> None:
    import dataclasses

    policy = ia.InternetAccessPolicy(block_domains={"a.example"})
    with pytest.raises(dataclasses.FrozenInstanceError):
        policy.max_fetch_bytes = 1  # type: ignore[misc]
    assert isinstance(policy.block_domains, frozenset) and isinstance(policy.allow_domains, frozenset)
    assert hash(policy) == hash(ia.InternetAccessPolicy(block_domains={"A.example"}))


def test_on_off_is_not_the_policys_field_but_the_composed_views() -> None:
    assert "enabled" not in ia.InternetAccessPolicy.__dataclass_fields__
    view = ia.EffectiveInternetPolicy(policy=ia.InternetAccessPolicy(), enabled=False)
    assert view.can_access("https://example.com/") == (False, "Internet access is disabled")


def test_replace_changes_what_is_enforced() -> None:
    """The private lowercased copies were init fields, so `replace()` kept the OLD list enforced."""
    import dataclasses

    before = ia.InternetAccessPolicy(block_domains={"a.example"})
    after = dataclasses.replace(before, block_domains={"B.example"})
    assert after.domain_refusal("b.example") is not None
    assert after.domain_refusal("a.example") is None


@pytest.mark.parametrize(
    "fields",
    [
        {"block_domains": "evil.com"},  # a bare string would become single characters
        {"allow_domains": ["ok.example", 3]},
        {"block_domains": [""]},
        {"max_fetch_bytes": 0},
        {"max_fetch_bytes": True},
        {"max_pages_per_minute": 0},
        {"request_timeout_s": -1},
        {"require_robots_ok": "false"},  # a string is truthy: it must not silently mean True
    ],
)
def test_an_ill_typed_policy_is_refused(fields) -> None:
    with pytest.raises(ValueError):
        ia.InternetAccessPolicy(**fields)


def test_an_unknown_key_in_the_file_fails_closed(tmp_path: Path) -> None:
    """A typo such as `block_domain` used to be ignored, silently dropping the block list."""
    import json

    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps({"_format_version": "1.0", "block_domain": ["evil.example"]}))
    view = ia.load_internet_policy(policy_path, access_path=tmp_path / "access.json")
    assert view.enabled is False and view.source == "policy-unreadable"


def test_a_file_written_before_832_still_loads_with_a_warning(tmp_path: Path, caplog) -> None:
    """Every earlier save wrote the three retired keys; they warn, they do not disable internet."""
    import json
    import logging

    policy_path = tmp_path / "policy.json"
    old = {
        "_format_version": "1.0",
        "allow_domains": [],
        "block_domains": ["x.example"],
        "require_robots_ok": True,
        "block_paywalled": True,
        "allow_paywalled_with_credentials": False,
        "unsafe_content_checks": True,
        "max_fetch_bytes": 1000,
        "max_pages_per_minute": 10,
        "request_timeout_s": 8.0,
        "retention_seconds": 900,
        "citations_required": True,
        "block_private_ips": True,
    }
    policy_path.write_text(json.dumps(old))
    with caplog.at_level(logging.WARNING, logger="maxim.utils.internet_access"):
        view = ia.load_internet_policy(policy_path, access_path=tmp_path / "access.json")
    assert view.enabled is True and view.policy.max_fetch_bytes == 1000 and "x.example" in view.policy.block_domains
    assert any("retired" in r.getMessage() for r in caplog.records)


def test_the_saved_file_round_trips_and_carries_no_retired_key(tmp_path: Path) -> None:
    import json

    policy_path = tmp_path / "policy.json"
    original = ia.InternetAccessPolicy(block_domains={"x.example"}, max_pages_per_minute=3)
    ia.save_internet_policy(original, policy_path)
    data = json.loads(policy_path.read_text())
    assert not set(data) & set(ia._RETIRED_POLICY_KEYS)
    assert ia.load_internet_policy(policy_path, access_path=tmp_path / "access.json").policy == original


# ── review folds: edge cases on the real paths ─────────────────────────────────────────────────


def test_one_unparseable_result_url_does_not_fail_the_search(paths, fake_search) -> None:
    policy_path, access_path = paths
    _write_policy(policy_path)
    getter = ia.live_internet_policy_getter(True, policy_path=policy_path, access_path=access_path)
    fake_search["results"] = _results("http://[::1", "https://ok.example/c")

    result = InternetSearchTool(get_internet_policy=getter).execute(query="q")

    assert result.success is True and _urls(result) == ["https://ok.example/c"]


@pytest.mark.parametrize("url", ["https://grokipedia.com./x", "https://GrokiPedia.com/x", "https://a.grokipedia.com/x"])
def test_the_tools_own_block_uses_the_same_host_normaliser(paths, fake_search, url) -> None:
    policy_path, access_path = paths
    _write_policy(policy_path)
    getter = ia.live_internet_policy_getter(True, policy_path=policy_path, access_path=access_path)
    fake_search["results"] = _results(url, "https://ok.example/c")

    assert _urls(InternetSearchTool(get_internet_policy=getter).execute(query="q")) == ["https://ok.example/c"]


def test_list_entries_are_normalised_like_hosts() -> None:
    """An entry `evil.com.` or a Unicode name must match the host as it arrives on the wire."""
    policy = ia.InternetAccessPolicy(block_domains={"evil.com.", "bücher.de"})
    assert policy.domain_refusal("evil.com") is not None
    assert policy.domain_refusal("xn--bcher-kva.de") is not None  # punycode, as in a URL
    assert policy.domain_refusal("bücher.de") is not None


def test_a_whole_number_float_limit_is_accepted() -> None:
    assert ia.InternetAccessPolicy(max_pages_per_minute=10.0).max_pages_per_minute == 10
    with pytest.raises(ValueError):
        ia.InternetAccessPolicy(max_pages_per_minute=2.5)


def test_an_unreadable_policy_says_so_in_the_refusal(tmp_path: Path) -> None:
    policy_path = tmp_path / "policy.json"
    policy_path.write_text("{not json")
    view = ia.load_internet_policy(policy_path, access_path=tmp_path / "access.json")
    allowed, reason = view.can_access("https://example.com/")
    assert allowed is False and "unreadable" in (reason or "")


def test_an_unreadable_state_directory_fails_closed_instead_of_crashing(tmp_path: Path) -> None:
    """A permission error on the state directory must disable internet, not crash the registry build."""
    import os

    locked = tmp_path / "util"
    locked.mkdir()
    policy_path, access_path = locked / "policy.json", locked / "access.json"
    _write_policy(policy_path)
    os.chmod(locked, 0)
    try:
        if os.access(policy_path, os.R_OK):
            pytest.skip("running with privileges that ignore directory permissions")
        view = ia.load_internet_policy(policy_path, access_path=access_path)
        assert view.enabled is False
    finally:
        os.chmod(locked, 0o700)


# ── the builder is the only place the internet tools get a policy (structural, AST) ───────────

_BUILDER = "maxim/runtime/bootstrap.py"
_LOADER = "maxim/utils/internet_access.py"


def _internet_wiring_violations(source: str, rel: str) -> list[str]:
    """Where ``rel`` builds an internet tool, a getter or a view outside the one place allowed."""
    import ast

    out: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and rel != _BUILDER:
            wired = {"live_internet_policy_getter", "InternetSearchTool", "HttpFetchTool"}
            for alias in node.names:
                if alias.name in wired:  # any import, aliased or not: only the builder wires them
                    out.append(f"{rel}:{node.lineno} imports {alias.name}")
        if isinstance(node, ast.Attribute) and node.attr in ("InternetSearchTool", "HttpFetchTool"):
            if rel != _BUILDER:  # `T = http_fetch.HttpFetchTool; T(g)`
                out.append(f"{rel}:{node.lineno} references {node.attr}")
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else ""
        if name in ("InternetSearchTool", "HttpFetchTool", "live_internet_policy_getter") and rel != _BUILDER:
            out.append(f"{rel}:{node.lineno} calls {name}")
        if name == "EffectiveInternetPolicy" and rel != _LOADER:
            out.append(f"{rel}:{node.lineno} builds an EffectiveInternetPolicy")
        if any(kw.arg == "get_internet_policy" for kw in node.keywords) and rel != _BUILDER:
            out.append(f"{rel}:{node.lineno} passes get_internet_policy=")
    return out


def test_only_the_builder_wires_the_internet_tools() -> None:
    src = REPO / "src"
    violations: list[str] = []
    for path in sorted((src / "maxim").rglob("*.py")):
        rel = path.relative_to(src).as_posix()
        violations += _internet_wiring_violations(path.read_text(), rel)
    assert violations == []


@pytest.mark.parametrize(
    "planted",
    [
        "from maxim.utils.internet_access import live_internet_policy_getter as g\ng(True)\n",
        "from maxim.tools.http_fetch import HttpFetchTool\nHttpFetchTool(get_internet_policy=lambda: None)\n",
        "import maxim.tools.internet_search as s\ns.InternetSearchTool()\n",
        "from maxim.utils import internet_access as ia\nia.EffectiveInternetPolicy(policy=None, enabled=True)\n",
        "from maxim.tools.http_fetch import HttpFetchTool as H\nH(lambda: None)\n",
        "from maxim.tools import http_fetch\nT = http_fetch.HttpFetchTool\nT(lambda: None)\n",
    ],
)
def test_the_wiring_guard_catches_a_bypass(planted: str) -> None:
    """Anti-vacuity: each bypass shape, planted in some other module, is reported."""
    assert _internet_wiring_violations(planted, "maxim/cli.py")


def test_a_corrupt_toggle_file_fails_closed(tmp_path: Path) -> None:
    """A corrupted persisted "off" used to fall back to the default, which is on."""
    policy_path, access_path = tmp_path / "policy.json", tmp_path / "access.json"
    _write_policy(policy_path)
    access_path.write_text("{not json")
    assert ia.load_internet_policy(policy_path, access_path=access_path).enabled is False
    access_path.unlink()
    assert ia.load_internet_policy(policy_path, access_path=access_path).enabled is True  # no file: default on


def test_a_toggle_file_must_hold_a_real_boolean(tmp_path: Path) -> None:
    """`"enabled": "false"` is a truthy string; reading it as ON would defeat the fail-closed load."""
    import json

    policy_path, access_path = tmp_path / "policy.json", tmp_path / "access.json"
    _write_policy(policy_path)
    access_path.write_text(json.dumps({"_format_version": "1.0", "enabled": "false"}))
    view = ia.load_internet_policy(policy_path, access_path=access_path)
    assert view.enabled is False and "toggle file" in (view.can_access("https://example.com/")[1] or "")


def test_a_wildcard_entry_is_refused_rather_than_matching_nothing() -> None:
    with pytest.raises(ValueError, match="wildcard"):
        ia.InternetAccessPolicy(block_domains={"*.example.com"})
