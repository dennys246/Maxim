"""#822 — the persisted internet policy and the access toggle must reach the live tools.

Before the fix both runtimes built a bare `InternetAccessPolicy(enabled=...)`, so the domain lists
and limits in `util/internet_policy.json` never applied and `internet_access_toggle` wrote a file
nothing read. These tests drive the real `HttpFetchTool` through the shared getter, with explicit
tmp paths (never the user's state), and a blocked URL is refused before any network call.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from maxim.tools.http_fetch import HttpFetchTool
from maxim.utils import internet_access as ia

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _fresh_policy_cache():
    ia._cached_policy = None
    ia._cached_policy_path = None
    yield
    ia._cached_policy = None
    ia._cached_policy_path = None


def _write_policy(path: Path, **fields) -> None:
    ia.save_internet_policy(ia.InternetAccessPolicy(enabled=True, **fields), path)


@pytest.fixture
def _public_dns(monkeypatch):
    """Every host resolves as public, so the DOMAIN LISTS decide -- not a DNS failure.

    Without this, an unresolvable test domain is treated as private and refused for the wrong
    reason, and a block-list test passes with no block list at all (#822 review blocker).
    """
    monkeypatch.setattr(ia.InternetAccessPolicy, "_is_private_ip", lambda self, host: False)


def test_block_list_from_the_policy_file_is_enforced_by_the_live_tool(tmp_path: Path, _public_dns) -> None:
    policy_path, access_path = tmp_path / "policy.json", tmp_path / "access.json"
    _write_policy(policy_path, block_domains={"blocked.example"})
    getter = ia.live_internet_policy_getter(True, policy_path=policy_path, access_path=access_path)

    result = HttpFetchTool(get_internet_policy=getter).execute(url="https://blocked.example/page")

    assert result.success is False
    assert result.metadata.get("policy_blocked") is True
    assert "blocked.example' is blocked" in (result.error or "")


def test_control_without_a_block_list_the_same_url_is_allowed(tmp_path: Path, _public_dns) -> None:
    """Anti-vacuity arm: the refusal above is caused by the list, nothing else."""
    policy_path, access_path = tmp_path / "policy.json", tmp_path / "access.json"
    _write_policy(policy_path)
    getter = ia.live_internet_policy_getter(True, policy_path=policy_path, access_path=access_path)
    assert getter().can_access("https://blocked.example/page") == (True, None)


def test_allow_list_refuses_an_unlisted_domain(tmp_path: Path, _public_dns) -> None:
    policy_path, access_path = tmp_path / "policy.json", tmp_path / "access.json"
    _write_policy(policy_path, allow_domains={"good.example"})
    policy = ia.live_internet_policy_getter(True, policy_path=policy_path, access_path=access_path)()
    assert policy.can_access("https://good.example/") == (True, None)
    allowed, reason = policy.can_access("https://other.example/")
    assert allowed is False and "not in allow list" in reason


def test_a_corrupt_policy_file_fails_closed(tmp_path: Path) -> None:
    policy_path = tmp_path / "policy.json"
    policy_path.write_text("{not json")
    policy = ia.load_internet_policy(policy_path, access_path=tmp_path / "access.json")
    assert policy.enabled is False


def test_an_ill_typed_domain_list_fails_closed_instead_of_splitting_characters(tmp_path: Path) -> None:
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps({"_format_version": "1.0", "block_domains": "evil.com"}))
    policy = ia.load_internet_policy(policy_path, access_path=tmp_path / "access.json")
    assert policy.enabled is False


def test_toggle_tool_requires_an_explicit_enabled() -> None:
    from maxim.tools.internet_search import InternetAccessTool

    calls: list[bool] = []
    tool = InternetAccessTool(set_internet_access_fn=lambda enabled, source: calls.append(enabled))
    assert tool.execute().success is False
    assert calls == []


def test_the_toggle_takes_effect_on_the_next_request(tmp_path: Path) -> None:
    policy_path, access_path = tmp_path / "policy.json", tmp_path / "access.json"
    _write_policy(policy_path)
    getter = ia.live_internet_policy_getter(True, policy_path=policy_path, access_path=access_path)
    assert getter().enabled is True

    ia.set_internet_access(False, source="tool", path=access_path)

    assert getter().enabled is False
    result = HttpFetchTool(get_internet_policy=getter).execute(url="https://example.com/")
    assert result.success is False
    assert result.metadata.get("policy_blocked") is True


def test_launch_cap_means_no_getter_and_no_internet_tools() -> None:
    assert ia.live_internet_policy_getter(False) is None


def test_policy_cache_is_keyed_by_file(tmp_path: Path) -> None:
    a, b, access = tmp_path / "a.json", tmp_path / "b.json", tmp_path / "access.json"
    _write_policy(a, block_domains={"a.example"})
    _write_policy(b, block_domains={"b.example"})
    assert "a.example" in ia.load_internet_policy(a, access_path=access).block_domains
    # Same mtime second is likely; the cache must still not hand back a's policy for b.
    assert "b.example" in ia.load_internet_policy(b, access_path=access).block_domains


def test_saved_policy_carries_format_version_and_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "policy.json"
    _write_policy(path, block_domains={"x.example"}, max_fetch_bytes=1234)
    data = json.loads(path.read_text())
    assert data["_format_version"] == "1.0"
    assert "enabled" not in data  # the access toggle owns it
    loaded = ia.load_internet_policy(path, access_path=tmp_path / "access.json")
    assert loaded.max_fetch_bytes == 1234 and "x.example" in loaded.block_domains


@pytest.mark.parametrize("rel", ["src/maxim/cli.py", "src/maxim/embodied_runtime/agentic_runtime.py"])
def test_runtimes_use_the_shared_getter_not_a_bare_policy(rel: str) -> None:
    """The composition guard: both runtimes route through `live_internet_policy_getter`."""
    source = (REPO / rel).read_text()
    assert "InternetAccessPolicy(enabled=" not in source
    assert "live_internet_policy_getter(" in source
