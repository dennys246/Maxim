"""Regression guards for C7a of config_unification.md.

Solo + cloud-key auto-detect: when role resolves to ``solo`` AND no
local LLM profile is set AND a cloud API key env var is present,
implicitly enable the cloud-LLM gates so ``maxim`` "just works" with
bare-API-key configuration.

Architectural framing pinned in config_unification.md C7a discussion:
role and LLM-source are independent axes. ``solo`` means "alone, not
in a mesh" — it does NOT imply "local LLM". The auto-detect just
removes the seven-flag incantation operators previously needed for
cloud-only setup.

C7b (cloud-backed leader serving peers) is deferred until
``mesh_usage_accounting.md`` ships, per the user-discussion fold.
"""

from __future__ import annotations

import logging
import os

import pytest

from maxim.cli_utils import (
    _CLOUD_API_KEY_TO_PROFILE,
    configure_cloud_solo_auto_detect,
)


@pytest.fixture
def clean_env(monkeypatch):
    """Strip every cloud-API-key + MAXIM_LLM_* + MAXIM_LANE_* + MAXIM_ROLE
    env var so each test starts from a known state."""
    for env_name, _ in _CLOUD_API_KEY_TO_PROFILE:
        monkeypatch.delenv(env_name, raising=False)
    for name in (
        "MAXIM_ROLE",
        "MAXIM_LLM_ENABLED",
        "MAXIM_LLM_CLOUD_ENABLED",
        "MAXIM_LLM_PROFILE",
        "MAXIM_LLM_REDACTION_POLICY",
        "MAXIM_MAX_CLOUD_LANES",
        "MAXIM_CLOUD_SESSION_BUDGET",
        "MAXIM_LANE_LARGE_REMOTE_URL",
    ):
        monkeypatch.delenv(name, raising=False)
    yield


# ─────────────────────────────────────────────────────────────────────────────
# Happy path — auto-detect fires for solo + cloud key
# ─────────────────────────────────────────────────────────────────────────────


class TestAutoDetectHappyPath:
    def test_anthropic_key_alone_picks_claude_sonnet(self, clean_env, monkeypatch, caplog):
        monkeypatch.setenv("MAXIM_ROLE", "solo")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        with caplog.at_level(logging.INFO):
            configure_cloud_solo_auto_detect(logging.getLogger("test"))

        assert os.environ["MAXIM_LLM_ENABLED"] == "1"
        assert os.environ["MAXIM_LLM_CLOUD_ENABLED"] == "1"
        assert os.environ["MAXIM_LLM_PROFILE"] == "claude-sonnet"
        assert os.environ["MAXIM_LLM_REDACTION_POLICY"] == "standard"
        assert os.environ["MAXIM_CLOUD_SESSION_BUDGET"] == "5.0"
        assert os.environ["MAXIM_MAX_CLOUD_LANES"] == "1"
        # Every implicit-set logged INFO
        infos = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
        assert any("auto-detect" in m for m in infos)

    def test_openai_key_alone_picks_gpt_4o(self, clean_env, monkeypatch):
        monkeypatch.setenv("MAXIM_ROLE", "solo")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-test")
        configure_cloud_solo_auto_detect(logging.getLogger("test"))
        assert os.environ["MAXIM_LLM_PROFILE"] == "gpt-4o"

    def test_groq_key_alone_picks_groq_profile(self, clean_env, monkeypatch):
        monkeypatch.setenv("MAXIM_ROLE", "solo")
        monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
        configure_cloud_solo_auto_detect(logging.getLogger("test"))
        assert os.environ["MAXIM_LLM_PROFILE"] == "groq-llama3-70b"

    def test_anthropic_wins_over_openai_when_both_present(self, clean_env, monkeypatch):
        """Priority order pinned per the bundled cloud profile catalog —
        Anthropic first, then OpenAI."""
        monkeypatch.setenv("MAXIM_ROLE", "solo")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-test")
        configure_cloud_solo_auto_detect(logging.getLogger("test"))
        assert os.environ["MAXIM_LLM_PROFILE"] == "claude-sonnet"
        # max_lanes counts present keys (capped at 3)
        assert os.environ["MAXIM_MAX_CLOUD_LANES"] == "2"

    def test_three_keys_caps_lanes_at_3(self, clean_env, monkeypatch):
        monkeypatch.setenv("MAXIM_ROLE", "solo")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
        monkeypatch.setenv("OPENAI_API_KEY", "k")
        monkeypatch.setenv("GOOGLE_API_KEY", "k")
        monkeypatch.setenv("GROQ_API_KEY", "k")
        configure_cloud_solo_auto_detect(logging.getLogger("test"))
        assert os.environ["MAXIM_MAX_CLOUD_LANES"] == "3"  # capped


# ─────────────────────────────────────────────────────────────────────────────
# Gating — auto-detect must NOT fire in these cases
# ─────────────────────────────────────────────────────────────────────────────


class TestAutoDetectGates:
    def test_leader_role_no_fire(self, clean_env, monkeypatch):
        """C7b (cloud-backed leader serving peers) is deferred —
        leader role must NOT trigger cloud auto-config."""
        monkeypatch.setenv("MAXIM_ROLE", "leader")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        configure_cloud_solo_auto_detect(logging.getLogger("test"))
        assert "MAXIM_LLM_PROFILE" not in os.environ
        assert "MAXIM_LLM_CLOUD_ENABLED" not in os.environ

    def test_peer_role_no_fire(self, clean_env, monkeypatch):
        monkeypatch.setenv("MAXIM_ROLE", "peer")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        configure_cloud_solo_auto_detect(logging.getLogger("test"))
        assert "MAXIM_LLM_PROFILE" not in os.environ

    def test_no_cloud_keys_no_fire(self, clean_env, monkeypatch):
        monkeypatch.setenv("MAXIM_ROLE", "solo")
        configure_cloud_solo_auto_detect(logging.getLogger("test"))
        assert "MAXIM_LLM_PROFILE" not in os.environ
        assert "MAXIM_LLM_CLOUD_ENABLED" not in os.environ

    def test_existing_llm_profile_preserved(self, clean_env, monkeypatch):
        """Operator already picked a model — auto-detect must respect it."""
        monkeypatch.setenv("MAXIM_ROLE", "solo")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        monkeypatch.setenv("MAXIM_LLM_PROFILE", "qwen-32b")
        configure_cloud_solo_auto_detect(logging.getLogger("test"))
        # Profile NOT overwritten
        assert os.environ["MAXIM_LLM_PROFILE"] == "qwen-32b"
        # Other cloud gates also skipped (the auto-detect early-returns)
        assert "MAXIM_LLM_CLOUD_ENABLED" not in os.environ

    def test_peer_routing_url_set_no_fire(self, clean_env, monkeypatch):
        """Routing to a peer leader — auto-detect skips so the peer
        flow takes precedence over cloud opt-in."""
        monkeypatch.setenv("MAXIM_ROLE", "solo")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        monkeypatch.setenv("MAXIM_LANE_LARGE_REMOTE_URL", "http://leader/v1")
        configure_cloud_solo_auto_detect(logging.getLogger("test"))
        assert "MAXIM_LLM_PROFILE" not in os.environ
        assert "MAXIM_LLM_CLOUD_ENABLED" not in os.environ

    def test_empty_role_treated_as_solo(self, clean_env, monkeypatch):
        """MAXIM_ROLE empty/unset — auto-detect proceeds because
        the no-role case behaves like solo for the operator (the
        normal startup path runs detect_and_apply_role BEFORE this
        function, so role is usually set; this case covers test
        callers that bypass detection)."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        configure_cloud_solo_auto_detect(logging.getLogger("test"))
        # Did fire — profile got set
        assert os.environ["MAXIM_LLM_PROFILE"] == "claude-sonnet"


# ─────────────────────────────────────────────────────────────────────────────
# Idempotency — operator overrides are respected for every implicit set
# ─────────────────────────────────────────────────────────────────────────────


class TestAutoDetectIdempotency:
    def test_existing_cloud_enabled_preserved(self, clean_env, monkeypatch):
        monkeypatch.setenv("MAXIM_ROLE", "solo")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        monkeypatch.setenv("MAXIM_LLM_CLOUD_ENABLED", "0")  # operator opted out
        configure_cloud_solo_auto_detect(logging.getLogger("test"))
        # Auto-detect respects the explicit opt-out
        assert os.environ["MAXIM_LLM_CLOUD_ENABLED"] == "0"

    def test_existing_session_budget_preserved(self, clean_env, monkeypatch):
        monkeypatch.setenv("MAXIM_ROLE", "solo")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        monkeypatch.setenv("MAXIM_CLOUD_SESSION_BUDGET", "10.00")
        configure_cloud_solo_auto_detect(logging.getLogger("test"))
        assert os.environ["MAXIM_CLOUD_SESSION_BUDGET"] == "10.00"

    def test_existing_max_cloud_lanes_preserved(self, clean_env, monkeypatch):
        monkeypatch.setenv("MAXIM_ROLE", "solo")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        monkeypatch.setenv("MAXIM_MAX_CLOUD_LANES", "5")
        configure_cloud_solo_auto_detect(logging.getLogger("test"))
        assert os.environ["MAXIM_MAX_CLOUD_LANES"] == "5"


# ─────────────────────────────────────────────────────────────────────────────
# Profile catalog consistency
# ─────────────────────────────────────────────────────────────────────────────


class TestCatalogConsistency:
    def test_anthropic_first_in_priority_order(self):
        """Anthropic should be the highest-priority default since
        Claude is currently the strongest cloud model for the agent-
        loop workloads Maxim runs."""
        assert _CLOUD_API_KEY_TO_PROFILE[0] == (
            "ANTHROPIC_API_KEY",
            "claude-sonnet",
        )

    def test_priority_table_matches_configuration_md_documentation(self):
        """The priority table is documented in docs/user/
        configuration.md — keep them in sync. The pair list should
        include exactly the eight cloud providers Maxim ships
        bundled profiles for."""
        env_names = {pair[0] for pair in _CLOUD_API_KEY_TO_PROFILE}
        expected = {
            "ANTHROPIC_API_KEY",
            "OPENAI_API_KEY",
            "GOOGLE_API_KEY",
            "GROQ_API_KEY",
            "TOGETHER_API_KEY",
            "FIREWORKS_API_KEY",
            "MISTRAL_API_KEY",
            "DEEPSEEK_API_KEY",
        }
        assert env_names == expected
