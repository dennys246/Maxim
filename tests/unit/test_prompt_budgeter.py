"""Unit tests for PromptBudgeter, truncation helpers, and ReasoningCarryover."""

from __future__ import annotations

import threading


from maxim.agents.prompt_budgeter import (
    PromptBudgeter,
    SectionPriority,
    _truncate_context_pool,
    _truncate_conversation,
    _truncate_reasoning_carryover,
)
from maxim.agents.llm_fallback import ReasoningCarryover, ReasoningEntry
from maxim.models.language.token_counter import (
    CharEstimateCounter,
    LlamaCppTokenCounter,
    TokenCounter,
)


# ── Helpers ─────────────────────────────────────────────────────────────────


class _ExactCounter:
    """Token counter where 1 word = 1 token (for deterministic testing)."""

    def count_tokens(self, text: str) -> int:
        return len(text.split())


class _MockLlama:
    """Mock llama model that returns fixed-size tokenization."""

    def tokenize(self, encoded: bytes) -> list[int]:
        text = encoded.decode("utf-8", errors="replace")
        # 1 token per 4 chars (same as CharEstimate but via tokenizer path)
        return list(range(max(1, len(text) // 4)))


class _BrokenLlama:
    """Mock llama model that always raises on tokenize."""

    def tokenize(self, encoded: bytes) -> list[int]:
        raise RuntimeError("Metal backend crashed")


def _make_budgeter(total: int, reserve: int = 0, counter=None) -> PromptBudgeter:
    return PromptBudgeter(
        total_budget=total,
        response_reserve=reserve,
        token_counter=counter or _ExactCounter(),
        template_overhead=0,
    )


# ── PromptBudgeter Tests ───────────────────────────────────────────────────


class TestPromptBudgeter:
    def test_all_sections_fit_large_budget(self):
        """With a large budget, nothing is dropped."""
        b = _make_budgeter(total=10000)
        b.add("mandatory", "Hello world", SectionPriority.MANDATORY)
        b.add("critical", "Tools list here", SectionPriority.CRITICAL)
        b.add("important", "Conversation history", SectionPriority.IMPORTANT)
        b.add("nice", "Foundational context", SectionPriority.NICE_TO_HAVE)

        text, dropped = b.build()
        assert dropped == []
        assert "Hello world" in text
        assert "Foundational context" in text

    def test_nice_to_have_dropped_when_over_budget(self):
        """NICE_TO_HAVE sections are dropped first under budget pressure."""
        b = _make_budgeter(total=6)  # 6 word budget
        b.add("mandatory", "Hello world", SectionPriority.MANDATORY)  # 2 words
        b.add("critical", "Tools list here", SectionPriority.CRITICAL)  # 3 words
        b.add("nice", "Foundational context extras", SectionPriority.NICE_TO_HAVE)  # 3 words

        text, dropped = b.build()
        assert "nice" in dropped
        assert "Hello world" in text
        assert "Tools list here" in text

    def test_mandatory_survives_extreme_pressure(self):
        """MANDATORY sections survive even with tiny budget."""
        b = _make_budgeter(total=3)  # Only 3 word budget
        b.add("mandatory", "User request here", SectionPriority.MANDATORY)  # 3 words
        b.add("critical", "identity info here", SectionPriority.CRITICAL)  # 3 words
        b.add("nice", "extra fluff", SectionPriority.NICE_TO_HAVE)  # 2 words

        text, dropped = b.build()
        assert "User request here" in text
        assert "critical" in dropped
        assert "nice" in dropped

    def test_truncation_before_drop(self):
        """Truncatable IMPORTANT sections are shortened before being dropped."""
        counter = _ExactCounter()

        def truncate_fn(content: str, max_tokens: int) -> str:
            words = content.split()
            # Keep only as many words as budget allows
            return " ".join(words[:max_tokens])

        b = _make_budgeter(total=8, counter=counter)
        b.add("mandatory", "Hello world", SectionPriority.MANDATORY)  # 2 words
        b.add(
            "important_big",
            "One two three four five six seven",  # 7 words
            SectionPriority.IMPORTANT,
            truncatable=True,
            min_tokens=2,
            truncate_fn=truncate_fn,
        )

        text, dropped = b.build()
        assert dropped == []
        assert "Hello world" in text
        # The truncated section should have fewer words
        assert "seven" not in text
        assert "One" in text

    def test_empty_sections_silently_ignored(self):
        """Adding empty content does not create a section."""
        b = _make_budgeter(total=100)
        b.add("empty1", "", SectionPriority.MANDATORY)
        b.add("empty2", "   ", SectionPriority.MANDATORY)
        b.add("real", "actual content", SectionPriority.MANDATORY)

        text, dropped = b.build()
        assert dropped == []
        assert text == "actual content"

    def test_insertion_order_preserved(self):
        """After budget cuts, surviving sections maintain their original order."""
        b = _make_budgeter(total=20)
        b.add("section_a", "AAA first", SectionPriority.MANDATORY)  # inserted first
        b.add("section_b", "BBB second", SectionPriority.NICE_TO_HAVE)  # inserted second
        b.add("section_c", "CCC third", SectionPriority.CRITICAL)  # inserted third
        b.add("section_d", "DDD fourth", SectionPriority.MANDATORY)  # inserted fourth

        text, dropped = b.build()
        # All should fit
        assert dropped == []
        # Check relative ordering in final text
        a_pos = text.index("AAA first")
        b_pos = text.index("BBB second")
        c_pos = text.index("CCC third")
        d_pos = text.index("DDD fourth")
        assert a_pos < b_pos < c_pos < d_pos

    def test_prompt_budget_property(self):
        """prompt_budget accounts for response reserve and template overhead."""
        b = PromptBudgeter(
            total_budget=4096,
            response_reserve=512,
            token_counter=_ExactCounter(),
            template_overhead=100,
        )
        assert b.prompt_budget == 4096 - 512 - 100

    def test_smollm_simulation(self):
        """Simulate SmolLM (2048 n_ctx) in active mode — NICE_TO_HAVE dropped."""
        counter = CharEstimateCounter()
        b = PromptBudgeter(
            total_budget=2048,
            response_reserve=512,
            token_counter=counter,
            template_overhead=100,
        )
        # Budget = 2048 - 512 - 100 = 1436 tokens

        # MANDATORY (~50 tokens)
        b.add("instructions", "Respond with JSON. " * 10, SectionPriority.MANDATORY)
        b.add("user_request", "What is the weather?", SectionPriority.MANDATORY)

        # CRITICAL (~200 tokens)
        b.add("identity", "You are Maxim. " * 50, SectionPriority.CRITICAL)
        b.add("tools", "tool_a: does stuff. " * 40, SectionPriority.CRITICAL)

        # IMPORTANT (~500 tokens)
        b.add(
            "conversation",
            "User said hello. Maxim replied. " * 50,
            SectionPriority.IMPORTANT,
            truncatable=True,
            min_tokens=30,
            truncate_fn=lambda c, m: _truncate_conversation(c, m, counter),
        )

        # NICE_TO_HAVE (~800 tokens) — should get dropped on small models
        b.add("foundational", "Constitution principles. " * 100, SectionPriority.NICE_TO_HAVE)
        b.add("agent_states", "Agent1: idle. " * 80, SectionPriority.NICE_TO_HAVE)

        text, dropped = b.build()
        # At least some NICE_TO_HAVE should be dropped
        assert any(name in dropped for name in ("foundational", "agent_states"))
        # MANDATORY sections must survive
        assert "What is the weather?" in text


# ── Token Counter Tests ─────────────────────────────────────────────────────


class TestTokenCounters:
    def test_char_estimate_counter(self):
        """CharEstimateCounter returns len(text) // 3 (conservative for structured text)."""
        c = CharEstimateCounter()
        assert c.count_tokens("hello world") == max(1, len("hello world") // 3)
        assert c.count_tokens("") == max(1, 0)  # min 1

    def test_llamacpp_counter_with_mock(self):
        """LlamaCppTokenCounter delegates to Llama.tokenize()."""
        mock_llm = _MockLlama()
        c = LlamaCppTokenCounter(mock_llm)
        result = c.count_tokens("hello world test")
        # _MockLlama returns len(text)//4 tokens (its own tokenizer logic)
        assert result == max(1, len("hello world test") // 4)

    def test_llamacpp_counter_fallback_on_error(self):
        """LlamaCppTokenCounter falls back to char estimate on error."""
        broken = _BrokenLlama()
        c = LlamaCppTokenCounter(broken)
        result = c.count_tokens("hello world test")
        # Should fall back to char estimate (len//3)
        assert result == max(1, len("hello world test") // 3)

    def test_token_counter_protocol(self):
        """CharEstimateCounter satisfies TokenCounter protocol."""
        c = CharEstimateCounter()
        assert isinstance(c, TokenCounter)


# ── Truncation Helper Tests ─────────────────────────────────────────────────


class TestTruncationHelpers:
    def test_truncate_conversation_drops_oldest(self):
        """_truncate_conversation drops oldest turns from front."""
        counter = _ExactCounter()
        content = (
            "User: Hello\n"
            "Maxim: Hi there\n"
            "User: What time is it?\n"
            "Maxim: It is noon\n"
            "User: Thanks\n"
            "Maxim: You're welcome"
        )
        result = _truncate_conversation(content, max_tokens=8, counter=counter)
        # Should have dropped earliest turns
        assert "Thanks" in result
        assert "You're welcome" in result

    def test_truncate_context_pool_drops_oldest(self):
        """_truncate_context_pool drops oldest lines from front."""
        counter = _ExactCounter()
        content = "line1 old\nline2 old\nline3 recent\nline4 newest"
        result = _truncate_context_pool(content, max_tokens=4, counter=counter)
        assert "newest" in result

    def test_truncate_reasoning_carryover_drops_oldest(self):
        """_truncate_reasoning_carryover drops oldest entries."""
        counter = _ExactCounter()
        content = (
            "=== Recent Decisions (Working Memory) ===\n"
            "- tool_a: old action [OK: result1]\n"
            "- tool_b: recent action [FAIL: error1]\n"
            "- tool_c: newest action [OK: result2]"
        )
        result = _truncate_reasoning_carryover(content, max_tokens=12, counter=counter)
        assert "Working Memory" in result
        assert "newest" in result


# ── ReasoningCarryover Tests ───────────────────────────────────────────────


class TestReasoningCarryover:
    def test_record_and_retrieve(self):
        """Basic record and get_prompt_text."""
        rc = ReasoningCarryover(max_entries=5)
        rc.record("internet_search", "User asked about weather", True, "Sunny 72F")
        text = rc.get_prompt_text()
        assert "Recent Decisions" in text
        assert "internet_search" in text
        assert "OK" in text

    def test_rolling_buffer_evicts_oldest(self):
        """Oldest entries are evicted past max_entries."""
        rc = ReasoningCarryover(max_entries=3)
        rc.record("tool_a", "reason_a", True, "result_a")
        rc.record("tool_b", "reason_b", False, "result_b")
        rc.record("tool_c", "reason_c", True, "result_c")
        rc.record("tool_d", "reason_d", True, "result_d")

        assert len(rc) == 3
        text = rc.get_prompt_text()
        assert "tool_a" not in text  # Evicted
        assert "tool_b" in text
        assert "tool_d" in text

    def test_empty_returns_empty_string(self):
        """Empty carryover returns empty string."""
        rc = ReasoningCarryover()
        assert rc.get_prompt_text() == ""

    def test_clear(self):
        """clear() wipes the buffer."""
        rc = ReasoningCarryover()
        rc.record("tool_a", "reason", True, "result")
        assert len(rc) == 1
        rc.clear()
        assert len(rc) == 0
        assert rc.get_prompt_text() == ""

    def test_prompt_line_formatting(self):
        """ReasoningEntry.to_prompt_line() produces expected format."""
        entry = ReasoningEntry(
            action_tool="internet_search",
            reasoning="User asked about weather",
            success=True,
            result_summary="Sunny 72F",
        )
        line = entry.to_prompt_line()
        assert line.startswith("- internet_search:")
        assert "OK" in line
        assert "Sunny 72F" in line

    def test_failed_entry_formatting(self):
        """Failed entries show FAIL status."""
        entry = ReasoningEntry(
            action_tool="write_file",
            reasoning="Saving code",
            success=False,
            result_summary="Permission denied",
        )
        line = entry.to_prompt_line()
        assert "FAIL" in line
        assert "Permission denied" in line

    def test_thread_safety(self):
        """Concurrent writes and reads don't corrupt state."""
        rc = ReasoningCarryover(max_entries=10)
        errors: list[str] = []

        def writer(thread_id: int):
            for i in range(50):
                try:
                    rc.record(f"tool_{thread_id}_{i}", f"reason_{i}", True, f"result_{i}")
                except Exception as e:
                    errors.append(f"write error: {e}")

        def reader():
            for _ in range(50):
                try:
                    rc.get_prompt_text()
                    _ = len(rc)
                except Exception as e:
                    errors.append(f"read error: {e}")

        threads = [
            threading.Thread(target=writer, args=(0,)),
            threading.Thread(target=writer, args=(1,)),
            threading.Thread(target=reader),
            threading.Thread(target=reader),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert errors == [], f"Thread safety errors: {errors}"
        assert len(rc) <= 10  # Never exceeds max


# ── Mode-Aware Manifest & Guidance Tests ──────────────────────────────────


class TestModeAwareManifest:
    def test_workspace_manifest_passive_mode(self, tmp_path):
        """Passive mode shows workspace files and CWD read-only context."""
        from maxim.agents.prompt_builder import build_workspace_manifest

        # Create a workspace inside tmp_path
        ws = tmp_path / ".maxim_workspace"
        ws.mkdir()
        (ws / "hello.py").write_text("print('hello')")
        # Add a CWD file so CWD context section appears
        (tmp_path / "app.py").write_text("import os")

        result = build_workspace_manifest(mode_name="passive", cwd=str(tmp_path))
        assert "EXISTING WORKSPACE" in result
        assert "hello.py" in result
        assert "proposed as plans" in result

    def test_workspace_manifest_active_mode(self, tmp_path):
        """Active mode shows workspace + CWD with 'requires approval' note."""
        from maxim.agents.prompt_builder import build_workspace_manifest

        ws = tmp_path / ".maxim_workspace"
        ws.mkdir()
        (ws / "draft.py").write_text("x = 1")
        (tmp_path / "main.py").write_text("print('main')")

        result = build_workspace_manifest(mode_name="active", cwd=str(tmp_path))
        assert "EXISTING WORKSPACE" in result
        assert "requires approval" in result

    def test_workspace_manifest_singularity_mode(self, tmp_path):
        """Singularity mode shows full CWD tree with full access language."""
        from maxim.agents.prompt_builder import build_workspace_manifest

        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text("print('main')")
        (tmp_path / "README.md").write_text("# Project")

        result = build_workspace_manifest(mode_name="singularity", cwd=str(tmp_path))
        assert "PROJECT DIRECTORY" in result
        assert "FULL read/write" in result
        assert ".maxim_workspace" not in result or "scratch" in result
        assert "read_file FIRST" in result

    def test_workspace_manifest_singularity_plan_hint(self, tmp_path):
        """Singularity with many files triggers PLAN FIRST hint."""
        from maxim.agents.prompt_builder import build_workspace_manifest

        for i in range(6):
            (tmp_path / f"file{i}.py").write_text(f"# file {i}")

        result = build_workspace_manifest(mode_name="singularity", cwd=str(tmp_path))
        assert "PLAN FIRST" in result

    def test_workspace_manifest_empty_workspace(self, tmp_path):
        """Empty workspace returns CWD context only for active mode."""
        from maxim.agents.prompt_builder import build_workspace_manifest

        (tmp_path / "app.py").write_text("import os")

        result = build_workspace_manifest(mode_name="active", cwd=str(tmp_path))
        assert "CWD Context" in result
        assert "EXISTING WORKSPACE" not in result


class TestModeAwareGuidance:
    def test_tool_guidance_core_passive(self):
        """Passive mode requires .maxim_workspace/ prefix."""
        from maxim.agents.prompt_builder import build_tool_guidance_core

        result = build_tool_guidance_core(mode_name="passive")
        assert ".maxim_workspace/" in result
        assert "proposed as plans" in result

    def test_tool_guidance_core_active(self):
        """Active mode requires .maxim_workspace/ prefix but allows CWD reads."""
        from maxim.agents.prompt_builder import build_tool_guidance_core

        result = build_tool_guidance_core(mode_name="active")
        assert ".maxim_workspace/" in result
        assert "requires approval" in result

    def test_tool_guidance_core_singularity(self):
        """Singularity mode allows writing any file, no prefix restriction."""
        from maxim.agents.prompt_builder import build_tool_guidance_core

        result = build_tool_guidance_core(mode_name="singularity")
        assert ".maxim_workspace/" not in result
        assert "ANY file" in result

    def test_tool_guidance_extended_singularity(self):
        """Singularity extended guidance mentions full project access."""
        from maxim.agents.prompt_builder import build_tool_guidance_extended

        result = build_tool_guidance_extended(mode_name="singularity")
        assert "full read/write" in result

    def test_tool_guidance_extended_passive(self):
        """Passive extended guidance only mentions workspace subdirs."""
        from maxim.agents.prompt_builder import build_tool_guidance_extended

        result = build_tool_guidance_extended(mode_name="passive")
        assert "full read/write" not in result
        assert "drafts/" in result


class TestTruncateManifest:
    def test_truncate_manifest_removes_entries(self):
        """_truncate_manifest removes file entries while keeping headers."""
        from maxim.agents.prompt_budgeter import _truncate_manifest

        counter = _ExactCounter()
        content = (
            "=== PROJECT DIRECTORY (5 entries) ===\n"
            "You have FULL read/write access.\n"
            "  src/main.py (2KB)\n"
            "  src/utils.py (1KB)\n"
            "  src/config.py (500B)\n"
            "  README.md (3KB)\n"
            "  tests/test_main.py (1KB)\n"
            "RULES:\n"
            "  1. Use read_file FIRST"
        )
        result = _truncate_manifest(content, max_tokens=10, counter=counter)
        assert "PROJECT DIRECTORY" in result
        assert "RULES" in result
        # Some entries should have been removed
        assert result.count("  src/") + result.count("  README") + result.count("  tests/") < 5
