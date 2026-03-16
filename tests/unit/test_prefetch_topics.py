"""Unit tests for topic-based file discovery in prefetch.py."""

from __future__ import annotations

import textwrap


from maxim.runtime.prefetch import (
    CandidateFile,
    DiscoveryPlan,
    PrefetchResult,
    SpeculativePrefetcher,
    TopicExtraction,
    discover_files_by_topic,
    extract_topics,
    select_files_for_context,
    summarize_file_structure,
)


# ── extract_topics ──────────────────────────────────────────────────────────


class TestExtractTopics:
    def test_memory_system(self):
        """'update the memory system' → topics=["memory"], dirs=["memory/"]."""
        result = extract_topics("update the memory system")
        assert "memory" in result.explicit_topics
        assert "memory/" in result.directory_hints
        assert result.action_intent == "modify"

    def test_multi_system(self):
        """'refactor agent pipeline and planning' → system_wide."""
        result = extract_topics("refactor agent pipeline and planning")
        assert "agent" in result.explicit_topics
        assert "planning" in result.explicit_topics or "plan" in result.explicit_topics
        assert result.complexity in ("multi_file", "system_wide")
        assert result.action_intent == "modify"

    def test_explicit_file_falls_through(self):
        """'update hello.py' has no topic matches (no domain keyword)."""
        result = extract_topics("update hello.py")
        # "hello" is not a domain keyword, "update" is not in the map
        # There should be no directory hints
        assert result.directory_hints == []

    def test_create_intent(self):
        """'create a new bridge module' detects create intent."""
        result = extract_topics("create a new bridge module")
        assert result.action_intent == "create"
        assert "bridge" in result.explicit_topics

    def test_explore_intent(self):
        """'explain the attention system' detects explore intent."""
        result = extract_topics("explain the attention system")
        assert result.action_intent == "explore"
        assert "attention" in result.explicit_topics

    def test_fuzzy_cwd_fallback(self):
        """Unrecognized words match against CWD directory names."""
        result = extract_topics(
            "update the foobar subsystem",
            cwd_dirs=["foobar/", "src/"],
        )
        assert "foobar" in result.explicit_topics
        assert "foobar/" in result.directory_hints


# ── discover_files_by_topic ─────────────────────────────────────────────────


class TestDiscoverFiles:
    def test_scoring_order(self, tmp_path):
        """Files matching topics rank higher than unrelated files."""
        mem = tmp_path / "memory"
        mem.mkdir()
        (mem / "hippocampus.py").write_text("class Hippocampus: pass")
        (mem / "base.py").write_text("class MemoryStore: pass")
        tools = tmp_path / "tools"
        tools.mkdir()
        (tools / "base.py").write_text("class ToolBase: pass")
        (tmp_path / "unrelated.txt").write_text("nothing")

        topics = TopicExtraction(
            explicit_topics=["memory", "hippocampus"],
            directory_hints=["memory/"],
            action_intent="modify",
            complexity="single_file",
        )
        candidates = discover_files_by_topic(topics, cwd=str(tmp_path))
        assert len(candidates) >= 2
        # hippocampus should be top — name_match + dir_topic + dir_match
        assert "hippocampus" in candidates[0].rel_path

    def test_init_deprioritized(self, tmp_path):
        """__init__.py ranks lower than named modules."""
        pkg = tmp_path / "memory"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("from .base import *")
        (pkg / "base.py").write_text("class MemoryStore: pass")

        topics = TopicExtraction(
            explicit_topics=["memory"],
            directory_hints=["memory/"],
            action_intent="modify",
            complexity="single_file",
        )
        candidates = discover_files_by_topic(topics, cwd=str(tmp_path))
        init_candidates = [c for c in candidates if "__init__" in c.rel_path]
        base_candidates = [c for c in candidates if "base.py" in c.rel_path]
        assert base_candidates and init_candidates
        assert base_candidates[0].score > init_candidates[0].score

    def test_empty_cwd(self, tmp_path):
        """Empty directory returns no candidates."""
        topics = TopicExtraction(
            explicit_topics=["memory"],
            directory_hints=["memory/"],
            action_intent="modify",
            complexity="single_file",
        )
        result = discover_files_by_topic(topics, cwd=str(tmp_path))
        assert result == []


# ── select_files_for_context ────────────────────────────────────────────────


class TestSelectFiles:
    def test_budget_respected(self, tmp_path):
        """Total content stays within max_total_chars."""
        d = tmp_path / "memory"
        d.mkdir()
        for name in ("a.py", "b.py", "c.py", "d.py"):
            (d / name).write_text("x = 1\n" * 100)  # ~600 chars each

        candidates = [
            CandidateFile(
                path=str(d / "a.py"),
                rel_path="memory/a.py",
                score=0.9,
                match_reasons=["topic"],
                size_bytes=600,
            ),
            CandidateFile(
                path=str(d / "b.py"),
                rel_path="memory/b.py",
                score=0.8,
                match_reasons=["topic"],
                size_bytes=600,
            ),
            CandidateFile(
                path=str(d / "c.py"),
                rel_path="memory/c.py",
                score=0.5,
                match_reasons=["dir"],
                size_bytes=600,
            ),
            CandidateFile(
                path=str(d / "d.py"),
                rel_path="memory/d.py",
                score=0.2,
                match_reasons=["dir"],
                size_bytes=600,
            ),
        ]
        plan = select_files_for_context(candidates, max_total_chars=1000)
        total = sum(len(v) for v in plan.full_content_files.values())
        total += sum(len(v) for v in plan.summary_files.values())
        assert total <= 1000

    def test_tiered_reading(self, tmp_path):
        """High-score files are full content, mid-score are summaries."""
        d = tmp_path / "src"
        d.mkdir()
        (d / "high.py").write_text("class High:\n    def run(self): pass\n")
        (d / "mid.py").write_text("class Mid:\n    def step(self): pass\n")
        (d / "low.py").write_text("x = 1")

        candidates = [
            CandidateFile(
                path=str(d / "high.py"),
                rel_path="src/high.py",
                score=0.9,
                match_reasons=["topic"],
                size_bytes=40,
            ),
            CandidateFile(
                path=str(d / "mid.py"),
                rel_path="src/mid.py",
                score=0.5,
                match_reasons=["dir"],
                size_bytes=40,
            ),
            CandidateFile(
                path=str(d / "low.py"),
                rel_path="src/low.py",
                score=0.2,
                match_reasons=["dir"],
                size_bytes=5,
            ),
        ]
        plan = select_files_for_context(candidates, max_total_chars=3000)
        assert "src/high.py" in plan.full_content_files
        assert "src/mid.py" in plan.summary_files
        assert "src/low.py" in plan.listed_files


# ── summarize_file_structure ────────────────────────────────────────────────


class TestSummarizeFileStructure:
    def test_extracts_classes_and_functions(self, tmp_path):
        """Extracts class and function names from a Python file."""
        f = tmp_path / "example.py"
        f.write_text(textwrap.dedent("""\
            \"\"\"Example module.\"\"\"

            class Foo:
                def bar(self): pass

            class Baz:
                pass

            def helper():
                pass
        """))
        summary = summarize_file_structure(str(f))
        assert "Foo" in summary
        assert "Baz" in summary
        # bar is an indented method — regex only captures top-level defs
        assert "helper" in summary

    def test_nonexistent_file(self):
        """Returns (unreadable) for missing files."""
        result = summarize_file_structure("/nonexistent/path.py")
        assert "unreadable" in result


# ── _format_discovery_plan ──────────────────────────────────────────────────


class TestFormatDiscoveryPlan:
    def test_single_file_banner(self):
        """Single-file complexity → 'FILE DISCOVERY' banner."""
        plan = DiscoveryPlan(
            topic_extraction=TopicExtraction(
                explicit_topics=["memory"],
                directory_hints=["memory/"],
                action_intent="modify",
                complexity="single_file",
            ),
            candidates=[],
            full_content_files={"memory/base.py": "class Base: pass"},
            summary_files={},
            listed_files=[],
        )
        output = SpeculativePrefetcher._format_discovery_plan(plan)
        assert "FILE DISCOVERY" in output
        assert "DISCOVERY PLAN" not in output
        assert "[FULL]" in output

    def test_system_wide_banner_and_propose(self):
        """System-wide complexity → 'DISCOVERY PLAN' + 'PROPOSE'."""
        plan = DiscoveryPlan(
            topic_extraction=TopicExtraction(
                explicit_topics=["memory", "agent"],
                directory_hints=["memory/", "agents/"],
                action_intent="modify",
                complexity="system_wide",
            ),
            candidates=[],
            full_content_files={"memory/base.py": "class Base: pass"},
            summary_files={"agents/llm.py": "llm.py (5KB):\n  Classes: LLMWorker"},
            listed_files=["agents/__init__.py"],
        )
        output = SpeculativePrefetcher._format_discovery_plan(plan)
        assert "DISCOVERY PLAN" in output
        assert "PROPOSE YOUR MODIFICATION PLAN" in output
        assert "[READ]" in output
        assert "[SUMMARY]" in output
        assert "[LIST]" in output


# ── Not-found safety net ────────────────────────────────────────────────────


class TestNotFoundGuidance:
    def test_not_found_refs_inject_search_guidance(self):
        """When file refs detected but not found, inject search directive."""
        result = PrefetchResult(
            file_references=[],
            intent="modify",
            not_found_refs=["chape_nii.py"],
        )
        prefetcher = SpeculativePrefetcher()
        output = prefetcher.format_prefetch_context(result)
        assert "NOT FOUND" in output
        assert "chape_nii.py" in output
        assert "glob" in output.lower() or "search" in output.lower()
        assert "Do NOT" in output

    def test_no_guidance_when_files_found(self):
        """No not-found guidance when file contents exist."""
        result = PrefetchResult(
            file_references=[],
            intent="modify",
            file_contents={"some/file.py": "content"},
        )
        prefetcher = SpeculativePrefetcher()
        output = prefetcher.format_prefetch_context(result)
        assert "NOT FOUND" not in output
