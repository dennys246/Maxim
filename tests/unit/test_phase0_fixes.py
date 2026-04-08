"""Tests for Phase 0 publication blockers.

Covers:
- 0a: maxim.run() signature fix (LLMWorker kwarg)
- 0b: Stub verbs emit warnings (campaign, benchmark, research)
- 0c: Error honesty (security URL inversion, warning logging)
- 0d: Missing type exports (RobotController, exception hierarchy)
- 0e: Hippocampus threading (queue full handling, flush reliability)
"""

from __future__ import annotations

import queue
import threading
import time
import warnings
from unittest import mock

import pytest


# ── 0a: maxim.run() signature fix ─────────────────────────────────────


class TestRunSignature:
    """maxim.run() should pass correct kwargs to LLMWorker."""

    def test_llm_worker_accepts_llm_kwarg(self):
        """LLMWorker should accept 'llm' not 'router'."""
        from maxim.agents.llm_worker import LLMWorker
        import inspect

        sig = inspect.signature(LLMWorker.__init__)
        params = list(sig.parameters.keys())
        assert "llm" in params, f"LLMWorker.__init__ should accept 'llm', got: {params}"
        assert "router" not in params, "LLMWorker should NOT have a 'router' parameter"

    def test_api_run_uses_llm_kwarg(self):
        """api.run() should call LLMWorker(llm=...) not LLMWorker(router=...)."""
        import ast
        from pathlib import Path

        api_source = Path("src/maxim/api.py").read_text()
        tree = ast.parse(api_source)

        # Find the LLMWorker call in the run function
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name) and func.id == "LLMWorker":
                    kw_names = [kw.arg for kw in node.keywords]
                    assert "llm" in kw_names, f"LLMWorker called with kwargs: {kw_names}"
                    assert "router" not in kw_names, "Should not use 'router' kwarg"


# ── 0b: Stub verbs emit warnings ──────────────────────────────────────


class TestStubWarnings:
    """Stub API verbs should emit UserWarning, not silently return empty."""

    def test_campaign_warns(self):
        import maxim

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = maxim.campaign("scenarios/campaigns/heist_v1.yaml")
            assert len(w) >= 1
            assert "not yet wired" in str(w[0].message).lower()
            assert result.campaign_name  # Has a name (actual value depends on YAML)

    def test_benchmark_warns(self):
        import maxim

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = maxim.benchmark(models=["mistral-7b"], suite="cognitive")
            assert len(w) >= 1
            assert "not yet wired" in str(w[0].message).lower()
            assert result.models == ["mistral-7b"]

    def test_research_warns(self):
        import maxim

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = maxim.research(goal="test memory")
            assert len(w) >= 1
            assert "not yet wired" in str(w[0].message).lower()
            assert result.goal == "test memory"


# ── 0c: Error honesty ─────────────────────────────────────────────────


class TestErrorHonesty:
    """Verify security and logging fixes in API surface."""

    def test_empty_url_not_cloud(self):
        """Empty/None URL should not be classified as cloud."""
        from maxim.runtime.lane_backends import _is_cloud_url

        assert _is_cloud_url("") is False
        assert _is_cloud_url(None) is False

    def test_localhost_not_cloud(self):
        """Localhost URLs should not be classified as cloud."""
        from maxim.runtime.lane_backends import _is_cloud_url

        assert _is_cloud_url("http://localhost:8080") is False
        assert _is_cloud_url("http://127.0.0.1:8080") is False

    def test_public_url_is_cloud(self):
        """Public URLs should be classified as cloud."""
        from maxim.runtime.lane_backends import _is_cloud_url

        assert _is_cloud_url("https://api.example.com/v1") is True

    def test_no_silent_pass_in_api_py(self):
        """api.py should have zero bare 'except Exception: pass' blocks."""
        from pathlib import Path
        import re

        source = Path("src/maxim/api.py").read_text()
        # Match: except Exception:\n    pass (with optional whitespace)
        matches = re.findall(r"except\s+Exception\s*:\s*\n\s*pass\b", source)
        assert len(matches) == 0, f"Found {len(matches)} bare except-pass in api.py"


# ── 0d: Missing type exports ──────────────────────────────────────────


class TestTypeExports:
    """All category-level types should be importable from maxim."""

    def test_robot_controller_importable(self):
        from maxim import RobotController
        assert RobotController is not None

    def test_connection_error_importable(self):
        from maxim import ConnectionError as MaximConnectionError
        assert issubclass(MaximConnectionError, Exception)

    def test_tool_not_found_error_importable(self):
        from maxim import ToolNotFoundError
        assert issubclass(ToolNotFoundError, Exception)

    def test_memory_error_importable(self):
        from maxim import MemoryError as MaximMemoryError
        assert issubclass(MaximMemoryError, Exception)

    def test_planning_error_importable(self):
        from maxim import PlanningError
        assert issubclass(PlanningError, Exception)

    def test_hardware_error_importable(self):
        from maxim import HardwareError
        assert issubclass(HardwareError, Exception)

    def test_runtime_error_importable(self):
        from maxim import RuntimeError as MaximRuntimeError
        assert issubclass(MaximRuntimeError, Exception)

    def test_exception_hierarchy(self):
        """All category exceptions should inherit from MaximError."""
        from maxim import (
            MaximError,
            ConfigurationError,
            ConnectionError as MC,
            ModelError,
            ToolExecutionError,
            MemoryError as MM,
            PlanningError,
            HardwareError,
            RuntimeError as MR,
        )
        for exc_cls in [ConfigurationError, MC, ModelError, ToolExecutionError,
                         MM, PlanningError, HardwareError, MR]:
            assert issubclass(exc_cls, MaximError), f"{exc_cls.__name__} should inherit MaximError"


# ── 0e: Hippocampus threading ─────────────────────────────────────────


class TestHippocampusCapture:
    """Test the fixed capture queue submission."""

    def test_capture_queue_full_doesnt_crash(self):
        """When queue is full, should drop oldest and not raise."""
        from maxim.memory.hippocampus import Hippocampus, HippocampusConfig

        hippo = Hippocampus(HippocampusConfig(capture_queue_size=2))
        hippo._capture_stop = threading.Event()

        # Fill the queue without processing
        for i in range(5):
            hippo.capture_from_loop_async(
                observation={"text": f"obs_{i}"},
                state={},
                intent="",
                decision="",
                action={"tool_name": "test"},
                result=None,
                evaluations={},
                run_id="test",
            )
        # Should not have raised — items may be dropped but no crash
        assert hippo._capture_queue.qsize() <= 2

    def test_capture_with_put_timeout(self):
        """The put() with timeout should not block indefinitely."""
        from maxim.memory.hippocampus import Hippocampus, HippocampusConfig

        hippo = Hippocampus(HippocampusConfig(capture_queue_size=1))
        hippo._capture_stop = threading.Event()

        start = time.time()
        for i in range(3):
            hippo.capture_from_loop_async(
                observation={"text": f"obs_{i}"},
                state={},
                intent="",
                decision="",
                action={"tool_name": "test"},
                result=None,
                evaluations={},
                run_id="test",
            )
        elapsed = time.time() - start
        # Should complete quickly (< 2s even with 3 items on queue of size 1)
        assert elapsed < 2.0


class TestHippocampusFlush:
    """Test the fixed flush() method."""

    def test_flush_empty_queue(self):
        """Flush on empty queue should return True immediately."""
        from maxim.memory.hippocampus import Hippocampus, HippocampusConfig

        hippo = Hippocampus(HippocampusConfig())
        assert hippo.flush(timeout=1.0) is True

    def test_flush_with_worker_processing(self):
        """Flush should wait for worker to process items."""
        from maxim.memory.hippocampus import Hippocampus, HippocampusConfig

        hippo = Hippocampus(HippocampusConfig(capture_queue_size=10))
        hippo._capture_stop = threading.Event()

        # Start the worker thread
        worker = threading.Thread(target=hippo._capture_worker, daemon=True)
        worker.start()

        # Add a few items
        hippo.store_observation("test memory 1")
        hippo.store_observation("test memory 2")

        # These go through the sync path (not async), so queue should be empty
        # But let's test flush returns true
        result = hippo.flush(timeout=5.0)
        assert result is True

        hippo._capture_stop.set()
        worker.join(timeout=2.0)

    def test_flush_timeout(self):
        """Flush should return False on timeout (no worker to drain)."""
        from maxim.memory.hippocampus import Hippocampus, HippocampusConfig

        hippo = Hippocampus(HippocampusConfig(capture_queue_size=10))
        hippo._capture_stop = threading.Event()

        # Manually put items WITHOUT starting worker
        hippo._capture_queue.put("fake_request")

        # Flush should timeout since no worker is draining
        result = hippo.flush(timeout=0.2)
        assert result is False


class TestHippocampusConcurrency:
    """Multi-thread capture shouldn't lose data or deadlock."""

    def test_concurrent_store_observation(self):
        """8 threads storing observations concurrently."""
        from maxim.memory.hippocampus import Hippocampus

        hippo = Hippocampus()
        errors = []

        def store_many(thread_id: int) -> None:
            try:
                for i in range(20):
                    hippo.store_observation(f"thread_{thread_id}_obs_{i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=store_many, args=(t,)) for t in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)

        assert len(errors) == 0, f"Errors during concurrent capture: {errors}"
        # 8 threads * 20 observations = 160 memories
        assert len(hippo) == 160

    def test_concurrent_store_and_recall(self):
        """Store and recall concurrently shouldn't deadlock."""
        from maxim.memory.hippocampus import Hippocampus

        hippo = Hippocampus()
        # Pre-populate
        for i in range(10):
            hippo.store_observation(f"seed memory {i}")

        errors = []
        stop = threading.Event()

        def writer() -> None:
            try:
                for i in range(50):
                    if stop.is_set():
                        break
                    hippo.store_observation(f"new memory {i}")
            except Exception as e:
                errors.append(("writer", e))

        def reader() -> None:
            try:
                for _ in range(50):
                    if stop.is_set():
                        break
                    hippo.recall(limit=5)
            except Exception as e:
                errors.append(("reader", e))

        threads = [
            threading.Thread(target=writer),
            threading.Thread(target=reader),
            threading.Thread(target=reader),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)

        # Check no thread is still alive (deadlock detection)
        for t in threads:
            assert not t.is_alive(), "Thread deadlocked!"

        assert len(errors) == 0, f"Errors: {errors}"
        stop.set()
