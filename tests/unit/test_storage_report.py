"""Tests for utils.storage — disk footprint reporting and download preflight.

Covers:
- StorageReport fields populated correctly
- Walk is one level deep per immediate subdir
- Cache TTL (second call within 60s is cached; force=True bypasses)
- Cache invalidated when data_home() changes
- can_download rejects insufficient fs_free
- can_download respects soft_budget_gb when set
- can_download tolerates failing disk_usage (returns True with fs-check skipped)
- format_report renders sorted subdirs
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from maxim.utils import storage
from maxim.utils.storage import (
    StorageReport,
    can_download,
    format_report,
    report_storage,
)


def _build_fake_maxim_home(root: Path) -> Path:
    """Populate a fake ~/.maxim/ layout for walk tests."""
    home = root / "dot_maxim"
    home.mkdir(parents=True)

    # models/: 100 MB "model1.gguf" and 50 MB "model2.gguf"
    models = home / "models"
    models.mkdir()
    (models / "model1.gguf").write_bytes(b"x" * (100 * 1024 * 1024))
    (models / "model2.gguf").write_bytes(b"y" * (50 * 1024 * 1024))

    # sim_reports/: one 60 MB file (just above the 50 MB display threshold)
    reports = home / "sim_reports"
    reports.mkdir()
    (reports / "report1.jsonl").write_bytes(b"z" * (60 * 1024 * 1024))

    # tiny/: 1 KB file (should NOT show up in subdir_sizes_gb because < 50 MB)
    tiny = home / "tiny"
    tiny.mkdir()
    (tiny / "small.txt").write_bytes(b"tiny")

    # empty/: no files (should not show up either)
    (home / "empty").mkdir()

    return home


class TestReportStorage:
    def setup_method(self):
        storage._reset_cache_for_tests()

    def test_walk_populates_subdir_sizes(self, tmp_path):
        home = _build_fake_maxim_home(tmp_path)
        with patch("maxim.utils.paths.data_home", return_value=home):
            report = report_storage(force=True)

        # models/ = 150 MB ≈ 0.15 GB
        assert "models" in report.subdir_sizes_gb
        assert 0.12 <= report.subdir_sizes_gb["models"] <= 0.16

        # sim_reports/ = 60 MB ≈ 0.06 GB
        assert "sim_reports" in report.subdir_sizes_gb

        # tiny/ and empty/ are below the 50 MB display threshold
        assert "tiny" not in report.subdir_sizes_gb
        assert "empty" not in report.subdir_sizes_gb

        # total_maxim_gb includes tiny/ even though it doesn't appear
        # in the detail dict — but it's tiny, so the total should still
        # be dominated by models/ + sim_reports/ ≈ 0.21 GB.
        assert 0.19 <= report.total_maxim_gb <= 0.24

    def test_walk_one_level_deep_with_shallow_nesting(self, tmp_path):
        """models/ is allowed to have one level of nesting (per-provider
        directories), and nested files should be counted."""
        home = tmp_path / "dot_maxim"
        home.mkdir()
        models = home / "models"
        (models / "provider_a").mkdir(parents=True)
        (models / "provider_a" / "model.gguf").write_bytes(b"x" * (100 * 1024 * 1024))

        with patch("maxim.utils.paths.data_home", return_value=home):
            report = report_storage(force=True)

        # The nested file's size should be summed into models/
        assert "models" in report.subdir_sizes_gb
        assert report.subdir_sizes_gb["models"] >= 0.09

    def test_cache_returns_same_instance_within_ttl(self, tmp_path):
        home = _build_fake_maxim_home(tmp_path)
        with patch("maxim.utils.paths.data_home", return_value=home):
            first = report_storage(force=True)
            second = report_storage(force=False)
        assert first is second

    def test_cache_bypassed_on_force(self, tmp_path):
        home = _build_fake_maxim_home(tmp_path)
        with patch("maxim.utils.paths.data_home", return_value=home):
            first = report_storage(force=True)
            # force=True should re-walk and produce a fresh report
            second = report_storage(force=True)
        assert first is not second
        # But the values should still match (same directory state)
        assert first.subdir_sizes_gb == second.subdir_sizes_gb

    def test_cache_invalidated_on_data_home_change(self, tmp_path):
        home1 = _build_fake_maxim_home(tmp_path / "run1")
        home2 = _build_fake_maxim_home(tmp_path / "run2")
        # Make run2 have a different footprint so we can verify
        (home2 / "models" / "model3.gguf").write_bytes(b"w" * (80 * 1024 * 1024))

        with patch("maxim.utils.paths.data_home", return_value=home1):
            first = report_storage(force=True)
        with patch("maxim.utils.paths.data_home", return_value=home2):
            second = report_storage(force=False)
        assert second.data_home == home2
        assert second.subdir_sizes_gb["models"] > first.subdir_sizes_gb["models"]

    def test_handles_missing_data_home_gracefully(self, tmp_path):
        # data_home() returning a non-existent path should not crash
        nonexistent = tmp_path / "does_not_exist"
        with patch("maxim.utils.paths.data_home", return_value=nonexistent):
            report = report_storage(force=True)
        assert report.total_maxim_gb == 0.0
        assert report.subdir_sizes_gb == {}


class TestCanDownload:
    def setup_method(self):
        storage._reset_cache_for_tests()

    def test_accepts_when_space_is_ample(self, tmp_path):
        home = _build_fake_maxim_home(tmp_path)
        with patch("maxim.utils.paths.data_home", return_value=home):
            ok, reason = can_download(size_gb=1.0, headroom_gb=2.0)
        assert ok is True
        assert "ok" in reason.lower()

    def test_rejects_when_fs_free_insufficient(self, tmp_path):
        home = _build_fake_maxim_home(tmp_path)

        fake_report = StorageReport(
            data_home=home,
            fs_free_gb=5.0,
            fs_total_gb=500.0,
            subdir_sizes_gb={},
            total_maxim_gb=0.0,
            walked_at=0.0,
        )
        with patch("maxim.utils.storage.report_storage", return_value=fake_report):
            ok, reason = can_download(size_gb=10.0, headroom_gb=2.0)
        assert ok is False
        assert "insufficient" in reason.lower()

    def test_respects_soft_budget(self, tmp_path):
        home = _build_fake_maxim_home(tmp_path)

        fake_report = StorageReport(
            data_home=home,
            fs_free_gb=500.0,  # plenty of fs space
            fs_total_gb=1000.0,
            subdir_sizes_gb={"models": 18.0},
            total_maxim_gb=18.0,
            walked_at=0.0,
        )
        with patch("maxim.utils.storage.report_storage", return_value=fake_report):
            # budget 20 GB, current 18 GB, wanting 5 GB → would be 23 GB, reject
            ok, reason = can_download(size_gb=5.0, soft_budget_gb=20.0)
        assert ok is False
        assert "soft budget" in reason.lower()

    def test_soft_budget_allows_fit(self, tmp_path):
        home = _build_fake_maxim_home(tmp_path)

        fake_report = StorageReport(
            data_home=home,
            fs_free_gb=500.0,
            fs_total_gb=1000.0,
            subdir_sizes_gb={"models": 10.0},
            total_maxim_gb=10.0,
            walked_at=0.0,
        )
        with patch("maxim.utils.storage.report_storage", return_value=fake_report):
            # budget 20 GB, current 10 GB, wanting 5 GB → 15 GB total, fits
            ok, reason = can_download(size_gb=5.0, soft_budget_gb=20.0)
        assert ok is True

    def test_infinite_free_skips_fs_check(self, tmp_path):
        """When disk_usage() fails the reporter fills fs_free_gb with
        infinity and the download should still proceed (the real
        download will fail if disk is truly full)."""
        home = _build_fake_maxim_home(tmp_path)

        fake_report = StorageReport(
            data_home=home,
            fs_free_gb=float("inf"),
            fs_total_gb=float("inf"),
            subdir_sizes_gb={},
            total_maxim_gb=0.0,
            walked_at=0.0,
        )
        with patch("maxim.utils.storage.report_storage", return_value=fake_report):
            ok, reason = can_download(size_gb=1000.0)
        assert ok is True


class TestFormatReport:
    def test_renders_subdirs_sorted_by_size(self, tmp_path):
        report = StorageReport(
            data_home=tmp_path / "dot_maxim",
            fs_free_gb=100.0,
            fs_total_gb=500.0,
            subdir_sizes_gb={"models": 12.3, "sim_reports": 0.5, "memory": 2.1},
            total_maxim_gb=14.9,
            walked_at=0.0,
        )
        out = format_report(report)
        assert "Filesystem: 100.0 GB free" in out
        assert "14.9 GB" in out
        # Sorted by size descending
        models_idx = out.find("models:")
        memory_idx = out.find("memory:")
        reports_idx = out.find("sim_reports:")
        assert 0 < models_idx < memory_idx < reports_idx

    def test_renders_empty_footprint(self, tmp_path):
        report = StorageReport(
            data_home=tmp_path / "dot_maxim",
            fs_free_gb=100.0,
            fs_total_gb=500.0,
            subdir_sizes_gb={},
            total_maxim_gb=0.0,
            walked_at=0.0,
        )
        out = format_report(report)
        assert "no subdirs over 50 MB" in out
