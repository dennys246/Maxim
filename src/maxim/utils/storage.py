"""Disk-footprint reporting for Maxim's data directory.

Used by:
- The model-download preflight (``P5`` in peer_leader_flexibility_plan)
  to decide whether a pending download will fit on disk AND stay within
  any configured soft budget.
- ``maxim doctor`` to show the user where their ``~/.maxim/`` space is
  going and suggest cleanups.

Design notes:
- Walks are **one level deep** (``os.scandir`` per top-level subdir,
  summing only immediate file sizes). Recursive walks on Maxim's
  ``sessions/`` and ``provenance/`` directories can involve thousands
  of small files and take multi-second latency on HDD or network
  drives. One-level-deep produces an approximation of the subdir
  total (missing size of nested subdirs) which is good enough for
  "where is space going" guidance — and the error bars are documented
  below.
- Results are cached for 60 seconds at module scope. ``force=True``
  bypasses the cache. Cache is invalidated automatically when
  ``data_home()`` returns a different path than the cached report
  (handles mid-session ``MAXIM_DATA_HOME`` relocation).
- All functions are best-effort: OSError from a single subdir is
  logged and the subdir is reported as 0.0GB rather than crashing
  the entire report.
"""

from __future__ import annotations

import logging
import os
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StorageReport:
    """Snapshot of disk usage at a moment in time.

    Fields:
        data_home: The ``~/.maxim`` (or ``$MAXIM_DATA_HOME``) path the
            report was taken against.
        fs_free_gb: Free space on the filesystem containing ``data_home``.
        fs_total_gb: Total size of the filesystem containing ``data_home``.
        subdir_sizes_gb: Approximate size of each immediate subdir of
            ``data_home`` (one level deep only — see module docstring).
            Keys are subdir names, values are GB. Subdirs under 0.05GB
            are omitted from this dict to keep the output readable.
        total_maxim_gb: Sum of ``subdir_sizes_gb`` values. Approximate
            for the same reason as ``subdir_sizes_gb``.
        walked_at: ``time.monotonic()`` when the walk happened. Used by
            the cache invalidation logic.
    """

    data_home: Path
    fs_free_gb: float
    fs_total_gb: float
    subdir_sizes_gb: dict[str, float] = field(default_factory=dict)
    total_maxim_gb: float = 0.0
    walked_at: float = 0.0


# ─── module-level cache ──────────────────────────────────────────────
_REPORT_CACHE: StorageReport | None = None
_CACHE_LOCK = threading.Lock()
_CACHE_TTL_S = 60.0


def report_storage(*, force: bool = False) -> StorageReport:
    """Walk ``data_home()`` one level deep and return a size report.

    The report is cached for ``_CACHE_TTL_S`` seconds at module scope.
    Pass ``force=True`` to bypass the cache (used by ``maxim doctor``
    when the user explicitly runs a storage check).

    Always returns a report — never raises. On filesystem errors, the
    affected field is populated with a conservative value (0.0 for
    sizes, ``fs_total_gb`` for ``fs_free_gb`` so preflight doesn't
    wrongly block downloads just because the walk failed).
    """
    from maxim.utils.paths import data_home

    current_home = data_home()
    now = time.monotonic()

    with _CACHE_LOCK:
        global _REPORT_CACHE
        if (
            not force
            and _REPORT_CACHE is not None
            and _REPORT_CACHE.data_home == current_home
            and now - _REPORT_CACHE.walked_at < _CACHE_TTL_S
        ):
            return _REPORT_CACHE

        report = _walk_storage(current_home, now)
        _REPORT_CACHE = report
        return report


def _walk_storage(data_home: Path, walked_at: float) -> StorageReport:
    """Do the actual walk. Called under the cache lock."""
    # Filesystem-level free / total
    fs_free_gb = 0.0
    fs_total_gb = 0.0
    try:
        usage = shutil.disk_usage(data_home)
        fs_free_gb = usage.free / (1024**3)
        fs_total_gb = usage.total / (1024**3)
    except OSError as e:
        logger.debug("shutil.disk_usage(%s) failed: %s", data_home, e)
        # On failure, report a huge free so can_download doesn't wrongly
        # refuse. The user will see the download actually fail if disk
        # is truly full.
        fs_free_gb = float("inf")
        fs_total_gb = float("inf")

    subdir_sizes_gb: dict[str, float] = {}
    total_bytes = 0

    if data_home.is_dir():
        try:
            entries = list(os.scandir(data_home))
        except OSError as e:
            logger.debug("scandir(%s) failed: %s", data_home, e)
            entries = []

        for entry in entries:
            if not entry.is_dir(follow_symlinks=False):
                continue
            subdir_bytes = _sum_immediate_files(Path(entry.path))
            if subdir_bytes <= 0:
                continue
            subdir_gb = subdir_bytes / (1024**3)
            total_bytes += subdir_bytes
            # Only surface subdirs >= 0.05 GB (50 MB) to keep the
            # output readable. Smaller dirs are still included in
            # total_maxim_gb but don't clutter the detail dict.
            if subdir_gb >= 0.05:
                subdir_sizes_gb[entry.name] = round(subdir_gb, 2)

    return StorageReport(
        data_home=data_home,
        fs_free_gb=round(fs_free_gb, 2) if fs_free_gb != float("inf") else fs_free_gb,
        fs_total_gb=round(fs_total_gb, 2) if fs_total_gb != float("inf") else fs_total_gb,
        subdir_sizes_gb=subdir_sizes_gb,
        total_maxim_gb=round(total_bytes / (1024**3), 2),
        walked_at=walked_at,
    )


def _sum_immediate_files(subdir: Path) -> int:
    """Sum the byte size of immediate files in ``subdir``.

    Does NOT recurse into nested subdirectories. This is a deliberate
    choice — see the module docstring. Returns 0 on any OSError
    (permission denied, broken symlink, etc.).
    """
    total = 0
    try:
        with os.scandir(subdir) as it:
            for entry in it:
                try:
                    if entry.is_file(follow_symlinks=False):
                        total += entry.stat(follow_symlinks=False).st_size
                    elif entry.is_dir(follow_symlinks=False):
                        # One-level-deep rule: also sum immediate
                        # files of subdirs-of-subdirs so ``models/``
                        # catches GGUFs nested under per-provider
                        # directories. This is a pragmatic exception —
                        # models/ typically has a shallow hierarchy
                        # (<= 2 levels), not pathological depth like
                        # sessions/.
                        try:
                            with os.scandir(entry.path) as nested:
                                for nested_entry in nested:
                                    try:
                                        if nested_entry.is_file(follow_symlinks=False):
                                            total += nested_entry.stat(
                                                follow_symlinks=False,
                                            ).st_size
                                    except OSError:
                                        continue
                        except OSError:
                            continue
                except OSError:
                    continue
    except OSError as e:
        logger.debug("_sum_immediate_files(%s) failed: %s", subdir, e)
    return total


def can_download(
    size_gb: float,
    *,
    headroom_gb: float = 2.0,
    soft_budget_gb: float | None = None,
) -> tuple[bool, str]:
    """Check whether a ``size_gb`` download can proceed.

    Args:
        size_gb: Size of the pending download in GB.
        headroom_gb: Required free space on top of the download itself,
            to keep the user's disk from running critically low. Default
            2.0 GB is a pragmatic buffer — enough to keep the OS happy
            and the user's other work unbroken.
        soft_budget_gb: Optional cap on Maxim's total disk usage. When
            set, the preflight also checks that
            ``total_maxim_gb + size_gb <= soft_budget_gb``. Sourced from
            ``MAXIM_DATA_BUDGET_GB`` at the call site if desired.

    Returns:
        ``(ok, reason)`` — ``ok`` is True iff the download should
        proceed. ``reason`` is a human-readable explanation, populated
        on both success (e.g., "24.3 GB free") and failure.
    """
    report = report_storage()

    # Filesystem-level check
    if report.fs_free_gb != float("inf"):
        needed = size_gb + headroom_gb
        if report.fs_free_gb < needed:
            return (
                False,
                (
                    f"insufficient disk space: {report.fs_free_gb:.1f} GB free, "
                    f"{needed:.1f} GB required ({size_gb:.1f} download + "
                    f"{headroom_gb:.1f} headroom)"
                ),
            )

    # Soft-budget check (opt-in)
    if soft_budget_gb is not None and soft_budget_gb > 0:
        projected = report.total_maxim_gb + size_gb
        if projected > soft_budget_gb:
            return (
                False,
                (
                    f"soft budget exceeded: Maxim would use {projected:.1f} GB "
                    f"(current: {report.total_maxim_gb:.1f} GB + download "
                    f"{size_gb:.1f} GB), budget: {soft_budget_gb:.1f} GB. "
                    f"Raise MAXIM_DATA_BUDGET_GB or free up existing files."
                ),
            )

    # Success — return a positive reason so callers can display it
    if report.fs_free_gb == float("inf"):
        return True, f"ok (fs check skipped; download {size_gb:.1f} GB)"
    return (
        True,
        f"ok ({report.fs_free_gb:.1f} GB free, download {size_gb:.1f} GB)",
    )


def format_report(report: StorageReport) -> str:
    """Human-readable summary for doctor / preflight messages."""
    lines = [f"Filesystem: {report.fs_free_gb:.1f} GB free of {report.fs_total_gb:.1f} GB"]
    lines.append(f"Maxim footprint ({report.total_maxim_gb:.1f} GB):")
    if report.subdir_sizes_gb:
        for subdir, size in sorted(report.subdir_sizes_gb.items(), key=lambda kv: -kv[1]):
            lines.append(f"  {subdir}: {size:.1f} GB")
    else:
        lines.append("  (no subdirs over 50 MB)")
    return "\n".join(lines)


def _reset_cache_for_tests() -> None:
    """Test-only helper to clear the module-level cache.

    Do not call from production code — the cache invalidates naturally
    after 60s or when data_home() changes.
    """
    global _REPORT_CACHE
    with _CACHE_LOCK:
        _REPORT_CACHE = None


__all__ = [
    "StorageReport",
    "report_storage",
    "can_download",
    "format_report",
]
