from __future__ import annotations

import logging
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Sequence

from maxim.utils.config import DEFAULT_SAVE_ROOT
from maxim.utils.logging import warn


def _is_truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in ("1", "true", "t", "yes", "y", "on")


def _matplotlib_preflight_command() -> list[str]:
    code = "\n".join(
        [
            "import matplotlib",
            "from matplotlib import font_manager as fm",
            "load = getattr(fm, '_load_fontmanager', None)",
            "if load is None:",
            "    fm.FontManager()",
            "else:",
            "    load(try_read_cache=False)",
            "print('ok')",
        ]
    )
    return [sys.executable, "-X", "faulthandler", "-c", code]


def preflight_matplotlib_fonts(
    *,
    cache_dir: str | os.PathLike[str] | None = None,
    logger: logging.Logger | None = None,
) -> bool:
    """
    Preflight Matplotlib font loading in a subprocess to avoid hard crashes
    when fonts or caches are corrupted on Linux/WSL.
    """
    if _is_truthy(os.getenv("MAXIM_SKIP_MPL_PREFLIGHT")):
        if cache_dir and not os.getenv("MPLCONFIGDIR"):
            try:
                Path(cache_dir).mkdir(parents=True, exist_ok=True)
                os.environ["MPLCONFIGDIR"] = Path(cache_dir).as_posix()
            except Exception as e:
                warn("Failed to create MPLCONFIGDIR '%s': %s (using default cache)", cache_dir, e, logger=logger)
        os.environ.setdefault("MPLBACKEND", "Agg")
        return True

    config_dir = os.getenv("MPLCONFIGDIR")
    if not config_dir and cache_dir:
        config_dir = Path(cache_dir).as_posix()

    if config_dir:
        try:
            Path(config_dir).mkdir(parents=True, exist_ok=True)
        except Exception as e:
            warn("Failed to create MPLCONFIGDIR '%s': %s (using default cache)", config_dir, e, logger=logger)
            config_dir = None

    env = os.environ.copy()
    if config_dir:
        env["MPLCONFIGDIR"] = config_dir
    env.setdefault("MPLBACKEND", "Agg")

    result = subprocess.run(
        _matplotlib_preflight_command(),
        env=env,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        warn(
            "Matplotlib font preflight failed (exit=%s). See README troubleshooting for font/cache fixes.",
            result.returncode,
            logger=logger,
        )
        if logger is not None and logger.isEnabledFor(logging.DEBUG):
            if result.stdout:
                logger.debug("Matplotlib preflight stdout: %s", result.stdout.strip())
            if result.stderr:
                logger.debug("Matplotlib preflight stderr: %s", result.stderr.strip())
        return False

    if config_dir:
        os.environ["MPLCONFIGDIR"] = config_dir
    os.environ.setdefault("MPLBACKEND", "Agg")
    return True


def preload_matplotlib_fonts(
    *,
    cache_dir: str | os.PathLike[str] | None = None,
    logger: logging.Logger | None = None,
) -> bool:
    """
    Load Matplotlib + font manager in-process early to lock shared libraries
    before other native deps (e.g., GStreamer/Ultralytics) initialize.
    """
    if _is_truthy(os.getenv("MAXIM_SKIP_MPL_PRELOAD")):
        if cache_dir and not os.getenv("MPLCONFIGDIR"):
            try:
                Path(cache_dir).mkdir(parents=True, exist_ok=True)
                os.environ["MPLCONFIGDIR"] = Path(cache_dir).as_posix()
            except Exception as e:
                warn("Failed to create MPLCONFIGDIR '%s': %s (using default cache)", cache_dir, e, logger=logger)
        os.environ.setdefault("MPLBACKEND", "Agg")
        return True

    config_dir = os.getenv("MPLCONFIGDIR")
    if not config_dir and cache_dir:
        config_dir = Path(cache_dir).as_posix()
    if config_dir:
        try:
            Path(config_dir).mkdir(parents=True, exist_ok=True)
            os.environ["MPLCONFIGDIR"] = config_dir
        except Exception as e:
            warn("Failed to create MPLCONFIGDIR '%s': %s (using default cache)", config_dir, e, logger=logger)

    os.environ.setdefault("MPLBACKEND", "Agg")

    try:
        import matplotlib
        from matplotlib import font_manager as fm

        load = getattr(fm, "_load_fontmanager", None)
        if load is None:
            fm.FontManager()
        else:
            load(try_read_cache=False)
        return True
    except Exception as e:
        warn("Failed to preload Matplotlib fonts: %s", e, logger=logger)
        return False


def _is_finite_number(value: object) -> bool:
    try:
        return math.isfinite(float(value))  # type: ignore[arg-type]
    except Exception:
        return False


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(content, encoding="utf-8")
    os.replace(tmp_path, path)

def _extract_metric_points(
    history: Sequence[dict],
    *,
    metric_key: str,
    max_points: int = 2000,
) -> list[tuple[int, float]]:
    points: list[tuple[int, float]] = []
    for idx, record in enumerate(history):
        if not isinstance(record, dict):
            continue
        metric = record.get(metric_key)
        if not _is_finite_number(metric):
            continue
        step = record.get("step", idx + 1)
        try:
            step_i = int(step)
        except Exception:
            step_i = idx + 1
        points.append((step_i, float(metric)))  # type: ignore[arg-type]

    if len(points) <= max_points:
        return points

    stride = max(1, len(points) // max_points)
    downsampled = points[::stride]
    if downsampled[-1] != points[-1]:
        downsampled.append(points[-1])
    return downsampled


def _extract_loss_points(history: Sequence[dict], *, max_points: int = 2000) -> list[tuple[int, float]]:
    return _extract_metric_points(history, metric_key="loss", max_points=max_points)

def _fmt_metric(v: float) -> str:
    if abs(v) >= 1000:
        return f"{v:.1f}"
    if abs(v) >= 10:
        return f"{v:.3f}"
    return f"{v:.6f}"


def render_loss_svg(
    points: Sequence[tuple[int, float]],
    *,
    title: str = "MotorCortex Loss",
    y_label: str = "loss",
    width: int = 900,
    height: int = 320,
    padding: int = 48,
) -> str:
    width = max(300, int(width))
    height = max(200, int(height))
    padding = max(20, int(padding))

    inner_w = max(1, width - 2 * padding)
    inner_h = max(1, height - 2 * padding)

    if not points:
        return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect x="0" y="0" width="{width}" height="{height}" fill="#0b0f19"/>
  <text x="{padding}" y="{padding}" fill="#e6e6e6" font-family="system-ui, -apple-system, Segoe UI, Roboto, sans-serif" font-size="18">{title}</text>
  <text x="{padding}" y="{padding + 28}" fill="#9aa4b2" font-family="system-ui, -apple-system, Segoe UI, Roboto, sans-serif" font-size="14">No data yet.</text>
</svg>"""

    steps = [p[0] for p in points]
    losses = [p[1] for p in points]

    min_step = min(steps)
    max_step = max(steps)
    min_loss = min(losses)
    max_loss = max(losses)

    if max_step == min_step:
        max_step = min_step + 1
    if max_loss == min_loss:
        max_loss = min_loss + 1.0

    def x_of(step: int) -> float:
        return padding + (float(step - min_step) / float(max_step - min_step)) * inner_w

    def y_of(loss: float) -> float:
        # Higher loss near the top; lower loss near the bottom.
        return padding + (float(max_loss - loss) / float(max_loss - min_loss)) * inner_h

    poly = " ".join(f"{x_of(s):.2f},{y_of(l):.2f}" for s, l in points)

    last_step, last_loss = points[-1]
    last_x = x_of(last_step)
    last_y = y_of(last_loss)

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <defs>
    <style>
      .bg {{ fill: #0b0f19; }}
      .axis {{ stroke: #2a3245; stroke-width: 1; }}
      .grid {{ stroke: #192033; stroke-width: 1; }}
      .line {{ fill: none; stroke: #7aa2f7; stroke-width: 2; }}
      .pt {{ fill: #7aa2f7; }}
      .title {{ fill: #e6e6e6; font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; font-size: 18px; }}
      .label {{ fill: #9aa4b2; font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; font-size: 12px; }}
      .value {{ fill: #e6e6e6; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; font-size: 12px; }}
    </style>
  </defs>

  <rect class="bg" x="0" y="0" width="{width}" height="{height}"/>

  <text class="title" x="{padding}" y="{padding - 18}">{title}</text>

  <!-- axes -->
  <line class="axis" x1="{padding}" y1="{padding}" x2="{padding}" y2="{height - padding}"/>
  <line class="axis" x1="{padding}" y1="{height - padding}" x2="{width - padding}" y2="{height - padding}"/>

  <!-- y labels -->
  <text class="label" x="{padding}" y="{padding - 6}">{y_label} (max)</text>
  <text class="value" x="{padding + 80}" y="{padding - 6}">{_fmt_metric(max_loss)}</text>
  <text class="label" x="{padding}" y="{height - padding + 18}">{y_label} (min)</text>
  <text class="value" x="{padding + 80}" y="{height - padding + 18}">{_fmt_metric(min_loss)}</text>

  <!-- x labels -->
  <text class="label" x="{width - padding - 160}" y="{height - padding + 18}">step</text>
  <text class="value" x="{width - padding - 110}" y="{height - padding + 18}">{min_step} → {max_step}</text>

  <!-- grid -->
  <line class="grid" x1="{padding}" y1="{padding + inner_h * 0.25}" x2="{width - padding}" y2="{padding + inner_h * 0.25}"/>
  <line class="grid" x1="{padding}" y1="{padding + inner_h * 0.50}" x2="{width - padding}" y2="{padding + inner_h * 0.50}"/>
  <line class="grid" x1="{padding}" y1="{padding + inner_h * 0.75}" x2="{width - padding}" y2="{padding + inner_h * 0.75}"/>

  <!-- series -->
  <polyline class="line" points="{poly}"/>
  <circle class="pt" cx="{last_x:.2f}" cy="{last_y:.2f}" r="3.5"/>

  <text class="label" x="{padding}" y="{padding + 20}">last</text>
  <text class="value" x="{padding + 40}" y="{padding + 20}">step={last_step} {y_label}={_fmt_metric(last_loss)}</text>
</svg>"""


def update_motor_cortex_loss_plot(
    history: Sequence[dict] | None,
    *,
    save_dir: str | os.PathLike[str] | None = None,
    filename: str = "motor_cortex_loss.svg",
    max_points: int = 2000,
) -> Path | None:
    if not history:
        return None

    if save_dir is None:
        save_dir = DEFAULT_SAVE_ROOT

    save_dir_path = Path(save_dir)
    path = save_dir_path / filename
    points = _extract_loss_points(history, max_points=max_points)
    svg = render_loss_svg(points, title="MotorCortex Loss", y_label="loss")
    _atomic_write_text(path, svg)
    return path


def update_motor_cortex_pixel_error_plot(
    history: Sequence[dict] | None,
    *,
    save_dir: str | os.PathLike[str] | None = None,
    filename: str = "motor_cortex_pixel_error.svg",
    max_points: int = 2000,
) -> Path | None:
    if not history:
        return None

    if save_dir is None:
        save_dir = DEFAULT_SAVE_ROOT

    save_dir_path = Path(save_dir)
    path = save_dir_path / filename
    points = _extract_metric_points(history, metric_key="pixel_error_px", max_points=max_points)
    if not points:
        return None
    svg = render_loss_svg(points, title="MotorCortex Pixel Error", y_label="px")
    _atomic_write_text(path, svg)
    return path
