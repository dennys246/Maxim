from __future__ import annotations

from typing import Any

import numpy as np


def score_detection_conf_area(obs: Any) -> tuple[float, float]:
    try:
        x1, y1, x2, y2 = float(obs[2]), float(obs[3]), float(obs[4]), float(obs[5])
        conf = float(obs[6])
    except Exception:
        return (float("-inf"), float("-inf"))

    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return (conf, area)


def maybe_scale_normalized_xyxy(
    x1_in: float,
    y1_in: float,
    x2_in: float,
    y2_in: float,
    width: int,
    height: int,
) -> tuple[float, float, float, float, bool]:
    """
    Some detectors return normalized [0,1] xyxy. Convert to pixels when that appears to be the case.
    """
    try:
        vals = [float(x1_in), float(y1_in), float(x2_in), float(y2_in)]
    except Exception:
        return x1_in, y1_in, x2_in, y2_in, False

    if not all(np.isfinite(v) for v in vals):
        return x1_in, y1_in, x2_in, y2_in, False

    x1_f, y1_f, x2_f, y2_f = vals
    looks_normalized = (
        0.0 <= x1_f <= 1.0
        and 0.0 <= x2_f <= 1.0
        and 0.0 <= y1_f <= 1.0
        and 0.0 <= y2_f <= 1.0
        and width > 2
        and height > 2
    )
    if not looks_normalized:
        return x1_in, y1_in, x2_in, y2_in, False

    scale_x = float(width - 1)
    scale_y = float(height - 1)
    return x1_f * scale_x, y1_f * scale_y, x2_f * scale_x, y2_f * scale_y, True
