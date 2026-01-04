from __future__ import annotations

import math
from typing import Optional

import numpy as np
from scipy.signal import resample, resample_poly


def to_int16(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.int16:
        return np.ascontiguousarray(arr)
    if np.issubdtype(arr.dtype, np.floating):
        clipped = np.clip(arr, -1.0, 1.0)
        return np.ascontiguousarray((clipped * 32767.0).astype(np.int16))
    return np.ascontiguousarray(np.clip(arr, -32768, 32767).astype(np.int16))


def resample_audio(sample: np.ndarray, input_rate: Optional[int], output_rate: Optional[int]) -> np.ndarray:
    if not input_rate or not output_rate or int(input_rate) == int(output_rate):
        return sample

    try:
        gcd = math.gcd(int(input_rate), int(output_rate))
        up = int(output_rate) // gcd
        down = int(input_rate) // gcd
        return resample_poly(sample, up, down, axis=0)
    except Exception:
        num_sample = int(int(output_rate) * len(sample) / int(input_rate))
        return resample(sample, num_sample)
