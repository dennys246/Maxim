from __future__ import annotations

import math
from typing import Any

def euclidian_distance(p1: Any, p2: Any):
    """
    Euclidean distance between 2D points.

    Supports:
    - Python/numpy-like sequences: returns float
    - TensorFlow tensors: returns a Tensor of shape (...) reduced over the last dim
    """
    try:
        import tensorflow as tf

        if isinstance(p1, (tf.Tensor, tf.Variable)) or isinstance(p2, (tf.Tensor, tf.Variable)):
            p1_t = tf.cast(p1, tf.float32)
            p2_t = tf.cast(p2, tf.float32)
            diff = p2_t - p1_t
            # Add epsilon so gradients don't become NaN at diff == 0.
            return tf.sqrt(tf.reduce_sum(tf.square(diff), axis=-1) + 1e-9)
    except Exception:
        pass

    try:
        import numpy as np

        p1_arr = np.asarray(p1, dtype=float)
        p2_arr = np.asarray(p2, dtype=float)
        diff = p2_arr - p1_arr
        return float(np.sqrt(np.sum(diff * diff) + 1e-9))
    except Exception:
        return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2 + 1e-9)
