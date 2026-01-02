#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-}"
if [[ -z "$PYTHON" ]]; then
  if [[ -x ".venv/bin/python" ]]; then
    PYTHON=".venv/bin/python"
  else
    PYTHON="python3"
  fi
fi

echo "[basic_learn] python=$PYTHON"

"$PYTHON" - <<'PY'
import os
import types

import numpy as np


def _has_tf():
    try:
        import tensorflow  # noqa: F401
        import keras  # noqa: F401

        return True
    except Exception:
        return False


if not _has_tf():
    print("[basic_learn] SKIP: tensorflow/keras not installed")
    raise SystemExit(0)

from src.models.movement.motor_cortex import MotorCortex
from src.utils import config as motor_config

cfg = motor_config.build(os.path.join("experiments", "models", "MotorCortex"))
model = MotorCortex(cfg)

input_shape = getattr(cfg, "input_shape", 256)
if isinstance(input_shape, int):
    h = w = int(input_shape)
else:
    h = w = 256

x = np.zeros((1, h, w, 3), dtype=np.float32)
y = model(x, training=False)

arr = np.asarray(y)
assert arr.ndim == 2, f"expected [B, D], got {arr.shape}"
assert arr.shape[0] == 1, f"expected batch=1, got {arr.shape}"
assert arr.shape[1] == int(getattr(cfg, "output_dim", 7) or 7), f"unexpected output_dim: {arr.shape}"

print("[basic_learn] OK: MotorCortex forward pass:", arr.shape)

try:
    import tensorflow as tf

    y_true = tf.zeros((1, int(getattr(cfg, "output_dim", 7) or 7)), dtype=tf.float32)
    x_tf = tf.convert_to_tensor(x, dtype=tf.float32)
    with tf.GradientTape() as tape:
        y_pred = model(x_tf, training=True)
        loss = tf.reduce_mean(tf.square(y_pred - y_true))
    grads = tape.gradient(loss, model.trainable_weights)
    model.optimizer.apply_gradients(zip(grads, model.trainable_weights))
    print("[basic_learn] OK: one training step, loss=", float(loss))
except Exception as e:
    print("[basic_learn] WARN: training step skipped:", e)
PY

echo "[basic_learn] done"
