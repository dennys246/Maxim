from __future__ import annotations

import time
from typing import Any, Callable

from maxim.utils.logging import warn


def coach_movement(
    model: Any,
    loss_fn: Callable[[Any, Any], Any],
    x: Any,
    y: Any,
    *,
    optimizer: Any | None = None,
    callbacks: Any | None = None,
    step: int | None = None,
) -> Any:

    try:
        import tensorflow as tf
    except Exception as e:
        raise RuntimeError(f"TensorFlow is required for coach_movement(): {e}") from e

    if optimizer is None:
        optimizer = getattr(model, "optimizer", None)
    if optimizer is None:
        optimizer = tf.keras.optimizers.Adam()
        try:
            model.optimizer = optimizer
        except Exception:
            pass

    callback_list = None
    if callbacks is not None:
        try:
            from maxim.training.callbacks import as_callback_list

            callback_list = as_callback_list(callbacks)
        except Exception as e:
            callback_list = None
            warn("Failed to initialize callbacks: %s", e)

    if step is None:
        step = int(getattr(model, "_train_step", 0) or 0) + 1
        try:
            setattr(model, "_train_step", step)
        except Exception:
            pass

    if callback_list is not None:
        callback_list.on_train_begin(model=model)
        callback_list.on_train_step_begin(step, logs=None, model=model)

    with tf.GradientTape() as tape:
        y_pred = model(x, training=True)
        loss_value = loss_fn(y_pred, y)
        loss_value = tf.reduce_mean(loss_value)

    variables = getattr(model, "trainable_variables", None) or []
    grads = tape.gradient(loss_value, variables)
    grads_and_vars = [(g, v) for g, v in zip(grads, variables) if g is not None]
    if grads_and_vars:
        optimizer.apply_gradients(grads_and_vars)

    if callback_list is not None:
        try:
            loss_scalar = float(loss_value)
        except Exception:
            loss_scalar = None
        callback_list.on_train_step_end(step, logs={"loss": loss_scalar}, model=model)

    # Best-effort: keep a tiny loss history on the model and update the MotorCortex loss plot.
    try:
        loss_scalar = float(loss_value)
    except Exception:
        loss_scalar = None

    if loss_scalar is not None:
        try:
            history = getattr(model, "loss_history", None)
            if history is None:
                history = []
                try:
                    setattr(model, "loss_history", history)
                except Exception:
                    history = []

            if isinstance(history, list):
                history.append({"time": time.time(), "step": int(step), "loss": float(loss_scalar)})

                try:
                    from maxim.utils.plotting import update_motor_cortex_loss_plot

                    cfg = getattr(model, "config", None)
                    save_dir = getattr(cfg, "save_dir", None) if cfg is not None else None
                    update_motor_cortex_loss_plot(history, save_dir=save_dir)
                except Exception:
                    pass
        except Exception:
            pass

    return loss_value
