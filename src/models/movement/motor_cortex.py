import os

import tensorflow as tf
import keras

from src.utils import config
from src.training import losses

class LayerScale(keras.layers.Layer):
    def __init__(self, init_value: float = 1e-6, **kwargs):
        super().__init__(**kwargs)
        self.init_value = float(init_value)

    def build(self, input_shape):
        channel_dim = int(input_shape[-1])
        self.gamma = self.add_weight(
            name="gamma",
            shape=(channel_dim,),
            initializer=keras.initializers.Constant(self.init_value),
            trainable=True,
        )

    def call(self, inputs):
        return inputs * self.gamma


def _resolve_input_shape(cfg):
    channels = int(getattr(cfg, "channels", 3) or 3)

    shape = getattr(cfg, "input_shape", None)
    if isinstance(shape, int):
        return (shape, shape, channels)

    if isinstance(shape, (list, tuple)):
        shape = tuple(shape)
        if len(shape) == 3:
            return shape
        if len(shape) == 2:
            return (shape[0], shape[1], channels)
        if len(shape) == 1:
            size = int(shape[0])
            return (size, size, channels)

    resolution = getattr(cfg, "resolution", None)
    if isinstance(resolution, (list, tuple)) and len(resolution) == 2:
        width, height = resolution
        try:
            return (int(height), int(width), channels)
        except (TypeError, ValueError):
            pass

    return (None, None, channels)


class MotorCortex(keras.Model):

    def __init__(self, config):
        super().__init__()

        self.config = config

        self.model = self._build_model()
        
        self.optimizer = self.get_optimizer()

    def call(self, inputs, training = False):
        return self.model(inputs, training = training)
    
    def _build_model(self):

        input_shape = _resolve_input_shape(self.config)
        inputs = keras.Input(shape=input_shape, name="image")

        x = inputs
        x = keras.layers.Rescaling(1.0 / 255.0, name="rescale")(x)

        # ConvNeXt-Tiny backbone: depths=[3, 3, 9, 3], dims=[96, 192, 384, 768]
        depths = (3, 3, 9, 3)
        dims = (96, 192, 384, 768)

        layer_scale_init_value = float(getattr(self.config, "layer_scale_init_value", 1e-6) or 1e-6)

        def convnext_block(tensor, dim: int, name: str):
            shortcut = tensor
            tensor = keras.layers.DepthwiseConv2D(
                kernel_size=7,
                padding="same",
                name=f"{name}_dwconv")(tensor)
            
            tensor = keras.layers.LayerNormalization(
                epsilon=1e-6,
                name=f"{name}_ln")(tensor)
            
            tensor = keras.layers.Dense(
                4 * dim,
                name=f"{name}_pwconv1")(tensor)
            
            tensor = keras.layers.Activation("gelu", name=f"{name}_gelu")(tensor)
            
            tensor = keras.layers.Dense(
                dim,
                name=f"{name}_pwconv2")(tensor)
            
            if layer_scale_init_value > 0:
                tensor = LayerScale(init_value=layer_scale_init_value, name=f"{name}_ls")(tensor)
            return keras.layers.Add(name=f"{name}_add")([shortcut, tensor])

        # Stem
        x = keras.layers.Conv2D(
            dims[0],
            kernel_size=4,
            strides=4,
            padding="valid",
            name="stem_conv")(x)
        
        x = keras.layers.LayerNormalization(epsilon=1e-6, name="stem_ln")(x)

        # Stages
        for stage_index, (dim, depth) in enumerate(zip(dims, depths)):
            
            for block_index in range(depth):
                x = convnext_block(x, dim=dim, name=f"stage{stage_index}_block{block_index}")
            
            if stage_index < len(dims) - 1:
                x = keras.layers.LayerNormalization(epsilon=1e-6, name=f"down{stage_index}_ln")(x)
                x = keras.layers.Conv2D(
                    dims[stage_index + 1],
                    kernel_size=2,
                    strides=2,
                    padding="valid",
                    name=f"down{stage_index}_conv")(x)

        # Head (regression)
        x = keras.layers.LayerNormalization(epsilon=1e-6, name="head_ln")(x)
        x = keras.layers.GlobalAveragePooling2D(name="avg_pool")(x)

        head_dim = int(getattr(self.config, "head_dim", 256) or 256)
        if head_dim > 0:
            x = keras.layers.Dense(head_dim, name="head_fc")(x)
            x = keras.layers.Activation("gelu", name="head_gelu")(x)
            x = keras.layers.Dropout(float(getattr(self.config, "dropout", 0.0) or 0.0), name="head_dropout")(x)

        output_dim = int(getattr(self.config, "output_dim", 7) or 7)
        
        final_activation = getattr(self.config, "final_activation", None) or None

        if output_dim == 7 and final_activation is None:
            # Default to bounded outputs for head-movement deltas.
            final_activation = "tanh"

        outputs = keras.layers.Dense(
            output_dim,
            activation=final_activation,
            name="movement")(x)

        return keras.Model(inputs=inputs, outputs=outputs, name="motor_cortex_convnext_t")

    def get_optimizer(self):
        return keras.optimizers.Adam(learning_rate = self.config.learning_rate, beta_1 = self.config.beta_1, beta_2 = self.config.beta_2)
    
    def get_loss(self, eye_center_coordinates, actual_coordinates):
        return losses.euclidian_distance(eye_center_coordinates, actual_coordinates)
    
def load_motor_model(checkpoint_path, cfg = None):
    if cfg is None:
        cfg = config.build(os.path.dirname(checkpoint_path))

        cfg.checkpoint_path = checkpoint_path
        cfg.save_dir = os.path.dirname(checkpoint_path)

    motor_model = MotorCortex(cfg)

    motor_model.model.build((None, *_resolve_input_shape(cfg)))

    return motor_model


# Backwards compatible alias (older code/checkpoints may still reference `motor_cortex`).
motor_cortex = MotorCortex
