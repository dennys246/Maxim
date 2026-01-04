import atexit
import copy
import json
import os
from pathlib import Path
from typing import Any, Optional

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SAVE_ROOT = _REPO_ROOT / "data" / "models" / "MotorCortex"
LEGACY_SAVE_ROOT = _REPO_ROOT / "sandbox" / "models" / "MotorCortex"
DEFAULT_CHECKPOINT_FILENAME = "motor_cortex.keras"
DEFAULT_CONFIG_FILENAME = "motor_cortex.json"


def _normalize_save_dir(path: Optional[str]) -> str:
    """
    Normalize save_dir inputs so downstream string concatenation keeps working.
    Always returns a POSIX-style path ending with '/' (many callers expect it).
    """
    if not path:
        path = DEFAULT_SAVE_ROOT.as_posix()
    save_dir = Path(path).as_posix()
    if not save_dir.endswith("/"):
        save_dir = f"{save_dir}/"
    return save_dir


def _normalize_checkpoint(save_dir: str, checkpoint: Optional[str], default_filename: str = DEFAULT_CHECKPOINT_FILENAME) -> str:
    """
    Resolve a checkpoint path so that it respects the configured save_dir while
    still honouring explicit absolute paths or already-resolved relatives.
    """
    save_dir_path = Path(save_dir)
    if not checkpoint:
        return (save_dir_path / default_filename).as_posix()

    checkpoint_path = Path(checkpoint)

    # Leave explicit absolute paths untouched.
    if checkpoint_path.is_absolute():
        return checkpoint_path.as_posix()

    # If the checkpoint already exists relative to the current working dir, keep it.
    if checkpoint_path.exists():
        return checkpoint_path.as_posix()

    # If checkpoint already points inside save_dir, keep relative layout.
    try:
        relative = checkpoint_path.relative_to(save_dir_path)
        return (save_dir_path / relative).as_posix()
    except ValueError:
        pass

    # Respect user-provided relative prefixes like "./" or "../" by leaving them alone.
    first_segment = checkpoint_path.parts[0] if checkpoint_path.parts else ""
    if first_segment in (".", ".."):
        return checkpoint_path.as_posix()

    # Otherwise treat the checkpoint as relative to save_dir.
    return (save_dir_path / checkpoint_path).as_posix()


config_template: dict[str, Any] = {
    # I/O
    "save_dir": DEFAULT_SAVE_ROOT.as_posix(),
    "checkpoint_path": (DEFAULT_SAVE_ROOT / DEFAULT_CHECKPOINT_FILENAME).as_posix(),
    "dataset": (Path("data") / "images").as_posix() + "/",
    # Data / input
    "resolution": [1920, 1080],  # [width, height] for camera frames
    "channels": 3,
    "input_shape": 256,  # int, [H,W], or [H,W,C]
    # Model
    "architecture": "convnext_tiny",
    "convnext_depths": [3, 3, 9, 3],
    "convnext_dims": [96, 192, 384, 768],
    "layer_scale_init_value": 1e-6,
    "head_dim": 256,
    "dropout": 0.0,
    "output_dim": 7,
    "final_activation": "tanh",
    # Motor outputs are ordered as: [x, y, z, roll, pitch, yaw, duration]
    "movement_delta_limits": [5.0, 5.0, 10.0, 10.0, 10.0, 10.0],  # per-step max delta (mm/deg)
    "movement_pose_limits": {
        "x": [-30.0, 30.0],
        "y": [-30.0, 30.0],
        "z": [-60.0, 60.0],
        "roll": [-30.0, 30.0],
        "pitch": [-30.0, 30.0],
        "yaw": [-45.0, 45.0],
    },
    "movement_duration_limits": [0.1, 2.0],
    # Optimizer / training
    "epochs": 10,
    "batch_size": 2,
    "learning_rate": 1e-5,
    "beta_1": 0.5,
    "beta_2": 0.9,
    # Misc
    "rebuild": False,
}


def _config_file_from_path(path: str | os.PathLike[str]) -> Path:
    requested = Path(path)
    if requested.suffix.lower() == ".json":
        return requested
    return requested / DEFAULT_CONFIG_FILENAME


class build:
    def __init__(self, config_filepath: str | os.PathLike[str]):
        self.config_filepath = _config_file_from_path(config_filepath).as_posix()
        self._config_dir = Path(self.config_filepath).parent.as_posix()

        requested = Path(config_filepath)

        config_path = Path(self.config_filepath)
        if config_path.exists() and config_path.is_file():
            print(f"Loading config file: {self.config_filepath}")
            config_json = self.load_config(self.config_filepath)
        else:
            legacy_config = LEGACY_SAVE_ROOT / DEFAULT_CONFIG_FILENAME
            is_default_path = config_path == (DEFAULT_SAVE_ROOT / DEFAULT_CONFIG_FILENAME)

            if is_default_path and legacy_config.exists():
                print(f"Loading legacy config file: {legacy_config.as_posix()}")
                config_json = self.load_config(legacy_config.as_posix())

                try:
                    config_json["save_dir"] = Path(self._config_dir).as_posix()
                    legacy_checkpoint = config_json.get("checkpoint_path") or config_json.get("checkpoint")
                    filename = Path(str(legacy_checkpoint)).name if legacy_checkpoint else DEFAULT_CHECKPOINT_FILENAME
                    config_json["checkpoint_path"] = (Path(config_json["save_dir"]) / filename).as_posix()
                except Exception:
                    pass
            else:
                print("WARNING: Config not found, building from default template...")
                config_json = copy.deepcopy(config_template)

                # If user passed a directory, default save_dir/checkpoint under it.
                try:
                    config_json["save_dir"] = Path(self._config_dir).as_posix()
                    config_json["checkpoint_path"] = (
                        Path(config_json["save_dir"]) / DEFAULT_CHECKPOINT_FILENAME
                    ).as_posix()
                except Exception:
                    pass

        # If the caller passed a directory (not a JSON file), treat that directory as the
        # canonical save_dir and keep checkpoint_path inside it.
        try:
            if requested.suffix.lower() != ".json":
                config_json["save_dir"] = Path(self._config_dir).as_posix()
                legacy_checkpoint = config_json.get("checkpoint_path") or config_json.get("checkpoint")
                filename = Path(str(legacy_checkpoint)).name if legacy_checkpoint else DEFAULT_CHECKPOINT_FILENAME
                config_json["checkpoint_path"] = (Path(config_json["save_dir"]) / filename).as_posix()
        except Exception:
            pass

        # Backwards compatibility: old repos stored datasets under experiments/maxim/.
        try:
            dataset = config_json.get("dataset")
            if isinstance(dataset, str) and "experiments/maxim/images" in dataset:
                config_json["dataset"] = config_template["dataset"]
        except Exception:
            pass

        # Backwards compatibility: old configs used "checkpoint".
        if "checkpoint_path" not in config_json and "checkpoint" in config_json:
            config_json["checkpoint_path"] = config_json.get("checkpoint")

        # Backwards compatibility: upgrade legacy 2D (u,v) motor outputs to 7D head movement outputs.
        try:
            legacy_output_dim = int(config_json.get("output_dim", 0) or 0)
        except Exception:
            legacy_output_dim = 0
        if legacy_output_dim == 2 and "movement_delta_limits" not in config_json:
            print("Upgrading motor cortex config: output_dim 2 -> 7 (head movement).")
            config_json["output_dim"] = config_template["output_dim"]
            if config_json.get("final_activation") is None:
                config_json["final_activation"] = config_template["final_activation"]
            config_json.setdefault("movement_delta_limits", config_template["movement_delta_limits"])
            config_json.setdefault("movement_pose_limits", config_template["movement_pose_limits"])
            config_json.setdefault("movement_duration_limits", config_template["movement_duration_limits"])

        self.configure(**config_json)

        atexit.register(self.save_config)

    def __repr__(self):
        return "\n".join([f"{key}: {value}" for key, value in self.__dict__.items()])

    def save_config(self, config_filepath: Optional[str] = None) -> None:
        if config_filepath:
            self.config_filepath = _config_file_from_path(config_filepath).as_posix()
            self._config_dir = Path(self.config_filepath).parent.as_posix()

        dest_dir = os.path.dirname(self.config_filepath) or "."
        os.makedirs(dest_dir, exist_ok=True)

        with open(self.config_filepath, "w", encoding="utf-8") as config_file:
            json.dump(self.dump(), config_file, indent=4)

    def load_config(self, config_path: str) -> dict[str, Any]:
        with open(config_path, "r", encoding="utf-8") as config_file:
            config_json = json.load(config_file)

        if not isinstance(config_json, dict):
            raise ValueError(
                f"Expected top-level JSON object in {config_path}, got {type(config_json).__name__}"
            )
        return config_json

    def configure(self, **kwargs: Any) -> None:
        save_dir = _normalize_save_dir(kwargs.get("save_dir"))
        checkpoint_path = kwargs.get("checkpoint_path")

        self.save_dir = save_dir
        self.checkpoint_path = _normalize_checkpoint(self.save_dir, checkpoint_path, DEFAULT_CHECKPOINT_FILENAME)

        self.dataset = kwargs.get("dataset") or config_template["dataset"]
        self.resolution = kwargs.get("resolution") or config_template["resolution"]
        self.channels = int(kwargs.get("channels", config_template["channels"]) or config_template["channels"])
        self.input_shape = kwargs.get("input_shape", config_template["input_shape"])

        self.architecture = kwargs.get("architecture") or config_template["architecture"]
        self.convnext_depths = list(kwargs.get("convnext_depths") or config_template["convnext_depths"])
        self.convnext_dims = list(kwargs.get("convnext_dims") or config_template["convnext_dims"])
        self.layer_scale_init_value = float(
            kwargs.get("layer_scale_init_value", config_template["layer_scale_init_value"])
            or config_template["layer_scale_init_value"]
        )

        self.head_dim = int(kwargs.get("head_dim", config_template["head_dim"]) or config_template["head_dim"])
        self.dropout = float(kwargs.get("dropout", config_template["dropout"]) or config_template["dropout"])
        self.output_dim = int(kwargs.get("output_dim", config_template["output_dim"]) or config_template["output_dim"])
        self.final_activation = kwargs.get("final_activation", config_template["final_activation"])

        self.movement_delta_limits = list(
            kwargs.get("movement_delta_limits") or config_template["movement_delta_limits"]
        )
        self.movement_pose_limits = dict(
            kwargs.get("movement_pose_limits") or config_template["movement_pose_limits"]
        )
        self.movement_duration_limits = list(
            kwargs.get("movement_duration_limits") or config_template["movement_duration_limits"]
        )

        self.epochs = int(kwargs.get("epochs", config_template["epochs"]) or config_template["epochs"])
        self.batch_size = int(kwargs.get("batch_size", config_template["batch_size"]) or config_template["batch_size"])
        self.learning_rate = float(
            kwargs.get("learning_rate", config_template["learning_rate"]) or config_template["learning_rate"]
        )
        self.beta_1 = float(kwargs.get("beta_1", config_template["beta_1"]) or config_template["beta_1"])
        self.beta_2 = float(kwargs.get("beta_2", config_template["beta_2"]) or config_template["beta_2"])

        self.rebuild = bool(kwargs.get("rebuild", config_template["rebuild"]))

    def dump(self) -> dict[str, Any]:
        return {
            "save_dir": self.save_dir,
            "checkpoint_path": self.checkpoint_path,
            "dataset": self.dataset,
            "resolution": self.resolution,
            "channels": self.channels,
            "input_shape": self.input_shape,
            "architecture": self.architecture,
            "convnext_depths": self.convnext_depths,
            "convnext_dims": self.convnext_dims,
            "layer_scale_init_value": self.layer_scale_init_value,
            "head_dim": self.head_dim,
            "dropout": self.dropout,
            "output_dim": self.output_dim,
            "final_activation": self.final_activation,
            "movement_delta_limits": self.movement_delta_limits,
            "movement_pose_limits": self.movement_pose_limits,
            "movement_duration_limits": self.movement_duration_limits,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "beta_1": self.beta_1,
            "beta_2": self.beta_2,
            "rebuild": self.rebuild,
        }


def load_config(config_filepath: str | os.PathLike[str]) -> build:
    return build(config_filepath)
