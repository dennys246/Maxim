"""CLIP vision encoder — image → 512-dim embedding for the P4 cross-modal path.

P4 Stage 2 implementation. Wraps ``sentence_transformers.SentenceTransformer``
loading the ``clip-ViT-B-32`` variant, which exposes both vision and text
encoders through the same class — encode a PIL image and you get a 512-dim
numpy array; encode a string and you get a 512-dim numpy array in the same
shared space. Stage 2's mug test consumes the vision side; Stage 3's Arm B
and Arm C consume both sides through the same underlying model instance.

Design notes
============

- **Single image at a time.** No batching in Stage 1 shape; callers that
  want to pre-encode Oxford Flowers-102 into arrays loop externally and
  handle the caching. Stage 3 may revisit if batching matters for wall
  clock, but premature batching is premature optimization — ~510 images
  for the Stage 2 calibration sweep run in seconds on a 5080.
- **Module-level singleton cache** matches the pattern in
  ``similarity/encoder.py``: first call loads the model, subsequent
  calls on the same process reuse it. Thread-safety follows the same
  "worst case two threads load simultaneously, second wins with identical
  model" contract.
- **No fallback.** LinguisticEncoder falls back to a bag-of-words hash
  when sentence-transformers is missing because test environments can
  exercise its pipeline without the dep. For CLIP there's no meaningful
  fallback — a hash of raw image bytes gives no cluster structure, and
  the Stage 2 mug test exists to measure CLIP-specific geometry.
  Callers without the ``semantic`` extra MUST ``pytest.importorskip`` at
  test time; production callers raise ``RuntimeError`` loudly.
- **Single-object only.** Multi-object detection + per-object cross-modal
  binding is P4-MV, deferred post-1.0 per the plan doc's "Deferred" section.
- **Lazy import at module level** so ``import maxim.models.vision.clip_encoder``
  does NOT trigger ``torch`` / ``sentence_transformers`` loading. The
  first call to ``VisionEncoder.encode_image`` triggers it.
- **Encoder identity is caller-visible**. Stage 3 arms A/B/C need to swap
  the underlying ``SentenceTransformer`` model name by configuration, not
  code — so ``VisionEncoder`` exposes ``model_name`` in its ``__init__``
  and the singleton cache is keyed by that name. Default ``clip-ViT-B-32``.

The companion encoder for Stage 3's Arm B text side is the SAME
``clip-ViT-B-32`` model — ``SentenceTransformer("clip-ViT-B-32").encode(str)``
returns text embeddings in the same 512-dim space as the image side.
Callers can instantiate a single ``VisionEncoder(model_name="clip-ViT-B-32")``
and use its underlying model for both image and text encoding via
``encode_image`` and ``encode_text`` respectively. This is how Arm B
(CLIP-text + CLIP-vision + hippocampus) and Arm C (CLIP-text + CLIP-vision
+ cosine baseline) share an encoder without instantiating two models.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    from PIL.Image import Image

logger = logging.getLogger(__name__)


# Module-level singleton cache. Keyed by model name so callers can
# instantiate multiple encoders with different model names (Stage 3
# Arms A/B/C) without stepping on each other's cached models.
_cached_models: dict[str, Any] = {}


def _load_model(model_name: str) -> Any:
    """Lazy-load a sentence-transformers model for CLIP vision + text.

    Raises ``RuntimeError`` if the ``semantic`` extra is not installed.
    No fallback — CLIP-specific geometry has no meaningful proxy.
    """
    cached = _cached_models.get(model_name)
    if cached is not None:
        return cached

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise RuntimeError(
            "VisionEncoder requires the 'semantic' optional extra. "
            "Install with: pip install 'pymaxim[semantic]' "
            "(pulls sentence-transformers + torch)."
        ) from e

    logger.info("VisionEncoder loading model: %s", model_name)
    model = SentenceTransformer(model_name)
    _cached_models[model_name] = model
    logger.info("VisionEncoder loaded model: %s", model_name)
    return model


@dataclass
class VisionEncoderConfig:
    """Configuration for the CLIP VisionEncoder."""

    # The sentence-transformers CLIP variant. ``clip-ViT-B-32`` is the
    # default because its published OpenCLIP numbers are the ones the P4
    # Stage 3 head-to-head will compare against. Do NOT change the default
    # without re-running the Stage 2 calibration sweep + updating
    # docs/experiments/p4_clip_calibration.md.
    model_name: str = "clip-ViT-B-32"


class VisionEncoder:
    """Encodes images into the substrate's vision-modality bucket.

    Stage 2 usage:

        encoder = VisionEncoder()
        embedding = encoder.encode_image(pil_image)  # 512-dim np.ndarray

    Stage 3 Arms B/C usage (text-side of CLIP through the same model):

        encoder = VisionEncoder()
        text_emb = encoder.encode_text("mug")
        image_emb = encoder.encode_image(pil_image)

    Both methods return 512-dim numpy arrays in CLIP's shared embedding
    space. Calling them on the same ``VisionEncoder`` instance reuses the
    cached underlying ``SentenceTransformer`` — no duplicate weight load.
    """

    def __init__(self, config: VisionEncoderConfig | None = None) -> None:
        self.config = config or VisionEncoderConfig()
        self._model: Any | None = None

    def _ensure_model(self) -> Any:
        if self._model is None:
            self._model = _load_model(self.config.model_name)
        return self._model

    def encode_image(self, image: "Image") -> "np.ndarray":
        """Encode a single PIL image to a 512-dim numpy array.

        Returns the raw unnormalized ``SentenceTransformer.encode`` output
        so callers that want cosine similarity normalize themselves (or
        pass ``normalize_embeddings=True`` to their own cosine helper).
        This matches the LinguisticEncoder convention.
        """
        model = self._ensure_model()
        result = model.encode(image)
        import numpy as np

        return np.asarray(result, dtype=np.float32)

    def encode_text(self, text: str) -> "np.ndarray":
        """Encode a single text string to a 512-dim numpy array via CLIP-text.

        Stage 3 Arms B and C use this to compute text-side CLIP embeddings
        that share the same space as ``encode_image`` output — that's the
        whole point of CLIP and the reason Arm B's "CLIP-text +
        hippocampus" claim is the load-bearing P4 commitment.
        """
        model = self._ensure_model()
        result = model.encode(text)
        import numpy as np

        return np.asarray(result, dtype=np.float32)

    @property
    def model_name(self) -> str:
        return self.config.model_name

    @property
    def embedding_dim(self) -> int:
        """CLIP-ViT-B-32 returns 512-dim embeddings. Hard-coded here
        because calling ``get_sentence_embedding_dimension()`` would
        force the model to load. If we ever swap CLIP variants this
        property needs an update."""
        return 512
