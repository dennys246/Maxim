"""Unit tests for ``maxim.models.vision.clip_encoder.VisionEncoder``.

Every test that actually loads the model is guarded by
``pytest.importorskip("sentence_transformers")`` so CI environments
without the ``semantic`` extra cleanly skip. The first model load is
cached at module level so subsequent tests in the same session reuse
it — running the whole file takes a few seconds on first load, then
~1ms per assertion thereafter.
"""

from __future__ import annotations

import pytest


def test_import_does_not_require_semantic_extra() -> None:
    """``import maxim.models.vision.clip_encoder`` must not pull torch or
    sentence_transformers. Lazy-loading is load-bearing for the
    ``import maxim`` path (pyproject core deps don't include torch)."""
    # Intentionally import WITHOUT importorskip — if this fails, the
    # module is eagerly importing a heavy dep.
    import maxim.models.vision.clip_encoder as ce  # noqa: F401

    # Construction must also be lazy.
    encoder = ce.VisionEncoder()
    assert encoder.model_name == "clip-ViT-B-32"
    assert encoder.embedding_dim == 512
    # Model must NOT be loaded yet.
    assert encoder._model is None


@pytest.mark.requires_model_cache
def test_encode_image_returns_512d_numpy_array() -> None:
    pytest.importorskip("sentence_transformers")
    pytest.importorskip("PIL")

    import numpy as np
    from PIL import Image

    from maxim.models.vision.clip_encoder import VisionEncoder

    encoder = VisionEncoder()
    # 64×64 RGB noise image is enough to exercise the encode path —
    # content doesn't matter for the shape/dtype assertion.
    rng = np.random.default_rng(42)
    img_array = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    img = Image.fromarray(img_array, mode="RGB")

    embedding = encoder.encode_image(img)
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (512,)
    assert embedding.dtype == np.float32
    # Non-zero output — if CLIP produces the zero vector something is
    # catastrophically wrong.
    assert float(np.linalg.norm(embedding)) > 0.0


@pytest.mark.requires_model_cache
def test_encode_text_returns_512d_numpy_array() -> None:
    pytest.importorskip("sentence_transformers")

    import numpy as np

    from maxim.models.vision.clip_encoder import VisionEncoder

    encoder = VisionEncoder()
    embedding = encoder.encode_text("a photograph of a mug")
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (512,)
    assert embedding.dtype == np.float32
    assert float(np.linalg.norm(embedding)) > 0.0


@pytest.mark.requires_model_cache
def test_encode_image_and_text_share_embedding_space() -> None:
    """CLIP's whole point: a text string and a matching image land close
    together in the same 512-dim space. This is the precondition for
    Stage 3 Arm C's baseline cosine-similarity retrieval AND the
    substrate-controlled Arm B's text encoder side.

    We don't assert a specific cosine value because real matching images
    aren't in the test fixture; we just assert both embeddings are in
    the same dim and that their cosine is a finite number. A stronger
    semantic-correctness test lives in Stage 2's calibration sweep."""
    pytest.importorskip("sentence_transformers")
    pytest.importorskip("PIL")

    import numpy as np
    from PIL import Image

    from maxim.models.vision.clip_encoder import VisionEncoder

    encoder = VisionEncoder()
    rng = np.random.default_rng(42)
    img = Image.fromarray(rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8), mode="RGB")

    img_emb = encoder.encode_image(img)
    txt_emb = encoder.encode_text("a photograph")

    assert img_emb.shape == txt_emb.shape == (512,)

    cos_sim = float(np.dot(img_emb, txt_emb) / (np.linalg.norm(img_emb) * np.linalg.norm(txt_emb)))
    assert np.isfinite(cos_sim)
    assert -1.0 <= cos_sim <= 1.0


@pytest.mark.requires_model_cache
def test_singleton_cache_reuses_model_across_instances() -> None:
    """Two ``VisionEncoder`` instances with the same model_name must
    share the underlying SentenceTransformer — instantiating multiple
    encoders should not reload weights. Load-bearing for Stage 3 where
    Arms A/B/C may instantiate separate ``VisionEncoder`` objects."""
    pytest.importorskip("sentence_transformers")

    from maxim.models.vision.clip_encoder import VisionEncoder, _cached_models

    _cached_models.pop("clip-ViT-B-32", None)  # clear for clean test

    enc1 = VisionEncoder()
    enc2 = VisionEncoder()

    # Trigger load on enc1, then enc2 should pick up the cached model.
    _ = enc1._ensure_model()
    cached_after_enc1 = _cached_models.get("clip-ViT-B-32")
    assert cached_after_enc1 is not None

    _ = enc2._ensure_model()
    # Same object identity — not just equal.
    assert enc1._model is enc2._model
    assert enc1._model is cached_after_enc1


def test_missing_semantic_extra_raises_runtime_error(monkeypatch) -> None:
    """When sentence_transformers is not installed, encode_image must
    raise a clear RuntimeError pointing at the install command. We
    simulate the missing-dep state by patching the import inside
    _load_model."""
    import sys

    from maxim.models.vision import clip_encoder as ce

    # Clear cache so _load_model will actually run its import block.
    ce._cached_models.pop("clip-ViT-B-32", None)

    # Temporarily hide sentence_transformers from the import machinery.
    # Save and restore so other tests in the same session aren't
    # affected.
    saved = sys.modules.pop("sentence_transformers", None)
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)

    try:
        encoder = ce.VisionEncoder()
        with pytest.raises(RuntimeError, match="semantic"):
            encoder._ensure_model()
    finally:
        if saved is not None:
            sys.modules["sentence_transformers"] = saved
        else:
            sys.modules.pop("sentence_transformers", None)
        ce._cached_models.pop("clip-ViT-B-32", None)
