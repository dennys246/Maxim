from __future__ import annotations

import logging
import os
import threading
from typing import Any

from maxim.utils.logging import warn
from maxim.utils.optional_deps import require_optional_dependency

logger = logging.getLogger(__name__)


class _LlamaCppBackend:
    """llama.cpp backend with full configuration support."""

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        self._llm = None
        self._lock = threading.Lock()
        self.requires_prompt_formatting = True

    def _ensure(self) -> bool:
        if self._llm is not None:
            return True

        with self._lock:
            # Double-check after acquiring lock
            if self._llm is not None:
                return True

            # Requested-but-missing dependency is a SETUP error: raise loudly
            # with an actionable hint instead of returning False and letting
            # the caller mask it as a generic "no local model" failure.
            llama_mod = require_optional_dependency(
                "llama_cpp", extra="llm-llama", feature="Local llama.cpp backend"
            )
            Llama = llama_mod.Llama

            model_path = str(self.cfg.model_path or "").strip()
            if not model_path or not os.path.exists(model_path):
                # Auto-download if model is in the registry
                profile = str(self.cfg.profile or self.cfg.model_base or "").strip()
                if profile:
                    try:
                        from maxim.models.download import download_llm, LLM_MODELS

                        if profile in LLM_MODELS:
                            logger.info("Model not found at %s — downloading %s...", model_path, profile)
                            if download_llm(profile):
                                logger.info("Download complete: %s", profile)
                            else:
                                warn("Auto-download failed for %s", profile)
                                return False
                        else:
                            warn("LLM model_path not found and no download available: %s", model_path)
                            return False
                    except Exception as e:
                        warn("Auto-download failed: %s", e)
                        return False
                else:
                    warn("LLM model_path not found: %s", model_path)
                    return False
            if not os.path.exists(model_path):
                # Re-resolve: downloader may save with different casing than profile expects
                from maxim.models.language.config import build_model_path

                model_base = str(self.cfg.model_base or "").strip()
                quant = str(self.cfg.quantization or "Q4_K_M").strip()
                resolved = build_model_path(model_base, quant)
                if os.path.exists(resolved):
                    model_path = resolved
                else:
                    warn("LLM model_path still not found after download: %s", model_path)
                    return False

            try:
                # Sanity-check n_ctx against model name hints to avoid
                # GGML_ASSERT crashes from over-sized KV cache allocations.
                n_ctx = int(self.cfg.n_ctx)
                model_lower = os.path.basename(model_path).lower()
                ctx_hints = {"2k": 2048, "4k": 4096, "8k": 8192, "16k": 16384}
                for hint, limit in ctx_hints.items():
                    if hint in model_lower and n_ctx > limit:
                        warn(
                            "n_ctx=%d exceeds model's advertised %s context (%s). "
                            "Clamping to %d to avoid backend allocation failure.",
                            n_ctx,
                            hint,
                            model_lower,
                            limit,
                        )
                        n_ctx = limit
                        break

                # Build kwargs with all supported options
                llama_kwargs: dict[str, Any] = {
                    "model_path": model_path,
                    "n_ctx": n_ctx,
                    "verbose": False,
                }

                # GPU layers (-1 = all available)
                n_gpu_layers = getattr(self.cfg, "n_gpu_layers", -1)
                if n_gpu_layers is not None:
                    llama_kwargs["n_gpu_layers"] = int(n_gpu_layers)

                # Thread count (None = auto)
                n_threads = getattr(self.cfg, "n_threads", None)
                if n_threads is not None:
                    llama_kwargs["n_threads"] = int(n_threads)

                # Seed (-1 = random)
                seed = getattr(self.cfg, "seed", -1)
                if seed is not None and seed != -1:
                    llama_kwargs["seed"] = int(seed)

                self._llm = Llama(**llama_kwargs)
                return True
            except Exception as e:
                warn("Failed to load LLM model (%s): %s", model_path, e)
                self._llm = None
                return False

    def warmup(self) -> bool:
        """Pre-load the model. Call at startup to avoid first-request latency."""
        return self._ensure()

    def complete(
        self,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float,
        stop: tuple[str, ...],
        top_p: float | None = None,
        top_k: int | None = None,
        repeat_penalty: float | None = None,
    ) -> str:
        if not self._ensure():
            return ""

        # Build generation kwargs
        gen_kwargs: dict[str, Any] = {
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "stop": list(stop) if stop else None,
        }

        # Optional parameters from config
        if top_p is not None:
            gen_kwargs["top_p"] = float(top_p)
        elif hasattr(self.cfg, "top_p"):
            gen_kwargs["top_p"] = float(self.cfg.top_p)

        if top_k is not None:
            gen_kwargs["top_k"] = int(top_k)
        elif hasattr(self.cfg, "top_k"):
            gen_kwargs["top_k"] = int(self.cfg.top_k)

        if repeat_penalty is not None:
            gen_kwargs["repeat_penalty"] = float(repeat_penalty)
        elif hasattr(self.cfg, "repeat_penalty"):
            gen_kwargs["repeat_penalty"] = float(self.cfg.repeat_penalty)

        out = self._llm(str(prompt), **gen_kwargs)
        try:
            return str(out["choices"][0]["text"])
        except Exception:
            return ""

    def unload(self) -> None:
        """Unload the model to free memory."""
        with self._lock:
            if self._llm is not None:
                del self._llm
                self._llm = None
