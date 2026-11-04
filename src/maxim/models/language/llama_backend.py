from __future__ import annotations

import logging
import os
import threading
from typing import Any

from maxim.utils.logging import warn
from maxim.utils.optional_deps import require_optional_dependency


logger = logging.getLogger(__name__)


def llama_kwargs(
    model_path: str,
    *,
    n_ctx: int,
    n_gpu_layers: int | None = -1,
    n_threads: int | None = None,
    seed: int | None = -1,
    vocab_only: bool = False,
) -> dict[str, Any]:
    """The ``Llama(**kwargs)`` for every in-process llama-cpp-python load: ``_LlamaCppBackend`` and ``closed_local``
    share it (#1315 §1A.3), so a future GGML flag lands once. The spawned ``llama-server``
    (``runtime/local_server_spawner.py``) takes command-line flags and builds its own.

    ``n_ctx`` is explicit, always: each caller owns its clamp (llama-cpp reads ``n_ctx=0`` as the model's full
    trained context; ``closed_local`` never passes it). ``n_gpu_layers`` -1 offloads every layer, ``n_threads``
    ``None`` lets llama.cpp choose, and ``seed`` -1 is random.
    """
    kwargs: dict[str, Any] = {"model_path": model_path, "n_ctx": int(n_ctx), "verbose": False}
    if n_gpu_layers is not None:
        kwargs["n_gpu_layers"] = int(n_gpu_layers)
    if n_threads is not None:
        kwargs["n_threads"] = int(n_threads)
    if seed is not None and seed != -1:
        kwargs["seed"] = int(seed)
    if vocab_only:
        kwargs["vocab_only"] = True
    return kwargs


class _LlamaCppBackend:
    """llama.cpp backend with full configuration support."""

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        self._llm = None
        self._lock = threading.Lock()
        self.requires_prompt_formatting = True
        # A missing model this backend may not download (#1343): asked once, then every later call fails fast.
        self._download_refused = False
        self._security_refusal: BaseException | None = None  # a remembered SECURITY_REFUSALS refusal (#1315 item 10b)

    def _fetch_missing_model(self, model_path: str) -> bool:
        """Put the profile's GGUF on disk through the consent gate, or refuse (#1343).

        Never prompts: this runs on the warmup thread and inside every ``complete()``, where a y/N read from stdin
        could take a line the user typed to the agent. Only a standing grant downloads (``--auto-download``,
        ``llm.auto_download``, ``MAXIM_AUTO_DOWNLOAD_MODELS``); the startup sites ask. A refusal is remembered, so
        later calls fail fast instead of asking the gate again, for this backend's life (a file that appears later, from
        another process, still loads: the path check runs first). An exception is reported and not remembered. A
        security refusal (the offline latch, #1315 item 10b) is remembered AND re-raised, the same one each call, so it
        stays typed without booking a new ledger entry per ``complete()``.
        """
        if self._security_refusal is not None:
            raise self._security_refusal
        if self._download_refused:
            return False
        profile = str(self.cfg.profile or self.cfg.model_base or "").strip()
        if not profile:
            warn("LLM model_path not found: %s", model_path)
            return False
        from maxim.models.download import ensure_available

        from maxim.exceptions import SECURITY_REFUSALS

        try:
            available = ensure_available(profile, interactive=False, logger=logger)
        except SECURITY_REFUSALS as refusal:
            self._security_refusal = refusal
            raise
        except Exception as e:
            warn("Model download for %s failed: %s", profile, e)
            return False
        if not available:
            self._download_refused = True
            warn("LLM model for %s is not on disk (%s) and was not downloaded", profile, model_path)
        return available

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
            llama_mod = require_optional_dependency("llama_cpp", extra="llm-llama", feature="Local llama.cpp backend")
            Llama = llama_mod.Llama

            model_path = str(self.cfg.model_path or "").strip()
            if (not model_path or not os.path.exists(model_path)) and not self._fetch_missing_model(model_path):
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

                self._llm = Llama(
                    **llama_kwargs(
                        model_path,
                        n_ctx=n_ctx,
                        n_gpu_layers=getattr(self.cfg, "n_gpu_layers", -1),
                        n_threads=getattr(self.cfg, "n_threads", None),
                        seed=getattr(self.cfg, "seed", -1),
                    )
                )
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
