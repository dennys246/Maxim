"""Central handling for optional third-party dependencies.

Maxim ships a small core (``numpy``, ``scipy``, ``pyyaml``, ``json-repair``)
and gates everything heavier behind optional extras (``llm-anthropic``,
``vision``, ``semantic``, ...). Historically each consumer rolled its own
``try: import X except ImportError: warn(...); return None`` block, and the
behaviour on a missing dependency was wildly inconsistent — some raised, some
logged a swallowed WARNING, some failed completely silently.

The 2026-06-05 cloud-dispatch incident is the canonical failure: the
``anthropic`` package was not installed, ``_AnthropicBackend._ensure_client``
logged a single WARNING and returned ``None``, the router treated the empty
response like a transient hiccup, fell through to "No eligible LLM providers",
and the whole sim completed with ``cost=$0`` and every action a
``_llm_unavailable`` fallback. The missing backbone was invisible.

This module makes the behaviour uniform and intentional, split by INTENT:

- :func:`require_optional_dependency` — for a feature the caller EXPLICITLY
  requested (a configured cloud LLM profile, ``--vision``, a selected TTS
  voice, a robot controller). A missing dependency here is a SETUP error, so
  this **raises** :class:`OptionalDependencyError` with an actionable
  ``pip install`` hint. The error subclasses ``ImportError`` (so legacy
  ``except ImportError`` sites still catch it) but is a dedicated type so
  fail-soft layers like the LLM router can distinguish "requested feature not
  installed → abort loudly" from "transient runtime failure → try next
  provider".

- :func:`optional_dependency_available` — for a capability PROBE (GPU
  detection, "use X if it happens to be present"). Absence is normal here, so
  this just returns a bool and never logs.

- :func:`warn_optional_fallback` — for a deliberate DEGRADATION path that has
  a real working fallback (``sentence-transformers`` → bag-of-words,
  ``spacy`` → identity decomposition, ``ddgs`` → regex scraping). The fallback
  is kept; this emits exactly ONE clear, deduped log so the degradation is
  visible instead of silent.

Regression guard: tests/unit/test_optional_deps.py.
"""

from __future__ import annotations

import importlib
import logging
from types import ModuleType

logger = logging.getLogger(__name__)

# Canonical ``import name → extra`` mapping. Generalises the partial
# ``_EXTRA_IMPORT_MAP`` that used to live in ``runtime/leader_proxy.py``.
# Keyed by the *import* name (what you ``import``), valued by the pyproject
# extra (what you ``pip install pymaxim[...]``). When an import name is not in
# this table the fix hint falls back to ``pip install <import_name>``.
EXTRA_FOR_IMPORT: dict[str, str] = {
    # vision
    "cv2": "vision",
    "onnxruntime": "vision",
    # audio
    "faster_whisper": "audio",
    "ctranslate2": "audio",
    "av": "audio",
    # robot
    "reachy_mini": "reachy",
    # search
    "ddgs": "search",
    "duckduckgo_search": "search",
    # training
    "tensorflow": "training",
    "keras": "training",
    # local LLM (llama.cpp)
    "llama_cpp": "llm-llama",
    # local LLM (torch / transformers)
    "torch": "llm-torch",
    "transformers": "llm-torch",
    "accelerate": "llm-torch",
    "bitsandbytes": "llm-torch",
    "sentencepiece": "llm-torch",
    "huggingface_hub": "llm-torch",
    # cloud LLM
    "anthropic": "llm-anthropic",
    "openai": "llm-openai",
    "tiktoken": "llm-openai",
    # tts
    "piper": "tts",
    # yolo
    "ultralytics": "yolo",
    "lap": "yolo",
    # comms
    "twilio": "comms",
    "fastapi": "comms",
    "uvicorn": "comms",
    # semantic
    "sentence_transformers": "semantic",
    "spacy": "semantic",
    # temporal
    "dateparser": "temporal",
    # database
    "psycopg": "database",
    "pgvector": "database",
}


def _fix_hint(import_name: str, extra: str | None) -> str:
    """Build the actionable ``pip install`` hint for a missing dependency."""
    if extra:
        return f"Fix: pip install pymaxim[{extra}]"
    return f"Fix: pip install {import_name}"


class OptionalDependencyError(ImportError):
    """An explicitly-requested feature's optional dependency is not installed.

    Subclasses ``ImportError`` so existing ``except ImportError`` handlers keep
    working, while giving fail-soft layers (the LLM router in particular) a
    dedicated type to re-raise instead of silently falling through.

    Access patterns: ``.import_name``, ``.extra``, ``.feature``, ``.fix_hint``.
    """

    def __init__(
        self,
        import_name: str,
        *,
        extra: str | None = None,
        feature: str | None = None,
        cause: BaseException | None = None,
    ) -> None:
        self.import_name = import_name
        self.extra = extra if extra is not None else EXTRA_FOR_IMPORT.get(import_name)
        self.feature = feature
        self.fix_hint = _fix_hint(import_name, self.extra)
        prefix = f"{feature}: " if feature else ""
        super().__init__(f"{prefix}required dependency '{import_name}' is not installed. {self.fix_hint}")
        if cause is not None:
            self.__cause__ = cause


def require_optional_dependency(
    import_name: str,
    *,
    extra: str | None = None,
    feature: str | None = None,
) -> ModuleType:
    """Import and return *import_name*, or raise :class:`OptionalDependencyError`.

    Use at sites where the feature was EXPLICITLY requested — a configured
    backend, a ``--vision`` flag, a selected TTS voice. A missing dependency
    here is a setup error that must fail loudly, never degrade silently.

    Args:
        import_name: the module to import (e.g. ``"anthropic"``, ``"cv2"``).
        extra: the pyproject extra to suggest. Defaults to the
            :data:`EXTRA_FOR_IMPORT` lookup, then to ``pip install <name>``.
        feature: human-readable feature name for the error message
            (e.g. ``"Anthropic backend"``).

    Returns:
        The imported module.

    Raises:
        OptionalDependencyError: if the module cannot be imported.
    """
    try:
        return importlib.import_module(import_name)
    except ImportError as e:
        raise OptionalDependencyError(import_name, extra=extra, feature=feature, cause=e) from e


def optional_dependency_available(import_name: str) -> bool:
    """Return ``True`` iff *import_name* is importable.

    Use for capability PROBES (GPU detection, "use X if present") where the
    absence of the dependency is a normal, expected condition rather than a
    setup error. Never logs.
    """
    try:
        importlib.import_module(import_name)
        return True
    except ImportError:
        return False


# Deduped "we degraded because X is missing" notices. Process-scoped warn-once,
# same shape as the ``_warned_envs`` set in runtime/config_loader.py.
_warned_fallbacks: set[str] = set()


def warn_optional_fallback(
    import_name: str,
    *,
    fallback: str,
    extra: str | None = None,
    feature: str | None = None,
    log: logging.Logger | None = None,
) -> None:
    """Emit exactly ONE clear, deduped log for a deliberate fallback path.

    Use at sites that have a real working fallback when an optional dependency
    is absent (``sentence-transformers`` → bag-of-words, ``spacy`` → identity,
    ``ddgs`` → regex). The fallback is kept by the caller; this just makes the
    degradation visible once per process instead of silent.

    Args:
        import_name: the absent dependency.
        fallback: short description of what the caller does instead
            (e.g. ``"using hash-based bag-of-words embeddings"``).
        extra / feature: as in :func:`require_optional_dependency`.
        log: logger to use; defaults to this module's logger.
    """
    key = f"{import_name}:{feature or ''}"
    if key in _warned_fallbacks:
        return
    _warned_fallbacks.add(key)
    resolved_extra = extra if extra is not None else EXTRA_FOR_IMPORT.get(import_name)
    hint = _fix_hint(import_name, resolved_extra)
    prefix = f"{feature}: " if feature else ""
    (log or logger).warning(
        "%soptional dependency '%s' not installed; %s. %s",
        prefix,
        import_name,
        fallback,
        hint,
    )


def _reset_optional_dep_warnings_for_test() -> None:
    """Clear the warn-once dedup set. Test-only (autouse fixture in conftest)."""
    _warned_fallbacks.clear()
