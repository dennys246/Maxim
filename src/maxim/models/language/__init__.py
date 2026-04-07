"""Language model helpers (optional).

This package provides lightweight, swappable backends for local LLM routing.
"""

from __future__ import annotations

from maxim.models.language.router import (
    DEFAULT_QUANTIZATION,
    LLMConfig,
    LLMRouter,
    QUANTIZATION_LEVELS,
    build_model_path,
    list_llm_profiles,
    list_prompt_styles,
    list_quantization_levels,
    load_llm_config,
    normalize_llm_profile,
)

__all__ = [
    "DEFAULT_QUANTIZATION",
    "LLMConfig",
    "LLMRouter",
    "QUANTIZATION_LEVELS",
    "build_model_path",
    "list_llm_profiles",
    "list_prompt_styles",
    "list_quantization_levels",
    "load_llm_config",
    "normalize_llm_profile",
]
