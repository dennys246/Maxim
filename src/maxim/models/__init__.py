"""Model definitions used by Maxim."""

from maxim.models.download import (
    download_llm,
    download_tts,
    list_models,
    check_models,
    enable_llm_config,
    LLM_MODELS,
    TTS_MODELS,
    DEFAULT_LLM,
    DEFAULT_TTS,
)

__all__ = [
    "download_llm",
    "download_tts",
    "list_models",
    "check_models",
    "enable_llm_config",
    "LLM_MODELS",
    "TTS_MODELS",
    "DEFAULT_LLM",
    "DEFAULT_TTS",
]
