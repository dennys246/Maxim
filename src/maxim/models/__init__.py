"""Model definitions used by Maxim."""

from typing import TYPE_CHECKING, Any

from maxim.models.download import (
    download_llm,
    download_tts,
    list_models,
    check_models,
    enable_llm_config,
    llm_models,
    TTS_MODELS,
    DEFAULT_LLM,
    DEFAULT_TTS,
)

if TYPE_CHECKING:
    from maxim.models.download import LLM_MODELS  # resolved through llm_models() at runtime (below)

__all__ = [
    "download_llm",
    "download_tts",
    "list_models",
    "check_models",
    "enable_llm_config",
    "llm_models",
    "LLM_MODELS",
    "TTS_MODELS",
    "DEFAULT_LLM",
    "DEFAULT_TTS",
]


def __getattr__(name: str) -> Any:
    # ``maxim.models.LLM_MODELS`` stays public, resolved through the ensuring accessor so the user's profiles.yml
    # downloads are merged before it is read (#1315 step 0a: the merge no longer runs at import).
    if name == "LLM_MODELS":
        return llm_models()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
