"""Shared utilities for Maxim.

The names below resolve on first access (PEP 562; #1315 step 0a, V9): importing ``maxim.utils`` or one of its
submodules no longer imports every subpackage, so a narrow import loads a narrow closure.
``from maxim.utils import <submodule>`` still works through Python's submodule fallback."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # the real types for mypy; at runtime each name is imported on first access
    from maxim.utils.structured_logging import (
        AbstractionBuffer,
        AgenticVerbosity,
        EventCategory,
        EVENT_VERBOSITY,
        LogRecord,
        StructuredFormatter,
        compact_detection,
        compact_vision_event,
        configure_agentic_verbosity,
        get_abstraction_buffer,
        log_agentic,
        log_structured,
    )
    from maxim.utils.internet_access import (
        Citation,
        EffectiveInternetPolicy,
        InternetAccessPolicy,
        InternetAccessState,
        load_internet_access,
        load_internet_policy,
        save_internet_access,
        save_internet_policy,
        set_internet_access,
    )
    from maxim.utils.web_cache import (
        CacheEntry,
        WebCache,
        get_web_cache,
        set_web_cache,
    )
    from maxim.utils.content_safety import (
        ContentSafetyChecker,
        SafetyCheckResult,
        SafetyViolation,
        check_content_safety,
        check_url_safety,
        get_content_safety_checker,
    )
    from maxim.utils.response_output import (
        ResponseOutput,
        print_response,
    )

# Each public name -> the module that defines it.
_LAZY: dict[str, str] = {
    "AbstractionBuffer": "maxim.utils.structured_logging",
    "AgenticVerbosity": "maxim.utils.structured_logging",
    "EventCategory": "maxim.utils.structured_logging",
    "EVENT_VERBOSITY": "maxim.utils.structured_logging",
    "LogRecord": "maxim.utils.structured_logging",
    "StructuredFormatter": "maxim.utils.structured_logging",
    "compact_detection": "maxim.utils.structured_logging",
    "compact_vision_event": "maxim.utils.structured_logging",
    "configure_agentic_verbosity": "maxim.utils.structured_logging",
    "get_abstraction_buffer": "maxim.utils.structured_logging",
    "log_agentic": "maxim.utils.structured_logging",
    "log_structured": "maxim.utils.structured_logging",
    "Citation": "maxim.utils.internet_access",
    "EffectiveInternetPolicy": "maxim.utils.internet_access",
    "InternetAccessPolicy": "maxim.utils.internet_access",
    "InternetAccessState": "maxim.utils.internet_access",
    "load_internet_access": "maxim.utils.internet_access",
    "load_internet_policy": "maxim.utils.internet_access",
    "save_internet_access": "maxim.utils.internet_access",
    "save_internet_policy": "maxim.utils.internet_access",
    "set_internet_access": "maxim.utils.internet_access",
    "CacheEntry": "maxim.utils.web_cache",
    "WebCache": "maxim.utils.web_cache",
    "get_web_cache": "maxim.utils.web_cache",
    "set_web_cache": "maxim.utils.web_cache",
    "ContentSafetyChecker": "maxim.utils.content_safety",
    "SafetyCheckResult": "maxim.utils.content_safety",
    "SafetyViolation": "maxim.utils.content_safety",
    "check_content_safety": "maxim.utils.content_safety",
    "check_url_safety": "maxim.utils.content_safety",
    "get_content_safety_checker": "maxim.utils.content_safety",
    "ResponseOutput": "maxim.utils.response_output",
    "print_response": "maxim.utils.response_output",
}

__all__ = [
    # Structured logging
    "AbstractionBuffer",
    "AgenticVerbosity",
    "EventCategory",
    "EVENT_VERBOSITY",
    "LogRecord",
    "StructuredFormatter",
    "compact_detection",
    "compact_vision_event",
    "configure_agentic_verbosity",
    "get_abstraction_buffer",
    "log_agentic",
    "log_structured",
    # Internet access
    "Citation",
    "EffectiveInternetPolicy",
    "InternetAccessPolicy",
    "InternetAccessState",
    "load_internet_access",
    "load_internet_policy",
    "save_internet_access",
    "save_internet_policy",
    "set_internet_access",
    # Web cache
    "CacheEntry",
    "WebCache",
    "get_web_cache",
    "set_web_cache",
    # Content safety
    "ContentSafetyChecker",
    "SafetyCheckResult",
    "SafetyViolation",
    "check_content_safety",
    "check_url_safety",
    "get_content_safety_checker",
    # Response output
    "ResponseOutput",
    "print_response",
]


def __getattr__(name: str) -> Any:
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module), name)
    globals()[name] = value  # cache: the next access is a plain module attribute
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY))
