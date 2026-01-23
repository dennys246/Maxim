"""Shared utilities for Maxim."""

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
