"""Tests for optional dependency error handling.

Verifies that importing modules that require optional deps (cv2, anthropic,
twilio) gives clear, actionable error messages instead of raw ImportErrors.
"""

from __future__ import annotations

import sys
from unittest.mock import patch

import pytest


def test_vision_rtm_engine_without_cv2():
    """RTMEngine._import_cv2() gives clear install instructions."""
    # Temporarily hide cv2
    saved = sys.modules.pop("cv2", None)
    try:
        with patch.dict(sys.modules, {"cv2": None}):
            from maxim.models.vision.rtm_engine import _import_cv2

            with pytest.raises(ImportError, match="pip install pymaxim\\[vision\\]"):
                _import_cv2()
    finally:
        if saved is not None:
            sys.modules["cv2"] = saved


def test_vision_ultralytics_without_cv2():
    """UltralyticsEngine._import_cv2() gives clear install instructions."""
    saved = sys.modules.pop("cv2", None)
    try:
        with patch.dict(sys.modules, {"cv2": None}):
            from maxim.models.vision.ultralytics_engine import _import_cv2

            with pytest.raises(ImportError, match="pip install pymaxim\\[vision\\]"):
                _import_cv2()
    finally:
        if saved is not None:
            sys.modules["cv2"] = saved


def test_validate_model_missing_anthropic_sdk():
    """_validate_model for Claude without anthropic SDK gives install hint."""
    import os

    from maxim.exceptions import ConfigurationError

    # Set the API key so we get past that check
    saved_key = os.environ.get("ANTHROPIC_API_KEY")
    os.environ["ANTHROPIC_API_KEY"] = "test-key"

    saved_mod = sys.modules.pop("anthropic", None)
    try:
        with patch.dict(sys.modules, {"anthropic": None}):
            # Need to reimport to get fresh validation
            from maxim.api import _validate_model

            with pytest.raises(ConfigurationError, match="pip install pymaxim\\[llm-anthropic\\]"):
                _validate_model("claude-sonnet")
    finally:
        if saved_mod is not None:
            sys.modules["anthropic"] = saved_mod
        if saved_key is not None:
            os.environ["ANTHROPIC_API_KEY"] = saved_key
        else:
            os.environ.pop("ANTHROPIC_API_KEY", None)


def test_validate_model_missing_openai_sdk():
    """_validate_model for GPT-4o without openai SDK gives install hint."""
    import os

    from maxim.exceptions import ConfigurationError

    saved_key = os.environ.get("OPENAI_API_KEY")
    os.environ["OPENAI_API_KEY"] = "test-key"

    saved_mod = sys.modules.pop("openai", None)
    try:
        with patch.dict(sys.modules, {"openai": None}):
            from maxim.api import _validate_model

            with pytest.raises(ConfigurationError, match="pip install pymaxim\\[llm-openai\\]"):
                _validate_model("gpt-4o")
    finally:
        if saved_mod is not None:
            sys.modules["openai"] = saved_mod
        if saved_key is not None:
            os.environ["OPENAI_API_KEY"] = saved_key
        else:
            os.environ.pop("OPENAI_API_KEY", None)
