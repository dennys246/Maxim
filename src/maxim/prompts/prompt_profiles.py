"""Prompt profile and injection utilities."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from maxim.utils.logging import warn


_BLOCKED_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+safety", re.IGNORECASE),
    re.compile(r"bypass\s+safety", re.IGNORECASE),
    re.compile(r"disable\s+safety", re.IGNORECASE),
    re.compile(r"ignore\s+constraints", re.IGNORECASE),
    re.compile(r"no\s+constraints", re.IGNORECASE),
    re.compile(r"override\s+rules", re.IGNORECASE),
    re.compile(r"disregard\s+policy", re.IGNORECASE),
]


def _sanitize_injection(text: str, slot: str, profile: str) -> str:
    cleaned = str(text or "").strip()
    if not cleaned:
        return ""
    for pattern in _BLOCKED_INJECTION_PATTERNS:
        if pattern.search(cleaned):
            warn("Prompt injection blocked in %s (%s)", slot, profile)
            return ""
    return cleaned


@dataclass(slots=True)
class PromptProfile:
    name: str
    system_suffix: str = ""
    context_prefix: str = ""

    @classmethod
    def from_config(cls, name: str, raw: dict[str, Any]) -> "PromptProfile":
        system_suffix = _sanitize_injection(raw.get("system_suffix", ""), "system_suffix", name)
        context_prefix = _sanitize_injection(raw.get("context_prefix", ""), "context_prefix", name)
        return cls(name=name, system_suffix=system_suffix, context_prefix=context_prefix)


@dataclass(slots=True)
class InjectedPrompt:
    system: str
    user: str
    messages: list[dict[str, str]]
    text_prompt: str


class PromptPrompt:
    """Base prompt class with injection support."""

    def build_system(self, context: Any) -> str:
        return ""

    def build_user(self, context: Any) -> str:
        return ""

    def build_messages(self, context: Any) -> list[dict[str, str]]:
        system = self.build_system(context)
        user = self.build_user(context)
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def build_text_prompt(self, context: Any) -> str:
        return self.build_user(context)

    def inject(self, context: Any, profile: PromptProfile | None = None) -> InjectedPrompt:
        profile = profile or PromptProfile(name="default")
        system = self.build_system(context).strip()
        user = self.build_user(context).strip()

        if profile.system_suffix:
            system = f"{system}\n\n{profile.system_suffix}".strip()
        if profile.context_prefix:
            user = f"{profile.context_prefix}\n\n{user}".strip()

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        return InjectedPrompt(
            system=system,
            user=user,
            messages=messages,
            text_prompt=user,
        )


class ExecutivePrompt(PromptPrompt):
    """ExecAgent prompt wrapper for goal proposal."""

    def __init__(self, system_prompt: str) -> None:
        self._system_prompt = system_prompt

    def build_system(self, context: Any) -> str:
        return self._system_prompt

    def build_user(self, context: Any) -> str:
        return str(context or "").strip()


def load_prompt_profile(
    prompt_profiles: dict[str, Any],
    name: str | None,
) -> PromptProfile:
    if not name:
        return PromptProfile(name="default")
    raw = prompt_profiles.get(name, {})
    if not isinstance(raw, dict):
        return PromptProfile(name=name)
    return PromptProfile.from_config(name, raw)


__all__ = [
    "PromptProfile",
    "InjectedPrompt",
    "PromptPrompt",
    "ExecutivePrompt",
    "load_prompt_profile",
]
