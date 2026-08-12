"""Cloud redaction filter for LLM prompts."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any


class DataSensitivity(Enum):
    PUBLIC = "public"
    INTERNAL = "internal"
    PRIVATE = "private"
    SECRET = "secret"


TOOL_SENSITIVITY: dict[str, DataSensitivity] = {
    "internet_search": DataSensitivity.PUBLIC,
    "http_fetch": DataSensitivity.PUBLIC,
    "read_file": DataSensitivity.PRIVATE,
    "write_file": DataSensitivity.PRIVATE,
    "list_directory": DataSensitivity.PRIVATE,
    "glob": DataSensitivity.PRIVATE,
    "bash": DataSensitivity.SECRET,
    "execute_file": DataSensitivity.SECRET,
    "maxim_command": DataSensitivity.INTERNAL,
    "respond": DataSensitivity.INTERNAL,
}

MAXIM_COMMAND_SENSITIVITY: dict[str, DataSensitivity] = {
    "request_shutdown": DataSensitivity.SECRET,
    "request_sleep": DataSensitivity.PRIVATE,
    "request_observe": DataSensitivity.INTERNAL,
    "goto_pose": DataSensitivity.PRIVATE,
    "move": DataSensitivity.SECRET,
    "look_at_image": DataSensitivity.PRIVATE,
}


@dataclass(slots=True)
class RedactionPolicy:
    name: str
    allow_internal: bool = True
    allow_private: bool = False
    allow_secret: bool = False


@dataclass(slots=True)
class RedactionResult:
    system: str
    user: str
    categories_sent: list[str]
    categories_redacted: list[str]
    policy: str


# Section rules are pinned against the headers PromptBuilder ACTUALLY
# emits (tests/unit/test_cloud_redaction_headers.py) — the 2026-08-12
# privacy audit found the previous rule set had silently drifted from
# the live prompt format: the ``memory_contents`` and
# ``workspace_manifest`` patterns matched no producer, so hippocampus
# recalls and workspace file listings shipped to cloud verbatim under
# every policy including ``strict``. When PromptBuilder renames or adds
# a sensitive section header, the structural test fails and forces a
# rule update here — pattern-based redaction with no header pin is a
# silent no-op waiting to happen.
_SECTION_RULES: list[tuple[re.Pattern, str, DataSensitivity, bool]] = [
    (re.compile(r"^=== Conversation History ===$"), "conversation_history", DataSensitivity.PRIVATE, True),
    (re.compile(r"^=== Recent Action Outcomes ===$"), "tool_outputs", DataSensitivity.PRIVATE, True),
    (re.compile(r"^=== Recent Speech ===$"), "speech_transcripts", DataSensitivity.PRIVATE, True),
    # workspace_manifest: legacy static header + the two live
    # parameterized headers from build_workspace_manifest.
    (re.compile(r"^=== Workspace Manifest ===$"), "workspace_manifest", DataSensitivity.PRIVATE, True),
    (
        re.compile(r"^=== PROJECT DIRECTORY \(\d+ entries, CWD: .*\) ===$"),
        "workspace_manifest",
        DataSensitivity.PRIVATE,
        True,
    ),
    (
        re.compile(r"^=== EXISTING WORKSPACE \(\d+ files?\) ===$"),
        "workspace_manifest",
        DataSensitivity.PRIVATE,
        True,
    ),
    (re.compile(r"^=== Context ===$"), "context_pool", DataSensitivity.PRIVATE, True),
    # memory_contents: live PromptBuilder header + legacy exec_agent
    # prompt marker (still emitted by the standalone-agent path).
    (re.compile(r"^=== Relevant Memories ===$"), "memory_contents", DataSensitivity.PRIVATE, True),
    (re.compile(r"^RELEVANT MEMORIES:$"), "memory_contents", DataSensitivity.PRIVATE, False),
    (re.compile(r"^RECENT CLI INPUTS:$"), "cli_inputs", DataSensitivity.PRIVATE, False),
    # inner_deliberation: the header itself classifies the section as
    # private, and the content is the LLM's own verbatim reasoning —
    # the exact leak class the hivemind bundle scrub closes on the
    # substrate side.
    (
        re.compile(r"^=== Your inner deliberation \(private — not speech\) ===$"),
        "inner_deliberation",
        DataSensitivity.PRIVATE,
        True,
    ),
    (re.compile(r"^=== Your prior reasoning ===$"), "inner_deliberation", DataSensitivity.PRIVATE, True),
]

_HEADER_PATTERN = re.compile(r"^=== .* ===$")
_PATH_PATTERN = re.compile(r"(?:(?:[A-Za-z]:\\)|/)[^ \n\t\r]+")
_SECRET_PATTERN = re.compile(r"(?i)(api[_-]?key|secret|token|bearer)\s*[:=]\s*[^ \n\t\r]+")


class CloudRedactionFilter:
    """Redacts sensitive sections before cloud dispatch."""

    def __init__(self, policy: RedactionPolicy) -> None:
        self._policy = policy

    @classmethod
    def from_config(
        cls,
        *,
        provider_cfg: dict[str, Any] | None = None,
        global_cfg: dict[str, Any] | None = None,
    ) -> "CloudRedactionFilter":
        provider_cfg = provider_cfg or {}
        global_cfg = global_cfg or {}
        policy_name = provider_cfg.get("redaction_policy") or global_cfg.get("policy") or "strict"
        policy = cls._policy_from_name(str(policy_name))
        return cls(policy)

    @staticmethod
    def _policy_from_name(name: str) -> RedactionPolicy:
        """Map a policy name to its behavior.

        ``standard`` currently has the same flags as ``strict`` (no
        distinct middle behavior is implemented yet) but keeps its own
        name so the cloud audit log reports the policy the operator
        actually configured instead of silently claiming ``strict``.
        Unknown names fall back to ``strict`` — over-redaction is the
        safe direction.
        """
        normalized = str(name or "").strip().lower()
        if normalized == "relaxed":
            return RedactionPolicy(name="relaxed", allow_internal=True, allow_private=True, allow_secret=False)
        if normalized == "standard":
            return RedactionPolicy(name="standard", allow_internal=True, allow_private=False, allow_secret=False)
        return RedactionPolicy(name="strict", allow_internal=True, allow_private=False, allow_secret=False)

    def redact(self, system: str, user: str) -> RedactionResult:
        categories_redacted: set[str] = set()
        categories_sent: set[str] = set()

        redacted_system, sys_redacted = self._redact_text(system)
        if sys_redacted:
            categories_redacted.update(sys_redacted)
        redacted_user, user_redacted = self._redact_sections(user)
        redacted_user, text_redacted = self._redact_text(redacted_user)
        categories_redacted.update(user_redacted)
        categories_redacted.update(text_redacted)

        # Everything not explicitly redacted is assumed sent
        categories_sent = self._infer_sent_categories(user, categories_redacted)

        return RedactionResult(
            system=redacted_system,
            user=redacted_user,
            categories_sent=sorted(categories_sent),
            categories_redacted=sorted(categories_redacted),
            policy=self._policy.name,
        )

    def _should_redact(self, sensitivity: DataSensitivity) -> bool:
        if sensitivity == DataSensitivity.PUBLIC:
            return False
        if sensitivity == DataSensitivity.INTERNAL:
            return not self._policy.allow_internal
        if sensitivity == DataSensitivity.PRIVATE:
            return not self._policy.allow_private
        if sensitivity == DataSensitivity.SECRET:
            return not self._policy.allow_secret
        return True

    def _redact_sections(self, text: str) -> tuple[str, set[str]]:
        lines = text.splitlines()
        redacted_categories: set[str] = set()
        out: list[str] = []
        i = 0
        while i < len(lines):
            line = lines[i]
            matched = None
            for pattern, category, sensitivity, bounded in _SECTION_RULES:
                if pattern.match(line.strip()):
                    matched = (category, sensitivity, bounded)
                    break
            if matched:
                category, sensitivity, bounded = matched
                if self._should_redact(sensitivity):
                    redacted_categories.add(category)
                    out.append(f"[REDACTED: {category}]")
                    i += 1
                    if bounded:
                        while i < len(lines) and not _HEADER_PATTERN.match(lines[i].strip()):
                            i += 1
                    else:
                        while i < len(lines) and lines[i].strip():
                            i += 1
                    continue
            out.append(line)
            i += 1
        return "\n".join(out), redacted_categories

    def _redact_text(self, text: str) -> tuple[str, set[str]]:
        redacted: set[str] = set()
        new_text = text
        if _SECRET_PATTERN.search(new_text):
            new_text = _SECRET_PATTERN.sub("[REDACTED_SECRET]", new_text)
            redacted.add("credentials")
        if _PATH_PATTERN.search(new_text):
            new_text = _PATH_PATTERN.sub("[REDACTED_PATH]", new_text)
            redacted.add("file_paths")
        return new_text, redacted

    def _infer_sent_categories(self, original: str, redacted: set[str]) -> set[str]:
        sent: set[str] = set()
        for line in original.splitlines():
            for pattern, category, _, _ in _SECTION_RULES:
                if category in redacted:
                    continue
                if pattern.match(line.strip()):
                    sent.add(category)
        # Always include core context as sent
        sent.add("system_context")
        return sent


__all__ = [
    "CloudRedactionFilter",
    "RedactionPolicy",
    "RedactionResult",
    "DataSensitivity",
    "TOOL_SENSITIVITY",
    "MAXIM_COMMAND_SENSITIVITY",
]
