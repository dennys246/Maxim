"""Content safety checks for web-fetched content.

Provides basic content safety filtering using keyword-based detection
and pattern matching. This is a foundational implementation that can
be extended with ML-based classifiers.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Safety Categories
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class SafetyViolation:
    """A detected safety violation in content."""

    category: str
    severity: str  # "low", "medium", "high"
    description: str
    matched_pattern: str = ""
    context: str = ""


@dataclass
class SafetyCheckResult:
    """Result of a content safety check."""

    is_safe: bool
    violations: list[SafetyViolation] = field(default_factory=list)
    categories_checked: list[str] = field(default_factory=list)

    @property
    def highest_severity(self) -> str | None:
        """Get the highest severity level among violations."""
        if not self.violations:
            return None
        severity_order = {"high": 3, "medium": 2, "low": 1}
        return max(self.violations, key=lambda v: severity_order.get(v.severity, 0)).severity

    def summary(self) -> str:
        """Get a human-readable summary."""
        if self.is_safe:
            return "Content passed safety checks"
        violations_by_cat = {}
        for v in self.violations:
            if v.category not in violations_by_cat:
                violations_by_cat[v.category] = []
            violations_by_cat[v.category].append(v)
        parts = [f"{cat}: {len(vs)} violation(s)" for cat, vs in violations_by_cat.items()]
        return f"Content flagged: {', '.join(parts)}"


# ─────────────────────────────────────────────────────────────────────────────
# Pattern Definitions
# ─────────────────────────────────────────────────────────────────────────────


# Patterns for potentially harmful content categories
# These are intentionally broad and conservative

HARMFUL_PATTERNS: dict[str, list[tuple[str, str, str]]] = {
    # (pattern, severity, description)
    "malware": [
        (r"<script[^>]*>.*?(eval|document\.write|unescape)\s*\(", "high", "Potentially malicious script"),
        (r"javascript:\s*void", "medium", "JavaScript void pattern"),
        (r"data:text/html;base64,", "high", "Base64 encoded HTML (potential XSS)"),
    ],
    "phishing": [
        (r"(verify|confirm|update)\s+(your\s+)?(account|password|credentials)", "medium", "Credential phishing language"),
        (r"(click|tap)\s+here\s+to\s+(verify|confirm|secure)", "low", "Phishing call-to-action"),
        (r"(suspended|locked|compromised)\s+(account|access)", "low", "Account fear language"),
    ],
    "explicit": [
        # Intentionally minimal - these are just markers for adult content sites
        (r"18\+|adults?\s+only|mature\s+content", "low", "Adult content warning"),
        (r"nsfw|not\s+safe\s+for\s+work", "low", "NSFW marker"),
    ],
    "violence": [
        (r"(how\s+to|instructions?\s+(for|to))\s+(make|build|create)\s+(a\s+)?(bomb|explosive|weapon)", "high", "Weapons instructions"),
        (r"(step.by.step|tutorial|guide)\s+(to|for)\s+(harm|hurt|kill)", "high", "Violence instructions"),
    ],
    "scam": [
        (r"(nigerian|foreign)\s+prince", "medium", "Classic scam pattern"),
        (r"(million|billion)\s+(dollars?|usd|euros?)\s+(inheritance|lottery|prize)", "medium", "Money scam pattern"),
        (r"(western\s+union|moneygram|wire\s+transfer).*(urgent|immediately)", "medium", "Wire transfer scam"),
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# Content Safety Checker
# ─────────────────────────────────────────────────────────────────────────────


class ContentSafetyChecker:
    """Checks content for potentially unsafe patterns.

    This is a rule-based system suitable for basic filtering.
    For production use, consider integrating with ML-based classifiers.
    """

    def __init__(
        self,
        enabled_categories: set[str] | None = None,
        severity_threshold: str = "medium",
        custom_patterns: dict[str, list[tuple[str, str, str]]] | None = None,
    ):
        """Initialize the safety checker.

        Args:
            enabled_categories: Categories to check (None = all)
            severity_threshold: Minimum severity to flag ("low", "medium", "high")
            custom_patterns: Additional patterns to check
        """
        self._enabled_categories = enabled_categories or set(HARMFUL_PATTERNS.keys())
        self._severity_threshold = severity_threshold
        self._severity_order = {"low": 1, "medium": 2, "high": 3}

        # Compile patterns
        self._patterns: dict[str, list[tuple[re.Pattern, str, str]]] = {}
        for category, patterns in HARMFUL_PATTERNS.items():
            if category in self._enabled_categories:
                self._patterns[category] = [
                    (re.compile(p, re.IGNORECASE | re.DOTALL), sev, desc)
                    for p, sev, desc in patterns
                ]

        if custom_patterns:
            for category, patterns in custom_patterns.items():
                if category not in self._patterns:
                    self._patterns[category] = []
                self._patterns[category].extend([
                    (re.compile(p, re.IGNORECASE | re.DOTALL), sev, desc)
                    for p, sev, desc in patterns
                ])

    def check(self, content: str, max_violations: int = 10) -> SafetyCheckResult:
        """Check content for safety violations.

        Args:
            content: The content to check
            max_violations: Maximum violations to return

        Returns:
            SafetyCheckResult with is_safe flag and any violations
        """
        violations: list[SafetyViolation] = []
        threshold_value = self._severity_order.get(self._severity_threshold, 2)

        for category, patterns in self._patterns.items():
            for pattern, severity, description in patterns:
                severity_value = self._severity_order.get(severity, 1)

                # Skip if below threshold
                if severity_value < threshold_value:
                    continue

                # Search for pattern
                match = pattern.search(content)
                if match:
                    # Extract context around match
                    start = max(0, match.start() - 50)
                    end = min(len(content), match.end() + 50)
                    context = content[start:end].strip()

                    violations.append(SafetyViolation(
                        category=category,
                        severity=severity,
                        description=description,
                        matched_pattern=match.group(0)[:100],
                        context=context[:200],
                    ))

                    if len(violations) >= max_violations:
                        break

            if len(violations) >= max_violations:
                break

        # Determine if content is safe
        # Safe if no violations at or above threshold
        is_safe = len(violations) == 0

        return SafetyCheckResult(
            is_safe=is_safe,
            violations=violations,
            categories_checked=list(self._patterns.keys()),
        )

    def check_url(self, url: str) -> SafetyCheckResult:
        """Quick check on URL itself (before fetching).

        Args:
            url: The URL to check

        Returns:
            SafetyCheckResult
        """
        violations: list[SafetyViolation] = []

        # Check for suspicious URL patterns
        url_lower = url.lower()

        # Data URLs with suspicious content
        if url_lower.startswith("data:"):
            violations.append(SafetyViolation(
                category="malware",
                severity="high",
                description="Data URL (potential malicious content)",
                matched_pattern=url[:50],
            ))

        # JavaScript URLs
        if url_lower.startswith("javascript:"):
            violations.append(SafetyViolation(
                category="malware",
                severity="high",
                description="JavaScript URL",
                matched_pattern=url[:50],
            ))

        # Suspicious file extensions
        suspicious_extensions = [".exe", ".scr", ".bat", ".cmd", ".msi", ".dll"]
        for ext in suspicious_extensions:
            if url_lower.endswith(ext):
                violations.append(SafetyViolation(
                    category="malware",
                    severity="medium",
                    description=f"Suspicious file extension: {ext}",
                    matched_pattern=url[-20:],
                ))
                break

        return SafetyCheckResult(
            is_safe=len(violations) == 0,
            violations=violations,
            categories_checked=["url_check"],
        )


# ─────────────────────────────────────────────────────────────────────────────
# Global Instance
# ─────────────────────────────────────────────────────────────────────────────


def _get_checker_singleton():
    """Lazy import to avoid circular dependency."""
    from maxim.utils.singleton import Singleton
    return Singleton("content_safety_checker")


def get_content_safety_checker(
    create_if_missing: bool = True,
) -> ContentSafetyChecker | None:
    """Get the global content safety checker instance."""
    singleton = _get_checker_singleton()
    checker = singleton.get()

    if checker is None and create_if_missing:
        checker = ContentSafetyChecker()
        singleton.set(checker)

    return checker


def check_content_safety(content: str) -> tuple[bool, str | None]:
    """Convenience function to check content safety.

    Returns (is_safe, reason) for easy integration with tools.
    """
    checker = get_content_safety_checker()
    if checker is None:
        return True, None

    result = checker.check(content)
    if result.is_safe:
        return True, None

    return False, result.summary()


def check_url_safety(url: str) -> tuple[bool, str | None]:
    """Convenience function to check URL safety.

    Returns (is_safe, reason) for easy integration with tools.
    """
    checker = get_content_safety_checker()
    if checker is None:
        return True, None

    result = checker.check_url(url)
    if result.is_safe:
        return True, None

    return False, result.summary()
