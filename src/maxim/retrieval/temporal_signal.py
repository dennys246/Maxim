"""Temporal reference signal - Extracts time mentions and queries SCN.

Parses natural language temporal references from text and retrieves
memories that occurred at or around those times.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

from maxim.retrieval.signals import RetrievalCandidate, RetrievalSignal
from maxim.time.temporal_signature import TemporalSignature

if TYPE_CHECKING:
    from maxim.time.scn import SCN

logger = logging.getLogger(__name__)

# Try to import dateparser, fall back to regex-only mode
try:
    import dateparser
    import dateparser.search

    DATEPARSER_AVAILABLE = True
except ImportError:
    DATEPARSER_AVAILABLE = False
    logger.debug("dateparser not installed, using regex-only temporal parsing")


@dataclass
class ParsedTemporalReference:
    """A parsed temporal reference from text.

    Attributes:
        phrase: The original text that was parsed
        datetime: The resolved datetime
        confidence: Parsing confidence (0.0-1.0)
        reference_type: Type of reference (absolute, relative, recurring)
    """

    phrase: str
    datetime: datetime
    confidence: float = 1.0
    reference_type: str = "absolute"


class TemporalReferenceSignal(RetrievalSignal):
    """Extract temporal references and retrieve temporally-relevant memories.

    Uses dateparser for natural language time parsing, with regex fallback
    for common patterns when dateparser isn't available.

    Supports:
    - Absolute times: "3pm", "January 5th", "2024-01-15"
    - Relative times: "yesterday", "last Tuesday", "two hours ago"
    - Recurring patterns: "every Monday", "mornings"

    Example:
        signal = TemporalReferenceSignal(scn=scn)

        # Query with temporal context
        candidates = signal.extract("What were we working on last Tuesday morning?")
        # Returns memories from last Tuesday morning

        # Multiple references
        candidates = signal.extract("Between Monday and Wednesday")
        # Returns memories from Mon, Tue, Wed
    """

    def __init__(
        self,
        scn: "SCN",
        *,
        tolerance_hours: int = 1,
        include_day_context: bool = True,
        min_confidence: float = 0.5,
    ) -> None:
        """Initialize the temporal reference signal.

        Args:
            scn: SCN instance for temporal queries
            tolerance_hours: Hours of tolerance for time matching (default 1)
            include_day_context: Include day-of-week in matching (default True)
            min_confidence: Minimum parsing confidence to use (default 0.5)
        """
        self._scn = scn
        self._tolerance_hours = tolerance_hours
        self._include_day_context = include_day_context
        self._min_confidence = min_confidence

        # Precompiled patterns for fallback parsing
        self._patterns = self._compile_patterns()

        # Dateparser settings
        self._dateparser_settings = {
            "PREFER_DATES_FROM": "past",
            "RELATIVE_BASE": datetime.now(),
            "RETURN_AS_TIMEZONE_AWARE": False,
        }

    @property
    def name(self) -> str:
        return "temporal_reference"

    @property
    def priority(self) -> int:
        # Run early since temporal context is often important
        return 70

    def extract(self, text: str) -> list[RetrievalCandidate]:
        """Extract temporal references and retrieve matching memories.

        Args:
            text: Input text to analyze

        Returns:
            List of RetrievalCandidate for temporally-relevant memories
        """
        if not text or not text.strip():
            return []

        # Parse temporal references from text
        references = self._parse_temporal_references(text)

        if not references:
            return []

        # Query SCN for each reference
        candidates: dict[str, RetrievalCandidate] = {}

        for ref in references:
            if ref.confidence < self._min_confidence:
                continue

            # Convert to TemporalSignature
            sig = TemporalSignature.from_timestamp(ref.datetime.timestamp())

            # Query with tolerance
            memory_ids = self._scn.query_similar_time(
                sig,
                tolerance=self._tolerance_hours,
            )

            # Optionally intersect with day-of-week
            if self._include_day_context:
                hour_bin, day_bin, _, _ = sig.to_bins()
                day_memories = self._scn.query_day(day_bin)
                memory_ids = memory_ids & day_memories

            # Create candidates
            for memory_id in memory_ids:
                score = ref.confidence * 0.8  # Scale down slightly
                if memory_id in candidates:
                    # Keep highest score
                    if score > candidates[memory_id].score:
                        candidates[memory_id] = RetrievalCandidate(
                            memory_id=memory_id,
                            score=score,
                            source=self.name,
                            metadata={
                                "phrase": ref.phrase,
                                "datetime": ref.datetime.isoformat(),
                                "reference_type": ref.reference_type,
                            },
                        )
                else:
                    candidates[memory_id] = RetrievalCandidate(
                        memory_id=memory_id,
                        score=score,
                        source=self.name,
                        metadata={
                            "phrase": ref.phrase,
                            "datetime": ref.datetime.isoformat(),
                            "reference_type": ref.reference_type,
                        },
                    )

        return list(candidates.values())

    def _parse_temporal_references(self, text: str) -> list[ParsedTemporalReference]:
        """Parse temporal references from text.

        Uses dateparser if available, falls back to regex patterns.
        """
        references: list[ParsedTemporalReference] = []

        # Check for recurring pattern keywords first
        recurring_refs = self._parse_recurring_patterns(text)
        references.extend(recurring_refs)

        if DATEPARSER_AVAILABLE:
            # Use dateparser.search for comprehensive parsing
            try:
                # Update relative base to current time
                self._dateparser_settings["RELATIVE_BASE"] = datetime.now()

                parsed = dateparser.search.search_dates(
                    text,
                    settings=self._dateparser_settings,
                    languages=["en"],
                )

                if parsed:
                    for phrase, dt in parsed:
                        # Skip if we already found this as recurring
                        if any(r.phrase.lower() in phrase.lower() for r in recurring_refs):
                            continue

                        references.append(
                            ParsedTemporalReference(
                                phrase=phrase,
                                datetime=dt,
                                confidence=0.9,
                                reference_type=self._classify_reference(phrase),
                            )
                        )
            except Exception as e:
                logger.debug("dateparser failed, using regex fallback: %s", e)
                references.extend(self._parse_with_regex(text))
        else:
            references.extend(self._parse_with_regex(text))

        return references

    def _parse_recurring_patterns(self, text: str) -> list[ParsedTemporalReference]:
        """Parse recurring time patterns like 'every Monday' or 'mornings'."""
        references: list[ParsedTemporalReference] = []
        text_lower = text.lower()
        now = datetime.now()

        # Day of week patterns
        days = {
            "monday": 0,
            "tuesday": 1,
            "wednesday": 2,
            "thursday": 3,
            "friday": 4,
            "saturday": 5,
            "sunday": 6,
        }

        for day_name, day_num in days.items():
            if f"every {day_name}" in text_lower or f"on {day_name}s" in text_lower:
                # Calculate the most recent occurrence of this day
                days_since = (now.weekday() - day_num) % 7
                if days_since == 0:
                    days_since = 7  # Use last week if today is the day
                target_date = now.replace(hour=12, minute=0, second=0, microsecond=0)
                from datetime import timedelta

                target_date = target_date - timedelta(days=days_since)

                references.append(
                    ParsedTemporalReference(
                        phrase=f"every {day_name}",
                        datetime=target_date,
                        confidence=0.7,
                        reference_type="recurring",
                    )
                )

        # Time of day patterns
        time_periods = {
            "morning": (8, 11),
            "mornings": (8, 11),
            "afternoon": (12, 17),
            "afternoons": (12, 17),
            "evening": (18, 21),
            "evenings": (18, 21),
            "night": (21, 23),
            "nights": (21, 23),
        }

        for period, (start_hour, end_hour) in time_periods.items():
            if period in text_lower:
                # Use middle of the period
                mid_hour = (start_hour + end_hour) // 2
                target_date = now.replace(hour=mid_hour, minute=0, second=0, microsecond=0)

                references.append(
                    ParsedTemporalReference(
                        phrase=period,
                        datetime=target_date,
                        confidence=0.6,
                        reference_type="recurring",
                    )
                )

        return references

    def _parse_with_regex(self, text: str) -> list[ParsedTemporalReference]:
        """Fallback regex-based parsing for common patterns."""
        references: list[ParsedTemporalReference] = []
        now = datetime.now()

        for pattern_name, pattern, handler in self._patterns:
            matches = pattern.findall(text.lower())
            for match in matches:
                try:
                    dt, confidence = handler(match, now)
                    if dt:
                        phrase = match if isinstance(match, str) else " ".join(match)
                        references.append(
                            ParsedTemporalReference(
                                phrase=phrase,
                                datetime=dt,
                                confidence=confidence,
                                reference_type=self._classify_reference(phrase),
                            )
                        )
                except Exception as e:
                    logger.debug("Regex handler failed for %s: %s", pattern_name, e)

        return references

    def _compile_patterns(
        self,
    ) -> list[tuple[str, re.Pattern, Any]]:
        """Compile regex patterns for fallback parsing."""
        from datetime import timedelta

        def handle_yesterday(match: str, now: datetime) -> tuple[datetime, float]:
            return now - timedelta(days=1), 0.95

        def handle_today(match: str, now: datetime) -> tuple[datetime, float]:
            return now, 0.95

        def handle_last_week(match: str, now: datetime) -> tuple[datetime, float]:
            return now - timedelta(weeks=1), 0.8

        def handle_n_days_ago(match: tuple, now: datetime) -> tuple[datetime, float]:
            n = int(match[0]) if match[0] else 1
            return now - timedelta(days=n), 0.85

        def handle_n_hours_ago(match: tuple, now: datetime) -> tuple[datetime, float]:
            n = int(match[0]) if match[0] else 1
            return now - timedelta(hours=n), 0.85

        def handle_last_day(match: str, now: datetime) -> tuple[datetime, float]:
            days = {
                "monday": 0,
                "tuesday": 1,
                "wednesday": 2,
                "thursday": 3,
                "friday": 4,
                "saturday": 5,
                "sunday": 6,
            }
            day_num = days.get(match.lower())
            if day_num is not None:
                days_since = (now.weekday() - day_num) % 7
                if days_since == 0:
                    days_since = 7
                return now - timedelta(days=days_since), 0.85
            return None, 0.0

        return [
            ("yesterday", re.compile(r"\byesterday\b"), handle_yesterday),
            ("today", re.compile(r"\btoday\b"), handle_today),
            ("last_week", re.compile(r"\blast week\b"), handle_last_week),
            (
                "n_days_ago",
                re.compile(r"\b(\d+)\s*days?\s*ago\b"),
                handle_n_days_ago,
            ),
            (
                "n_hours_ago",
                re.compile(r"\b(\d+)\s*hours?\s*ago\b"),
                handle_n_hours_ago,
            ),
            (
                "last_day",
                re.compile(r"\blast\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b"),
                lambda m, n: handle_last_day(m[0] if isinstance(m, tuple) else m, n),
            ),
        ]

    def _classify_reference(self, phrase: str) -> str:
        """Classify the type of temporal reference."""
        phrase_lower = phrase.lower()

        if any(word in phrase_lower for word in ["every", "always", "usually", "often", "mornings", "evenings"]):
            return "recurring"
        elif any(word in phrase_lower for word in ["ago", "last", "yesterday", "previous", "before"]):
            return "relative"
        else:
            return "absolute"

    def warmup(self) -> None:
        """Preload dateparser if available."""
        if DATEPARSER_AVAILABLE:
            # Trigger lazy loading
            try:
                dateparser.parse("yesterday")
            except Exception:
                pass


__all__ = ["TemporalReferenceSignal", "ParsedTemporalReference"]
