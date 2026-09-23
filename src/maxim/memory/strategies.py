"""Memory management strategies for sleep consolidation.

This module provides different strategies for scoring memories and
deciding which to retain, compress, or remove during sleep consolidation.

Strategies:
- AccessBasedStrategy: Prioritizes frequently and recently accessed memories
- ImportanceBasedStrategy: Prioritizes high-salience and novel memories
- CompositeStrategy: Weighted combination of multiple strategies
- StrengthStrategy: Bjork storage/retrieval strength on the experience clock
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Mapping

from maxim.memory.encoding import S_BASE_DEFAULT

if TYPE_CHECKING:
    from maxim.memory.experience_clock import ExperienceClock
    from maxim.memory.types import MemoryRecord
    from maxim.time.scn import SCN


# ── the retrieval update (memory-strength plan, §Retrieval + Phase 2 decision 5) ──────────────
#
# ``S <- S * (1 + a * w_src * (1 - R) * (S / s_base)^-w)``, then ``R = 1``.
#
# The plan writes the saturation factor as ``S^-w``, which is NOT dimensionless: S is carried in the
# experience clock's microseconds, where ``S^-0.5`` is ~3e-4 and every retrieval gain silently
# rounds to nothing. Taking it relative to ``s_base`` (2c-3 decision) makes the factor 1.0 at the
# base strength and preserves what the plan's worked example was about -- the gain SHRINKS as a
# trace grows well-learned, so spaced retrievals cannot compound S without bound. The plan's
# absolute numbers (S0 = 10 ticks) do not carry over; its DIRECTION does, and Phase 5 calibrates.
RETRIEVAL_GAIN = 1.0  # a
RETRIEVAL_SATURATION = 0.5  # w (FSRS's exponent)

# An activation within this much EXPERIENCE of the last credited one is recorded and credits
# nothing. Without it a trace re-rendered every tick out-earns a genuinely spaced one over the same
# span (the plan's massed-exposure correction): within a tick dt = 0 so each gain is ~0, but
# ``1 - e^-x <= x``, so many tiny gains beat one large one.
CREDITED_GAP_US = 2_000_000  # 2 s of experience

# How much each kind of use is worth (the testing effect: effortful recall beats re-exposure;
# internal reactivation, often unconsumed, sits at the low end). This is a second copy of
# ``types.ACTIVATION_SOURCES``, held equal by a test rather than by a comment: a source missing
# here credits SILENTLY nothing, so a sixth activation source would quietly never strengthen
# anything (review round, Architecture N2 -- the earlier comment called that "louder", which it
# is not).
RETRIEVAL_SOURCE_WEIGHTS: "Mapping[str, float]" = {
    "tool": 1.0,
    "replan": 0.8,
    "planner": 0.8,
    "enrichment": 0.5,
    "prediction": 0.5,
}

# Protection is a FLOOR under retrievability, never immortality (plan §Protection). A strongly
# tagged trace stays retrievable while its tag still speaks for it -- and the tag's voice FADES, on
# the same experience clock, ``TAG_FADE_MULTIPLIER`` times slower than R itself (the fading-affect
# bias: emotional memories fade too, just slowly). A floor that never lifted would rebuild exactly
# the ``access_count >= 10`` immortality this plan exists to remove.
#
# The stamped tag is HISTORY and is never rewritten: the fade is computed as a VIEW from the same
# ``dt`` R uses, so re-tuning changes what happens next rather than what already happened.
PROTECTION_FLOOR_WEIGHT = 0.5
TAG_FADE_MULTIPLIER = 10.0


class MemoryStrategy(ABC):
    """Abstract base class for memory management strategies.

    Strategies determine how memories are scored for retention and
    whether they should be compressed to save space.

    Memory lifecycle during sleep():
    1. score_for_retention() evaluates each memory
    2. Low scores → removal (permanent deletion)
    3. Medium scores → compression (if should_compress() agrees)
    4. High scores → preservation (full record kept)

    A strategy also declares what it NEEDS, so no consumer has to name a strategy by string to find
    out (memory-strength 2c-3). ``requires_experience_clock`` is the one capability today: a store
    whose model runs on experience time is broken -- silently, keeping everything forever -- if
    nothing advances that clock, so ``MemoryHub`` asserts the clock moved for strategies that say
    so. A third-party strategy gets the same treatment by setting the flag; comparing against the
    name ``"strength"`` would have been a second source of truth it could never satisfy.
    """

    #: Whether this model reads an :class:`~maxim.memory.experience_clock.ExperienceClock` that
    #: something else must advance. Default False: nothing asserts anything about the clock.
    requires_experience_clock: bool = False

    def activation_now(self) -> float:
        """The time to hand ``on_activation``, read ONCE before any lock is taken.

        Default 0.0: a strategy that does not model time does not need one. ``StrengthStrategy``
        returns its experience clock, so every record in one ``activate`` call is credited against
        the same instant rather than a clock that drifted mid-loop.
        """
        return 0.0

    def on_activation(self, record: MemoryRecord, now: float, source: str) -> bool:
        """React to one honest activation EVENT. Returns whether the record changed.

        The event hook, not Phase 1's ``activation_count``: that count is massed and
        un-deduplicated (deliberation re-renders the same top 3 every cycle), so it can only say
        "used a lot", never "used after a gap" -- and the spacing effect is entirely about the gap.
        A no-op by default, which is what keeps today's retention byte-identical.
        """
        return False

    @abstractmethod
    def score_for_retention(
        self,
        record: MemoryRecord,
        now: float,
        degree: int = 0,
    ) -> float:
        """Score a memory for retention (0.0 = remove, 1.0 = definitely keep).

        Args:
            record: The memory record to evaluate (full or compressed).
            now: Current timestamp for age calculations.
            degree: Graph degree (number of connections) for centrality boost.

        Returns:
            Retention score 0.0-1.0 where:
            - < retention_threshold → remove
            - < compression_threshold → compress
            - >= compression_threshold → preserve
        """

    @abstractmethod
    def should_compress(
        self,
        record: MemoryRecord,
        now: float,
        degree: int = 0,
    ) -> bool:
        """Determine if a full memory should be compressed.

        Only called for EpisodicMemory records that score between
        retention_threshold and compression_threshold.

        Args:
            record: The memory record to evaluate.
            now: Current timestamp.
            degree: Graph degree (number of connections).

        Returns:
            True if should compress to CompressedMemory.
        """


class AccessBasedStrategy(MemoryStrategy):
    """Memory strategy based on access recency and frequency.

    Memories that haven't been accessed recently and are infrequently
    accessed are candidates for removal. High-access memories are
    preserved regardless of age.

    Good for: Systems with heavy query loads where frequently-used
    memories should always be available.
    """

    def __init__(
        self,
        max_age_without_access: float = 7 * 24 * 3600,  # 1 week default
        compression_age: float = 24 * 3600,  # Compress after 1 day
        access_count_threshold: int = 5,  # High-access threshold
        centrality_weight: float = 0.3,  # How much connectivity matters
    ):
        """Initialize AccessBasedStrategy.

        Args:
            max_age_without_access: Seconds without access before removal.
            compression_age: Seconds after creation before compression eligible.
            access_count_threshold: Access count to consider "frequently used".
            centrality_weight: Weight for graph centrality boost (0-1).
        """
        self.max_age_without_access = max_age_without_access
        self.compression_age = compression_age
        self.access_count_threshold = access_count_threshold
        self.centrality_weight = centrality_weight

    def score_for_retention(
        self,
        record: MemoryRecord,
        now: float,
        degree: int = 0,
    ) -> float:
        """Score based on access patterns and graph importance."""
        from maxim.memory.types import CompressedMemory, EpisodicMemory

        # Time since last access
        time_since_access = now - record.accessed_at
        max_age = self.max_age_without_access

        # Base score from access recency (exponential decay)
        if time_since_access >= max_age:
            recency_score = 0.0
        else:
            recency_score = 1.0 - (time_since_access / max_age)

        # Boost from access frequency
        access_boost = min(1.0, record.access_count / self.access_count_threshold)

        # Boost from graph centrality
        # Normalize: assume max useful degree is ~20
        centrality_boost = min(1.0, degree / 20) if degree > 0 else 0.0

        # Combine scores
        score = recency_score * 0.5 + access_boost * 0.2 + centrality_boost * self.centrality_weight

        # Always keep if accessed many times
        if record.access_count >= self.access_count_threshold * 2:
            score = max(score, 0.8)

        # Always keep successful user interactions (check for EpisodicMemory)
        if isinstance(record, EpisodicMemory):
            if record.perception.cli_input or record.perception.transcript:
                if record.outcome.success:
                    score = max(score, 0.7)
        elif isinstance(record, CompressedMemory):
            if record.had_user_input and record.success:
                score = max(score, 0.7)

        return min(1.0, score)

    def should_compress(
        self,
        record: MemoryRecord,
        now: float,
        degree: int = 0,
    ) -> bool:
        """Compress based on age and access patterns."""
        from maxim.memory.types import CompressedMemory, EpisodicMemory

        # Already compressed
        if isinstance(record, CompressedMemory):
            return False

        age = now - record.created_at
        if age < self.compression_age:
            return False

        # Keep full records for frequently accessed memories
        if record.access_count >= self.access_count_threshold:
            return False

        # Keep full records for user interactions longer
        if isinstance(record, EpisodicMemory):
            if record.perception.cli_input or record.perception.transcript:
                return age > self.compression_age * 3  # 3 days

        return True


class ImportanceBasedStrategy(MemoryStrategy):
    """Memory strategy based on importance and outcomes.

    Preserves memories that are:
    - High novelty (rare/unique experiences)
    - High salience (contextually important)
    - Successful outcomes (worth learning from)
    - User interactions (direct commands/conversations)

    Good for: Learning systems where memorable experiences
    matter more than recent ones.
    """

    def __init__(
        self,
        max_age: float = 30 * 24 * 3600,  # 30 days default
        compression_age: float = 3 * 24 * 3600,  # Compress after 3 days
        novelty_weight: float = 0.2,
        success_weight: float = 0.2,
    ):
        """Initialize ImportanceBasedStrategy.

        Args:
            max_age: Maximum age before hard removal (seconds).
            compression_age: Age before compression eligible (seconds).
            novelty_weight: Weight for novelty score (0-1).
            success_weight: Weight for success bonus (0-1).
        """
        self.max_age = max_age
        self.compression_age = compression_age
        self.novelty_weight = novelty_weight
        self.success_weight = success_weight

    def score_for_retention(
        self,
        record: MemoryRecord,
        now: float,
        degree: int = 0,
    ) -> float:
        """Score based on importance and outcomes."""
        from maxim.memory.semantic_types import CompressedSemantic, SemanticMemory
        from maxim.memory.types import CompressedMemory, EpisodicMemory

        age = now - record.created_at

        # Hard cutoff for very old, rarely accessed memories
        if age > self.max_age and record.access_count < 3:
            return 0.0

        # Base score from age (linear decay)
        age_score = max(0.0, 1.0 - (age / self.max_age))

        # Success bonus
        if isinstance(record, EpisodicMemory):
            success_score = 1.0 if record.outcome.success else 0.3
            novelty_score = record.perception.novelty
            had_user_input = bool(record.perception.cli_input or record.perception.transcript)
        elif isinstance(record, CompressedMemory):
            success_score = 1.0 if record.success else 0.3
            novelty_score = record.novelty
            had_user_input = record.had_user_input
        elif isinstance(record, (SemanticMemory, CompressedSemantic)):
            # A concept's importance is how well the world has supported it, not an outcome it
            # never had: confidence carries the success/novelty weights, and a concept met again
            # and again takes the place of the interaction bonus. Without this branch every
            # concept scored by AGE alone (they all fell to the constant ``else``), so the ATL
            # could not tell a heavily-reinforced concept from an untouched one — found when
            # Phase 2c-1 let the ATL honour this strategy for the first time.
            success_score = record.confidence
            novelty_score = record.confidence
            had_user_input = getattr(record, "reinforcement_count", 1) >= 3
        else:
            success_score = 0.5
            novelty_score = 0.5
            had_user_input = False

        # User interaction bonus
        user_bonus = 0.3 if had_user_input else 0.0

        # Combine
        score = age_score * 0.4 + success_score * self.success_weight + novelty_score * self.novelty_weight + user_bonus

        return min(1.0, score)

    def should_compress(
        self,
        record: MemoryRecord,
        now: float,
        degree: int = 0,
    ) -> bool:
        """Compress based on age and importance."""
        from maxim.memory.types import CompressedMemory, EpisodicMemory

        if isinstance(record, CompressedMemory):
            return False

        age = now - record.created_at
        if age < self.compression_age:
            return False

        if isinstance(record, EpisodicMemory):
            # Keep full records for user interactions
            if record.perception.cli_input or record.perception.transcript:
                return age > self.compression_age * 3  # 9 days

            # Keep full records for high-novelty events
            if record.perception.novelty > 0.8:
                return age > self.compression_age * 2  # 6 days

        return True


class CompositeStrategy(MemoryStrategy):
    """Combines multiple strategies with configurable weights.

    Useful for balancing multiple concerns (e.g., both access patterns
    and importance should matter).
    """

    def __init__(
        self,
        strategies: list[tuple[MemoryStrategy, float]],
    ):
        """Initialize CompositeStrategy.

        Args:
            strategies: List of (strategy, weight) tuples.
                Weights are normalized automatically.
        """
        total_weight = sum(w for _, w in strategies)
        if total_weight == 0:
            raise ValueError("Total weight must be > 0")

        # Normalize weights
        self.strategies = [(s, w / total_weight) for s, w in strategies]

    @property
    def requires_experience_clock(self) -> bool:  # type: ignore[override]
        """A blend needs the clock if ANY member does -- the member would read a frozen one."""
        return any(s.requires_experience_clock for s, _ in self.strategies)

    def activation_now(self) -> float:
        """The members' clock, if any member has one; blends are otherwise timeless."""
        return max((s.activation_now() for s, _ in self.strategies), default=0.0)

    def on_activation(self, record: MemoryRecord, now: float, source: str) -> bool:
        """Every member sees the event. Non-modelling members are no-ops, so this stays free."""
        changed = False
        for strategy, _ in self.strategies:
            changed = strategy.on_activation(record, now, source) or changed
        return changed

    def score_for_retention(
        self,
        record: MemoryRecord,
        now: float,
        degree: int = 0,
    ) -> float:
        """Weighted average of all strategy scores."""
        total = 0.0
        for strategy, weight in self.strategies:
            total += strategy.score_for_retention(record, now, degree) * weight
        return total

    def should_compress(
        self,
        record: MemoryRecord,
        now: float,
        degree: int = 0,
    ) -> bool:
        """Compress if majority of strategies agree."""
        votes = sum(1 for strategy, _ in self.strategies if strategy.should_compress(record, now, degree))
        return votes > len(self.strategies) / 2


class TemporalAwareStrategy(MemoryStrategy):
    """SCN-integrated strategy for temporally-aware memory consolidation.

    Wraps a base strategy and adds temporal context from SCN:
    - Protects sole representatives of time slots
    - Boosts memories in rhythmic patterns
    - Enables temporal clustering for batch operations

    This is the most efficient strategy for long-running systems with
    established temporal patterns.
    """

    def __init__(
        self,
        scn: "SCN",
        base_strategy: MemoryStrategy | None = None,
        sole_representative_boost: float = 1.5,
        rhythmic_pattern_boost: float = 1.3,
        min_rhythmic_occurrences: int = 5,
    ):
        """Initialize TemporalAwareStrategy.

        Args:
            scn: SCN instance for temporal lookups
            base_strategy: Underlying strategy (defaults to AccessBasedStrategy)
            sole_representative_boost: Score multiplier for sole representatives
            rhythmic_pattern_boost: Score multiplier for rhythmic patterns
            min_rhythmic_occurrences: Min memories in bin to consider rhythmic
        """
        self.scn = scn
        self.base = base_strategy or AccessBasedStrategy()
        self.sole_representative_boost = sole_representative_boost
        self.rhythmic_pattern_boost = rhythmic_pattern_boost
        self.min_rhythmic_occurrences = min_rhythmic_occurrences

        # Pre-computed bin populations (set during prepare())
        self._bin_populations: dict[tuple[int, int], int] = {}
        self._prepared = False

    def prepare(self) -> None:
        """Pre-compute bin populations for efficient lookups.

        Call this once at the start of sleep() for O(1) lookups.
        """
        self._bin_populations = self.scn.get_bin_populations()
        self._prepared = True

    @property
    def requires_experience_clock(self) -> bool:  # type: ignore[override]
        """The SCN wrapper adds temporal boosts; what the base needs, this needs."""
        return self.base.requires_experience_clock

    def activation_now(self) -> float:
        return self.base.activation_now()

    def on_activation(self, record: MemoryRecord, now: float, source: str) -> bool:
        """Retrieval strengthening belongs to the base model, not to the SCN boost."""
        return self.base.on_activation(record, now, source)

    def score_for_retention(
        self,
        record: MemoryRecord,
        now: float,
        degree: int = 0,
    ) -> float:
        """Score with temporal context boosts."""
        base_score = self.base.score_for_retention(record, now, degree)

        # Get temporal bins for this memory
        bins = self.scn.get_bins(record.id)
        if bins is None:
            return base_score  # No temporal data, use base score

        # Apply temporal boosts
        score = base_score

        # Boost 1: Sole representative of time slot (preserve coverage)
        if self._prepared:
            population = self._bin_populations.get(bins, 0)
            if population == 1:
                score *= self.sole_representative_boost
        else:
            if self.scn.is_sole_representative(record.id):
                score *= self.sole_representative_boost

        # Boost 2: Part of rhythmic pattern (learned behavior)
        if self.scn.is_rhythmic_bin(record.id, self.min_rhythmic_occurrences):
            score *= self.rhythmic_pattern_boost

        return min(1.0, score)

    def should_compress(
        self,
        record: MemoryRecord,
        now: float,
        degree: int = 0,
    ) -> bool:
        """Defer to base strategy for compression decisions."""
        return self.base.should_compress(record, now, degree)

    def select_cluster_representative(
        self,
        cluster: set[str],
        records: dict[str, MemoryRecord],
        now: float,
    ) -> str | None:
        """Select best representative from a temporal cluster.

        Used for temporal clustering: instead of keeping all memories
        in a cluster, keep the best one and remove/compress the rest.

        Args:
            cluster: Set of memory_ids in the cluster
            records: Dict of memory_id -> record for scoring
            now: Current timestamp

        Returns:
            memory_id of best representative, or None if cluster is empty
        """
        if not cluster:
            return None

        best_id = None
        best_score = -1.0

        for memory_id in cluster:
            record = records.get(memory_id)
            if record is None:
                continue

            score = self.base.score_for_retention(record, now, 0)

            # Prefer long-term memories
            if getattr(record, "long_term", False):
                score *= 1.5

            # Prefer user interactions
            from maxim.memory.types import CompressedMemory, EpisodicMemory

            if isinstance(record, EpisodicMemory):
                if record.perception.cli_input or record.perception.transcript:
                    score *= 1.3
            elif isinstance(record, CompressedMemory):
                if record.had_user_input:
                    score *= 1.3

            if score > best_score:
                best_score = score
                best_id = memory_id

        return best_id


class StrengthStrategy(MemoryStrategy):
    """Bjork storage/retrieval strength on the EXPERIENCE clock (memory-strength Phase 2c-3).

    Two quantities per trace, and only one of them is stored:

    - **Storage strength ``S``** -- how well-learned. Stamped at capture from what the trace was
      encoded WITH (``S0 = s_base * (1 + k * tag)``, Phase 2c-2) and raised by each CREDITED
      retrieval. It is carried in the experience clock's own integer microseconds, so ``dt / S``
      needs no conversion and no conversion can be got wrong.
    - **Retrievability ``R = exp(-dt / S)``** -- how accessible the trace is NOW. Derived, never
      stored: ``dt`` is the experience elapsed since the trace's anchor (its capture, or its last
      credited retrieval, which is what "then R = 1" means).

    ``dt`` is EXPERIENCE, never wall-clock, which is the whole point: a robot switched off for a
    month must not wake with its memory wiped, while an agent that lived through a month forgets.
    So this strategy ignores the ``now`` its ``score_for_retention`` signature is handed -- that
    argument is ``time.time()`` from the sleep path -- and reads its clock instead. It says so by
    declaring ``requires_experience_clock``, and it refuses to be built without one: a strategy
    holding no clock would score every trace at ``R = 1`` forever, which is not "no forgetting"
    but "silently broken", and looks identical from the outside.

    **Protection, not immortality.** A strongly tagged trace keeps a FLOOR under its retrievability,
    and that floor fades on the same clock, ``TAG_FADE_MULTIPLIER`` times slower than ``R``. So
    one-shot fear survives its neighbours by a wide margin and still, eventually, fades -- unlike
    ``AccessBasedStrategy``'s ``access_count >= 10`` floor, which never lifts.

    **Not in Phase 2c-3, deliberately:** this is the HIPPOCAMPUS' model (ATL concepts carry no
    ``S``; plan decision 5); the sleep-time replay, downscale and the unlinked/uncited conjunction
    in the forgetting rule are Phase 3; ``degree`` is accepted and unused, since schema linkage
    enters through that conjunction rather than as a score term. The constants are UNVALIDATED
    placeholders in the plan's own words -- Phase 5 earns them.
    """

    requires_experience_clock = True

    def __init__(
        self,
        clock: "ExperienceClock",
        *,
        s_base: float = S_BASE_DEFAULT,
        gain: float = RETRIEVAL_GAIN,
        saturation: float = RETRIEVAL_SATURATION,
        credited_gap_us: int = CREDITED_GAP_US,
        source_weights: "Mapping[str, float] | None" = None,
        protection_floor_weight: float = PROTECTION_FLOOR_WEIGHT,
        tag_fade_multiplier: float = TAG_FADE_MULTIPLIER,
    ) -> None:
        """Build the model. The clock is REQUIRED and positional -- see the class docstring."""
        if clock is None:
            raise ValueError(
                "StrengthStrategy needs an ExperienceClock: without one every trace scores R = 1 "
                "forever, which is a silent no-op, not a retention policy"
            )
        if not math.isfinite(s_base) or s_base <= 0.0:
            raise ValueError(f"s_base must be finite and positive, got {s_base!r}")
        if not math.isfinite(tag_fade_multiplier) or tag_fade_multiplier < 1.0:
            raise ValueError(f"tag_fade_multiplier must be >= 1 (the tag fades SLOWER), got {tag_fade_multiplier!r}")
        if credited_gap_us < 0:
            raise ValueError(f"credited_gap_us cannot be negative, got {credited_gap_us!r}")
        self.clock = clock
        self.s_base = float(s_base)
        self.gain = float(gain)
        self.saturation = float(saturation)
        self.credited_gap_us = int(credited_gap_us)
        self.source_weights = dict(source_weights if source_weights is not None else RETRIEVAL_SOURCE_WEIGHTS)
        self.protection_floor_weight = float(protection_floor_weight)
        self.tag_fade_multiplier = float(tag_fade_multiplier)

    # ── the model ────────────────────────────────────────────────────────

    def retrievability(self, record: MemoryRecord, now_us: int | None = None) -> float:
        """``R = exp(-dt / S)`` for one trace, as of ``now_us`` (default: the clock's now)."""
        return self._retrievability_and_floor(record, self.clock.now_us() if now_us is None else now_us)[0]

    def _strength_of(self, record: MemoryRecord) -> tuple[float, int | None]:
        """``(S, anchor)``, with the unstamped case named rather than guessed.

        A record that carries no strength FIELDS at all reached the wrong store: raise, because the
        silent alternative is scoring every ATL concept at ``R = 1`` -- immortality dressed as a
        default. A record that carries them UNSTAMPED (``None``) is an ordinary pre-2c trace: it is
        read as freshly encoded at ``s_base``, never as a zero-strength trace to be thrown away.
        """
        if not hasattr(record, "storage_strength"):
            raise TypeError(
                f"{type(record).__name__} carries no storage strength; StrengthStrategy models the "
                "Hippocampus' episodes only (plan decision 5 -- ATL concepts get S with their own path)"
            )
        strength = record.storage_strength
        return (self.s_base if strength is None else float(strength), record.retrievability_anchor_us)

    def _retrievability_and_floor(self, record: MemoryRecord, now_us: int) -> tuple[float, float]:
        strength, anchor = self._strength_of(record)
        # A clock that restarted at 0 (a corrupt record: see hippocampus_persistence) must never
        # make a trace MORE retrievable than when it was stored. Experience does not run backwards.
        dt = 0.0 if anchor is None else float(max(0, now_us - anchor))
        retrievability = math.exp(-dt / strength)
        tag = record.encoding_tag or 0.0
        floor = self.protection_floor_weight * tag * math.exp(-dt / (strength * self.tag_fade_multiplier))
        return retrievability, floor

    # ── the strategy protocol ────────────────────────────────────────────

    def score_for_retention(self, record: MemoryRecord, now: float, degree: int = 0) -> float:
        """Retrievability, floored by what the trace was encoded with. ``now`` is IGNORED."""
        retrievability, floor = self._retrievability_and_floor(record, self.clock.now_us())
        # ``min(1.0, ...)`` is unreachable while the dt clamp holds (R <= 1, floor <= 0.5). Kept as
        # the belt, and remembered as a lesson: it is exactly what HID the clamp's absence when the
        # clamp was deleted to test it, which is why that guard asserts raw retrievability instead.
        return min(1.0, max(retrievability, floor))

    def should_compress(self, record: MemoryRecord, now: float, degree: int = 0) -> bool:
        """Compress a fading trace to gist; leave one held up by its tag whole.

        A trace still scoring on its own retrievability is one the agent is simply losing, and gist
        is what survives that. A trace scoring on its protection floor is being KEPT for what it
        meant -- and detail is most of what "it meant" is made of.

        It answers "fading rather than tag-held", so it is only meaningful for a record the caller
        has already placed in the compression band -- a fresh untagged trace answers ``True`` here
        (R = 1, floor = 0) and is saved only by its score. A ``CompositeStrategy`` containing this
        model VOTES on the raw answer, so it would read that as a near-constant yes.
        """
        from maxim.memory.types import CompressedMemory

        if isinstance(record, CompressedMemory):
            return False
        retrievability, floor = self._retrievability_and_floor(record, self.clock.now_us())
        return floor <= retrievability

    # ── the retrieval update ─────────────────────────────────────────────

    def activation_now(self) -> float:
        """Read the clock ONCE per ``activate`` call, before any lock (plan decision 5)."""
        return float(self.clock.now_us())

    def on_activation(self, record: MemoryRecord, now: float, source: str) -> bool:
        """``S <- S * (1 + a * w_src * (1 - R) * (S/s_base)^-w)``, then ``R = 1``, if CREDITED.

        Credited means: this source is worth something, and the trace has not been credited within
        ``credited_gap_us`` of experience. An uncredited activation is still RECORDED (Phase 1's
        counters already ran) and changes no strength -- that is what stops a trace re-rendered
        every deliberation cycle from out-earning a genuinely spaced recall.

        The gain is largest when the trace had faded (``1 - R``) and shrinks as it grows
        well-learned, so repeated spaced retrievals converge rather than compound.
        """
        weight = self.source_weights.get(source, 0.0)
        if weight <= 0.0:
            return False
        now_us = int(now)

        def compute(strength: float | None, anchor: int | None) -> tuple[float, int] | None:
            if anchor is not None and now_us - anchor < self.credited_gap_us:
                return None  # massed: recorded by the counters, credited by nothing
            current = self.s_base if strength is None else float(strength)
            # No clamp needed: the gap check above already returned for every anchor at or ahead of
            # ``now_us``, so a credited update always has ``now_us - anchor >= credited_gap_us >= 0``.
            # (The clamp in ``_retrievability_and_floor`` is the live one -- scoring has no such
            # guard in front of it.)
            dt = 0.0 if anchor is None else float(now_us - anchor)
            retrievability = math.exp(-dt / current)
            # ``min(1.0, ...)``: within one tuning ``S >= s_base`` always holds (S0 = s_base *
            # (1 + k*tag) with tag in [0,1], k >= 0, and the update only raises S), so the clamp is
            # a no-op on the intended path. It bites after an operator RAISES ``memory.s_base``:
            # an already-stamped trace then has S < s_base, the factor exceeds 1, and a single
            # retrieval could multiply S by ~11x where this module documents a bound of 1 + a*w
            # (review round, Executor #7). It also forecloses a 0.0 ** -w underflow.
            saturating = min(1.0, (current / self.s_base) ** -self.saturation)
            return current * (1.0 + self.gain * weight * (1.0 - retrievability) * saturating), now_us

        from maxim.memory.types import update_strength_atomically

        return update_strength_atomically(record, compute)


__all__ = [
    "MemoryStrategy",
    "AccessBasedStrategy",
    "ImportanceBasedStrategy",
    "CompositeStrategy",
    "TemporalAwareStrategy",
    "StrengthStrategy",
    "CREDITED_GAP_US",
    "PROTECTION_FLOOR_WEIGHT",
    "RETRIEVAL_GAIN",
    "RETRIEVAL_SATURATION",
    "RETRIEVAL_SOURCE_WEIGHTS",
    "TAG_FADE_MULTIPLIER",
]
