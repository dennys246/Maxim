"""Unit tests for NoveltyTracker — instance-level and class-level novelty.

Tests the core novelty tracking system including:
- Instance novelty (focus/decay/recovery)
- Class-level novelty (categorical habituation)
- Class modulation of instance novelty
- Thread-safe wrapper
- Cleanup behavior with class mappings
"""

from __future__ import annotations

import time

import pytest

from maxim.inference.segment_vision import NoveltyTracker
from maxim.salience.novelty import ThreadSafeNoveltyTracker


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tracker():
    """Default NoveltyTracker with standard settings."""
    return NoveltyTracker()


@pytest.fixture
def fast_tracker():
    """NoveltyTracker with fast decay for timing tests."""
    return NoveltyTracker(focus_decay_seconds=0.1, recovery_seconds=0.2)


@pytest.fixture
def class_tracker():
    """NoveltyTracker with strong class weighting for clear test signals."""
    return NoveltyTracker(class_novelty_weight=0.5, class_novelty_halflife=5)


# ---------------------------------------------------------------------------
# Instance-Level Novelty
# ---------------------------------------------------------------------------

class TestInstanceNovelty:
    """Tests for instance-level (track_id) novelty."""

    def test_unseen_track_has_max_novelty(self, tracker):
        """A never-seen track_id should have maximum novelty."""
        assert tracker.get_novelty(999) == tracker.max_novelty

    def test_none_track_has_max_novelty(self, tracker):
        """None track_id should return max novelty."""
        assert tracker.get_novelty(None) == tracker.max_novelty

    def test_focused_track_retains_max_novelty(self, tracker):
        """A recently focused track should keep max novelty during decay window."""
        tracker.focus(1)
        assert tracker.get_novelty(1) == tracker.max_novelty

    def test_novelty_decays_after_focus_window(self, fast_tracker):
        """Novelty should decay after focus_decay_seconds passes."""
        fast_tracker.focus(1)
        time.sleep(0.35)  # Past focus_decay + some recovery time
        novelty = fast_tracker.get_novelty(1)
        assert novelty < fast_tracker.max_novelty

    def test_update_does_not_affect_novelty(self, tracker):
        """Calling update() should not change novelty — only focus() does."""
        tracker.update(1)
        assert tracker.get_novelty(1) == tracker.max_novelty

    def test_focus_resets_novelty(self, fast_tracker):
        """Focusing on a previously decayed track resets novelty."""
        fast_tracker.focus(1)
        time.sleep(0.35)
        decayed = fast_tracker.get_novelty(1)
        assert decayed < fast_tracker.max_novelty

        fast_tracker.focus(1)
        assert fast_tracker.get_novelty(1) == fast_tracker.max_novelty


# ---------------------------------------------------------------------------
# Class-Level Novelty
# ---------------------------------------------------------------------------

class TestClassNovelty:
    """Tests for class-level (COCO class) novelty."""

    def test_unseen_class_has_max_novelty(self, tracker):
        """A never-seen class should have maximum novelty."""
        assert tracker.get_class_novelty(41) == tracker.max_novelty  # cup

    def test_class_novelty_decays_with_instances(self, tracker):
        """Seeing more unique instances of a class should reduce its novelty."""
        # Register 10 unique chairs (class 56)
        for i in range(10):
            tracker.update_with_class(f"chair_{i}", 56)

        chair_novelty = tracker.get_class_novelty(56)
        cup_novelty = tracker.get_class_novelty(41)  # Never seen

        assert chair_novelty < cup_novelty
        assert chair_novelty < tracker.max_novelty

    def test_class_novelty_halves_at_halflife(self, tracker):
        """Class novelty should roughly halve after halflife instances."""
        halflife = tracker.class_novelty_halflife  # Default: 10

        for i in range(halflife):
            tracker.update_with_class(f"obj_{i}", 99)

        novelty = tracker.get_class_novelty(99)
        expected = tracker.max_novelty * 0.5
        # Should be close to half (or floored)
        assert novelty <= expected + 0.01

    def test_class_novelty_floor(self, tracker):
        """Class novelty should never drop below floor * max_novelty."""
        # Register many instances
        for i in range(200):
            tracker.update_with_class(f"obj_{i}", 56)

        novelty = tracker.get_class_novelty(56)
        floor = tracker.class_novelty_floor * tracker.max_novelty
        assert novelty >= floor - 0.001  # Allow floating point tolerance

    def test_same_track_id_counted_once(self, tracker):
        """Re-seeing the same track_id should not re-increment class count."""
        tracker.update_with_class("chair_1", 56)
        tracker.update_with_class("chair_1", 56)
        tracker.update_with_class("chair_1", 56)

        assert tracker._class_instance_counts[56] == 1

    def test_different_classes_tracked_separately(self, tracker):
        """Different classes should have independent novelty scores."""
        for i in range(15):
            tracker.update_with_class(f"chair_{i}", 56)
        for i in range(3):
            tracker.update_with_class(f"cup_{i}", 41)

        chair_novelty = tracker.get_class_novelty(56)
        cup_novelty = tracker.get_class_novelty(41)

        assert chair_novelty < cup_novelty  # More chairs seen


# ---------------------------------------------------------------------------
# Class Modulation of Instance Novelty
# ---------------------------------------------------------------------------

class TestClassModulation:
    """Tests that class novelty modulates instance novelty correctly."""

    def test_class_modulation_reduces_familiar_category(self, class_tracker):
        """New instance of a familiar class should have lower novelty than new instance of rare class."""
        # Register many chairs
        for i in range(20):
            class_tracker.update_with_class(f"chair_{i}", 56)

        # New chair (never focused, max instance novelty) but familiar class
        new_chair_novelty = class_tracker.get_novelty("new_chair", class_id=56)
        # New cup (never focused, max instance novelty) and rare class
        new_cup_novelty = class_tracker.get_novelty("new_cup", class_id=41)

        assert new_chair_novelty < new_cup_novelty

    def test_no_class_id_skips_modulation(self, class_tracker):
        """Without class_id, novelty should equal pure instance novelty."""
        # Even with many chairs registered
        for i in range(20):
            class_tracker.update_with_class(f"chair_{i}", 56)

        # New track without class_id should get full max novelty
        assert class_tracker.get_novelty("new_track") == class_tracker.max_novelty

    def test_zero_weight_disables_class_modulation(self, tracker):
        """With class_novelty_weight=0, class_id should have no effect."""
        tracker.class_novelty_weight = 0.0

        for i in range(50):
            tracker.update_with_class(f"obj_{i}", 56)

        novelty_with_class = tracker.get_novelty("new", class_id=56)
        novelty_without_class = tracker.get_novelty("new")

        assert novelty_with_class == novelty_without_class

    def test_novel_class_preserves_full_novelty(self, class_tracker):
        """A never-seen class should not reduce instance novelty."""
        # class_id=99 never seen → class_novelty = max → class_factor = 1.0
        novelty = class_tracker.get_novelty("fresh_obj", class_id=99)
        assert novelty == class_tracker.max_novelty

    def test_modulation_blends_correctly(self, tracker):
        """Verify the blend formula: instance * ((1-w) + w * class_factor)."""
        w = tracker.class_novelty_weight

        # Never-seen class: class_factor = 1.0, so modulation = 1.0
        novelty_novel = tracker.get_novelty("x", class_id=999)
        assert novelty_novel == tracker.max_novelty

        # Seen class: class_factor < 1.0
        for i in range(tracker.class_novelty_halflife):
            tracker.update_with_class(f"seen_{i}", 56)

        class_nov = tracker.get_class_novelty(56)
        class_factor = class_nov / tracker.max_novelty
        expected = tracker.max_novelty * ((1 - w) + w * class_factor)

        actual = tracker.get_novelty("y", class_id=56)
        assert abs(actual - expected) < 0.001


# ---------------------------------------------------------------------------
# Update With Class
# ---------------------------------------------------------------------------

class TestUpdateWithClass:
    """Tests for update_with_class method."""

    def test_registers_class_first_seen(self, tracker):
        """First instance of a class should record first-seen timestamp."""
        tracker.update_with_class("obj_1", 56)
        assert 56 in tracker._class_first_seen

    def test_increments_instance_count(self, tracker):
        """Each new track_id for a class should increment the count."""
        tracker.update_with_class("a", 56)
        tracker.update_with_class("b", 56)
        tracker.update_with_class("c", 56)
        assert tracker._class_instance_counts[56] == 3

    def test_none_track_id_handled(self, tracker):
        """None track_id should not crash update_with_class."""
        tracker.update_with_class(None, 56)
        # Should not add to class counts
        assert tracker._class_instance_counts.get(56, 0) == 0

    def test_track_to_class_mapping(self, tracker):
        """Should maintain track_id -> class_id mapping."""
        tracker.update_with_class("obj_1", 56)
        assert tracker._track_to_class["obj_1"] == 56


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

class TestCleanup:
    """Tests that cleanup handles class mappings correctly."""

    def test_cleanup_removes_stale_track_class_mapping(self):
        """Cleanup should remove _track_to_class for cleaned entries."""
        tracker = NoveltyTracker(
            cleanup_interval=5,
            max_age=0.1,  # Very short age for test
            adaptive_cleanup=False,
        )

        tracker.update_with_class("old_obj", 56)
        time.sleep(0.15)  # Let it age out

        # Trigger cleanup by making enough updates
        for i in range(6):
            tracker.update_with_class(f"new_{i}", 41)

        # Old entry should be cleaned from track_to_class
        assert "old_obj" not in tracker._track_to_class

    def test_cleanup_preserves_class_counts(self):
        """Cleanup should NOT decrement class instance counts — they're lifetime totals."""
        tracker = NoveltyTracker(
            cleanup_interval=5,
            max_age=0.1,
            adaptive_cleanup=False,
        )

        tracker.update_with_class("old_chair", 56)
        time.sleep(0.15)

        for i in range(6):
            tracker.update_with_class(f"new_{i}", 41)

        # Chair count should still be 1 even though the track was cleaned
        assert tracker._class_instance_counts.get(56, 0) == 1


# ---------------------------------------------------------------------------
# ThreadSafeNoveltyTracker
# ---------------------------------------------------------------------------

class TestThreadSafeWrapper:
    """Tests for the thread-safe wrapper."""

    def test_wraps_existing_tracker(self):
        """Should wrap a provided tracker."""
        inner = NoveltyTracker()
        safe = ThreadSafeNoveltyTracker(inner)
        assert safe._tracker is inner

    def test_creates_default_tracker(self):
        """Should create a tracker if none provided."""
        safe = ThreadSafeNoveltyTracker()
        assert isinstance(safe._tracker, NoveltyTracker)

    def test_get_novelty_with_class_id(self):
        """get_novelty should pass class_id through."""
        safe = ThreadSafeNoveltyTracker()

        # Register some instances
        for i in range(10):
            safe.update_with_class(f"obj_{i}", 56)

        # Verify class modulation works through wrapper
        novelty_familiar = safe.get_novelty("new", class_id=56)
        novelty_novel = safe.get_novelty("new", class_id=99)
        assert novelty_familiar < novelty_novel

    def test_update_with_class(self):
        """update_with_class should work through wrapper."""
        safe = ThreadSafeNoveltyTracker()
        safe.update_with_class("obj_1", 56)
        safe.update_with_class("obj_2", 56)

        assert safe.get_class_novelty(56) < safe.max_novelty

    def test_get_class_novelty(self):
        """get_class_novelty should work through wrapper."""
        safe = ThreadSafeNoveltyTracker()
        assert safe.get_class_novelty(99) == safe.max_novelty

    def test_get_novelty_batch_with_class_ids(self):
        """get_novelty_batch should accept class_ids dict."""
        safe = ThreadSafeNoveltyTracker()

        for i in range(15):
            safe.update_with_class(f"chair_{i}", 56)

        track_ids = ["a", "b"]
        class_ids = {"a": 56, "b": 99}
        results = safe.get_novelty_batch(track_ids, class_ids=class_ids)

        assert results["a"] < results["b"]  # Chair familiar, 99 novel

    def test_properties_accessible(self):
        """All properties should work through wrapper."""
        safe = ThreadSafeNoveltyTracker()
        assert safe.max_novelty == 2.0
        assert safe.min_novelty == 0.5
        assert safe.tracked_count == 0
        assert safe.focus_decay_seconds > 0
        assert safe.recovery_seconds > 0


# ---------------------------------------------------------------------------
# Full-Spectrum Detection (Integration-like)
# ---------------------------------------------------------------------------

class TestFullSpectrumDetection:
    """Tests verifying the full-spectrum detection design."""

    def test_all_coco_classes_get_novelty_scores(self, tracker):
        """Any COCO class ID (0-79) should produce valid novelty scores."""
        for cls_id in range(80):
            novelty = tracker.get_class_novelty(cls_id)
            assert novelty == tracker.max_novelty  # All unseen

    def test_interest_classes_vs_non_interest_novelty(self, tracker):
        """Interest and non-interest classes should have same novelty behavior.

        The interest/non-interest distinction happens in PerceptionAgent's
        _compute_salience (2x multiplier), NOT in NoveltyTracker.
        """
        # Both should behave identically in the tracker
        tracker.update_with_class("person_1", 0)   # Interest class
        tracker.update_with_class("cup_1", 41)      # Non-interest class

        person_novelty = tracker.get_class_novelty(0)
        cup_novelty = tracker.get_class_novelty(41)

        # Both seen once — same novelty
        assert person_novelty == cup_novelty

    def test_habituation_over_session(self, tracker):
        """Simulates a session where the robot sees many chairs and few cups."""
        # 25 unique chairs seen over session
        for i in range(25):
            tracker.update_with_class(f"chair_{i}", 56)

        # 2 cups seen
        for i in range(2):
            tracker.update_with_class(f"cup_{i}", 41)

        # A new chair should be less novel than a new cup
        new_chair = tracker.get_novelty("chair_26", class_id=56)
        new_cup = tracker.get_novelty("cup_3", class_id=41)

        assert new_chair < new_cup

        # But neither should be zero — floor ensures attention
        assert new_chair > 0
        assert new_cup > 0


# ---------------------------------------------------------------------------
# Sensitization
# ---------------------------------------------------------------------------

class TestSensitization:
    """Tests for value-gated sensitization via modulation_lookup callback."""

    def test_no_modulation_lookup_natural_decay(self, tracker):
        """Without modulation lookup, class novelty should follow pure count-based decay."""
        for i in range(10):
            tracker.update_with_class(f"obj_{i}", 56)

        # Should be natural decay — no amplification
        novelty = tracker.get_class_novelty(56)
        expected = tracker.max_novelty * 0.5  # halflife = 10
        assert abs(novelty - expected) < 0.01

    def test_modulation_above_one_amplifies(self):
        """Modulation > 1.0 should amplify class novelty above natural level."""
        tracker = NoveltyTracker(_modulation_lookup=lambda _: 1.3)

        for i in range(10):
            tracker.update_with_class(f"obj_{i}", 56)

        natural = tracker.max_novelty * 0.5  # Without modulation
        actual = tracker.get_class_novelty(56)
        assert actual > natural
        assert abs(actual - natural * 1.3) < 0.01

    def test_modulation_one_has_no_effect(self):
        """Modulation = 1.0 should have no effect on class novelty."""
        tracker = NoveltyTracker(_modulation_lookup=lambda _: 1.0)

        for i in range(10):
            tracker.update_with_class(f"obj_{i}", 56)

        natural = tracker.max_novelty * 0.5
        actual = tracker.get_class_novelty(56)
        assert abs(actual - natural) < 0.001

    def test_sensitization_ceiling_clamps(self):
        """Modulation should be clamped to sensitization_ceiling * max_novelty."""
        tracker = NoveltyTracker(
            sensitization_ceiling=1.5,
            _modulation_lookup=lambda _: 5.0,  # Extreme
        )

        for i in range(5):
            tracker.update_with_class(f"obj_{i}", 56)

        actual = tracker.get_class_novelty(56)
        ceiling = tracker.max_novelty * tracker.sensitization_ceiling
        assert actual <= ceiling + 0.001

    def test_broken_lookup_graceful_fallback(self):
        """Broken modulation lookup should fall back to unmodulated score."""
        def bad_lookup(_):
            raise RuntimeError("broken")

        tracker = NoveltyTracker(_modulation_lookup=bad_lookup)

        for i in range(10):
            tracker.update_with_class(f"obj_{i}", 56)

        # Should fall back to natural decay
        natural = tracker.max_novelty * 0.5
        actual = tracker.get_class_novelty(56)
        assert abs(actual - natural) < 0.01

    def test_set_none_disables_modulation(self):
        """Setting lookup to None should disable modulation."""
        tracker = NoveltyTracker(_modulation_lookup=lambda _: 1.5)

        for i in range(10):
            tracker.update_with_class(f"obj_{i}", 56)

        amplified = tracker.get_class_novelty(56)

        tracker.set_modulation_lookup(None)
        natural = tracker.get_class_novelty(56)

        assert amplified > natural

    def test_unseen_class_max_novelty_regardless(self):
        """Unseen class (count=0) should return max_novelty regardless of lookup."""
        tracker = NoveltyTracker(_modulation_lookup=lambda _: 1.5)
        assert tracker.get_class_novelty(99) == tracker.max_novelty

    def test_thread_safe_wrapper_passthrough(self):
        """ThreadSafe wrapper should pass through set_modulation_lookup."""
        safe = ThreadSafeNoveltyTracker()
        safe.set_modulation_lookup(lambda _: 1.3)

        for i in range(10):
            safe.update_with_class(f"obj_{i}", 56)

        natural = safe.max_novelty * 0.5
        actual = safe.get_class_novelty(56)
        assert actual > natural

        assert safe.sensitization_ceiling == 1.5

    def test_extreme_positive_sensitizes(self):
        """100% success rate (extreme positive) should produce modulation > 1."""
        # Simulate the modulation formula
        success_rate = 1.0
        total = 10
        min_interactions = 5
        scale = 0.5

        extremity = abs(success_rate - 0.5) * 2  # 1.0
        confidence = min(1.0, total / min_interactions)  # 1.0
        modulation = 1.0 + extremity * confidence * scale  # 1.5

        tracker = NoveltyTracker(_modulation_lookup=lambda _: modulation)

        for i in range(10):
            tracker.update_with_class(f"obj_{i}", 56)

        natural = tracker.max_novelty * 0.5
        actual = tracker.get_class_novelty(56)
        assert actual > natural

    def test_extreme_negative_also_sensitizes(self):
        """0% success rate (extreme negative) should also produce modulation > 1."""
        success_rate = 0.0
        total = 10
        min_interactions = 5
        scale = 0.5

        extremity = abs(success_rate - 0.5) * 2  # 1.0
        confidence = min(1.0, total / min_interactions)  # 1.0
        modulation = 1.0 + extremity * confidence * scale  # 1.5

        tracker = NoveltyTracker(_modulation_lookup=lambda _: modulation)

        for i in range(10):
            tracker.update_with_class(f"obj_{i}", 56)

        natural = tracker.max_novelty * 0.5
        actual = tracker.get_class_novelty(56)
        assert actual > natural

    def test_neutral_interaction_no_sensitization(self):
        """50% success rate (neutral) should produce modulation = 1.0."""
        success_rate = 0.5
        total = 10
        min_interactions = 5
        scale = 0.5

        extremity = abs(success_rate - 0.5) * 2  # 0.0
        confidence = min(1.0, total / min_interactions)  # 1.0
        modulation = 1.0 + extremity * confidence * scale  # 1.0

        assert modulation == 1.0

        tracker = NoveltyTracker(_modulation_lookup=lambda _: modulation)

        for i in range(10):
            tracker.update_with_class(f"obj_{i}", 56)

        natural = tracker.max_novelty * 0.5
        actual = tracker.get_class_novelty(56)
        assert abs(actual - natural) < 0.001
