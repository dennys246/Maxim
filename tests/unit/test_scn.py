"""Unit tests for Suprachiasmatic Nucleus (SCN) - Temporal rhythm indexing.

Tests the core functionality of temporal indexing, queries, and
sleep consolidation support.
"""

from __future__ import annotations


class TestSCNRegistration:
    """Test memory registration and indexing."""

    def test_register_adds_to_all_bins(self, scn, temporal_signature_morning):
        """Registration populates circadian, weekly, monthly, annual bins."""
        scn.register("mem_1", temporal_signature_morning)

        # Check circadian (hour 9)
        assert "mem_1" in scn.query_hour(9)

        # Check weekly (Monday = 0)
        assert "mem_1" in scn.query_day(0)

    def test_register_stores_signature(self, scn, temporal_signature_morning):
        """Registration stores the temporal signature for retrieval."""
        scn.register("mem_1", temporal_signature_morning)

        sig = scn.get_signature("mem_1")
        assert sig is not None
        assert sig == temporal_signature_morning

    def test_unregister_removes_from_all_bins(self, scn, temporal_signature_morning):
        """Unregistration removes from all indices."""
        scn.register("mem_1", temporal_signature_morning)
        scn.unregister("mem_1")

        assert "mem_1" not in scn.query_hour(9)
        assert "mem_1" not in scn.query_day(0)
        assert scn.get_signature("mem_1") is None

    def test_unregister_nonexistent_is_safe(self, scn):
        """Unregistering non-existent memory doesn't raise."""
        scn.unregister("nonexistent")  # Should not raise

    def test_remove_memory_is_alias_for_unregister(self, scn, temporal_signature_morning):
        """remove_memory works as callback-compatible alias."""
        scn.register("mem_1", temporal_signature_morning)
        scn.remove_memory("mem_1")

        assert "mem_1" not in scn.query_hour(9)


class TestSCNQueries:
    """Test temporal queries."""

    def test_query_hour_returns_matching_memories(self, scn, temporal_signature_morning, temporal_signature_evening):
        """Query by hour returns memories at that hour."""
        scn.register("morning", temporal_signature_morning)  # 9am
        scn.register("evening", temporal_signature_evening)  # 8pm

        morning_mems = scn.query_hour(9)
        evening_mems = scn.query_hour(20)

        assert "morning" in morning_mems
        assert "morning" not in evening_mems
        assert "evening" in evening_mems
        assert "evening" not in morning_mems

    def test_query_day_returns_matching_memories(self, scn, temporal_signature_morning, temporal_signature_evening):
        """Query by day returns memories on that day."""
        scn.register("monday", temporal_signature_morning)  # Monday
        scn.register("friday", temporal_signature_evening)  # Friday

        monday_mems = scn.query_day(0)
        friday_mems = scn.query_day(4)

        assert "monday" in monday_mems
        assert "friday" in friday_mems

    def test_query_intersection_returns_correct_memories(self, scn):
        """Intersection query returns only memories matching ALL criteria."""
        from maxim.time.temporal_signature import TemporalSignature

        # Monday 9am
        sig_mon_9 = TemporalSignature(
            timestamp=1704103200.0, circadian_phase=9 / 24, weekly_phase=0 / 7, monthly_phase=0.25, annual_phase=0.0
        )
        # Monday 10am
        sig_mon_10 = TemporalSignature(
            timestamp=1704106800.0, circadian_phase=10 / 24, weekly_phase=0 / 7, monthly_phase=0.25, annual_phase=0.0
        )
        # Tuesday 9am
        sig_tue_9 = TemporalSignature(
            timestamp=1704189600.0, circadian_phase=9 / 24, weekly_phase=1 / 7, monthly_phase=0.25, annual_phase=0.0
        )

        scn.register("mon_9am", sig_mon_9)
        scn.register("mon_10am", sig_mon_10)
        scn.register("tue_9am", sig_tue_9)

        result = scn.query_intersection(hour=9, day=0)

        assert result == {"mon_9am"}

    def test_query_intersection_with_no_criteria_returns_empty(self, scn):
        """Intersection with no criteria returns empty set."""
        result = scn.query_intersection()
        assert result == set()

    def test_query_similar_time_with_tolerance(self, scn):
        """Similar time query respects tolerance parameter."""
        from maxim.time.temporal_signature import TemporalSignature

        sig_9 = TemporalSignature(
            timestamp=1704103200.0, circadian_phase=9 / 24, weekly_phase=0 / 7, monthly_phase=0.25, annual_phase=0.0
        )
        sig_10 = TemporalSignature(
            timestamp=1704106800.0, circadian_phase=10 / 24, weekly_phase=0 / 7, monthly_phase=0.25, annual_phase=0.0
        )
        sig_12 = TemporalSignature(
            timestamp=1704114000.0, circadian_phase=12 / 24, weekly_phase=0 / 7, monthly_phase=0.25, annual_phase=0.0
        )

        scn.register("9am", sig_9)
        scn.register("10am", sig_10)
        scn.register("12pm", sig_12)

        result = scn.query_similar_time(sig_9, tolerance=1)

        assert "9am" in result
        assert "10am" in result  # Within tolerance
        assert "12pm" not in result  # Outside tolerance


class TestSCNSleepConsolidation:
    """Test features used by Hippocampus sleep consolidation."""

    def test_is_sole_representative_true_for_single_memory(self, scn):
        """Single memory in a time slot is sole representative."""
        from maxim.time.temporal_signature import TemporalSignature

        sig = TemporalSignature(
            timestamp=1704103200.0, circadian_phase=3 / 24, weekly_phase=6 / 7, monthly_phase=0.5, annual_phase=0.5
        )
        scn.register("only_one", sig)

        assert scn.is_sole_representative("only_one") is True

    def test_is_sole_representative_false_when_others_exist(self, scn):
        """Not sole representative when other memories share the slot."""
        from maxim.time.temporal_signature import TemporalSignature

        sig = TemporalSignature(
            timestamp=1704103200.0, circadian_phase=9 / 24, weekly_phase=0 / 7, monthly_phase=0.25, annual_phase=0.0
        )
        scn.register("mem_1", sig)
        scn.register("mem_2", sig)

        assert scn.is_sole_representative("mem_1") is False
        assert scn.is_sole_representative("mem_2") is False

    def test_is_rhythmic_bin_detects_patterns(self, scn):
        """Detects bins with many memories as rhythmic."""
        from maxim.time.temporal_signature import TemporalSignature

        # Create many memories at 9am
        sig_9am = TemporalSignature(
            timestamp=1704103200.0, circadian_phase=9 / 24, weekly_phase=0 / 7, monthly_phase=0.25, annual_phase=0.0
        )
        for i in range(10):
            scn.register(f"mem_{i}", sig_9am)

        assert scn.is_rhythmic_bin("mem_0", min_occurrences=5) is True

    def test_is_rhythmic_bin_false_for_sparse_bins(self, scn):
        """Sparse bins are not rhythmic."""
        from maxim.time.temporal_signature import TemporalSignature

        sig = TemporalSignature(
            timestamp=1704103200.0, circadian_phase=3 / 24, weekly_phase=6 / 7, monthly_phase=0.5, annual_phase=0.5
        )
        scn.register("lonely", sig)

        assert scn.is_rhythmic_bin("lonely", min_occurrences=5) is False

    def test_get_all_clusters_groups_correctly(self, scn):
        """Clusters group memories by (hour, day)."""
        from maxim.time.temporal_signature import TemporalSignature

        sig_1 = TemporalSignature(
            timestamp=1704103200.0, circadian_phase=9 / 24, weekly_phase=0 / 7, monthly_phase=0.25, annual_phase=0.0
        )
        sig_2 = TemporalSignature(
            timestamp=1704106800.0, circadian_phase=10 / 24, weekly_phase=0 / 7, monthly_phase=0.25, annual_phase=0.0
        )

        scn.register("a", sig_1)
        scn.register("b", sig_1)  # Same cluster as 'a'
        scn.register("c", sig_2)  # Different cluster

        clusters = scn.get_all_clusters()

        assert len(clusters[(9, 0)]) == 2
        assert len(clusters[(10, 0)]) == 1

    def test_get_bins_returns_correct_tuple(self, scn, temporal_signature_morning):
        """get_bins returns (hour, day) tuple."""
        scn.register("mem", temporal_signature_morning)

        bins = scn.get_bins("mem")

        assert bins is not None
        hour, day = bins
        assert hour == 9
        assert day == 0

    def test_get_bins_returns_none_for_unregistered(self, scn):
        """get_bins returns None for unregistered memory."""
        assert scn.get_bins("nonexistent") is None

    def test_get_bin_populations(self, scn):
        """get_bin_populations returns correct counts."""
        from maxim.time.temporal_signature import TemporalSignature

        sig = TemporalSignature(
            timestamp=1704103200.0, circadian_phase=9 / 24, weekly_phase=0 / 7, monthly_phase=0.25, annual_phase=0.0
        )
        for i in range(5):
            scn.register(f"mem_{i}", sig)

        populations = scn.get_bin_populations()

        assert populations[(9, 0)] == 5


class TestSCNPatterns:
    """Test rhythmic pattern detection."""

    def test_find_rhythmic_patterns(self, scn):
        """Identifies bins with repeated activity."""
        from maxim.time.temporal_signature import TemporalSignature

        # Create pattern: many at 9am Monday
        sig = TemporalSignature(
            timestamp=1704103200.0, circadian_phase=9 / 24, weekly_phase=0 / 7, monthly_phase=0.25, annual_phase=0.0
        )
        for i in range(10):
            scn.register(f"mem_{i}", sig)

        patterns = scn.find_rhythmic_patterns(min_occurrences=5)

        # Should detect circadian pattern at hour 9
        assert any(hour == 9 and count >= 5 for hour, count in patterns["circadian"])

    def test_threshold_adjustment_based_on_data(self, scn, temporal_signature_morning):
        """Threshold adjustment reflects data availability."""
        # Few memories = higher threshold
        scn.register("single", temporal_signature_morning)
        adjustment = scn.get_threshold_adjustment(temporal_signature_morning)
        assert adjustment >= 1.0  # Higher when little data


class TestSCNPersistence:
    """Test save/load roundtrip."""

    def test_save_load_preserves_registrations(self, scn, temporal_signature_morning, tmp_path):
        """Registrations survive save/load cycle."""
        scn.register("mem_1", temporal_signature_morning)

        path = tmp_path / "scn.json"
        scn.save(str(path))

        from maxim.time.scn import SCN

        scn2 = SCN()
        scn2.load(str(path))

        assert "mem_1" in scn2.query_hour(9)
        assert scn2.get_signature("mem_1") is not None

    def test_save_load_preserves_priors(self, scn, tmp_path):
        """Temporal priors survive save/load cycle."""
        scn.add_temporal_prior("morning_greeting", 9)

        path = tmp_path / "scn.json"
        scn.save(str(path))

        from maxim.time.scn import SCN

        scn2 = SCN()
        scn2.load(str(path))

        assert 9 in scn2._priors.get("morning_greeting", set())


class TestSCNStats:
    """Test statistics collection."""

    def test_stats_returns_expected_keys(self, scn, temporal_signature_morning):
        """Stats returns all expected keys."""
        scn.register("mem", temporal_signature_morning)

        stats = scn.stats()

        assert "total_signatures" in stats
        assert "circadian_bins_used" in stats
        assert "weekly_bins_used" in stats

    def test_len_returns_signature_count(self, scn, temporal_signature_morning):
        """__len__ returns number of registered memories."""
        assert len(scn) == 0

        scn.register("mem_1", temporal_signature_morning)
        assert len(scn) == 1

        scn.register("mem_2", temporal_signature_morning)
        assert len(scn) == 2
