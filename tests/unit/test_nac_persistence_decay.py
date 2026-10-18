"""Offline (decay-on-load) semantics for NAc cross-session persistence.

**Default since 2026-10-18 (owner decision 2026-10-17, experience_clock_decay.md decision 2 + Q7): NO
forgetting while switched off.** ``load()``/``load_safe()`` restore every map exactly as saved, whatever
wall-clock time passed. The 1.3-era decay (docs/plans/archive/nac_cross_session_persistence.md) is the
Dormant opt-in ``NACConfig(offline_wall_decay=True)``, unchanged: ``load()`` ages the bias surfaces by
elapsed WALL-CLOCK time since the payload's ``saved_at`` stamp, on two half-life schedules:

- ``cluster_bias_wall_decay_half_life_s`` (1 day) — ``cluster_reward_bias``.
- ``bias_wall_decay_half_life_s`` (7 days) — ``reward_bias`` / ``goal_reward_bias`` /
  ``percept_valences`` / ``cluster_fear``.

Under either setting causal links + Welford variance are NOT decayed on load, pre-1.3 payloads (no
``saved_at``) load undecayed, and ``load_state`` (the dump-shape entry used by hivemind merge) never decays.
The ``apply_decay=`` keyword is deprecated (removed in 1.5.0): it warns, and True/False decides that one load.
Every assertion this file changed at S1 names the owner decision it follows.
"""

from __future__ import annotations

import json

import pytest

from maxim.decisions.nac import _NAC_FORMAT_VERSION, NAc, NACConfig, Valence

DAY_S = 86400.0
WEEK_S = 604800.0


def _populated_nac(path: str) -> NAc:
    nac = NAc(NACConfig(persistence_path=path))
    nac.observe("tool_call", "tool_a", "result", "success", Valence.POSITIVE, 0.5, context={"agent_id": "a1"})
    with nac._lock:
        nac._reward_bias[("a1", "node_1")] = 0.2
        nac._goal_reward_bias["find warmth"] = 0.4
        nac._percept_valences[("a1", "dragon", "burn")] = -0.5
        nac._cluster_reward_bias[("a1", "cluster_1", "tool:warm")] = 0.8
    return nac


def _opted_in(path: str) -> NAc:
    """A loader on the Dormant opt-in (owner decision 2026-10-17, Q7: offline decay is opt-in only)."""
    return NAc(NACConfig(persistence_path=path, offline_wall_decay=True))


def _rewrite_saved_at(path: str, saved_at) -> None:
    with open(path) as f:
        state = json.load(f)
    if saved_at is None:
        state.pop("saved_at", None)
    else:
        state["saved_at"] = saved_at
    with open(path, "w") as f:
        json.dump(state, f)


class TestSavedAtStamp:
    def test_save_carries_wall_clock_saved_at_but_dump_does_not(self, tmp_path):
        """``saved_at`` is a persistence-envelope concern of the file
        writer — ``dump()`` stays a pure state surface (two dumps of
        identical state compare equal, per test_bio_system_snapshot.py)."""
        import time

        path = str(tmp_path / "nac.json")
        nac = NAc(NACConfig(persistence_path=path))
        assert "saved_at" not in nac.dump()
        before = time.time()
        nac.save()
        with open(path) as f:
            state = json.load(f)
        assert before <= state["saved_at"] <= time.time()

    def test_format_version_is_1_3(self, tmp_path):
        path = str(tmp_path / "nac.json")
        _populated_nac(path).save()
        with open(path) as f:
            assert json.load(f)["_format_version"] == _NAC_FORMAT_VERSION


class TestDefaultNoOfflineForgetting:
    """Owner decision 2026-10-17 (decision 2): no forgetting while the agent is switched off."""

    def test_six_month_gap_loads_every_map_verbatim_by_default(self, tmp_path):
        import time

        path = str(tmp_path / "nac.json")
        nac = _populated_nac(path)
        nac.record_cluster_fear("a1", "cluster_1", "drive:health", 0.8)
        assert nac._cluster_fear, "the fear write must land, or the fear half is vacuous"
        before = nac.dump()
        nac.save()
        _rewrite_saved_at(path, time.time() - 180 * DAY_S)

        nac2 = NAc(NACConfig(persistence_path=path))
        nac2.load()
        assert nac2.dump() == before
        nac3 = NAc(NACConfig(persistence_path=path))
        assert nac3.load_safe() == (True, None)
        assert nac3.dump() == before

    def test_default_load_emits_no_deprecation_warning(self, tmp_path):
        import warnings

        path = str(tmp_path / "nac.json")
        _populated_nac(path).save()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            NAc(NACConfig(persistence_path=path)).load()
            NAc(NACConfig(persistence_path=path)).load_safe()
        assert not [w for w in caught if issubclass(w.category, DeprecationWarning)]

    def test_orphan_source_sweep_still_runs_on_load_with_decay_off(self, tmp_path):
        """A2: a file can carry a cluster_reward_source whose bias is gone; the sweep is load_state's, not
        the decay's, so it runs with offline decay off (the default)."""
        path = str(tmp_path / "nac.json")
        _populated_nac(path).save()
        with open(path) as f:
            state = json.load(f)
        state["cluster_reward_source"] = {
            "a1\x1fcluster_1\x1ftool:warm": "operant",
            "a1\x1fgone\x1ftool:warm": "operant",  # no matching bias: an orphan
        }
        with open(path, "w") as f:
            json.dump(state, f)

        nac2 = NAc(NACConfig(persistence_path=path))
        nac2.load()
        assert nac2._cluster_reward_source == {("a1", "cluster_1", "tool:warm"): "operant"}


class TestOfflineDecayOptIn:
    """The Dormant opt-in (owner decision 2026-10-17, Q7): the 1.3-era behaviour, unchanged."""

    def test_one_day_gap_halves_cluster_bias_and_barely_touches_slow_surfaces(self, tmp_path):
        import time

        path = str(tmp_path / "nac.json")
        _populated_nac(path).save()
        _rewrite_saved_at(path, time.time() - DAY_S)

        nac2 = _opted_in(path)  # owner decision 2026-10-17 (Q7): decay is opt-in
        nac2.load()

        # cluster: half-life 1 day → 0.8 * 0.5 = 0.4
        assert nac2.cluster_reward_bias("a1", "cluster_1", "tool:warm") == pytest.approx(0.4, rel=0.02)
        # slow surfaces: half-life 7 days → factor 0.5^(1/7) ≈ 0.906
        slow = 0.5 ** (1.0 / 7.0)
        assert nac2._reward_bias[("a1", "node_1")] == pytest.approx(0.2 * slow, rel=0.02)
        assert nac2._goal_reward_bias["find warmth"] == pytest.approx(0.4 * slow, rel=0.02)
        assert nac2.get_percept_valence("dragon", "burn", agent_id="a1") == pytest.approx(-0.5 * slow, rel=0.02)

    def test_six_month_gap_prunes_biases_but_keeps_links(self, tmp_path):
        import time

        path = str(tmp_path / "nac.json")
        _populated_nac(path).save()
        _rewrite_saved_at(path, time.time() - 180 * DAY_S)

        nac2 = _opted_in(path)  # owner decision 2026-10-17 (Q7): decay is opt-in
        nac2.load()

        # A 180-day gap: cluster 0.8*0.5^180 → 0; slow 0.5^(180/7) ≈ 1.8e-8 → pruned
        assert nac2._cluster_reward_bias == {}
        assert nac2._reward_bias == {}
        assert nac2._goal_reward_bias == {}
        assert nac2._percept_valences == {}
        # Causal links are accumulated statistics — not activations — and
        # take the flat decay_all(0.95) haircut at session end instead.
        assert len(nac2.get_links_for_event("tool_a")) == 1

    def test_missing_saved_at_loads_undecayed(self, tmp_path):
        path = str(tmp_path / "nac.json")
        _populated_nac(path).save()
        _rewrite_saved_at(path, None)  # pre-1.3 payload shape

        nac2 = _opted_in(path)  # owner decision 2026-10-17 (Q7): decay is opt-in
        nac2.load()
        assert nac2.cluster_reward_bias("a1", "cluster_1", "tool:warm") == pytest.approx(0.8)
        assert nac2._reward_bias[("a1", "node_1")] == pytest.approx(0.2)

    def test_load_state_never_decays(self, tmp_path):
        """The dump-shape entry point (hivemind nac_merge, snapshot
        restores) must be byte-faithful — decay lives ONLY on the file
        entry point ``load()``."""
        import time

        path = str(tmp_path / "nac.json")
        nac = _populated_nac(path)
        state = nac.dump()
        state["saved_at"] = time.time() - 180 * DAY_S

        nac2 = NAc(NACConfig())
        nac2.load_state(state)
        assert nac2.cluster_reward_bias("a1", "cluster_1", "tool:warm") == pytest.approx(0.8)
        assert nac2._percept_valences[("a1", "dragon", "burn")] == pytest.approx(-0.5)

    def test_clock_skew_negative_elapsed_is_noop(self, tmp_path):
        import time

        path = str(tmp_path / "nac.json")
        _populated_nac(path).save()
        _rewrite_saved_at(path, time.time() + 3600.0)  # file "from the future"

        nac2 = _opted_in(path)  # owner decision 2026-10-17 (Q7): decay is opt-in
        nac2.load()
        assert nac2.cluster_reward_bias("a1", "cluster_1", "tool:warm") == pytest.approx(0.8)

    def test_boolean_saved_at_does_not_decay(self, tmp_path):
        """JSON ``true`` passes isinstance(x, (int, float)) and would read
        as epoch-second 1 → ~56 years elapsed → everything pruned from a
        corrupt-but-parseable file (review fold, Exec #6)."""
        path = str(tmp_path / "nac.json")
        _populated_nac(path).save()
        _rewrite_saved_at(path, True)

        nac2 = _opted_in(path)  # owner decision 2026-10-17 (Q7): decay is opt-in
        nac2.load()
        assert nac2.cluster_reward_bias("a1", "cluster_1", "tool:warm") == pytest.approx(0.8)

    def test_corrupt_string_bias_is_skipped_not_loaded(self, tmp_path):
        """A string-valued bias must not load 'successfully' and then
        poison every arithmetic consumer (decay-on-load, the per-tick
        agent_loop §8.5 decay block) with a TypeError (review fold,
        Exec #1). float-coercible strings coerce; garbage is skipped."""
        path = str(tmp_path / "nac.json")
        _populated_nac(path).save()
        with open(path) as f:
            state = json.load(f)
        state["reward_bias"]["a1:node_1"] = "not-a-number"
        state["cluster_reward_bias"]["a1\x1fcluster_1\x1ftool:warm"] = None
        with open(path, "w") as f:
            json.dump(state, f)

        nac2 = NAc(NACConfig(persistence_path=path))
        ok, err = nac2.load_safe()
        assert ok, err
        assert ("a1", "node_1") not in nac2._reward_bias
        assert nac2.cluster_reward_bias("a1", "cluster_1", "tool:warm") == 0.0
        # The rest of the file still loads.
        assert len(nac2.get_links_for_event("tool_a")) == 1
        assert nac2._percept_valences[("a1", "dragon", "burn")] == pytest.approx(-0.5)

    def test_load_safe_recovery_resets_all_bias_surfaces(self, tmp_path):
        """A corrupt file must not leave half-loaded bias surfaces behind
        — 'starting with empty causal model' must be literally true
        (review fold, Exec #1 + Arch #5)."""
        path = str(tmp_path / "nac.json")
        with open(path, "w") as f:
            f.write('{"links": "not-a-dict"}')

        nac = _populated_nac(str(tmp_path / "other.json"))  # pre-populated in-memory state
        ok, err = nac.load_safe(path)
        assert not ok
        assert nac._links == {}
        assert nac._reward_bias == {}
        assert nac._goal_reward_bias == {}
        assert nac._cluster_reward_bias == {}
        assert nac._percept_valences == {}
        assert nac._event_outcome_welford == {}

    def test_apply_wall_clock_decay_reports_pruned_counts(self):
        nac = NAc(NACConfig())
        with nac._lock:
            nac._cluster_reward_bias[("a1", "c1", "t1")] = 0.8
            nac._reward_bias[("a1", "n1")] = 0.2
        results = nac.apply_wall_clock_decay(180 * DAY_S)
        assert results == {
            "reward_bias_pruned": 1,
            "cluster_reward_bias_pruned": 1,
        }


class TestApplyDecayDeprecated:
    """``apply_decay=`` keeps working through 1.4.x and is removed in 1.5.0 (owner decision 2026-10-17, Q7)."""

    def _old_file(self, tmp_path) -> str:
        import time

        path = str(tmp_path / "nac.json")
        _populated_nac(path).save()
        _rewrite_saved_at(path, time.time() - 180 * DAY_S)
        return path

    @pytest.mark.single_agent_only
    def test_apply_decay_false_warns_and_switches_the_opt_in_off(self, tmp_path):
        """The --resume-sim / read-only-observer opt-out still loads disk truth, now with a warning."""
        path = self._old_file(tmp_path)
        nac2 = _opted_in(path)
        with pytest.warns(DeprecationWarning, match=r"removed in 1\.5\.0"):
            nac2.load(apply_decay=False)
        assert nac2.cluster_reward_bias("a1", "cluster_1", "tool:warm") == pytest.approx(0.8)
        assert nac2._reward_bias[("a1", "node_1")] == pytest.approx(0.2)

    @pytest.mark.single_agent_only
    def test_apply_decay_true_decays_that_one_load_on_a_default_config(self, tmp_path):
        """Owner decision 2026-10-18: during the 1.4.x window apply_decay=True enables offline decay for that one
        load, whatever the config; the next keyword-free load of the same file is verbatim again."""
        path = self._old_file(tmp_path)
        nac2 = NAc(NACConfig(persistence_path=path))
        with pytest.warns(DeprecationWarning):
            nac2.load(apply_decay=True)
        assert nac2._cluster_reward_bias == {}
        assert nac2._reward_bias == {}
        nac3 = NAc(NACConfig(persistence_path=path))
        nac3.load()
        assert nac3.cluster_reward_bias("a1", "cluster_1", "tool:warm") == pytest.approx(0.8)

    def test_apply_decay_false_on_a_default_config_loads_verbatim(self, tmp_path):
        path = self._old_file(tmp_path)
        nac2 = NAc(NACConfig(persistence_path=path))
        with pytest.warns(DeprecationWarning):
            nac2.load(apply_decay=False)
        assert nac2.cluster_reward_bias("a1", "cluster_1", "tool:warm") == pytest.approx(0.8)

    @pytest.mark.single_agent_only
    def test_apply_decay_true_under_the_opt_in_still_decays(self, tmp_path):
        path = self._old_file(tmp_path)
        nac2 = _opted_in(path)
        with pytest.warns(DeprecationWarning):
            nac2.load(apply_decay=True)
        assert nac2._cluster_reward_bias == {}

    @pytest.mark.single_agent_only
    def test_load_safe_warns_and_passes_the_switch_through(self, tmp_path):
        path = self._old_file(tmp_path)
        nac2 = _opted_in(path)
        with pytest.warns(DeprecationWarning, match="load_safe"):
            assert nac2.load_safe(apply_decay=False) == (True, None)
        assert nac2.cluster_reward_bias("a1", "cluster_1", "tool:warm") == pytest.approx(0.8)
        nac3 = NAc(NACConfig(persistence_path=path))  # default config: True still decays (owner decision 2026-10-18)
        with pytest.warns(DeprecationWarning):
            assert nac3.load_safe(apply_decay=True) == (True, None)
        assert nac3._cluster_reward_bias == {}
        with pytest.warns(DeprecationWarning):  # warns even when there is no file to load
            assert NAc(NACConfig()).load_safe(str(tmp_path / "missing.json"), apply_decay=False) == (True, None)

    @pytest.mark.parametrize("version", ["1.5.0", "1.5.0rc1", "2.0.0"])
    def test_the_keyword_is_gone_from_1_5(self, tmp_path, monkeypatch, version):
        """The removal is mechanized: the release that bumps to 1.5 finds these calls raising."""
        import maxim

        path = self._old_file(tmp_path)
        monkeypatch.setattr(maxim, "__version__", version)
        with pytest.raises(TypeError, match=r"removed in 1\.5\.0"):
            NAc(NACConfig(persistence_path=path)).load(apply_decay=False)
        with pytest.raises(TypeError, match="load_safe"):
            NAc(NACConfig(persistence_path=path)).load_safe(apply_decay=False)
        NAc(NACConfig(persistence_path=path)).load()  # the keyword-free call is unaffected


def _resume_dir_with_a_saved_nac(home, name: str) -> None:
    d = home / "sim_reports" / name
    d.mkdir(parents=True)
    (d / "report.json").write_text('{"goal": "g"}')
    _populated_nac(str(d / "aut_nac.json")).save()


@pytest.mark.single_agent_only
def test_the_resume_restore_loads_a_real_nac_today(tmp_path, monkeypatch):
    """The control arm for the 1.5 gate below: the same call, at today's version, loads."""
    from maxim.simulation.orchestrator import _restore_aut_from_session

    monkeypatch.setenv("MAXIM_DATA_HOME", str(tmp_path))
    _resume_dir_with_a_saved_nac(tmp_path, "prior")
    nac = NAc(NACConfig())
    record = _restore_aut_from_session(
        "prior", persistent_agent=None, aut_hippocampus=None, aut_nac=nac, aut_memory_hub=None
    )
    assert record["stores"]["nac"] == "loaded"
    assert nac.cluster_reward_bias("a1", "cluster_1", "tool:warm") == pytest.approx(0.8)


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="flips at S3a, which removes apply_decay= from orchestrator._restore_aut_from_session (fenced); until "
    "then a 1.5 version makes NAc.load raise TypeError there, _restore catches it at DEBUG as failed:TypeError, "
    "and a --resume-sim silently starts with an empty NAc",
)
@pytest.mark.single_agent_only
def test_the_resume_restore_still_loads_the_nac_at_1_5(tmp_path, monkeypatch):
    import maxim
    from maxim.simulation.orchestrator import _restore_aut_from_session

    monkeypatch.setenv("MAXIM_DATA_HOME", str(tmp_path))
    _resume_dir_with_a_saved_nac(tmp_path, "prior")
    monkeypatch.setattr(maxim, "__version__", "1.5.0")
    nac = NAc(NACConfig())
    record = _restore_aut_from_session(
        "prior", persistent_agent=None, aut_hippocampus=None, aut_nac=nac, aut_memory_hub=None
    )
    assert record["stores"]["nac"] == "loaded", record["stores"]
