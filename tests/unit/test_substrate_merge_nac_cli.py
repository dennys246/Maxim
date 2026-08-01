"""Stage-4b regression guards: ``maxim substrate merge-nac``.

live_audio_orient_wiring.md Stage 4b (decision #3: ONE-SHOT CLI VERB).
The import is a MERGE via ``hivemind.nac_merge`` — never a
``load_safe`` replace, which would clobber the runtime NAc's other
learning. Pins:

- round-trip: merging a trained orient policy into a populated runtime
  nac.json preserves BOTH the orient cluster biases and the pre-existing
  learning, and the per-bin argmax matches the source policy;
- the policy-meta sidecar travels with the import;
- a DISAGREEING target sidecar aborts before any mutation;
- pre-merge backup of an existing target;
- reserved ``_``-prefixed contributor IDs are rejected.
"""

from __future__ import annotations

import json

import pytest

from maxim.decisions.nac import NAc, NACConfig
from maxim.hivemind.cli import run_substrate_subcommand

# The trained orient policy shape (Exp 45c): cluster_reward_bias keyed on
# (agent_id, az-bin, tool_signature).
_ORIENT_BIASES = {
    ("reachy", "near_left", "turn_left"): 0.8,
    ("reachy", "near_left", "turn_left_big"): 0.2,
    ("reachy", "far_left", "turn_left_big"): 0.9,
    ("reachy", "far_left", "turn_left"): 0.3,
}

_POLICY_META = {"bin_boundary": 0.328, "gain": 0.57, "action_deltas": {"turn_left": 0.3, "turn_left_big": 0.9}}


def _save_policy(path):
    nac = NAc(config=NACConfig())
    for key, bias in _ORIENT_BIASES.items():
        nac._cluster_reward_bias[key] = bias
    nac.save(str(path))


def _save_runtime(path):
    nac = NAc(config=NACConfig())
    nac._reward_bias[("reachy", "node-preexisting")] = 0.12
    nac._cluster_reward_bias[("reachy", "warm-cluster", "seek_warmth")] = 0.5
    nac.save(str(path))


def _load_merged(path):
    state = json.loads(path.read_text(encoding="utf-8"))
    state.pop("_format_version", None)
    out = NAc(config=NACConfig())
    out.load_state(state)
    return out


def _meta_path(path):
    return path.with_name(path.name[: -len(".json")] + ".meta.json")


class TestMergeNacRoundTrip:
    def test_orient_policy_and_preexisting_learning_both_survive(self, tmp_path):
        src = tmp_path / "nac_reachy_flip.json"
        tgt = tmp_path / "nac.json"
        _save_policy(src)
        _save_runtime(tgt)

        rc = run_substrate_subcommand(["merge-nac", str(src), "--into", str(tgt), "--source-id", "reachy-orient-45c"])
        assert rc == 0

        merged = _load_merged(tgt)
        # Pre-existing runtime learning survives (a load_safe replace would
        # have clobbered these — the exact failure the MERGE decision avoids).
        assert merged._reward_bias[("reachy", "node-preexisting")] == pytest.approx(0.12)
        assert merged._cluster_reward_bias[("reachy", "warm-cluster", "seek_warmth")] == pytest.approx(0.5)
        # Orient policy arrived intact (no matching target keys → passthrough).
        for key, bias in _ORIENT_BIASES.items():
            assert merged._cluster_reward_bias[key] == pytest.approx(bias), key

    def test_per_bin_argmax_matches_source_policy(self, tmp_path):
        src = tmp_path / "policy.json"
        tgt = tmp_path / "nac.json"
        _save_policy(src)
        _save_runtime(tgt)
        assert run_substrate_subcommand(["merge-nac", str(src), "--into", str(tgt), "--source-id", "p"]) == 0
        merged = _load_merged(tgt)

        def argmax_action(nac, az_bin):
            candidates = {tool: b for (agent, b_bin, tool), b in nac._cluster_reward_bias.items() if b_bin == az_bin}
            return max(candidates, key=candidates.get)

        assert argmax_action(merged, "near_left") == "turn_left"
        assert argmax_action(merged, "far_left") == "turn_left_big"

    def test_merge_into_absent_target_creates_it(self, tmp_path):
        src = tmp_path / "policy.json"
        tgt = tmp_path / "fresh" / "nac.json"
        _save_policy(src)
        rc = run_substrate_subcommand(["merge-nac", str(src), "--into", str(tgt), "--source-id", "p"])
        assert rc == 0
        merged = _load_merged(tgt)
        for key, bias in _ORIENT_BIASES.items():
            assert merged._cluster_reward_bias[key] == pytest.approx(bias)
        # No backup for a target that didn't exist.
        assert not (tgt.parent / "nac.json.pre-merge.bak").exists()

    def test_format_version_restamped_on_output(self, tmp_path):
        from maxim.decisions.nac import _NAC_FORMAT_VERSION

        src = tmp_path / "policy.json"
        tgt = tmp_path / "nac.json"
        _save_policy(src)
        assert run_substrate_subcommand(["merge-nac", str(src), "--into", str(tgt), "--source-id", "p"]) == 0
        data = json.loads(tgt.read_text(encoding="utf-8"))
        assert data["_format_version"] == _NAC_FORMAT_VERSION

    def test_backup_written_when_target_existed(self, tmp_path):
        src = tmp_path / "policy.json"
        tgt = tmp_path / "nac.json"
        _save_policy(src)
        _save_runtime(tgt)
        original = tgt.read_text(encoding="utf-8")
        assert run_substrate_subcommand(["merge-nac", str(src), "--into", str(tgt), "--source-id", "p"]) == 0
        backup = tmp_path / "nac.json.pre-merge.bak"
        assert backup.is_file()
        assert backup.read_text(encoding="utf-8") == original


class TestPolicyMetaSidecar:
    def test_sidecar_travels_with_the_import(self, tmp_path):
        src = tmp_path / "policy.json"
        tgt = tmp_path / "nac.json"
        _save_policy(src)
        _meta_path(src).write_text(json.dumps(_POLICY_META), encoding="utf-8")
        assert run_substrate_subcommand(["merge-nac", str(src), "--into", str(tgt), "--source-id", "p"]) == 0
        copied = json.loads(_meta_path(tgt).read_text(encoding="utf-8"))
        # atomic_write_json may add the CC1 stamp; the policy fields must survive.
        for k, v in _POLICY_META.items():
            assert copied[k] == v

    def test_disagreeing_target_sidecar_aborts_before_mutation(self, tmp_path):
        src = tmp_path / "policy.json"
        tgt = tmp_path / "nac.json"
        _save_policy(src)
        _save_runtime(tgt)
        _meta_path(src).write_text(json.dumps(_POLICY_META), encoding="utf-8")
        _meta_path(tgt).write_text(json.dumps({**_POLICY_META, "bin_boundary": 0.5}), encoding="utf-8")
        before = tgt.read_text(encoding="utf-8")

        rc = run_substrate_subcommand(["merge-nac", str(src), "--into", str(tgt), "--source-id", "p"])
        assert rc == 2
        assert tgt.read_text(encoding="utf-8") == before, "target mutated despite state-space mismatch"
        assert not (tmp_path / "nac.json.pre-merge.bak").exists()

    def test_matching_sidecars_proceed(self, tmp_path):
        src = tmp_path / "policy.json"
        tgt = tmp_path / "nac.json"
        _save_policy(src)
        _save_runtime(tgt)
        _meta_path(src).write_text(json.dumps(_POLICY_META), encoding="utf-8")
        _meta_path(tgt).write_text(json.dumps(_POLICY_META), encoding="utf-8")
        assert run_substrate_subcommand(["merge-nac", str(src), "--into", str(tgt), "--source-id", "p"]) == 0


class TestArgumentValidation:
    def test_reserved_source_id_rejected(self, tmp_path):
        src = tmp_path / "policy.json"
        tgt = tmp_path / "nac.json"
        _save_policy(src)
        rc = run_substrate_subcommand(["merge-nac", str(src), "--into", str(tgt), "--source-id", "_consensus"])
        assert rc == 2
        assert not tgt.exists()

    def test_missing_source_file_is_a_clean_error(self, tmp_path):
        rc = run_substrate_subcommand(
            ["merge-nac", str(tmp_path / "nope.json"), "--into", str(tmp_path / "nac.json"), "--source-id", "p"]
        )
        assert rc == 2


class TestFoldRobustness:
    """Pre-merge review fold: corrupt inputs get the clean rc=2 contract
    (never a traceback), the decay clock survives the merge, and the
    copied sidecar is CC1-stamped without breaking the equality gate."""

    def test_corrupt_target_json_is_a_clean_error(self, tmp_path):
        src = tmp_path / "policy.json"
        tgt = tmp_path / "nac.json"
        _save_policy(src)
        tgt.write_text("{truncated", encoding="utf-8")
        rc = run_substrate_subcommand(["merge-nac", str(src), "--into", str(tgt), "--source-id", "p"])
        assert rc == 2
        assert tgt.read_text(encoding="utf-8") == "{truncated"  # untouched

    def test_list_rooted_target_is_a_clean_error(self, tmp_path):
        src = tmp_path / "policy.json"
        tgt = tmp_path / "nac.json"
        _save_policy(src)
        tgt.write_text("[1, 2, 3]", encoding="utf-8")
        rc = run_substrate_subcommand(["merge-nac", str(src), "--into", str(tgt), "--source-id", "p"])
        assert rc == 2

    def test_corrupt_sidecar_is_a_clean_error(self, tmp_path):
        src = tmp_path / "policy.json"
        tgt = tmp_path / "nac.json"
        _save_policy(src)
        _save_runtime(tgt)
        before = tgt.read_text(encoding="utf-8")
        _meta_path(src).write_text("not json at all", encoding="utf-8")
        rc = run_substrate_subcommand(["merge-nac", str(src), "--into", str(tgt), "--source-id", "p"])
        assert rc == 2
        assert tgt.read_text(encoding="utf-8") == before

    def test_saved_at_decay_clock_survives_the_merge(self, tmp_path):
        """nac_merge drops saved_at; the verb must restore it or the next
        boot's load_safe(apply_decay=True) silently skips one whole
        wall-clock decay cycle for ALL runtime biases."""
        src = tmp_path / "policy.json"
        tgt = tmp_path / "nac.json"
        _save_policy(src)
        _save_runtime(tgt)
        tgt_saved_at = json.loads(tgt.read_text(encoding="utf-8")).get("saved_at")
        assert tgt_saved_at, "fixture assumption: NAc.save stamps saved_at"
        assert run_substrate_subcommand(["merge-nac", str(src), "--into", str(tgt), "--source-id", "p"]) == 0
        merged = json.loads(tgt.read_text(encoding="utf-8"))
        # The TARGET's clock wins — it times the pre-existing state's decay.
        assert merged.get("saved_at") == tgt_saved_at

    def test_copied_sidecar_is_format_version_stamped(self, tmp_path):
        src = tmp_path / "policy.json"
        tgt = tmp_path / "nac.json"
        _save_policy(src)
        _meta_path(src).write_text(json.dumps(_POLICY_META), encoding="utf-8")
        assert run_substrate_subcommand(["merge-nac", str(src), "--into", str(tgt), "--source-id", "p"]) == 0
        copied = json.loads(_meta_path(tgt).read_text(encoding="utf-8"))
        assert "_format_version" in copied  # CC1: every persisted JSON is stamped

    def test_stamped_target_sidecar_vs_unstamped_source_still_matches(self, tmp_path):
        """The equality gate compares stamp-stripped essence — stamping
        history alone must never abort a legitimate import."""
        src = tmp_path / "policy.json"
        tgt = tmp_path / "nac.json"
        _save_policy(src)
        _save_runtime(tgt)
        _meta_path(src).write_text(json.dumps(_POLICY_META), encoding="utf-8")
        _meta_path(tgt).write_text(json.dumps({**_POLICY_META, "_format_version": "1.0"}), encoding="utf-8")
        assert run_substrate_subcommand(["merge-nac", str(src), "--into", str(tgt), "--source-id", "p"]) == 0
