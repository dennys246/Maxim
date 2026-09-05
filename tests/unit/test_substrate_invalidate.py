"""Gate 1's migrate half — invalidate stale-geometry EC nodes + prune their biases.

1.1.3 shipped the reject half (geometry-mismatched nodes are masked out of
scans and warned about once, D1); nothing invalidated them. EC stores
centroids, not raw readings, so migration for sensor substrate is loud
invalidation: nodes removed by NAMED stale geometry, the NAc biases keyed on
the removed cluster ids pruned in the same operation (else they are minted
as permanently dangling — the D2 shape), everything removed recorded in a
tombstone sidecar. 1.1.4's A4 moved the world geometry tag, so the case is
live, not hypothetical.

Callers: `maxim substrate invalidate` (hivemind/cli.py::_run_invalidate).
"""

from __future__ import annotations

import json
from pathlib import Path

from maxim.hivemind.cli import run_substrate_subcommand
from maxim.hivemind.merge import (
    NAC_KEY_SEP,
    invalidate_stale_geometry_nodes,
    prune_nac_cluster_biases,
)

OLD_GEOM = "world:v1:aaaa"
NEW_GEOM = "world:v2:bbbb"


def _nodes() -> dict[str, dict]:
    return {
        "stale-1": {"embedding": [1.0, 0.0], "modality": "world", "geometry": OLD_GEOM},
        "stale-2": {"embedding": [0.0, 1.0], "modality": "world", "geometry": OLD_GEOM},
        "live-1": {"embedding": [0.5, 0.5], "modality": "world", "geometry": NEW_GEOM},
        "unstamped-1": {"embedding": [0.2, 0.8], "modality": "world", "geometry": None},
        "other-mod": {"embedding": [0.9, 0.1], "modality": "audio", "geometry": OLD_GEOM},
    }


class TestInvalidateStaleGeometryNodes:
    def test_drops_only_the_named_geometry_in_the_named_modality(self):
        kept, removed = invalidate_stale_geometry_nodes(_nodes(), modality="world", drop_geometry=OLD_GEOM)
        assert sorted(removed) == ["stale-1", "stale-2"]
        assert sorted(kept) == ["live-1", "other-mod", "unstamped-1"]

    def test_unstamped_nodes_are_never_touched(self):
        """geometry: None is the permissive class (D66) — not deletable by accident."""
        kept, removed = invalidate_stale_geometry_nodes(_nodes(), modality="world", drop_geometry="None")
        assert removed == []
        assert len(kept) == 5

    def test_pure_input_not_mutated(self):
        nodes = _nodes()
        invalidate_stale_geometry_nodes(nodes, modality="world", drop_geometry=OLD_GEOM)
        assert len(nodes) == 5


class TestPruneNacClusterBiases:
    def _state(self) -> dict:
        j = NAC_KEY_SEP.join
        return {
            "cluster_reward_bias": {
                j(("a1", "stale-1", "tool:x_turn")): 0.4,
                j(("a1", "live-1", "tool:x_turn")): 0.2,
            },
            "cluster_reward_source": {
                j(("a1", "stale-1", "tool:x_turn")): "drive:hunger",
            },
            "reward_bias": {"a1:stale-2": 0.3, "a1:live-1": 0.1},
            "links": {"untouched": []},
        }

    def test_prunes_all_three_surfaces(self):
        new, pruned = prune_nac_cluster_biases(self._state(), {"stale-1", "stale-2"})
        assert pruned == 3
        assert list(new["cluster_reward_bias"]) == [NAC_KEY_SEP.join(("a1", "live-1", "tool:x_turn"))]
        assert new["cluster_reward_source"] == {}
        assert new["reward_bias"] == {"a1:live-1": 0.1}
        assert new["links"] == {"untouched": []}

    def test_pure_input_not_mutated(self):
        state = self._state()
        prune_nac_cluster_biases(state, {"stale-1", "stale-2"})
        assert len(state["cluster_reward_bias"]) == 2

    def test_malformed_keys_are_kept_not_validated(self):
        new, pruned = prune_nac_cluster_biases({"cluster_reward_bias": {"weird-key": 1.0}}, {"stale-1"})
        assert pruned == 0
        assert new["cluster_reward_bias"] == {"weird-key": 1.0}


class TestInvalidateCli:
    def _session(self, tmp_path: Path) -> Path:
        session = tmp_path / "session"
        session.mkdir()
        (session / "aut_ec.json").write_text(
            json.dumps({"_format_version": "1.0", "substrate_nodes": _nodes(), "signatures": {"keep": "me"}})
        )
        j = NAC_KEY_SEP.join
        (session / "aut_nac.json").write_text(
            json.dumps(
                {
                    "_format_version": "1.2",
                    "cluster_reward_bias": {
                        j(("a1", "stale-1", "tool:x_turn")): 0.4,
                        j(("a1", "live-1", "tool:x_turn")): 0.2,
                    },
                    "reward_bias": {"a1:stale-2": 0.3},
                }
            )
        )
        return session

    def test_census_when_no_geometry_named(self, tmp_path, capsys):
        session = self._session(tmp_path)
        rc = run_substrate_subcommand(["invalidate", "--session", str(session)])
        assert rc == 0
        out = capsys.readouterr().out
        assert OLD_GEOM in out and NEW_GEOM in out and "(unstamped)" in out

    def test_dry_run_by_default_writes_nothing(self, tmp_path, capsys):
        session = self._session(tmp_path)
        before = (session / "aut_ec.json").read_text()
        rc = run_substrate_subcommand(
            ["invalidate", "--session", str(session), "--modality", "world", "--drop-geometry", OLD_GEOM]
        )
        assert rc == 0
        assert "DRY RUN" in capsys.readouterr().out
        assert (session / "aut_ec.json").read_text() == before
        assert not list(session.glob("aut_ec.invalidated.*.json"))

    def test_apply_removes_prunes_and_tombstones(self, tmp_path):
        session = self._session(tmp_path)
        rc = run_substrate_subcommand(
            [
                "invalidate",
                "--session",
                str(session),
                "--modality",
                "world",
                "--drop-geometry",
                OLD_GEOM,
                "--apply",
            ]
        )
        assert rc == 0
        ec = json.loads((session / "aut_ec.json").read_text())
        assert sorted(ec["substrate_nodes"]) == ["live-1", "other-mod", "unstamped-1"]
        assert ec["signatures"] == {"keep": "me"}  # the rest of the payload survives
        nac = json.loads((session / "aut_nac.json").read_text())
        assert list(nac["cluster_reward_bias"]) == [NAC_KEY_SEP.join(("a1", "live-1", "tool:x_turn"))]
        assert nac["reward_bias"] == {}
        tombstones = list(session.glob("aut_ec.invalidated.*.json"))
        assert len(tombstones) == 1
        ts = json.loads(tombstones[0].read_text())
        # everything removed is recorded VERBATIM — auditable and hand-reversible
        assert sorted(ts["removed_nodes"]) == ["stale-1", "stale-2"]
        assert ts["removed_nodes"]["stale-1"]["embedding"] == [1.0, 0.0]
        assert ts["pruned_nac_entries"]["reward_bias"] == {"a1:stale-2": 0.3}
        assert ts["drop_geometry"] == OLD_GEOM

    def test_apply_requires_the_full_target(self, tmp_path):
        session = self._session(tmp_path)
        assert run_substrate_subcommand(["invalidate", "--session", str(session), "--apply"]) == 2
        assert (
            run_substrate_subcommand(["invalidate", "--session", str(session), "--drop-geometry", OLD_GEOM]) == 2
        )  # --drop-geometry requires --modality

    def test_nothing_to_invalidate_is_a_clean_noop(self, tmp_path, capsys):
        session = self._session(tmp_path)
        rc = run_substrate_subcommand(
            [
                "invalidate",
                "--session",
                str(session),
                "--modality",
                "world",
                "--drop-geometry",
                "no-such-geom",
                "--apply",
            ]
        )
        assert rc == 0
        assert "nothing to invalidate" in capsys.readouterr().out
        assert not list(session.glob("aut_ec.invalidated.*.json"))


class TestInvalidateCliRound2Folds:
    """Round-2 review folds: rc=2 on malformed persisted data, census
    modality filter (no silently ignored flags), dry-run leaves NAc alone."""

    def _session(self, tmp_path: Path) -> Path:
        session = tmp_path / "session"
        session.mkdir()
        (session / "aut_ec.json").write_text(json.dumps({"_format_version": "1.0", "substrate_nodes": _nodes()}))
        (session / "aut_nac.json").write_text(
            json.dumps({"_format_version": "1.2", "reward_bias": {"a1:stale-2": 0.3}})
        )
        return session

    def test_malformed_ec_json_reports_not_tracebacks(self, tmp_path):
        session = self._session(tmp_path)
        (session / "aut_ec.json").write_text("{not json")
        rc = run_substrate_subcommand(["invalidate", "--session", str(session)])
        assert rc == 2

    def test_malformed_nac_json_reports_not_tracebacks(self, tmp_path):
        session = self._session(tmp_path)
        (session / "aut_nac.json").write_text("{not json")
        rc = run_substrate_subcommand(
            ["invalidate", "--session", str(session), "--modality", "world", "--drop-geometry", OLD_GEOM]
        )
        assert rc == 2

    def test_census_is_filtered_by_modality_and_counts_are_exact(self, tmp_path, capsys):
        session = self._session(tmp_path)
        rc = run_substrate_subcommand(["invalidate", "--session", str(session), "--modality", "audio"])
        assert rc == 0
        out = capsys.readouterr().out
        assert f"audio        {OLD_GEOM}: 1" in out
        # no world-MODALITY rows in an audio-scoped census (the geometry
        # STRINGS legitimately contain "world", so mask those first)
        assert "world" not in out.replace("world:v", "G:v")
        assert NEW_GEOM not in out  # NEW_GEOM only exists on world-modality nodes

    def test_census_survives_non_string_geometry(self, tmp_path, capsys):
        session = self._session(tmp_path)
        nodes = _nodes()
        nodes["weird"] = {"embedding": [0.1, 0.9], "modality": "world", "geometry": 7}
        nodes["broken"] = "not-a-dict"
        (session / "aut_ec.json").write_text(json.dumps({"_format_version": "1.0", "substrate_nodes": nodes}))
        rc = run_substrate_subcommand(["invalidate", "--session", str(session)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "7: 1" in out
        assert "malformed non-dict node entry skipped" in out

    def test_dry_run_leaves_nac_untouched_too(self, tmp_path):
        session = self._session(tmp_path)
        before = (session / "aut_nac.json").read_text()
        rc = run_substrate_subcommand(
            ["invalidate", "--session", str(session), "--modality", "world", "--drop-geometry", OLD_GEOM]
        )
        assert rc == 0
        assert (session / "aut_nac.json").read_text() == before
