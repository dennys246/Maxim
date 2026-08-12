"""Guards for the S4 non-stationarity analyzer (scripts/exp44/analyze_nonstationarity.py).

The load-bearing test is the ROUND-TRIP: the parser is fed output from the REAL
production renderer (``prompts/cluster_bias_annotation.py::compose_cluster_bias_
annotation_section``), not a hand-written imitation. If the rendering contract
changes — column padding, band phrasing, the "from prior experience" suffix — this
fails here instead of silently reporting "no annotation" across a whole campaign
(the same failure class as the pilot's has_cluster_bias nesting bug).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from maxim.prompts.cluster_bias_annotation import compose_cluster_bias_annotation_section

REPO = Path(__file__).resolve().parents[2]

_spec = importlib.util.spec_from_file_location("_exp44_s4", REPO / "scripts/exp44/analyze_nonstationarity.py")
s4 = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(s4)


class TestParserRoundTripsProductionRenderer:
    def test_every_band_round_trips(self):
        """One tool per band, rendered by production, parsed back exactly."""
        biases = [
            ("tool:strong_pos", 0.9),  # strongly rewarding
            ("tool:mild_pos", 0.2),  # mildly rewarding
            ("tool:neutralish", 0.0),  # neutral / mixed  (no suffix)
            ("tool:mild_neg", -0.2),  # mildly aversive
            ("tool:strong_neg", -0.9),  # strongly aversive
        ]
        section = compose_cluster_bias_annotation_section(biases)
        parsed = s4.parse_annotation(f"PREAMBLE\n\n{section}\n\nTRAILING PROMPT TEXT\n")
        assert parsed is not None
        assert parsed["strong_pos"] == "strongly rewarding"
        assert parsed["mild_pos"] == "mildly rewarding"
        assert parsed["neutralish"] == "neutral / mixed"
        assert parsed["mild_neg"] == "mildly aversive"
        assert parsed["strong_neg"] == "strongly aversive"
        assert all(s4.band_tier(b) is not None for b in parsed.values())

    def test_tiers_are_ordered(self):
        assert s4.band_tier("strongly rewarding") > s4.band_tier("mildly rewarding")
        assert s4.band_tier("mildly rewarding") > s4.band_tier("neutral / mixed")
        assert s4.band_tier("neutral / mixed") > s4.band_tier("mildly aversive")
        assert s4.band_tier("mildly aversive") > s4.band_tier("strongly aversive")

    def test_column_padding_does_not_break_parsing(self):
        """Production pads names to a common width; a very long name changes the
        padding of every other row."""
        section = compose_cluster_bias_annotation_section(
            [("tool:a", 0.9), ("tool:an_extremely_long_affordance_name_here", 0.6)]
        )
        parsed = s4.parse_annotation(section)
        assert parsed == {
            "a": "strongly rewarding",
            "an_extremely_long_affordance_name_here": "strongly rewarding",
        }

    def test_s1_source_gloss_round_trips(self):
        """S1 renderer: the credit-source gloss after the em-dash must be
        stripped by the parser — every band recovers exactly, glossed or
        not. This is the guard that keeps the S1 format change from
        silently reading as 'no annotation' across a campaign."""
        biases = [
            ("tool:green_flame_warm_self", 0.9),
            ("tool:purple_flame_observe", 0.3),
            ("tool:unglossed", 0.7),
            ("tool:risky", -0.9),
            ("tool:meh", 0.0),
        ]
        sources = {
            "tool:green_flame_warm_self": "drive_relief",
            "tool:purple_flame_observe": "tool_success",
            "tool:risky": "operant",
        }
        section = compose_cluster_bias_annotation_section(biases, sources)
        # Sanity: the gloss is actually present in the rendered text.
        assert "— relieved a bodily need]" in section
        parsed = s4.parse_annotation(f"PREAMBLE\n\n{section}\n\nTRAILING\n")
        assert parsed == {
            "green_flame_warm_self": "strongly rewarding",
            "purple_flame_observe": "mildly rewarding",
            "unglossed": "strongly rewarding",
            "risky": "strongly aversive",
            "meh": "neutral / mixed",
        }
        assert all(s4.band_tier(b) is not None for b in parsed.values())

    def test_parser_separator_matches_composer_constant(self):
        """The parser's separator (import-with-fallback) must equal the
        composer's shared constant — two literals agreeing by luck is
        exactly what this pins."""
        from maxim.prompts.cluster_bias_annotation import ANNOTATION_SOURCE_SEPARATOR

        assert s4._SOURCE_SEP == ANNOTATION_SOURCE_SEPARATOR

    def test_absent_annotation_returns_none(self):
        assert s4.parse_annotation("a prompt with no substrate section at all") is None

    def test_block_ends_at_following_prose(self):
        section = compose_cluster_bias_annotation_section([("tool:x", 0.9)])
        parsed = s4.parse_annotation(f"{section}\nSome following instruction line.\n  indented prose\n")
        assert parsed == {"x": "strongly rewarding"}


class TestEndToEnd:
    def _capture(self, tmp_path: Path, tiers: list[float]) -> Path:
        """A capture file whose annotation decays across decisions."""
        p = tmp_path / "capture.jsonl"
        lines = []
        for i, bias in enumerate(tiers):
            section = compose_cluster_bias_annotation_section([("tool:green_flame_warm_self", bias)])
            lines.append(
                json.dumps(
                    {
                        "decision_id": i,
                        "world_state": {"has_cluster_bias": True},
                        "prompt_full": f"stuff\n{section}\nmore",
                        "prompt_ablated": "stuff\nmore",
                    }
                )
            )
        p.write_text("\n".join(lines))
        return p

    def test_detects_band_decay_across_run(self, tmp_path, capsys):
        """Strong -> mild: both bands render, so tier drift is measurable."""
        cap = self._capture(tmp_path, [0.9] * 4 + [0.2] * 4)
        import sys

        sys.argv = ["s4", "--capture", str(cap), "--tools", "green_flame_warm_self", "--json", str(tmp_path / "r.json")]
        assert s4.main() == 0
        rep = json.loads((tmp_path / "r.json").read_text())["tools"]["green_flame_warm_self"]
        assert rep["mean_tier_first_half"] == 2.0  # strongly rewarding
        assert rep["mean_tier_second_half"] == 1.0  # mildly rewarding
        assert rep["tier_drift"] == -1.0  # decayed
        assert "drift" in capsys.readouterr().out

    def test_detects_annotation_vanishing(self, tmp_path, capsys):
        """The SHARP form of F6: production suppresses the whole section once every
        bias decays into the neutral band, so late decisions are UNTREATED, not
        weakly treated. Discovered by round-tripping the real renderer — a
        hand-written fixture would have asserted 'neutral' and missed it."""
        cap = self._capture(tmp_path, [0.9] * 4 + [0.0] * 4)
        import sys

        sys.argv = [
            "s4",
            "--capture",
            str(cap),
            "--tools",
            "green_flame_warm_self",
            "--json",
            str(tmp_path / "rv.json"),
        ]
        assert s4.main() == 0
        rep = json.loads((tmp_path / "rv.json").read_text())
        assert rep["n_annotated"] == 4
        assert rep["last_annotated_index"] == 3
        assert rep["tools"]["green_flame_warm_self"]["present_count"] == 4
        assert rep["tools"]["green_flame_warm_self"]["mean_tier_second_half"] is None
        out = capsys.readouterr().out
        assert "ANNOTATION VANISHES (trailing cliff)" in out
        assert "4 trailing decision(s) UNTREATED" in out

    def test_joins_results_and_buckets_flips(self, tmp_path, capsys):
        cap = self._capture(tmp_path, [0.9] * 4 + [0.0] * 4)
        res = tmp_path / "res.jsonl"
        # flips only while the annotation is strong — the dose-response signature
        res.write_text(
            "\n".join(
                json.dumps(
                    {
                        "decision_id": i,
                        "action_full": "a" if i < 4 else "b",
                        "action_ablated": "b",
                        "flipped": i < 4,
                    }
                )
                for i in range(8)
            )
        )
        import sys

        sys.argv = [
            "s4",
            "--capture",
            str(cap),
            "--results",
            str(res),
            "--tools",
            "green_flame_warm_self",
            "--json",
            str(tmp_path / "r2.json"),
        ]
        assert s4.main() == 0
        rep = json.loads((tmp_path / "r2.json").read_text())
        assert rep["flip_by_half"]["first"] == 1.0
        assert rep["flip_by_half"]["second"] == 0.0
        # second half has NO annotation at all (all-neutral -> section suppressed),
        # so those decisions bucket as "absent" (tier key None), not as tier 0.
        assert rep["flip_by_tier"]["2"]["rate"] == 1.0
        assert rep["flip_by_tier"]["None"]["rate"] == 0.0
        assert "NON-STATIONARY" in capsys.readouterr().out

    def test_capture_error_rows_skipped(self, tmp_path):
        cap = self._capture(tmp_path, [0.9, 0.9])
        with open(cap, "a") as f:
            f.write("\n" + json.dumps({"decision_id": 99, "capture_error": "boom"}) + "\n")
        import sys

        sys.argv = ["s4", "--capture", str(cap), "--json", str(tmp_path / "r3.json")]
        assert s4.main() == 0
        assert json.loads((tmp_path / "r3.json").read_text())["n_decisions"] == 2


class TestUntreatedShapeAndNoiseProbe:
    """Two fixes from the first REAL run (30/36 annotated, last annotated index 35):
    the 'trailing cliff' headline was misleading when gaps are scattered, and a raw
    half-split confounds effect decay with untreated decisions landing late."""

    def _cap(self, tmp_path, biases_by_index):
        from maxim.prompts.cluster_bias_annotation import compose_cluster_bias_annotation_section

        p = tmp_path / "capture.jsonl"
        rows = []
        for i, b in enumerate(biases_by_index):
            sec = compose_cluster_bias_annotation_section([("tool:t", b)])
            rows.append(json.dumps({"decision_id": i, "prompt_full": f"x\n{sec}\ny", "prompt_ablated": "x\ny"}))
        p.write_text("\n".join(rows))
        return p

    def test_scattered_gaps_not_called_a_cliff(self, tmp_path, capsys):
        # annotated, missing, annotated, missing, annotated  -> scattered
        cap = self._cap(tmp_path, [0.9, 0.0, 0.9, 0.0, 0.9])
        import sys

        sys.argv = ["s4", "--capture", str(cap), "--json", str(tmp_path / "s.json")]
        assert s4.main() == 0
        out = capsys.readouterr().out
        assert "SCATTERED" in out and "ANNOTATION VANISHES" not in out
        assert json.loads((tmp_path / "s.json").read_text())["untreated_shape"] == "scattered"

    def test_trailing_cliff_still_detected(self, tmp_path, capsys):
        cap = self._cap(tmp_path, [0.9, 0.9, 0.9, 0.0, 0.0])
        import sys

        sys.argv = ["s4", "--capture", str(cap), "--json", str(tmp_path / "t.json")]
        assert s4.main() == 0
        assert "ANNOTATION VANISHES (trailing cliff)" in capsys.readouterr().out
        assert json.loads((tmp_path / "t.json").read_text())["untreated_shape"] == "trailing_cliff"

    def test_annotated_only_split_and_noise_probe(self, tmp_path, capsys):
        # 4 annotated (2 flip) + 2 untreated (0 flip, identical prompts)
        cap = self._cap(tmp_path, [0.9, 0.9, 0.9, 0.9, 0.0, 0.0])
        res = tmp_path / "r.jsonl"
        res.write_text(
            "\n".join(
                json.dumps(
                    {"decision_id": i, "action_full": "a" if i < 2 else "b", "action_ablated": "b", "flipped": i < 2}
                )
                for i in range(6)
            )
        )
        import sys

        sys.argv = ["s4", "--capture", str(cap), "--results", str(res), "--json", str(tmp_path / "n.json")]
        assert s4.main() == 0
        rep = json.loads((tmp_path / "n.json").read_text())
        assert rep["untreated_noise_rate"] == 0.0 and rep["untreated_n"] == 2
        # raw second half includes the 2 untreated; annotated-only excludes them
        assert rep["flip_by_half_annotated"]["n_first"] == 3
        assert rep["flip_by_half_annotated"]["n_second"] == 1
        assert "free determinism probe" in capsys.readouterr().out
