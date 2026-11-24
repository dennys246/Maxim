"""#1789: persisted id indexes are written sorted, so a saved store's bytes do not follow PYTHONHASHSEED.

Each writer below kept a ``set[str]`` and wrote ``list(v)``. String hashing is randomised per process, so the same
store saved by two processes gave two different files (found by world clock A-pre PR 0's WC5 round trip). A
same-process test cannot see this class of bug: every check here runs the save in TWO child processes with different
``PYTHONHASHSEED``s and compares their bytes (CLAUDE.md's stable-hash lesson).

Each source store holds many ids under one key, so an unsorted writer would almost surely differ between the seeds.
The guard was proven by reverting the fix: every case failed.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src"
SEEDS = ("1", "2")
IDS = [f"mem-{i:02d}-{'abcdefgh'[i % 8]}" for i in range(12)]


def _save_in_two_processes(tmp_path: Path, body: str, source: Path | None = None) -> list[bytes]:
    """Run *body* in a fresh interpreter per seed; it reads ``SOURCE`` (optional) and writes ``OUT``."""
    outputs = []
    for seed in SEEDS:
        out = tmp_path / f"out-{seed}.json"
        home = tmp_path / f"home-{seed}"
        home.mkdir()
        env = {**os.environ, "PYTHONHASHSEED": seed, "PYTHONPATH": str(SRC), "HOME": str(home)}
        script = "import time\ntime.time = lambda: 1_764_000_000.5\n" + textwrap.dedent(body)
        subprocess.run(
            [sys.executable, "-c", script, str(source or ""), str(out)],
            env=env,
            check=True,
            capture_output=True,
            timeout=120,
        )
        outputs.append(out.read_bytes())
    return outputs


def test_hippocampus_context_index_is_byte_stable_across_processes(tmp_path: Path) -> None:
    from maxim.memory.encoding import EncodingSignals
    from maxim.memory.hippocampus import Hippocampus, HippocampusConfig
    from maxim.memory.types import Action, Context, Decision, Outcome, Perception

    config = HippocampusConfig(persistence_path=None, enable_sleep_consolidation=False, auto_save_after_sleep=False)
    hippocampus = Hippocampus(config)
    for i in range(12):
        hippocampus.capture(
            perception=Perception(
                observations={"i": i},
                detected_objects=["mug"],
                detected_people=[],
                salience=0.5,
                novelty=0.5,
                cli_input=None,
                transcript=None,
            ),
            context=Context(active_goal="find mug", active_mode="exploration"),
            decision=Decision(intent={"goal": "find mug"}, reasoning="look", confidence=0.7),
            action=Action(tool_name="look_around", tool_params={}),
            outcome=Outcome(success=True, result={}, error=None, evaluations=[]),
            encoding=EncodingSignals.unmeasured("api"),
        )
    source = tmp_path / "hippocampus.json"
    hippocampus.save(str(source))
    assert max(len(v) for v in json.loads(source.read_text())["context_index"].values()) >= 12

    first, second = _save_in_two_processes(
        tmp_path,
        """
        import sys
        from maxim.memory.hippocampus import Hippocampus, HippocampusConfig
        h = Hippocampus(HippocampusConfig(persistence_path=None, enable_sleep_consolidation=False,
                                          auto_save_after_sleep=False))
        h.load(sys.argv[1])
        h.save(sys.argv[2])
        """,
        source,
    )
    assert first == second


def test_nac_outcome_index_is_byte_stable_across_processes(tmp_path: Path) -> None:
    from maxim.decisions.causal_link import Valence
    from maxim.decisions.nac import NAc, NACConfig

    nac = NAc(NACConfig())
    for i in range(12):
        nac.observe(
            event_type="tool_call",
            event_signature=f"tool:t{i}",
            outcome_type="tool_result",
            outcome_signature="done",
            outcome_valence=Valence.POSITIVE,
            delta_seconds=1.0,
        )
    source = tmp_path / "nac.json"
    nac.save(str(source))
    assert len(json.loads(source.read_text())["outcome_index"]["done"]) == 12

    first, second = _save_in_two_processes(
        tmp_path,
        """
        import sys
        from maxim.decisions.nac import NAc, NACConfig
        n = NAc(NACConfig())
        n.load(sys.argv[1])
        n.save(sys.argv[2])
        """,
        source,
    )
    assert first == second


def test_ec_inverted_indices_are_byte_stable_across_processes(tmp_path: Path) -> None:
    source = tmp_path / "inverted.json"
    source.write_text(
        json.dumps(
            {
                "by_tool": {"look": IDS},
                "by_outcome": {"success": IDS},
                "by_mode": {"explore": IDS},
                "by_hour": {"9": IDS},
                "by_day": {"2": IDS},
            }
        )
    )
    first, second = _save_in_two_processes(
        tmp_path,
        """
        import json, sys
        from maxim.similarity.indices import InvertedIndices
        data = json.load(open(sys.argv[1]))
        open(sys.argv[2], "w").write(json.dumps(InvertedIndices.from_dict(data).to_dict()))
        """,
        source,
    )
    assert first == second


def test_similarity_index_bands_are_byte_stable_across_processes(tmp_path: Path) -> None:
    first, second = _save_in_two_processes(
        tmp_path,
        f"""
        import sys
        from maxim.memory.context_index import SimilarityIndex
        index = SimilarityIndex()
        for memory_id in {IDS!r}:
            index.register(memory_id, "the robot looked at the red mug on the table")
        index.save(sys.argv[2])
        """,
    )
    assert first == second


@pytest.mark.parametrize("writer", ["hippocampus", "nac", "inverted", "similarity"])
def test_the_saved_lists_are_sorted(tmp_path: Path, writer: str) -> None:
    """Same-process half: the order is the sorted one, not merely the same twice."""
    if writer == "inverted":
        from maxim.similarity.indices import InvertedIndices

        out = InvertedIndices.from_dict({"by_tool": {"look": list(reversed(IDS))}}).to_dict()
        assert out["by_tool"]["look"] == sorted(IDS)
    elif writer == "similarity":
        from maxim.memory.context_index import SimilarityIndex

        index = SimilarityIndex()
        for memory_id in reversed(IDS):
            index.register(memory_id, "the robot looked at the red mug on the table")
        path = tmp_path / "index.json"
        index.save(str(path))
        for band in json.loads(path.read_text())["bands"]:
            for ids in band.values():
                assert ids == sorted(ids)
    elif writer == "nac":
        from maxim.decisions.causal_link import Valence
        from maxim.decisions.nac import NAc, NACConfig

        nac = NAc(NACConfig())
        for i in reversed(range(6)):
            nac.observe(
                event_type="tool_call",
                event_signature=f"tool:t{i}",
                outcome_type="tool_result",
                outcome_signature="done",
                outcome_valence=Valence.POSITIVE,
                delta_seconds=1.0,
            )
        path = tmp_path / "nac.json"
        nac.save(str(path))
        ids = json.loads(path.read_text())["outcome_index"]["done"]
        assert ids == sorted(ids)
    else:
        from maxim.memory.hippocampus import Hippocampus, HippocampusConfig

        hippocampus = Hippocampus(HippocampusConfig(persistence_path=None))
        hippocampus._context_index["goal:x"] = set(reversed(IDS))
        path = tmp_path / "h.json"
        hippocampus.save(str(path))
        assert json.loads(path.read_text())["context_index"]["goal:x"] == sorted(IDS)
