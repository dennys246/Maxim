"""Two-process hash-stability tests for persisted hash values.

Python's builtin ``hash()`` for str is randomized per process
(PYTHONHASHSEED).  Any value derived from it that crosses a process
boundary — persisted to disk and reloaded, or recomputed for a query
against persisted state — silently stops matching after a restart.

Measured pre-fix (2026-07-30): an identical situation scored 0.825
same-process vs 0.425 after reload through ``SituationSignature``
(structural/context hash mismatch), straddling the ``min_similarity=0.5``
gate NAc uses for EC-mediated causal-link matching.  A reloaded
``SimilarityIndex`` reported ``len == 1`` but returned ``[]`` for the
exact text it stored.

These tests MUST be two-process with differing PYTHONHASHSEED — a
same-process test passes over the bug because both sides share one hash
seed.  Each test was verified to FAIL against the pre-fix code.

Persistence boundaries pinned here:
- ``similarity/signature.py`` structural_hash / context_hash →
  ``EC.save()`` / ``EC.load()`` (live via ``--resume-sim``)
- ``memory/context_index.py`` MinHash → ``SimilarityIndex.save/load``
- ``similarity/lsh.py::SemanticLSH.hash`` → ``SituationSignature.semantic_hash``
  → ``EC.save()`` signatures + LSH tables
- ``similarity/semantic.py::NeuralSemanticLSH._fallback_hash`` → hash bits
  stored in the persisted ``EmbeddingStore`` npz (memory_hub semantic path)
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

REPO_SRC = str(Path(__file__).resolve().parents[2] / "src")


def _run_in_subprocess(code: str, hashseed: str) -> dict:
    """Run ``code`` in a fresh interpreter with a pinned PYTHONHASHSEED.

    The snippet must print a single JSON object on its last stdout line.
    """
    import os

    env = dict(os.environ)
    env["PYTHONHASHSEED"] = hashseed
    env["PYTHONPATH"] = REPO_SRC + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    assert proc.returncode == 0, f"subprocess failed (seed={hashseed}):\n{proc.stderr}"
    return json.loads(proc.stdout.strip().splitlines()[-1])


_SIGNATURE_SNIPPET = """
import json
from types import SimpleNamespace

from maxim.similarity.signature import SituationSignature

memory = SimpleNamespace(
    decision=SimpleNamespace(intent={"goal": "find the red cup"}, reasoning="kitchen sweep"),
    action=SimpleNamespace(tool_name="internet_search"),
    outcome=SimpleNamespace(success=True),
    timestamp=1753800000.0,
    context=SimpleNamespace(active_mode="autonomous"),
    perception=SimpleNamespace(detected_objects=["cup", "table"], detected_people=[]),
)
sig = SituationSignature.from_memory(memory)
print(json.dumps(sig.to_dict()))
"""


class TestSituationSignatureCrossProcess:
    def test_structural_and_context_hash_stable_across_hashseeds(self):
        """The persisted exact-match int fields must be process-independent.

        ``_structural_match`` / ``_context_match`` compare these ints for
        exact equality; ``similarity()`` weights them at 0.25 + 0.15 —
        enough to straddle NAc's 0.5 EC gate when they stop matching.
        """
        a = _run_in_subprocess(_SIGNATURE_SNIPPET, hashseed="1")
        b = _run_in_subprocess(_SIGNATURE_SNIPPET, hashseed="2")
        assert a["structural_hash"] == b["structural_hash"], (
            "structural_hash differs across PYTHONHASHSEED — persisted "
            "signatures can never structurally match after a restart"
        )
        assert a["context_hash"] == b["context_hash"], (
            "context_hash differs across PYTHONHASHSEED — persisted signatures can never context-match after a restart"
        )

    def test_reloaded_signature_similarity_is_self_identical(self):
        """A persisted signature must score 1.0 against a fresh recompute
        of the identical situation in a different process."""
        a = _run_in_subprocess(_SIGNATURE_SNIPPET, hashseed="1")
        b = _run_in_subprocess(_SIGNATURE_SNIPPET, hashseed="2")

        from maxim.similarity.signature import SituationSignature

        sim = SituationSignature.from_dict(a).similarity(SituationSignature.from_dict(b))
        assert sim >= 0.99, (
            f"identical situation across two processes scored {sim:.3f} — "
            "below NAc's min_similarity=0.5 EC gate territory"
        )


_INDEX_STORE_SNIPPET = """
import json
from maxim.memory.context_index import SimilarityIndex

idx = SimilarityIndex(num_hashes=64, num_bands=16)
idx.register("mem_1", "the robot saw a person near the table in the kitchen")
idx.save({path!r})
print(json.dumps({{"stored": len(idx)}}))
"""

_INDEX_QUERY_SNIPPET = """
import json
from maxim.memory.context_index import SimilarityIndex

idx = SimilarityIndex.load({path!r})
results = idx.query_similar("the robot saw a person near the table in the kitchen", min_similarity=0.9)
print(json.dumps({{"count": len(idx), "results": [[mid, sim] for mid, sim in results]}}))
"""


class TestSimilarityIndexCrossProcess:
    def test_reloaded_index_finds_exact_stored_text(self, tmp_path):
        """A reloaded index must return the exact text it stored.

        Pre-fix: ``len(idx) == 1`` but ``query_similar`` returns ``[]``
        because new-process MinHash values never equal the persisted ones.
        This is also the root cause of the ~2.5% CI flake in
        tests/unit/test_context_index.py::TestSimilarityIndexQueries::
        test_similar_text_found (randomized MinHash occasionally lands an
        unlucky band split even same-process).
        """
        path = str(tmp_path / "context_index.json")
        stored = _run_in_subprocess(_INDEX_STORE_SNIPPET.format(path=path), hashseed="1")
        assert stored["stored"] == 1

        out = _run_in_subprocess(_INDEX_QUERY_SNIPPET.format(path=path), hashseed="2")
        assert out["count"] == 1, "reloaded index lost its entry entirely"
        result_ids = {mid for mid, _sim in out["results"]}
        assert "mem_1" in result_ids, (
            "reloaded SimilarityIndex reports len==1 but cannot find the "
            "exact text it stored — MinHash values are process-local"
        )
        # Exact text must estimate ~identical Jaccard, not merely clear a
        # low bar: pre-fix the estimate is ~0.0.
        sim_by_id = dict(out["results"])
        assert sim_by_id["mem_1"] >= 0.99


_SEMANTIC_LSH_SNIPPET = """
import json
from maxim.similarity.lsh import SemanticLSH

hasher = SemanticLSH()
print(json.dumps(list(hasher.hash("the robot saw a person near the table"))))
"""

_NEURAL_FALLBACK_SNIPPET = """
import json
from maxim.similarity.semantic import NeuralSemanticLSH

hasher = NeuralSemanticLSH.__new__(NeuralSemanticLSH)
from maxim.similarity.semantic import SemanticEmbedderConfig
hasher.config = SemanticEmbedderConfig()
print(json.dumps(list(hasher._fallback_hash("the robot saw a person near the table"))))
"""


class TestSemanticHashersCrossProcess:
    def test_semantic_lsh_hash_stable_across_hashseeds(self):
        """``SemanticLSH`` seeds its per-plane salt through builtin
        ``hash()`` — the ``seed`` parameter looked deterministic but was
        not.  Its output lands in ``SituationSignature.semantic_hash``,
        persisted via ``EC.save()`` and compared against fresh query
        hashes (ec.py `_semantic_hasher.hash(query)`)."""
        a = _run_in_subprocess(_SEMANTIC_LSH_SNIPPET, hashseed="1")
        b = _run_in_subprocess(_SEMANTIC_LSH_SNIPPET, hashseed="2")
        assert a == b, "SemanticLSH.hash differs across PYTHONHASHSEED"

    def test_neural_fallback_hash_stable_across_hashseeds(self):
        """``NeuralSemanticLSH._fallback_hash`` (model-unavailable path)
        feeds hash bits into the persisted EmbeddingStore npz via
        memory_hub's semantic capture path."""
        try:
            a = _run_in_subprocess(_NEURAL_FALLBACK_SNIPPET, hashseed="1")
            b = _run_in_subprocess(_NEURAL_FALLBACK_SNIPPET, hashseed="2")
        except AssertionError as e:
            if "ModuleNotFoundError" in str(e) or "ImportError" in str(e):
                import pytest

                pytest.skip("similarity.semantic optional deps not installed")
            raise
        assert a == b, "NeuralSemanticLSH._fallback_hash differs across PYTHONHASHSEED"
