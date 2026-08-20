"""Evidence-integrity guards (bugs ledger D24 + D25).

Substrate sweeps produce EVIDENCE — the numbers in `docs/experiments/results/`
back graduated behavioral claims. Two failure modes, both observed live on
2026-08-20, are guarded here:

D24 — the sweeps wrote those committed records as a side effect of running, so
an ordinary run replaced real evidence. One offline run rewrote
`p1_recognition_sweep.json` from `seeds_passing: 7` to `0`; it was caught in
`git status` by luck, not by a guard.

D25 — with the encoder weights absent, `LinguisticEncoder` degraded to hash
embeddings and the sweep asserted on the SCIENCE ("Substrate only beats random
by -0.9%") instead of erroring on the APPARATUS. That reads as a refutation of
the project's central claim; the true cause was a missing file.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from maxim.exceptions import ModelLoadError
from maxim.similarity.encoder import require_semantic_encoder

SUBSTRATE_TESTS = Path(__file__).parent
COMMITTED_RESULTS = SUBSTRATE_TESTS.parent.parent / "docs" / "experiments" / "results"

# The sweeps whose output is committed evidence.
EVIDENCE_SUITES = (
    "test_p1_recognition.py",
    "test_p2_reward_modulation.py",
    "test_concept_decomposition_validation.py",
)


class TestNoDirectEvidenceWrites:
    """D24: no test may write the committed records except through the fixture."""

    def test_no_suite_writes_the_committed_results_dir(self) -> None:
        offenders = []
        for path in SUBSTRATE_TESTS.rglob("test_*.py"):
            if path.name == Path(__file__).name:
                continue
            src = path.read_text()
            # A direct write is a results path joined to a filename and handed
            # to a writer. The fixture is the only sanctioned route.
            if 'experiments" / "results"' in src or "experiments/results" in src.replace(" ", ""):
                if "publish_sweep_results" not in src:
                    offenders.append(path.name)
        assert not offenders, (
            f"{offenders} reference the committed results tree without going through "
            "publish_sweep_results — a sweep must not overwrite evidence as a side effect (D24)"
        )

    def test_evidence_suites_use_the_publishing_fixture(self) -> None:
        for name in EVIDENCE_SUITES:
            src = (SUBSTRATE_TESTS / name).read_text()
            assert "publish_sweep_results" in src, f"{name} must publish through the fixture (D24)"

    def test_fixture_defaults_to_tmp_and_gates_the_committed_write(self) -> None:
        src = (SUBSTRATE_TESTS / "conftest.py").read_text()
        assert "--write-experiment-results" in src
        # The committed write must be behind the flag AND the apparatus check.
        publish = src.split("def _publish", 1)[1]
        flag_idx = publish.index("--write-experiment-results")
        apparatus_idx = publish.index("require_semantic_encoder")
        committed_idx = publish.index("COMMITTED_RESULTS_DIR /")
        assert flag_idx < committed_idx, "the flag must gate the committed write"
        assert apparatus_idx < committed_idx, "the apparatus check must precede publishing evidence"


class TestApparatusAssertion:
    """D25: a measurement path must refuse to run on the fallback encoder."""

    def test_missing_model_raises_typed_error_not_a_science_assertion(self, monkeypatch) -> None:
        import maxim.similarity.encoder as enc

        monkeypatch.setattr(enc, "_get_encoder", lambda *a, **k: None)
        with pytest.raises(ModelLoadError) as excinfo:
            require_semantic_encoder(context="guard test")

        message = str(excinfo.value)
        # The operator must learn WHICH run was refused, WHY, and HOW to fix it.
        assert "guard test" in message
        assert "refusing to measure" in message
        assert "hash fallback" in message
        assert "semantic" in message and "HF_HUB_OFFLINE" in message

    def test_real_encoder_passes_silently(self, monkeypatch) -> None:
        import maxim.similarity.encoder as enc

        monkeypatch.setattr(enc, "_get_encoder", lambda *a, **k: object())
        require_semantic_encoder(context="guard test")  # must not raise

    def test_evidence_suites_check_the_apparatus_before_measuring(self) -> None:
        """The check must be an autouse fixture, so no sweep can skip it."""
        for name in ("test_p1_recognition.py", "test_p2_reward_modulation.py"):
            src = (SUBSTRATE_TESTS / name).read_text()
            assert "require_semantic_encoder" in src, f"{name} must assert its apparatus (D25)"
            tree = ast.parse(src)
            found = False
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and "require_semantic_encoder" in ast.dump(node):
                    decorators = ast.dump(ast.Module(body=node.decorator_list, type_ignores=[]))
                    if "autouse" in decorators:
                        found = True
            assert found, f"{name}'s apparatus check must be autouse, not opt-in per test"


class TestCommittedEvidenceIsWellFormed:
    """A cheap tripwire: the records this policy protects must stay readable."""

    @pytest.mark.parametrize(
        "filename",
        [
            "p1_recognition_sweep.json",
            "p2_reward_modulation_sweep.json",
            "concept_decomposition_validation.json",
        ],
    )
    def test_record_parses_and_is_not_a_degraded_stub(self, filename: str) -> None:
        path = COMMITTED_RESULTS / filename
        if not path.exists():
            pytest.skip(f"{filename} not committed")
        payload = json.loads(path.read_text())
        assert isinstance(payload, dict) and payload, f"{filename} is empty or malformed"
