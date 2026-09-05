"""Evidence-integrity guards (bugs ledger D25 + D26).

Substrate sweeps produce EVIDENCE — the numbers in `docs/experiments/results/`
back graduated behavioral claims. Two failure modes, both observed live on
2026-08-20, are guarded here:

D25 — the sweeps wrote those committed records as a side effect of running, so
an ordinary run replaced real evidence. One offline run rewrote
`p1_recognition_sweep.json` from `seeds_passing: 7` to `0`; it was caught in
`git status` by luck, not by a guard.

D26 — with the encoder weights absent, `LinguisticEncoder` degraded to hash
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
    """D25: no test may write the committed records except through the fixture."""

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
            "publish_sweep_results — a sweep must not overwrite evidence as a side effect (D25)"
        )

    def test_evidence_suites_use_the_publishing_fixture(self) -> None:
        for name in EVIDENCE_SUITES:
            src = (SUBSTRATE_TESTS / name).read_text()
            assert "publish_sweep_results" in src, f"{name} must publish through the fixture (D25)"

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
    """D26: a measurement path must refuse to run on the fallback encoder."""

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
        """Each evidence suite must CALL the assertion from an autouse fixture.

        This looks for an `ast.Call` and not the bare name: an earlier version
        matched `"require_semantic_encoder" in src`, which the `import` line
        alone satisfies — so deleting the actual call still passed. A guard
        that cannot detect its own removal is the failure mode this file exists
        to prevent.
        """
        for name in EVIDENCE_SUITES:
            src = (SUBSTRATE_TESTS / name).read_text()
            tree = ast.parse(src)
            called_in_autouse = False
            for node in ast.walk(tree):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                is_autouse = "autouse" in ast.dump(ast.Module(body=node.decorator_list, type_ignores=[]))
                if not is_autouse:
                    continue
                for inner in ast.walk(node):
                    if (
                        isinstance(inner, ast.Call)
                        and isinstance(inner.func, ast.Name)
                        and inner.func.id == "require_semantic_encoder"
                    ):
                        called_in_autouse = True
            assert called_in_autouse, (
                f"{name} must CALL require_semantic_encoder() from an autouse fixture, "
                "so a sweep cannot measure on the hash fallback (D26)"
            )


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


class TestScriptsEvidenceWritePolicy:
    """D27 (1.2 gate 8(a)): the same policy, extended to the `scripts/` surface.

    D25/D26 closed the class for tests/substrate/; seven `scripts/` harnesses
    kept overwriting committed S4 evidence unconditionally. They now route
    every committed-tree write through `_provenance.evidence_out_paths_or_exit`
    (opt-in via --write-experiment-results; dirty-tree refusal exits 3 on the
    opt-in), and the LinguisticEncoder measurement scripts assert their
    apparatus via `require_semantic_encoder` before measuring.

    KNOWN SCAN LIMIT (stated, not hidden): the scan keys on the RESULTS tree
    (`experiments/results`). A new script writing only a committed
    `docs/experiments/<report>.md` — the paired-artifact shape the seven also
    had — is caught only if it also touches results/; a reliable .md scan
    would false-positive on the countless docstring references to experiment
    reports. The seven's own .md writes ARE routed (their paths go through
    the same helper call), and reviewer attention owns the residual class.
    """

    SCRIPTS = SUBSTRATE_TESTS.parent.parent / "scripts"
    # Files that may reference the committed results tree WITHOUT the opt-in
    # helper, each with a stated reason (strict-grep allowlist pattern —
    # additions need a reason, not just a name).
    ALLOWLIST: dict[str, str] = {
        "_provenance.py": "defines the policy (evidence_out_paths lives here)",
    }
    # LinguisticEncoder measurement entry points — a degraded hash-fallback
    # encoder must error on the APPARATUS, never publish over the science.
    SEMANTIC_MEASUREMENT_SCRIPTS = (
        "fine_sweep_phase_2.py",
        "measure_p1_at_threshold.py",
        "diagnose_roy_paraphrase_collapse.py",
    )

    @staticmethod
    def _references_results_tree(src: str) -> bool:
        flat = src.replace('"', "").replace("'", "").replace(" ", "")
        return "experiments/results" in flat

    def test_scripts_touching_the_results_tree_use_the_optin_helper(self) -> None:
        offenders = []
        for path in sorted(self.SCRIPTS.rglob("*.py")):
            src = path.read_text()
            if not self._references_results_tree(src):
                continue
            if path.name in self.ALLOWLIST:
                continue
            if "evidence_out_path" not in src:
                offenders.append(str(path.relative_to(self.SCRIPTS)))
        assert not offenders, (
            f"{offenders} reference docs/experiments/results without routing through "
            "_provenance.evidence_out_paths — a harness must not overwrite committed "
            "evidence as a side effect (D25/D27). Route the write through the helper, "
            "or add an ALLOWLIST entry WITH a reason."
        )

    def test_semantic_measurement_scripts_assert_their_apparatus(self) -> None:
        missing = [
            name
            for name in self.SEMANTIC_MEASUREMENT_SCRIPTS
            if "require_semantic_encoder" not in (self.SCRIPTS / name).read_text()
        ]
        assert not missing, (
            f"{missing} build a LinguisticEncoder without require_semantic_encoder — "
            "a hash-fallback run would measure noise and (pre-D27) publish it (D26)."
        )

    def test_linguistic_encoder_scripts_are_all_accounted_for(self) -> None:
        """A NEW LinguisticEncoder measurement script must join the list above
        (or explain itself) — the scan that keeps the D27 encoder half closed."""
        known = set(self.SEMANTIC_MEASUREMENT_SCRIPTS) | {
            "exp_d8_read_mutation.py",  # calls require_semantic_encoder itself (checked below)
        }
        builders = {
            str(p.relative_to(self.SCRIPTS))
            for p in self.SCRIPTS.rglob("*.py")
            if "LinguisticEncoder(" in p.read_text()
        }
        unknown = builders - known
        assert not unknown, (
            f"{unknown} construct LinguisticEncoder but are not in the D27 accounting — "
            "add require_semantic_encoder to the measurement entry point and list the file."
        )
        assert "require_semantic_encoder" in (self.SCRIPTS / "exp_d8_read_mutation.py").read_text()


class TestEvidenceOutPathsHelper:
    """The helper's own contract (D27)."""

    @staticmethod
    def _prov():
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "_provenance_under_test", SUBSTRATE_TESTS.parent.parent / "scripts" / "_provenance.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_without_the_flag_governed_paths_redirect_names_preserved(self, tmp_path, capsys) -> None:
        prov = self._prov()
        repo = tmp_path
        committed_json = repo / "docs" / "experiments" / "results" / "x.json"
        committed_md = repo / "docs" / "experiments" / "x.md"
        out = prov.evidence_out_paths(repo, [committed_md, committed_json], write_experiment_results=False)
        assert [p.name for p in out] == ["x.md", "x.json"]
        assert out[0].parent == out[1].parent  # paired artifacts share one temp dir
        assert not str(out[0]).startswith(str(repo))
        assert "NOT updating committed record" in capsys.readouterr().out

    def test_ungoverned_paths_pass_through_untouched(self, tmp_path) -> None:
        prov = self._prov()
        elsewhere = tmp_path / "out" / "free.json"
        out = prov.evidence_out_paths(tmp_path, [elsewhere], write_experiment_results=False)
        assert out == [elsewhere.resolve()]

    def test_with_the_flag_clean_tree_returns_committed(self, tmp_path, monkeypatch) -> None:
        prov = self._prov()
        monkeypatch.setattr(prov, "working_tree_dirty", lambda *a, **k: False)
        committed = tmp_path / "docs" / "experiments" / "results" / "x.json"
        out = prov.evidence_out_paths(tmp_path, [committed], write_experiment_results=True)
        assert out == [committed.resolve()]

    def test_with_the_flag_dirty_tree_refuses(self, tmp_path, monkeypatch) -> None:
        prov = self._prov()
        monkeypatch.setattr(prov, "working_tree_dirty", lambda *a, **k: True)
        committed = tmp_path / "docs" / "experiments" / "results" / "x.json"
        with pytest.raises(prov.DirtyTreeError):
            prov.evidence_out_paths(tmp_path, [committed], write_experiment_results=True)

    def test_dirty_tree_with_explicit_allowance_writes_committed(self, tmp_path, monkeypatch) -> None:
        prov = self._prov()
        monkeypatch.setattr(prov, "working_tree_dirty", lambda *a, **k: True)
        committed = tmp_path / "docs" / "experiments" / "results" / "x.json"
        out = prov.evidence_out_paths(tmp_path, [committed], write_experiment_results=True, allow_dirty=True)
        assert out == [committed.resolve()]
