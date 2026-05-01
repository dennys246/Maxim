"""Integration test for the V1 substrate-attribution report block.

Asserts that ``simulation/report.py::build_report`` populates the
``confound_quarantine`` block correctly under two configurations:

- **Phase A (substrate-only baseline):** all four ``MAXIM_DISABLE_*=1``
  flags set + ``MAXIM_DATA_HOME`` to a tmpdir. The report's metrics
  show zero tokens for every gated scaffold and ``persona_active=None``.

- **Phase G (control, today's behavior):** all flags unset. The report
  records non-zero token estimates for the active scaffolds and the
  persona name passed by the caller.

The LLM is mocked. We exercise ``build_report`` directly with a stub
bridge — no real sim is run. Per ``docs/plans/confound_quarantine.md``
this is the metric-shape assertion only; the actual phased re-run with
real LLMs is run by ``scripts/run_v1_phases.sh`` in a separate session.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from maxim.simulation.report import build_report


@pytest.fixture
def stub_bridge():
    bridge = MagicMock()
    bridge.turn_count = 1
    bridge.get_all_actions.return_value = []
    return bridge


class TestPhaseA:
    """Phase A — substrate-only baseline."""

    def test_all_flags_set_zeros_token_counts(self, stub_bridge, monkeypatch, tmp_path):
        monkeypatch.setenv("MAXIM_DISABLE_PFC_PREAMBLE", "1")
        monkeypatch.setenv("MAXIM_DISABLE_ACTING_COACH", "1")
        monkeypatch.setenv("MAXIM_DISABLE_SIM_SANDBOX_TEXT", "1")
        monkeypatch.setenv("MAXIM_NO_DEFAULT_PERSONA", "1")
        monkeypatch.setenv("MAXIM_DATA_HOME", str(tmp_path))
        monkeypatch.setenv("MAXIM_V1_PHASE", "A")

        report = build_report(
            goal="recall the password from session 1",
            persona="neutral",
            bridge=stub_bridge,
            duration_s=1.0,
            finish_reason="completed",
            entity_ref=None,
            arc_name=None,
        )

        block = report.confound_quarantine
        assert block["phase"] == "A"
        assert block["isolated_data_home"] is True
        assert block["flags"]["MAXIM_DISABLE_PFC_PREAMBLE"] == "1"
        assert block["flags"]["MAXIM_DISABLE_ACTING_COACH"] == "1"
        assert block["flags"]["MAXIM_DISABLE_SIM_SANDBOX_TEXT"] == "1"
        assert block["flags"]["MAXIM_NO_DEFAULT_PERSONA"] == "1"
        assert block["flags"]["MAXIM_DATA_HOME"] == str(tmp_path)

        metrics = block["metrics"]
        assert metrics["tokens_in_pfc_preamble"] == 0
        assert metrics["tokens_in_acting_coach"] == 0
        assert metrics["tokens_in_sim_sandbox"] == 0
        assert metrics["persona_active"] is None
        assert metrics["embodiment_ref"] is None
        assert metrics["arc_active"] is None

    def test_block_present_in_asdict_export(self, stub_bridge, monkeypatch, tmp_path):
        # The dataclass must serialize the block — without this, save_report
        # writes a JSON without the field and the harness analysis breaks.
        from dataclasses import asdict

        monkeypatch.setenv("MAXIM_DISABLE_PFC_PREAMBLE", "1")
        monkeypatch.setenv("MAXIM_DATA_HOME", str(tmp_path))

        report = build_report(
            goal="test",
            persona="neutral",
            bridge=stub_bridge,
            duration_s=1.0,
            finish_reason="completed",
        )
        d = asdict(report)
        assert "confound_quarantine" in d
        assert d["confound_quarantine"]["isolated_data_home"] is True


class TestPhaseG:
    """Phase G — today's behavior, control run."""

    def test_no_flags_set_records_persona_and_estimates_tokens(self, stub_bridge, monkeypatch, tmp_path):
        # Clear the V1 phase label so the report shows empty phase
        monkeypatch.delenv("MAXIM_V1_PHASE", raising=False)
        # Phase G uses an isolated data home (per protocol) so isolation
        # alone is the only difference from "real production".
        monkeypatch.setenv("MAXIM_DATA_HOME", str(tmp_path))

        report = build_report(
            goal="recall the password from session 1",
            persona="adversarial",
            bridge=stub_bridge,
            duration_s=1.0,
            finish_reason="completed",
            entity_ref="bodies/base_humanoid",
            arc_name=None,
        )

        block = report.confound_quarantine
        assert block["phase"] == ""
        assert block["isolated_data_home"] is True
        # Flags absent → empty string sentinel in the snapshot
        assert block["flags"]["MAXIM_DISABLE_PFC_PREAMBLE"] == ""
        assert block["flags"]["MAXIM_DISABLE_ACTING_COACH"] == ""
        assert block["flags"]["MAXIM_DISABLE_SIM_SANDBOX_TEXT"] == ""
        assert block["flags"]["MAXIM_NO_DEFAULT_PERSONA"] == ""

        metrics = block["metrics"]
        # All three scaffolds should report SOME estimated tokens (positive).
        # Exact counts depend on token-counter availability; the >0 lower
        # bound is the structural assertion.
        assert metrics["tokens_in_pfc_preamble"] > 0
        assert metrics["tokens_in_acting_coach"] > 0
        assert metrics["tokens_in_sim_sandbox"] > 0
        assert metrics["persona_active"] == "adversarial"
        assert metrics["embodiment_ref"] == "bodies/base_humanoid"


class TestPartialPhases:
    """Spot-check the per-flag attribution surface."""

    def test_only_pfc_disabled_zeros_pfc_only(self, stub_bridge, monkeypatch, tmp_path):
        # Phase B (PFC OFF, others ON) — the inverse of Phase A's PFC
        # contribution — should zero only the PFC token count.
        monkeypatch.setenv("MAXIM_DISABLE_PFC_PREAMBLE", "1")
        monkeypatch.setenv("MAXIM_DATA_HOME", str(tmp_path))

        report = build_report(
            goal="test",
            persona="adversarial",
            bridge=stub_bridge,
            duration_s=1.0,
            finish_reason="completed",
        )

        m = report.confound_quarantine["metrics"]
        assert m["tokens_in_pfc_preamble"] == 0
        assert m["tokens_in_acting_coach"] > 0
        assert m["tokens_in_sim_sandbox"] > 0

    def test_only_persona_disabled_records_none_persona(self, stub_bridge, monkeypatch, tmp_path):
        monkeypatch.setenv("MAXIM_NO_DEFAULT_PERSONA", "1")
        monkeypatch.setenv("MAXIM_DATA_HOME", str(tmp_path))

        report = build_report(
            goal="test",
            persona="neutral",
            bridge=stub_bridge,
            duration_s=1.0,
            finish_reason="completed",
        )

        m = report.confound_quarantine["metrics"]
        # persona_active is None when the flag is set, regardless of
        # what string was passed in.
        assert m["persona_active"] is None
        # Other token counts are still nonzero.
        assert m["tokens_in_pfc_preamble"] > 0
        assert m["tokens_in_acting_coach"] > 0
