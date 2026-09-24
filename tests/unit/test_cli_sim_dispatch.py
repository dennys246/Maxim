"""Guards for two ``--sim`` dispatch defects in ``cli.py`` (#863 review round).

1. ``--interactive=false`` was ignored. Explicitness was detected by scanning
   argv for the literal token ``"--interactive"``, so the ``=`` spelling (and
   any argparse abbreviation) fell through to TTY auto-detection and a human
   at a terminal got interactive mode anyway.
2. A DM campaign that failed to load or run was swallowed by the auto-detect
   ``try`` and the same YAML was then re-run as a plain scenario, hiding the
   DM failure behind an unrelated second run.
"""

from __future__ import annotations

import pytest

from maxim.cli import _is_dm_campaign_yaml, _maybe_run_dm_campaign, _resolve_interactive_mode
from maxim.cli_parser import _build_parser
from maxim.cli_utils import normalize_args


def _args(*argv: str):
    args = _build_parser().parse_args(["--sim", "some goal", *argv])
    normalize_args(args)
    return args


class TestInteractiveResolution:
    @pytest.mark.parametrize(
        "argv",
        [
            ("--interactive", "false"),
            ("--interactive=false",),
            ("--interactive=off",),
            ("--interac", "false"),  # argparse abbreviation
        ],
    )
    def test_an_explicit_off_wins_over_a_tty(self, argv):
        assert _resolve_interactive_mode(_args(*argv), is_dm=True, is_tty=True) == "off"

    @pytest.mark.parametrize("argv", [("--interactive",), ("--interactive=true",)])
    def test_an_explicit_on_wins_over_no_tty(self, argv):
        assert _resolve_interactive_mode(_args(*argv), is_dm=False, is_tty=False) == "on"

    @pytest.mark.parametrize(
        ("is_dm", "is_tty", "expected"),
        [(False, False, "off"), (False, True, "on"), (True, False, "on")],
    )
    def test_unset_auto_detects(self, is_dm, is_tty, expected):
        assert _resolve_interactive_mode(_args(), is_dm=is_dm, is_tty=is_tty) == expected


_DM_YAML = "campaign:\n  name: t\nencounters: []\n"
_SCENARIO_YAML = "name: s\npercepts: []\n"


class TestDmCampaignProbe:
    def test_probe_recognises_a_campaign(self, tmp_path):
        f = tmp_path / "c.yaml"
        f.write_text(_DM_YAML)
        assert _is_dm_campaign_yaml(f) is True

    @pytest.mark.parametrize(
        "content",
        [
            _SCENARIO_YAML,
            "[1, 2]\n",
            "key: [unclosed\n",
            "d: 2026-13-45\n",  # PyYAML raises ValueError, not YAMLError
            "[" * 5000 + "]" * 5000 + "\n",  # RecursionError
        ],
    )
    def test_probe_rejects_non_campaigns_and_bad_yaml(self, tmp_path, content):
        f = tmp_path / "s.yaml"
        f.write_text(content)
        assert _is_dm_campaign_yaml(f) is False

    def test_probe_rejects_a_missing_file(self, tmp_path):
        assert _is_dm_campaign_yaml(tmp_path / "nope.yaml") is False


class TestDmCampaignFailuresPropagate:
    def test_a_load_failure_is_raised_not_rerun_as_a_scenario(self, tmp_path, monkeypatch):
        import maxim.simulation.dm_schema as dm_schema

        def _boom(*_a, **_kw):
            raise RuntimeError("campaign load failed")

        monkeypatch.setattr(dm_schema, "load_campaign", _boom)
        f = tmp_path / "c.yaml"
        f.write_text(_DM_YAML)
        with pytest.raises(RuntimeError, match="campaign load failed"):
            _maybe_run_dm_campaign(f, _args(), debug=False, entity_ref=None)

    def test_a_run_failure_is_raised(self, tmp_path, monkeypatch):
        import maxim.simulation.dm_schema as dm_schema
        import maxim.simulation.orchestrator as orchestrator

        class _Campaign:
            name = "t"

        monkeypatch.setattr(dm_schema, "load_campaign", lambda *_a, **_kw: _Campaign())
        monkeypatch.setattr(dm_schema, "validate_campaign", lambda _c: [])

        def _boom(**_kw):
            raise RuntimeError("dm run failed")

        monkeypatch.setattr(orchestrator, "start_simulation_mode", _boom)
        f = tmp_path / "c.yaml"
        f.write_text(_DM_YAML)
        with pytest.raises(RuntimeError, match="dm run failed"):
            _maybe_run_dm_campaign(f, _args(), debug=False, entity_ref=None)

    def test_a_scenario_yaml_is_left_for_the_scenario_path(self, tmp_path, monkeypatch):
        import maxim.simulation.dm_schema as dm_schema

        def _must_not_load(*_a, **_kw):
            raise AssertionError("a scenario must not be loaded as a campaign")

        monkeypatch.setattr(dm_schema, "load_campaign", _must_not_load)
        f = tmp_path / "s.yaml"
        f.write_text(_SCENARIO_YAML)
        assert _maybe_run_dm_campaign(f, _args(), debug=False, entity_ref=None) is None
