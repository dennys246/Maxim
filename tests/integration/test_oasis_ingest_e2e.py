"""End-to-end proof for the 1.2 Oasis ingestion path, on the real archive.

The contract's §8 closing requirement: compose the taught seed-43 bundle
from the SHA-manifested ``docs/experiments/data/53_agents/`` archive via the
REAL CLI export (``--body-ref --body-yaml``), and ingest it into a FRESH
``create.agent()`` receiver (D28: actually fresh) through the REAL CLI
ingest verb — V3's default refusal on the pre-stamp archive, the dry-run
default, the journal, the replay refusal, and the DV read at the real
consumer (``NAc.cluster_reward_bias`` after a real ``load_state``), where
the taught positive biases must land byte-untouched.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
ARCHIVE = REPO / "docs" / "experiments" / "data" / "53_agents" / "taught_seed43"

# The `--body-yaml` flag takes an EMBODIMENT-spec YAML (`body:` root, the
# load_spec shape); the infant_operant body exists only as a SEM component
# (`_data/components/bodies/infant_operant.yaml`, loaded through the
# component registry — a different loader). This spec mirrors the real
# component's naming exactly (entity `infant_operant`, modulator `orient`,
# affordances `turn_left`/`turn_right` from base_humanoid), so
# derive_capability_map runs the REAL tool-naming path and emits the very
# `tool:infant_operant_turn_*` keys the archive's biases carry.
_INFANT_OPERANT_BODY_SPEC = """\
body:
  name: infant_operant
  entity_type: body
  modulators:
    orient:
      abstract: true
      affordances:
        turn_left:
          params: {}
          description: "Turn to face left."
        turn_right:
          params: {}
          description: "Turn to face right."
"""


@pytest.fixture()
def api_home(tmp_path, monkeypatch):
    """An isolated MAXIM_DATA_HOME with the path caches reset around it."""
    from maxim.utils.paths import _reset_caches

    monkeypatch.setenv("MAXIM_DATA_HOME", str(tmp_path))
    monkeypatch.setenv("MAXIM_LLM_ENABLED", "0")
    _reset_caches()
    yield tmp_path
    _reset_caches()


def test_taught_seed43_ingests_into_fresh_agent_end_to_end(api_home, caplog):
    from maxim import create
    from maxim.decisions.nac import NAc, NACConfig
    from maxim.hivemind.cli import run_substrate_subcommand

    caplog.set_level(logging.WARNING)

    # 1. Compose the bundle from the archive via the real CLI export path.
    body_yaml = api_home / "infant_operant_body.yaml"
    body_yaml.write_text(_INFANT_OPERANT_BODY_SPEC, encoding="utf-8")
    bundle = api_home / "taught_seed43.zip"
    rc = run_substrate_subcommand(
        [
            "export",
            str(bundle),
            "--session",
            str(ARCHIVE),
            "--contributor-id",
            "nursery-43",
            "--body-ref",
            "infant_operant",
            "--body-yaml",
            str(body_yaml),
        ]
    )
    assert rc == 0, "real-archive export must succeed"
    assert bundle.is_file()

    # The derived capability map carries the very keys the archive's biases
    # key on (gate 7's forward-insurance half).
    from maxim.hivemind.bundle import read_bundle_manifest

    manifest = read_bundle_manifest(bundle)
    assert manifest["capability_map"].get("tool:infant_operant_turn_left") == "orient/turn_left"

    # 2. A FRESH receiver through the real API (D28 made this actually fresh).
    agent = create.agent("receiver43")
    agent.shutdown()
    receiver_dir = api_home / "agents" / "receiver43"
    assert (receiver_dir / "nac.json").is_file()
    assert (receiver_dir / "ec.json").is_file()

    common = [
        "ingest",
        str(bundle),
        "--session",
        str(receiver_dir),
        "--trust",
        "nursery-43",
        "--receiver-body",
        "infant_operant",
        "--receiver-agent-id",
        "receiver43",
    ]
    journal_path = receiver_dir / "substrate_ingest_journal.json"

    # 3. V3 default: the pre-stamp archive's unstamped nodes REFUSE without
    # the explicit legacy override.
    assert run_substrate_subcommand(list(common)) == 2

    # 4. Dry-run by default: rc 0, and NOTHING was written.
    nac_bytes_before = (receiver_dir / "nac.json").read_bytes()
    assert run_substrate_subcommand(common + ["--allow-unstamped-geometry"]) == 0
    assert not journal_path.exists()
    assert (receiver_dir / "nac.json").read_bytes() == nac_bytes_before

    # 5. Apply.
    assert run_substrate_subcommand(common + ["--allow-unstamped-geometry", "--apply"]) == 0
    journal = json.loads(journal_path.read_text())
    assert journal["_format_version"] == "1.0"
    assert len(journal["entries"]) == 1
    assert journal["entries"][0]["contributor_id"] == "nursery-43"
    assert journal["entries"][0]["biases_tightened"] == 0  # taught = positive only
    assert (receiver_dir / "nac.json.pre-ingest.bak").is_file()

    # 6. The DV at the real consumer: a real NAc loads the receiver's state
    # and reads the taught want. Fresh receiver → donor nodes insert under
    # their own ids → biases re-key on the agent axis only, and the
    # POSITIVE fold is byte-untouched (the sign-scope guarantee).
    donor_nac = json.loads((ARCHIVE / "aut_nac.json").read_text())
    donor_ec_ids = set(json.loads((ARCHIVE / "aut_ec.json").read_text())["substrate_nodes"].keys())
    expected = {}
    for key, value in donor_nac["cluster_reward_bias"].items():
        _aid, cid, tsig = key.split("\x1f")
        if cid in donor_ec_ids:  # a bias whose cluster shipped (D2: others drop)
            expected[(cid, tsig)] = value
    assert len(expected) >= 2, "the taught archive must carry cluster biases with their clusters"

    receiver_state = json.loads((receiver_dir / "nac.json").read_text())
    receiver_state.pop("_format_version", None)
    nac = NAc(config=NACConfig())
    nac.load_state(receiver_state)
    for (cid, tsig), value in expected.items():
        assert nac.cluster_reward_bias("receiver43", cid, tsig) == value, (
            f"taught bias for ({cid[:8]}…, {tsig}) must land byte-untouched at the real consumer"
        )

    # The donor's EC nodes are all present in the receiver's substrate.
    receiver_ec = json.loads((receiver_dir / "ec.json").read_text())
    assert donor_ec_ids <= set(receiver_ec["substrate_nodes"].keys())

    # 7. Replay of the same digest refuses (V8) — idempotence by refusal.
    assert run_substrate_subcommand(common + ["--allow-unstamped-geometry", "--apply"]) == 2
    assert len(json.loads(journal_path.read_text())["entries"]) == 1
