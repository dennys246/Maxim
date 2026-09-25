"""Provenance at export (owner decision (c), 2026-09-25): round trips through the REAL export
(`compose_bundle`) and the REAL ingest (`ingest_bundle`).

Before: an exporter that had ever ingested someone else's material shipped their ids verbatim, and
every receiver REFUSED the bundle (V1 accepts only the manifest's own contributor or "local").
Default now: an agent exports ONLY its own learning. ``reauthor=True`` (release composition, the
Queen publishing merged contributions): every row is re-stamped as the author's.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

from maxim.hivemind.bundle import compose_bundle
from maxim.hivemind.ingest import IngestionJournal
from maxim.hivemind.merge import NAC_KEY_SEP
from tests.unit.test_hivemind_ingest import BODY, DONOR, _ingest, _link, _nac_state, _node

UPSTREAM = ("alice-private-id", "bob-private-id")


def _hub_state():
    """A hub that learned one link itself, merged one with others, and ingested one node."""
    own = _link("tool:probe")
    own_by_id = _link("tool:mine", source=DONOR, contributors=[DONOR])  # stamped with the exporter's id
    merged = _link("tool:other", source="_consensus", contributors=["local", *UPSTREAM])
    received = _link("tool:heard", source=UPSTREAM[1], contributors=[])  # foreign by SOURCE alone
    co_voted = _link("tool:joint", source="local", contributors=["local", UPSTREAM[0]])  # by CONTRIBUTORS alone
    nac = _nac_state(
        links={
            "tool:probe": [own],
            "tool:mine": [own_by_id],
            "tool:other": [merged],
            "tool:heard": [received],
            "tool:joint": [co_voted],
        },
        cluster_fear={
            NAC_KEY_SEP.join(("agent", "n_own", "drive:oxygen")): -0.5,
            NAC_KEY_SEP.join(("agent", "n_foreign", "drive:oxygen")): -0.7,
        },
    )
    ec = {"n_own": _node(), "n_foreign": _node(source=UPSTREAM[0], contributors=[UPSTREAM[0]])}
    return nac, ec


def _export(tmp_path: Path, *, reauthor: bool) -> Path:
    nac, ec = _hub_state()
    out = tmp_path / ("release.zip" if reauthor else "contribution.zip")
    compose_bundle(
        nac_state=nac,
        ec_substrate_nodes=ec,
        output_path=out,
        contributor_id=DONOR,
        body_ref=BODY,
        apply_identity_filter=False,
        reauthor=reauthor,
    )
    return out


def _slices(path: Path) -> tuple[dict, dict, str]:
    with zipfile.ZipFile(path) as z:
        raw = "".join(z.read(n).decode() for n in z.namelist())
        return json.loads(z.read("nac.json")), json.loads(z.read("ec.json")), raw


def test_by_default_an_agent_exports_only_its_own_learning_and_the_bundle_is_admitted(tmp_path):
    bundle = _export(tmp_path, reauthor=False)
    nac, ec, raw = _slices(bundle)
    assert set(nac["links"]) == {"tool:probe", "tool:mine"}  # own (local or own id) kept; merged + received dropped
    assert set(ec["substrate_nodes"]) == {"n_own"}  # the ingested node is dropped
    assert all("n_foreign" not in k for k in nac.get("cluster_fear", {}))  # and rows naming it
    assert not any(u in raw for u in UPSTREAM)  # no upstream id is published
    _ingest(bundle, IngestionJournal(tmp_path / "j.json"))  # a receiver ADMITS it (raised before)


def test_a_release_reauthors_every_row_and_is_admitted_without_leaking_upstream_ids(tmp_path):
    bundle = _export(tmp_path, reauthor=True)
    nac, ec, raw = _slices(bundle)
    assert set(nac["links"]) == {"tool:probe", "tool:mine", "tool:other", "tool:heard", "tool:joint"}
    assert set(ec["substrate_nodes"]) == {"n_own", "n_foreign"}
    rows = [link for links in nac["links"].values() for link in links] + list(ec["substrate_nodes"].values())
    assert all(r["source"] == "local" and r["contributors"] == [] for r in rows)
    assert not any(u in raw for u in UPSTREAM) and "_consensus" not in raw
    _ingest(bundle, IngestionJournal(tmp_path / "j.json"))


def test_a_purely_local_state_exports_unchanged_either_way(tmp_path):
    nac = _nac_state(links={"tool:probe": [_link()]})
    for reauthor in (False, True):
        out = tmp_path / f"b{reauthor}.zip"
        compose_bundle(
            nac_state=nac,
            ec_substrate_nodes={"n1": _node()},
            output_path=out,
            contributor_id=DONOR,
            body_ref=BODY,
            apply_identity_filter=False,
            reauthor=reauthor,
        )
        shipped, ec, _ = _slices(out)
        assert shipped["links"]["tool:probe"][0]["source"] == "local" and set(ec["substrate_nodes"]) == {"n1"}


def test_the_cli_refuses_an_unsigned_release_and_reports_what_a_contribution_drops(tmp_path, capsys):
    from maxim.hivemind.cli import run_substrate_subcommand

    nac, ec = _hub_state()
    session = tmp_path / "session"
    session.mkdir()
    (session / "aut_nac.json").write_text(json.dumps(nac))
    (session / "aut_ec.json").write_text(json.dumps({"substrate_nodes": ec}))
    base = ["export", "--session", str(session), "--contributor-id", DONOR, "--no-identity-filter"]
    assert run_substrate_subcommand([*base, "--release", str(tmp_path / "r.zip")]) == 2
    assert "must be signed" in capsys.readouterr().err
    assert run_substrate_subcommand([*base, str(tmp_path / "c.zip")]) == 0
    assert "dropped 3 link(s) and 1 EC node(s)" in capsys.readouterr().out
