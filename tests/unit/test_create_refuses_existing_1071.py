"""#1071: ``create.hippocampus`` / ``create.atl`` refuse an existing store at construction (owner decision 2026-10-04).

``create.*`` means a NEW store. Before, an existing path opened silently empty and the refusal came only at
``save()`` (#939's guard), so the caller worked against an empty store it believed was its own. Now construction
raises ``StoreOverwriteRefused`` and names ``maxim.load.*``; ``overwrite=True`` declares a deliberate replace.
"""

from __future__ import annotations

import pytest

import maxim
from maxim.exceptions import StoreOverwriteRefused
from maxim.memory.encoding import EncodingSignals
from maxim.memory.types import Outcome, Perception

ENC = EncodingSignals.unmeasured("api")


def _hippocampus_file(path) -> None:
    h = maxim.create.hippocampus(persistence_path=str(path))
    for i in range(3):
        h.capture(perception=Perception(salience=0.5, observations={"text": f"m{i}"}), outcome=Outcome(), encoding=ENC)
    h.save()


def _atl_file(path) -> None:
    a = maxim.create.atl(persistence_path=str(path))
    a.find_or_create("wolf", category="creature")
    a.save()


@pytest.mark.parametrize(
    "make,create,load",
    [
        (_hippocampus_file, maxim.create.hippocampus, maxim.load.hippocampus),
        (_atl_file, maxim.create.atl, maxim.load.atl),
    ],
)
def test_create_refuses_an_existing_store_at_construction(tmp_path, make, create, load) -> None:
    path = tmp_path / "store.json"
    make(path)
    before = path.read_bytes()
    with pytest.raises(StoreOverwriteRefused, match="maxim.load"):
        create(persistence_path=str(path))
    assert path.read_bytes() == before  # nothing touched
    assert load(str(path)) is not None  # the store is still there to load


def test_a_tilde_path_is_checked_where_it_points(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    _hippocampus_file(tmp_path / "mem.json")
    with pytest.raises(StoreOverwriteRefused):
        maxim.create.hippocampus(persistence_path="~/mem.json")


def test_overwrite_declares_a_deliberate_replace(tmp_path) -> None:
    path = tmp_path / "store.json"
    _hippocampus_file(path)
    h = maxim.create.hippocampus(persistence_path=str(path), overwrite=True)
    h.save()  # declared: the replace goes through
    assert len(maxim.load.hippocampus(str(path)).recall(limit=10)) == 0


def test_overwrite_on_atl_and_its_declared_warning(tmp_path, caplog) -> None:
    import logging

    path = tmp_path / "atl.json"
    _atl_file(path)
    with caplog.at_level(logging.WARNING):
        a = maxim.create.atl(persistence_path=str(path), overwrite=True)
    assert any("without reading it (declared)" in r.getMessage() for r in caplog.records)
    a.save()
    assert len(maxim.load.atl(str(path))) == 0  # the declared replace dropped the old concept


def test_a_directory_at_the_path_is_named_as_one(tmp_path) -> None:
    (tmp_path / "d").mkdir()
    with pytest.raises(StoreOverwriteRefused, match="a directory is at"):
        maxim.create.hippocampus(persistence_path=str(tmp_path / "d"))


def test_a_new_path_and_no_path_still_create(tmp_path) -> None:
    assert maxim.create.hippocampus(persistence_path=str(tmp_path / "new.json")) is not None
    assert maxim.create.hippocampus() is not None
    assert maxim.create.atl(persistence_path=str(tmp_path / "new_atl.json")) is not None
