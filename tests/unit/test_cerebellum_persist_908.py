"""#908: the Cerebellum is saved where it is loaded from, through the real bio-stack.

``build_bio_stack`` built ``Cerebellum(config=CerebellumConfig())`` and loaded ``<home>/cerebellum.json``
by a hard-coded path, while ``BioStack.save_cerebellum`` saved only to ``config.persistence_path``, which
nothing set. So every session-end save was a no-op and forward-model learning was lost each session. The
three call sites that "save" at session end guarded a no-op. Owner decision 2026-10-04: once the
Cerebellum persists, it also gets the #971 store guard.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from maxim.exceptions import StoreOverwriteRefused


_RED_908 = pytest.mark.xfail(strict=True, reason="#908: the Cerebellum is never saved, and has no store guard")


def _stack(home: Path, **kw):
    from maxim.runtime.bio_stack import build_bio_stack

    return build_bio_stack(persistence_dir=str(home), agent_id="default_agent", **kw)


def _close(stack) -> None:
    stack.on_session_end()
    stack.memory_hub.shutdown()


def _train(cerebellum) -> None:
    for load in (0.40, 0.44, 0.42):
        cerebellum.observe_from_action("body/arm", "motor", "lift", {"force": 0.5}, {"arm.load": load})


def _models(cerebellum) -> dict:
    return cerebellum.export_state()["models"]


@_RED_908
def test_forward_models_survive_a_session_through_the_real_bio_stack(tmp_path: Path) -> None:
    first = _stack(tmp_path)
    _train(first.cerebellum)
    learned = _models(first.cerebellum)
    assert learned
    _close(first)

    assert (tmp_path / "cerebellum.json").exists()
    second = _stack(tmp_path)
    try:
        assert _models(second.cerebellum) == learned
    finally:
        _close(second)


def test_a_write_but_dont_read_stack_restores_nothing(tmp_path: Path) -> None:
    first = _stack(tmp_path)
    _train(first.cerebellum)
    _close(first)

    fresh = _stack(tmp_path, load_persisted=False)
    try:
        assert _models(fresh.cerebellum) == {}
    finally:
        _close(fresh)


@_RED_908
def test_an_unreadable_file_is_kept_as_a_copy_and_the_stack_starts_empty(tmp_path: Path) -> None:
    (tmp_path / "cerebellum.json").write_text('{"models": {"a|b|c|d": "not-a-model"}}')
    stack = _stack(tmp_path)
    try:
        assert _models(stack.cerebellum) == {}
        copies = list(tmp_path.glob("cerebellum.json.corrupt-*"))
        assert len(copies) == 1
        assert "not-a-model" in copies[0].read_text()
    finally:
        _close(stack)


@_RED_908
def test_a_cerebellum_never_saves_over_a_file_it_did_not_read(tmp_path: Path) -> None:
    from maxim.embodiment.cerebellum import Cerebellum, CerebellumConfig

    path = tmp_path / "cerebellum.json"
    writer = Cerebellum(CerebellumConfig(persistence_path=str(path)))
    _train(writer)
    writer.save()
    before = path.read_bytes()

    stranger = Cerebellum(CerebellumConfig(persistence_path=str(path)))
    with pytest.raises(StoreOverwriteRefused):
        stranger.save()
    assert path.read_bytes() == before

    reader = Cerebellum(CerebellumConfig(persistence_path=str(path)))
    assert reader.load() is True
    reader.save()  # it read the file, so it may write it


@_RED_908
def test_a_refused_session_end_save_logs_at_error_and_still_cleans_up(tmp_path: Path, monkeypatch, caplog) -> None:
    """A load that hit an OSError leaves the file unowned; the session-end save is then refused. That must be
    loud (ERROR, like every auto-save site) and must not skip the distributor's cleanup."""
    import logging

    from maxim.embodiment.cerebellum import Cerebellum

    (tmp_path / "cerebellum.json").write_text("{}")

    def _unreachable(self, path=None):
        raise OSError("disk went away")

    monkeypatch.setattr(Cerebellum, "load", _unreachable)
    stack = _stack(tmp_path)
    cleaned: list[bool] = []
    monkeypatch.setattr(stack.distributor, "cleanup_session", lambda: cleaned.append(True))
    try:
        with caplog.at_level(logging.ERROR):
            stack.on_session_end()
    finally:
        stack.memory_hub.shutdown()
    assert cleaned == [True]
    assert any("Cerebellum not saved" in r.getMessage() and r.levelno == logging.ERROR for r in caplog.records)
    assert (tmp_path / "cerebellum.json").read_text() == "{}"


@_RED_908
def test_a_failed_session_end_write_is_logged_and_does_not_abort_the_session_end(
    tmp_path: Path, monkeypatch, caplog
) -> None:
    """A harness stages its measured NAc/EC right after ``on_session_end``: a Cerebellum write failure (state
    nothing reads yet) must be loud but must not abort it, nor skip the distributor's cleanup."""
    import logging

    from maxim.embodiment.cerebellum import Cerebellum

    stack = _stack(tmp_path)
    cleaned: list[bool] = []
    monkeypatch.setattr(stack.distributor, "cleanup_session", lambda: cleaned.append(True))

    def _disk_full(self, path=None, *, overwrite=False):
        raise OSError("disk full")

    monkeypatch.setattr(Cerebellum, "save", _disk_full)
    try:
        with caplog.at_level(logging.ERROR):
            stack.on_session_end()
        assert cleaned == [True]
        assert any("Cerebellum not saved (write failed)" in r.getMessage() for r in caplog.records)
    finally:
        stack.memory_hub.shutdown()


@_RED_908
def test_a_defect_in_the_session_end_save_still_runs_the_cleanup(tmp_path: Path, monkeypatch) -> None:
    """A defect (not a refusal or an OSError) propagates, but never skips the distributor's cleanup."""
    from maxim.embodiment.cerebellum import Cerebellum

    stack = _stack(tmp_path)
    cleaned: list[bool] = []
    monkeypatch.setattr(stack.distributor, "cleanup_session", lambda: cleaned.append(True))

    def _defect(self, path=None, *, overwrite=False):
        raise RuntimeError("dictionary changed size during iteration")

    monkeypatch.setattr(Cerebellum, "save", _defect)
    try:
        with pytest.raises(RuntimeError):
            stack.on_session_end()
        assert cleaned == [True]
    finally:
        stack.memory_hub.shutdown()
