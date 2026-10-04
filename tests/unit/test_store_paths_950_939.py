"""#950 + #939 -- a store opens the file it was pointed at, and never writes over one it did not read.

#950: `~` was expanded by the existence check but not by the loader, so `maxim.load.hippocampus("~/...")`
returned an EMPTY store, `load.nac("~/...")` raised, and a `persistence_path="~/..."` saved into a
literal `./~/` directory under the working directory.

#939: nothing stopped a store from saving over a file it never read. `create.hippocampus` /
`create.atl` with an existing path, the plain constructor, and `from_config` with the path on the
config all started empty and clobbered the file on the next save; `create.agent` over an existing
agent did the same across every store. `load.*` on a corrupt file raised a raw `JSONDecodeError`.

Owner decisions 2026-09-28: a guard at SAVE (a store refuses to overwrite an existing file it neither
read nor created, unless told to); `~` expanded everywhere; `load()` of a missing file raises;
`create.agent` refuses up front over an agent that already has persisted state.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from maxim.exceptions import MemoryCorruptionError, StoreOverwriteRefused


@pytest.fixture
def home(tmp_path: Path, monkeypatch) -> Path:
    """A temp HOME, and a separate working directory, so a literal `./~` would be visible."""
    home_dir, work_dir = tmp_path / "home", tmp_path / "work"
    home_dir.mkdir()
    work_dir.mkdir()
    monkeypatch.setenv("HOME", str(home_dir))
    monkeypatch.chdir(work_dir)
    return home_dir


def _hippo(**kw):
    from maxim.memory.hippocampus import Hippocampus, HippocampusConfig

    kw.setdefault("auto_save_after_sleep", False)
    return Hippocampus(HippocampusConfig(**kw))


def _store_three(path: Path) -> None:
    h = _hippo(persistence_path=str(path))
    for text in ("wolf at the cave", "river is cold", "berries are safe"):
        h.store_observation(text)
    h.save()


def _count_on_disk(path: Path) -> int:
    return len(json.loads(path.read_text())["memories"])


def _atl(**kw):
    from maxim.memory.atl import ATL, ATLConfig

    return ATL(ATLConfig(**kw))


def _store_concepts(path: Path, names=("wolf", "cave")) -> None:
    a = _atl(persistence_path=str(path))
    for name in names:
        a.find_or_create(name, "object")
    a.save()


def _concepts_on_disk(path: Path) -> int:
    return len(json.loads(path.read_text())["concepts"])


# ── #950: `~` means the home directory, everywhere ─────────────────────────────────────────────


def test_load_hippocampus_expands_the_home_directory(home: Path) -> None:
    import maxim

    _store_three(home / "hippocampus.json")
    assert len(maxim.load.hippocampus("~/hippocampus.json")) == 3  # was 0: an empty store


def test_load_nac_and_atl_expand_the_home_directory(home: Path) -> None:
    import maxim
    from maxim.decisions.nac import NAc, NACConfig

    NAc(NACConfig(persistence_path=str(home / "nac.json"))).save()
    assert maxim.load.nac("~/nac.json") is not None  # raised FileNotFoundError
    _store_concepts(home / "atl.json")
    assert len(maxim.load.atl("~/atl.json")) == 2


def test_a_tilde_persistence_path_saves_under_home_not_the_working_directory(home: Path) -> None:
    h = _hippo(persistence_path="~/mem/hippocampus.json")
    h.store_observation("a note")
    h.save()
    a = _atl(persistence_path="~/mem/atl.json")
    a.find_or_create("wolf", "object")
    a.save()
    assert (home / "mem" / "hippocampus.json").exists() and (home / "mem" / "atl.json").exists()
    assert not (Path.cwd() / "~").exists()  # the literal ./~ directory


def test_an_agent_persistence_dir_expands_the_home_directory(home: Path) -> None:
    from maxim.runtime.agent_factory import AgentConfig, AgentFactory

    AgentFactory()._resolve_persistence_dir(AgentConfig(agent_id="scout", persistence_dir="~/agents/scout"))
    assert (home / "agents" / "scout").is_dir() and not (Path.cwd() / "~").exists()


# ── #939: a store never writes over a file it did not read ─────────────────────────────────────


def test_create_hippocampus_over_an_existing_store_refuses_to_clobber_it(tmp_path: Path) -> None:
    import maxim

    path = tmp_path / "memory.json"
    _store_three(path)
    # #1071: create.* now refuses at construction (before, at save(), after work against an empty store).
    with pytest.raises(StoreOverwriteRefused, match="load"):
        maxim.create.hippocampus(persistence_path=str(path), auto_save_after_sleep=False)
    assert _count_on_disk(path) == 3  # was 1: two memories lost without a word


def test_create_atl_over_an_existing_store_refuses_to_clobber_it(tmp_path: Path) -> None:
    import maxim

    path = tmp_path / "atl.json"
    _store_concepts(path)
    with pytest.raises(StoreOverwriteRefused):  # #1071: at construction
        maxim.create.atl(persistence_path=str(path))
    assert _concepts_on_disk(path) == 2


def test_the_plain_constructor_cannot_clobber_either(tmp_path: Path) -> None:
    """The issue's comment: the clobber sits in the constructor, one level below `create`."""
    path = tmp_path / "memory.json"
    _store_three(path)
    h = _hippo(persistence_path=str(path))
    h.store_observation("x")
    with pytest.raises(StoreOverwriteRefused):
        h.save()
    with pytest.raises(StoreOverwriteRefused):
        h.save(str(path))  # an explicit path is no different
    assert _count_on_disk(path) == 3


def test_a_store_that_read_its_file_saves_back_to_it(tmp_path: Path) -> None:
    import maxim

    path = tmp_path / "memory.json"
    _store_three(path)
    h = maxim.load.hippocampus(str(path))
    h.store_observation("a fourth")
    h.save(str(path))
    assert _count_on_disk(path) == 4


def test_from_config_reads_a_path_given_only_on_the_config(tmp_path: Path) -> None:
    from maxim.memory.hippocampus import Hippocampus, HippocampusConfig

    path = tmp_path / "memory.json"
    _store_three(path)
    h = Hippocampus.from_config(HippocampusConfig(persistence_path=str(path), auto_save_after_sleep=False))
    assert len(h) == 3  # was 0: the path on the config was never read
    h.store_observation("a fourth")
    h.save()
    assert _count_on_disk(path) == 4


def test_a_store_that_created_its_file_keeps_saving_to_it(tmp_path: Path) -> None:
    path = tmp_path / "new.json"
    h = _hippo(persistence_path=str(path))
    h.store_observation("one")
    h.save()
    h.store_observation("two")
    h.save()
    assert _count_on_disk(path) == 2


def test_overwrite_is_an_explicit_choice(tmp_path: Path) -> None:
    path = tmp_path / "memory.json"
    _store_three(path)
    h = _hippo(persistence_path=str(path))
    h.store_observation("x")
    h.save(overwrite=True)
    assert _count_on_disk(path) == 1

    other = tmp_path / "atl.json"
    _store_concepts(other)
    a = _atl(persistence_path=str(other))
    a.allow_overwrite()  # the write-but-don't-read declaration
    a.save()
    assert _concepts_on_disk(other) == 0


def test_the_sleep_auto_save_refusal_is_loud_and_does_not_lose_the_sleep(tmp_path: Path, caplog) -> None:
    from maxim.memory.hippocampus import Hippocampus, HippocampusConfig

    path = tmp_path / "memory.json"
    _store_three(path)
    h = Hippocampus(HippocampusConfig(persistence_path=str(path), auto_save_after_sleep=True))
    h.store_observation("x")
    with caplog.at_level(logging.ERROR):
        results = h.sleep()
    assert isinstance(results, dict)
    assert _count_on_disk(path) == 3
    assert any("refus" in r.getMessage().lower() for r in caplog.records if r.levelno >= logging.ERROR)


def test_the_write_but_dont_read_agent_declares_its_overwrite(tmp_path: Path) -> None:
    """The sim NPC (`load_persisted=False`) saves over last run's files on purpose; it says so."""
    from maxim.runtime.bio_stack import build_bio_stack

    _store_three(tmp_path / "hippocampus.json")  # last run's files: the NPC must be able to replace them
    _store_concepts(tmp_path / "atl.json")
    stack = build_bio_stack(persistence_dir=str(tmp_path), agent_id="default_agent", load_persisted=False)
    try:
        assert len(stack.hippocampus) == 0  # it did not read them
        assert stack.hippocampus.may_write(str(tmp_path / "hippocampus.json"))
        assert stack.atl.may_write(str(tmp_path / "atl.json"))
    finally:
        shutdown = getattr(stack.memory_hub, "shutdown", None)
        if callable(shutdown):
            shutdown()


# ── #939: create.agent over an existing agent ─────────────────────────────────────────────────


def test_create_agent_refuses_an_agent_that_already_has_persisted_state(tmp_path: Path) -> None:
    import maxim

    first = maxim.create.agent("scout", persistence_dir=str(tmp_path / "scout"))
    first.shutdown()
    before = {p.name: p.read_bytes() for p in (tmp_path / "scout").glob("*.json")}
    assert before, "the first agent persisted nothing; the refusal below would be vacuous"

    with pytest.raises(FileExistsError, match="load.agent"):
        maxim.create.agent("scout", persistence_dir=str(tmp_path / "scout"))
    after = {p.name: p.read_bytes() for p in (tmp_path / "scout").glob("*.json")}
    assert after == before  # nothing built, nothing written


def test_create_agent_in_an_empty_home_still_works(tmp_path: Path) -> None:
    import maxim

    (tmp_path / "fresh").mkdir()  # an empty directory is not persisted state
    maxim.create.agent("fresh", persistence_dir=str(tmp_path / "fresh")).shutdown()


# ── #939: load.* report corruption as MemoryCorruptionError ────────────────────────────────────


@pytest.mark.parametrize("which", ["hippocampus", "nac", "atl"])
def test_load_of_a_corrupt_file_raises_memory_corruption_error(tmp_path: Path, which: str) -> None:
    import maxim

    path = tmp_path / f"{which}.json"
    path.write_text("{not json")
    with pytest.raises(MemoryCorruptionError, match=str(path.name)):
        getattr(maxim.load, which)(str(path))


# ── missing files: load() raises; `missing_ok` is the explicit load-if-present ─────────────────


def test_load_of_a_missing_file_raises(tmp_path: Path) -> None:
    missing = str(tmp_path / "nope.json")
    with pytest.raises(FileNotFoundError):
        _hippo().load(missing)
    with pytest.raises(FileNotFoundError):
        _atl().load(missing)
    _hippo().load(missing, missing_ok=True)  # the explicit load-if-present form
    _atl().load(missing, missing_ok=True)


def test_a_fresh_agent_session_start_does_not_warn_about_a_missing_atl(tmp_path: Path, caplog) -> None:
    """`MemoryHub.on_session_start` loads the ATL if present; a new agent has none, which is normal."""
    import maxim

    with caplog.at_level(logging.WARNING):
        agent = maxim.create.agent("newcomer", persistence_dir=str(tmp_path / "newcomer"))
        agent.shutdown()
    assert not any("Failed to load ATL" in r.getMessage() for r in caplog.records)


# ── review folds: the recovery policy, the remaining `~` paths, the exception ──────────────────


def test_recovery_from_a_corrupt_file_keeps_a_copy_then_saves(tmp_path: Path) -> None:
    """Owner decision: the corrupt file is copied to `<name>.corrupt-<UTC>` and the store, started
    empty, persists normally (preserving it forever left the agent half-persisted until #971)."""
    path = tmp_path / "memory.json"
    path.write_text("{not json")
    h = _hippo(persistence_path=str(path))
    ok, err = h.load_with_recovery()
    assert ok is True and err is not None and len(h) == 0
    kept = list(tmp_path.glob("memory.json.corrupt-*"))
    assert len(kept) == 1 and kept[0].read_text() == "{not json"
    h.store_observation("new")
    h.save()
    assert _count_on_disk(path) == 1


def test_restoring_from_the_backup_claims_the_primary(tmp_path: Path) -> None:
    """`restore_backup` is the caller's choice to replace the corrupt primary with the backup's state."""
    path = tmp_path / "memory.json"
    _store_three(path)
    (tmp_path / "memory.json.backup").write_text(path.read_text())
    path.write_text("{not json")
    h = _hippo(persistence_path=str(path))
    ok, _ = h.load_with_recovery(on_error="restore_backup")
    assert ok is True and len(h) == 3
    h.save()
    assert _count_on_disk(path) == 3


def test_atl_load_safe_keeps_a_copy_of_a_corrupt_file_then_saves(tmp_path: Path) -> None:
    path = tmp_path / "atl.json"
    path.write_text("{not json")
    a = _atl(persistence_path=str(path))
    ok, _ = a.load_safe()
    assert ok is False
    assert [p.read_text() for p in tmp_path.glob("atl.json.corrupt-*")] == ["{not json"]
    a.find_or_create("wolf", "object")
    a.save()
    assert _concepts_on_disk(path) == 1


def test_a_copy_that_cannot_be_made_leaves_the_file_unsaved_over(tmp_path: Path, monkeypatch) -> None:
    """If the evidence cannot be kept, the store must not destroy the only copy."""
    import shutil

    path = tmp_path / "memory.json"
    path.write_text("{not json")

    def _fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(shutil, "copy2", _fail)
    h = _hippo(persistence_path=str(path))
    h.load_with_recovery()
    h.store_observation("x")
    with pytest.raises(StoreOverwriteRefused):
        h.save()
    assert path.read_text() == "{not json"


def test_load_agent_fresh_keeps_a_copy_and_the_agent_saves(tmp_path: Path) -> None:
    import maxim

    maxim.create.agent("scout", persistence_dir=str(tmp_path / "scout")).shutdown()
    hippo_file = tmp_path / "scout" / "hippocampus.json"
    hippo_file.write_text("{not json")
    agent = maxim.load.agent("scout", base_dir=str(tmp_path), on_corrupt="fresh")
    agent.hippocampus.store_observation("after the fresh start")
    agent.shutdown()
    assert [p.read_text() for p in (tmp_path / "scout").glob("hippocampus.json.corrupt-*")] == ["{not json"]
    assert _count_on_disk(hippo_file) == 1


def test_a_tilde_bio_stack_home_keeps_one_directory_across_runs(home: Path) -> None:
    """`build_bio_stack(persistence_dir="~/...")`: every store under $HOME, and run 2 reads run 1."""
    from maxim.runtime.bio_stack import build_bio_stack

    def _run(note: str) -> int:
        stack = build_bio_stack(persistence_dir="~/agentx", agent_id="default_agent")
        try:
            restored = len(stack.hippocampus)
            stack.hippocampus.store_observation(note)
            stack.hippocampus.save()
            return restored
        finally:
            shutdown = getattr(stack.memory_hub, "shutdown", None)
            if callable(shutdown):
                shutdown()

    assert _run("one") == 0
    assert _run("two") == 1  # run 2 found run 1's memories (it looked under $HOME)
    assert not (Path.cwd() / "~").exists()


def test_save_with_backup_refuses_before_taking_the_backup(tmp_path: Path) -> None:
    path = tmp_path / "memory.json"
    _store_three(path)
    h = _hippo(persistence_path=str(path))
    with pytest.raises(StoreOverwriteRefused):
        h.save_with_backup()
    assert not (tmp_path / "memory.json.backup").exists()


def test_load_agent_and_the_factory_base_dir_expand_the_home_directory(home: Path) -> None:
    import maxim
    from maxim.runtime.agent_factory import AgentFactory

    maxim.create.agent("scout", persistence_dir=str(home / "agents" / "scout")).shutdown()
    agent = maxim.load.agent("scout", base_dir="~/agents")
    agent.shutdown()
    AgentFactory(base_data_dir="~/other")
    assert (home / "other").is_dir() and not (Path.cwd() / "~").exists()


def test_create_agent_ignores_unrelated_files_in_its_directory(tmp_path: Path) -> None:
    """Only the stores' own files count as persisted agent state, not any JSON that happens to be there."""
    import maxim

    home_dir = tmp_path / "scout"
    home_dir.mkdir()
    (home_dir / "notes.json").write_text("{}")
    maxim.create.agent("scout", persistence_dir=str(home_dir)).shutdown()


def test_the_refusal_is_a_picklable_file_exists_error_naming_the_file() -> None:
    import copy
    import pickle

    e = StoreOverwriteRefused("refused", path="/data/m.json", store="hippocampus")
    for clone in (pickle.loads(pickle.dumps(e)), copy.copy(e)):
        assert isinstance(clone, FileExistsError) and clone.path == "/data/m.json" and str(clone) == "refused"
    assert e.filename == "/data/m.json"  # what a plain FileExistsError handler reads


def test_the_same_corrupt_file_is_copied_once(tmp_path: Path) -> None:
    """create_full_agent's factory and bio stack both meet it; one copy of identical bytes is enough."""
    path = tmp_path / "memory.json"
    path.write_text("{not json")
    _hippo(persistence_path=str(path)).load_with_recovery()
    _hippo(persistence_path=str(path)).load_with_recovery()
    assert len(list(tmp_path.glob("memory.json.corrupt-*"))) == 1


# ── verification-pass folds: a failed load leaves nothing behind; one definition of "unreadable" ─


def test_a_partly_loaded_hippocampus_is_emptied_before_it_saves(tmp_path: Path) -> None:
    """The load fails AFTER memories were swapped in; "starting empty" must be true before the store
    may save over the file, or a partial rebuild replaces it labelled fresh."""
    path = tmp_path / "memory.json"
    _store_three(path)
    data = json.loads(path.read_text())
    data["episodes"] = [{"bogus": 1}]
    path.write_text(json.dumps(data))
    h = _hippo(persistence_path=str(path))
    ok, err = h.load_with_recovery()
    assert ok is True and err is not None
    assert len(h) == 0
    h.save()
    assert _count_on_disk(path) == 0


def test_an_atl_file_that_is_not_an_object_is_copied_aside_like_any_corrupt_file(tmp_path: Path) -> None:
    """`[]` raises AttributeError, which ATL.load_safe used not to treat as corruption."""
    path = tmp_path / "atl.json"
    path.write_text("[]")
    a = _atl(persistence_path=str(path))
    ok, _ = a.load_safe()
    assert ok is False and a.may_write()


def test_an_unreachable_file_is_never_licensed_for_replacement(tmp_path: Path, monkeypatch) -> None:
    """An OSError on load (EIO, permissions) is not corruption: the file may be perfectly good."""
    from maxim.memory import hippocampus_persistence as hp

    path = tmp_path / "memory.json"
    _store_three(path)
    real_open = open

    def _eio(file, *args, **kwargs):
        if str(file) == str(path):
            raise OSError(5, "I/O error")
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(hp, "open", _eio, raising=False)
    h = _hippo(persistence_path=str(path))
    h.load_with_recovery()
    monkeypatch.undo()
    assert not h.may_write()
    assert list(tmp_path.glob("memory.json.corrupt-*")) == []


def test_a_recurring_corruption_leaves_one_copy_per_occurrence(tmp_path: Path) -> None:
    """Dedup covers two loaders in one construction, not a corruption that happens again later."""
    from maxim.utils import store_ownership as store_mod  # the registry's home (#971)

    path = tmp_path / "memory.json"
    path.write_text("")
    _hippo(persistence_path=str(path)).load_with_recovery()
    store_mod._COPIES_MADE_THIS_PROCESS.clear()  # a later run is a new process
    path.write_text("")
    _hippo(persistence_path=str(path)).load_with_recovery()
    assert len(list(tmp_path.glob("memory.json.corrupt-*"))) == 2
