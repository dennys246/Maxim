"""`memory.strategy`: one resolver, and a typo can never quietly keep today's model.

memory-strength Phase 2c-1. The plan's guarantee is that selecting a retention model is explicit
and that today's default stays byte-identical until Phase 5 earns the flip — so an unknown name
RAISES at every door instead of falling back, and every store's config comes from one resolver.
"""

from __future__ import annotations

import pytest

from maxim.memory.atl import ATL, ATLConfig
from maxim.memory.hippocampus import Hippocampus, HippocampusConfig
from maxim.runtime.config_loader import (
    _VALID_MEMORY_STRATEGIES,
    ConfigurationError,
    MaximConfig,
    MemoryConfigSection,
    resolve_memory_strategy,
)

# DERIVED, not a second copy: a name added to the config set and to no store would otherwise leave
# both stores untested -- which is the 2c-1 defect's exact shape, one layer up (review round,
# Architecture N3). `strength` became valid in 2c-3, WITH the strategy that implements it.
_VALID = tuple(sorted(_VALID_MEMORY_STRATEGIES))


def test_todays_model_is_the_default():
    assert MaximConfig().memory.strategy == "access_based"
    assert resolve_memory_strategy() == "access_based"
    assert HippocampusConfig().memory_strategy == "access_based"
    assert ATLConfig().memory_strategy == "access_based"


@pytest.mark.parametrize("name", _VALID)
def test_every_valid_name_is_accepted(name):
    assert MemoryConfigSection(strategy=name).strategy == name


@pytest.mark.parametrize("name", _VALID)
def test_every_valid_name_builds_a_working_model_in_BOTH_stores(name):
    """A name config accepts and a store cannot build crashes at the first consolidation, far from
    the command that set it. Parametrized over the config's OWN set, so a new name cannot be added
    without answering for both stores."""
    from maxim.memory.strategies import MemoryStrategy

    hippo = Hippocampus(HippocampusConfig(persistence_path=None, memory_strategy=name))
    assert isinstance(hippo._get_memory_strategy(), MemoryStrategy)
    hippo.sleep()

    atl = ATL(ATLConfig(persistence_path=None, memory_strategy=name))
    assert isinstance(atl._get_memory_strategy(), MemoryStrategy)
    atl.consolidate()


def test_every_declared_section_is_actually_parsed():
    """The bug this file's own section shipped with, and the `console` bug before it: a section
    declared on MaximConfig passes the unknown-key gate and is then silently DROPPED at parse, so
    `maxim config set` writes a value the loader ignores and the next write erases. Structural, so
    the next section cannot repeat it a third time."""
    import ast
    import dataclasses
    import inspect
    import typing

    from maxim.runtime import config_loader

    # ``from __future__ import annotations`` makes f.type a STRING, so is_dataclass(f.type) is
    # always False — resolve the annotations instead of trusting the field object.
    hints = typing.get_type_hints(config_loader.MaximConfig)
    sections = [
        f.name for f in dataclasses.fields(config_loader.MaximConfig) if dataclasses.is_dataclass(hints[f.name])
    ]
    assert "memory" in sections and len(sections) > 5  # the walk below is only as good as this list
    source = inspect.getsource(config_loader._parse_config_dict)
    tree = ast.parse(source)
    read = {
        node.args[0].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "get"
        and node.args
        and isinstance(node.args[0], ast.Constant)
    }
    # ...and PASSED to the constructor: "parsed but not passed" has the identical symptom.
    passed = {
        kw.arg
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "MaximConfig"
        for kw in node.keywords
    }
    assert set(sections) <= read, f"declared but never parsed: {sorted(set(sections) - read)}"
    assert set(sections) <= passed, f"parsed but never passed to MaximConfig: {sorted(set(sections) - passed)}"


def test_the_setting_survives_a_write_then_load(tmp_path, monkeypatch):
    import json

    from maxim.runtime import config_writer
    from maxim.runtime.config_loader import load_config

    path = tmp_path / "config.json"
    monkeypatch.setattr(config_writer, "config_path", lambda: path)
    config_writer.set_field("memory.strategy", "composite")
    assert json.loads(path.read_text())["memory"]["strategy"] == "composite"
    assert load_config(path).memory.strategy == "composite"  # the loader must not drop it
    config_writer.set_field("llm.n_ctx", "4096")  # a later write must not erase it
    assert load_config(path).memory.strategy == "composite"


def test_a_typo_in_the_config_file_raises_rather_than_defaulting(tmp_path):
    import json

    from maxim.runtime.config_loader import load_config

    path = tmp_path / "config.json"
    path.write_text(json.dumps({"_format_version": "1.0", "memory": {"strategy": "strenght"}}))
    with pytest.raises(ConfigurationError, match="memory.strategy"):
        load_config(path)


def test_a_typo_raises_at_the_config_door():
    with pytest.raises(ConfigurationError, match="memory.strategy"):
        MemoryConfigSection(strategy="strenght")


def test_a_typo_raises_at_the_writer_door(tmp_path, monkeypatch):
    from maxim.runtime import config_writer

    monkeypatch.setattr(config_writer, "config_path", lambda: tmp_path / "config.json")
    with pytest.raises(ConfigurationError):
        config_writer.set_field("memory.strategy", "strenght")
    config_writer.set_field("memory.strategy", "composite")  # a real name still writes


def test_a_typo_raises_in_the_env_door(monkeypatch):
    monkeypatch.setenv("MAXIM_MEMORY_STRATEGY", "strenght")
    with pytest.raises(ConfigurationError):
        resolve_memory_strategy()
    monkeypatch.setenv("MAXIM_MEMORY_STRATEGY", "composite")
    assert resolve_memory_strategy() == "composite"


def test_a_typo_reaching_a_store_raises_rather_than_scoring_the_old_way():
    # the hole this closes: the selector used to fall through to access_based
    with pytest.raises(ValueError, match="unknown memory strategy"):
        Hippocampus(HippocampusConfig(persistence_path=None, memory_strategy="strenght"))._get_memory_strategy()
    with pytest.raises(ValueError, match="unknown memory strategy"):
        ATL(ATLConfig(persistence_path=None, memory_strategy="strenght"))._get_memory_strategy()


def test_a_missing_value_still_falls_back_to_the_default():
    from maxim.memory.strategies import AccessBasedStrategy

    # only a MISSING value falls back; that is the dataclass default, not a silent rescue
    assert isinstance(Hippocampus(HippocampusConfig(persistence_path=None))._get_memory_strategy(), AccessBasedStrategy)


def test_every_builder_threads_the_one_resolver(monkeypatch, tmp_path):
    """A builder that forgot the resolver would leave that store on the old model (the
    shared-builder silent no-op): each one is checked on the BUILT object.

    The strength KNOBS are checked on the same objects, and that is the whole reason
    ``resolve_hippocampus_memory_kwargs`` is a bundle rather than three resolvers: a builder that
    threaded the strategy and forgot ``memory.s_base`` would run the operator's chosen model on
    default tuning, silently.
    """
    monkeypatch.setenv("MAXIM_MEMORY_STRATEGY", "composite")
    monkeypatch.setenv("MAXIM_MEMORY_S_BASE", "1234.0")
    monkeypatch.setenv("MAXIM_MEMORY_K", "3.5")

    import maxim.create as create_api
    from maxim.runtime.bio_stack import build_bio_stack

    def check(hippo):
        assert hippo.config.memory_strategy == "composite"
        assert (hippo.config.strength_s_base, hippo.config.strength_k) == (1234.0, 3.5)

    check(create_api.hippocampus())
    assert create_api.atl().config.memory_strategy == "composite"  # ATL takes only the strategy

    bio = build_bio_stack(agent_id="test-agent", persistence_dir=str(tmp_path / "bio"))
    check(bio.hippocampus)
    # NOT `or bio.atl is None`: build_bio_stack swallows ATL construction failures, so a dropped
    # kwarg would make that arm pass vacuously.
    assert bio.atl is not None and bio.atl.config.memory_strategy == "composite"

    import maxim.load as load_api

    saved = tmp_path / "saved.json"
    create_api.hippocampus(persistence_path=str(saved)).save()
    check(load_api.hippocampus(str(saved)))


def test_the_agent_factory_threads_it_too(monkeypatch, tmp_path):
    monkeypatch.setenv("MAXIM_MEMORY_STRATEGY", "composite")
    monkeypatch.setenv("MAXIM_MEMORY_S_BASE", "1234.0")
    from maxim.runtime.agent_factory import AgentConfig, AgentFactory

    instance = AgentFactory(base_data_dir=tmp_path).create_agent(AgentConfig(agent_id="probe"))
    try:
        assert instance.hippocampus.config.memory_strategy == "composite"
        assert instance.hippocampus.config.strength_s_base == 1234.0
        if getattr(instance, "atl", None) is not None:
            assert instance.atl.config.memory_strategy == "composite"
    finally:
        instance.shutdown()  # the factory starts a worker thread


# ── the strength knobs (2c-3: the keys ship WITH the code that reads them) ────


def test_an_unset_knob_leaves_the_models_own_default_as_the_single_source():
    """`None` is not written into the schema as a number: the equation's default lives in
    ``memory/encoding.py`` alone, where it cannot drift out of step with the equation."""
    from maxim.memory.encoding import K_DEFAULT, S_BASE_DEFAULT
    from maxim.runtime.config_loader import resolve_hippocampus_memory_kwargs

    assert MaximConfig().memory.s_base is None
    assert MaximConfig().memory.k is None
    assert resolve_hippocampus_memory_kwargs() == {"memory_strategy": "access_based"}
    assert HippocampusConfig().strength_s_base == S_BASE_DEFAULT
    assert HippocampusConfig().strength_k == K_DEFAULT


def test_a_set_knob_reaches_the_stamp_through_the_resolver(monkeypatch):
    from maxim.memory.encoding import EncodingSignals
    from maxim.runtime.config_loader import resolve_hippocampus_memory_kwargs

    monkeypatch.setenv("MAXIM_MEMORY_S_BASE", "2000.0")
    monkeypatch.setenv("MAXIM_MEMORY_K", "0.0")
    hippo = Hippocampus(HippocampusConfig(persistence_path=None, **resolve_hippocampus_memory_kwargs()))
    memory = hippo.get(hippo.capture(encoding=EncodingSignals.unmeasured("loop")))
    assert memory.storage_strength == pytest.approx(2000.0)


@pytest.mark.parametrize("value", ["0", "-1", "nan", "banana"])
def test_an_unusable_s_base_raises_where_it_was_set(value, monkeypatch, tmp_path):
    """S divides dt: a zero would be a ZeroDivisionError at the first score, far from the command
    that set it, and on the async capture worker whose broad handler LOSES the memory."""
    from maxim.runtime import config_writer
    from maxim.runtime.config_loader import resolve_hippocampus_memory_kwargs

    monkeypatch.setenv("MAXIM_MEMORY_S_BASE", value)
    with pytest.raises(ConfigurationError):
        resolve_hippocampus_memory_kwargs()

    monkeypatch.delenv("MAXIM_MEMORY_S_BASE")
    monkeypatch.setattr(config_writer, "config_path", lambda: tmp_path / "config.json")
    with pytest.raises(ConfigurationError):
        config_writer.set_field("memory.s_base", value)


def test_an_unusable_knob_raises_at_the_config_door_too():
    """config_writer only coerces STRINGS, so a Python-API caller would slip past it."""
    with pytest.raises(ConfigurationError, match="memory.s_base"):
        MemoryConfigSection(s_base=0.0)
    with pytest.raises(ConfigurationError, match="memory.k"):
        MemoryConfigSection(k=-1.0)


def test_the_knobs_survive_a_write_then_load(tmp_path, monkeypatch):
    import json

    from maxim.runtime import config_writer
    from maxim.runtime.config_loader import load_config

    path = tmp_path / "config.json"
    monkeypatch.setattr(config_writer, "config_path", lambda: path)
    config_writer.set_field("memory.strategy", "strength")
    config_writer.set_field("memory.s_base", "5e6")
    assert json.loads(path.read_text())["memory"] == {"strategy": "strength", "s_base": 5e6, "k": None}
    loaded = load_config(path)
    assert (loaded.memory.strategy, loaded.memory.s_base) == ("strength", 5e6)


def test_a_nonsense_knob_in_the_config_file_raises_rather_than_defaulting(tmp_path):
    import json

    from maxim.runtime.config_loader import load_config

    path = tmp_path / "config.json"
    path.write_text(json.dumps({"_format_version": "1.0", "memory": {"s_base": -3.0}}))
    with pytest.raises(ConfigurationError, match="memory.s_base"):
        load_config(path)


# ── the name that landed with its strategy ───────────────────────────────────


def test_the_strength_name_reaches_a_working_model_in_every_store():
    """2c-1's review: a name accepted by config and unimplemented in the store crashes at the
    first consolidation, far from the command that set it. Both stores answer for it."""
    from maxim.memory.strategies import AccessBasedStrategy, StrengthStrategy

    hippo = Hippocampus(HippocampusConfig(persistence_path=None, memory_strategy="strength"))
    assert isinstance(hippo._get_memory_strategy(), StrengthStrategy)
    hippo.sleep()  # the consolidation that 2c-1's review said must not crash

    # The ATL keeps today's model under this name ON PURPOSE (plan decision 5: Phase 2's strength
    # model is the hippocampal one). Named, not fallen through: silence would be the band-aid.
    atl = ATL(ATLConfig(persistence_path=None, memory_strategy="strength"))
    assert isinstance(atl._get_memory_strategy(), AccessBasedStrategy)
    atl.consolidate()


def test_the_atl_honours_the_name_rather_than_validating_and_ignoring_it():
    from maxim.memory.strategies import AccessBasedStrategy, CompositeStrategy, ImportanceBasedStrategy

    def base_of(name: str):
        return ATL(ATLConfig(persistence_path=None, memory_strategy=name))._get_memory_strategy()

    assert isinstance(base_of("access_based"), AccessBasedStrategy)
    assert isinstance(base_of("importance_based"), ImportanceBasedStrategy)
    assert isinstance(base_of("composite"), CompositeStrategy)

    # ...and the honoured strategy can actually tell two concepts apart on a semantic record:
    # isinstance alone passed while every concept scored by age (the round-2 finding).
    import time

    from maxim.memory.semantic_types import Concept

    now = time.time()
    established = Concept(id="a", timestamp=now - 3 * 86400, created_at=now - 3 * 86400, confidence=0.9)
    established.reinforcement_count = 9
    barely = Concept(id="b", timestamp=now - 3 * 86400, created_at=now - 3 * 86400, confidence=0.2)
    scorer = base_of("importance_based")
    assert scorer.score_for_retention(established, now, 2) > scorer.score_for_retention(barely, now, 2)


def test_no_store_is_built_without_the_resolved_strategy():
    """Outside memory/ itself, a bare ``Hippocampus()`` / ``ATL()`` would silently keep the old
    model whatever the setting says — the shared-builder silent no-op, one layer up. Parsed, not
    grepped, so docstring examples do not count.

    A BELT, not the braces: it catches only the bare-call shape, in ``src/maxim/`` (tests and
    scripts legitimately build stores with no ambient config, and the conftest scrub pins them to
    the default). What proves each builder actually threads the resolver is the behavioural test
    above, on the built object."""
    import ast
    from pathlib import Path

    import maxim

    root = Path(maxim.__file__).resolve().parent
    offenders = []
    for path in root.rglob("*.py"):
        if path.parent.name == "memory":  # the stores' own internals resolve nothing (layering)
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in ("Hippocampus", "ATL")
                and not node.args
                and not node.keywords
            ):
                offenders.append(f"{path.relative_to(root)}:{node.lineno}")
    assert not offenders, "built without the resolved strategy: " + ", ".join(offenders)


def test_todays_retention_is_untouched_by_the_section(complete_memory_args):
    """The section exists, but the default path scores exactly as before."""
    hippo = Hippocampus(HippocampusConfig(persistence_path=None))
    mid = hippo.capture(**complete_memory_args)
    [record] = hippo.recall_by_ids([mid])
    strategy = hippo._get_memory_strategy()
    from maxim.memory.strategies import AccessBasedStrategy, TemporalAwareStrategy

    assert isinstance(strategy, (AccessBasedStrategy, TemporalAwareStrategy))
    # compression_age never enters score_for_retention, so pin it directly: a selector regression
    # to the ATL's 7-day value would otherwise score identically here.
    assert strategy.compression_age == hippo.config.compression_age
    # scored away from the degenerate point (aged, with a graph degree), so the numbers can differ
    aged = record.accessed_at + 3 * 86400
    assert strategy.score_for_retention(record, aged, 4) == pytest.approx(
        AccessBasedStrategy(
            max_age_without_access=hippo.config.max_age_without_access,
            compression_age=hippo.config.compression_age,
        ).score_for_retention(record, aged, 4)
    )
