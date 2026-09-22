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
    ConfigurationError,
    MaximConfig,
    MemoryConfigSection,
    resolve_memory_strategy,
)

# `strength` is NOT valid until 2c-3 ships the strategy: accepting it earlier would take the
# setting and then crash at the first consolidation.
_VALID = ("access_based", "importance_based", "composite")


def test_todays_model_is_the_default():
    assert MaximConfig().memory.strategy == "access_based"
    assert resolve_memory_strategy() == "access_based"
    assert HippocampusConfig().memory_strategy == "access_based"
    assert ATLConfig().memory_strategy == "access_based"


@pytest.mark.parametrize("name", _VALID)
def test_every_valid_name_is_accepted(name):
    assert MemoryConfigSection(strategy=name).strategy == name


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
    shared-builder silent no-op): each one is checked on the BUILT object."""
    monkeypatch.setenv("MAXIM_MEMORY_STRATEGY", "composite")

    import maxim.create as create_api
    from maxim.runtime.bio_stack import build_bio_stack

    assert create_api.hippocampus().config.memory_strategy == "composite"
    assert create_api.atl().config.memory_strategy == "composite"

    bio = build_bio_stack(agent_id="test-agent", persistence_dir=str(tmp_path / "bio"))
    assert bio.hippocampus.config.memory_strategy == "composite"
    # NOT `or bio.atl is None`: build_bio_stack swallows ATL construction failures, so a dropped
    # kwarg would make that arm pass vacuously.
    assert bio.atl is not None and bio.atl.config.memory_strategy == "composite"


def test_the_agent_factory_threads_it_too(monkeypatch, tmp_path):
    monkeypatch.setenv("MAXIM_MEMORY_STRATEGY", "composite")
    from maxim.runtime.agent_factory import AgentConfig, AgentFactory

    instance = AgentFactory(base_data_dir=tmp_path).create_agent(AgentConfig(agent_id="probe"))
    try:
        assert instance.hippocampus.config.memory_strategy == "composite"
        if getattr(instance, "atl", None) is not None:
            assert instance.atl.config.memory_strategy == "composite"
    finally:
        instance.shutdown()  # the factory starts a worker thread


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
