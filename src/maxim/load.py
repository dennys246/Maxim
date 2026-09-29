"""Deserialization namespace for persisted Maxim objects.

Provides ``maxim.load.*`` functions that reconstruct subsystems, agents,
and sessions from disk.  This is the **single canonical way** to restore
persisted state.  ``maxim.create.*`` always makes new, empty objects.

Example::

    import maxim

    hippo = maxim.load.hippocampus("/path/to/hippocampus.json")
    agent = maxim.load.agent("scout")
    session = maxim.load.session("20260408")

    for s in maxim.load.sessions(limit=5):
        print(s.id, s.goal)
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from maxim.decisions.nac import NAc
    from maxim.memory.atl import ATL
    from maxim.memory.hippocampus import Hippocampus
    from maxim.runtime.agent_factory import AgentInstance
    from maxim.session import Session

__all__ = ["hippocampus", "nac", "atl", "session", "sessions", "agent", "entity"]


def _unreadable_errors() -> tuple[type[Exception], ...]:
    """What a store's loader raises for a file it cannot parse or reconstruct: the one shared definition
    (``memory/store.py::UNREADABLE_STORE_ERRORS``), imported lazily like every facade import.
    OSError (permissions, a directory) is not corruption and propagates as itself."""
    from maxim.memory.store import UNREADABLE_STORE_ERRORS

    return UNREADABLE_STORE_ERRORS


def _existing_store_file(path: str, label: str) -> str:
    """The path with ``~`` expanded -- the SAME path that is then loaded (#950: the check expanded
    it and the loader did not, so ``~/...`` loaded an empty store). Raises ``FileNotFoundError``."""
    import os

    from maxim.utils.paths import store_file_path

    resolved = store_file_path(path)
    if not os.path.exists(resolved):
        raise FileNotFoundError(f"{label} file not found: {path}")
    return resolved


def _corrupt(label: str, path: str, exc: Exception) -> Exception:
    """The ``MemoryCorruptionError`` ``load.agent`` raises for the same file (#939)."""
    from maxim.exceptions import MemoryCorruptionError

    return MemoryCorruptionError(
        f"{label} file {path} could not be read ({type(exc).__name__}: {exc}); repair or move it",
        context={"path": path, "subsystem": label},
    )


def hippocampus(path: str) -> "Hippocampus":
    """Load a Hippocampus from a persisted JSON file.

    Args:
        path: Path to the hippocampus JSON file.

    Returns:
        ``Hippocampus`` instance with memories restored.

    Raises:
        FileNotFoundError: If the file does not exist.
        MemoryCorruptionError: If the file cannot be read as a Hippocampus.

    Example::

        hippo = maxim.load.hippocampus("~/.maxim/agents/scout/hippocampus.json")
        memories = hippo.recall(query="wolf", limit=5)
    """
    resolved = _existing_store_file(path, "Hippocampus")

    # The retention model is runtime POLICY, not persisted state: a loaded store scores by the
    # CURRENT ``memory.strategy``, not whatever was set when the file was written (contrast
    # ``load.nac``, which deliberately skips decay-on-load so a resumed run is not double-decayed).
    from maxim.memory.hippocampus import Hippocampus, HippocampusConfig
    from maxim.runtime.config_loader import resolve_hippocampus_memory_kwargs

    h = Hippocampus(HippocampusConfig(**resolve_hippocampus_memory_kwargs()))
    try:
        h.load(resolved)
    except _unreadable_errors() as e:
        raise _corrupt("Hippocampus", resolved, e) from e
    return h


def nac(path: str) -> "NAc":
    """Load a NAc from a persisted JSON file.

    Args:
        path: Path to the NAc JSON file.

    Returns:
        ``NAc`` instance with causal links restored.

    Raises:
        FileNotFoundError: If the file does not exist.
        MemoryCorruptionError: If the file cannot be read as a NAc.

    Example::

        nac = maxim.load.nac("~/.maxim/agents/scout/nac.json")
        prediction = nac.predict("action", "ate_mushroom")
    """
    resolved = _existing_store_file(path, "NAc")

    from maxim.decisions.nac import NAc

    n = NAc()
    # apply_decay=False: a read-only load must report disk truth — the
    # same file inspected on two different days must give the same
    # numbers, and a load→save round-trip must not compound decay.
    try:
        n.load(resolved, apply_decay=False)
    except _unreadable_errors() as e:
        raise _corrupt("NAc", resolved, e) from e
    return n


def atl(path: str) -> "ATL":
    """Load an ATL from a persisted JSON file.

    Args:
        path: Path to the ATL JSON file.

    Returns:
        ``ATL`` instance with semantic concepts restored.

    Raises:
        FileNotFoundError: If the file does not exist.
        MemoryCorruptionError: If the file cannot be read as an ATL.

    Example::

        atl = maxim.load.atl("~/.maxim/agents/scout/atl.json")
        concepts = atl.recall(limit=10)
    """
    resolved = _existing_store_file(path, "ATL")

    # As in ``load.hippocampus``: the model is current policy, never restored from the file.
    from maxim.memory.atl import ATL, ATLConfig
    from maxim.runtime.config_loader import resolve_memory_strategy

    a = ATL(ATLConfig(memory_strategy=resolve_memory_strategy()))
    try:
        a.load(resolved)
    except _unreadable_errors() as e:
        raise _corrupt("ATL", resolved, e) from e
    return a


def session(session_id: str) -> "Session":
    """Load a persisted simulation session by ID.

    This is the canonical way to access past sessions. ``session_id`` is an ID
    or a path, resolved like every other run directory
    (``utils/paths.py::resolve_run_dir``); an ID that names no directory may be
    a unique prefix (e.g. ``"20260408"`` matches ``"20260408_143022"``).

    Args:
        session_id: Session ID, unique ID prefix, or path to the session directory.

    Returns:
        ``Session`` with metadata loaded from report.json.

    Raises:
        FileNotFoundError: If no matching session is found.
        maxim.RunDirAmbiguous: If the ID or prefix matches more than one
            session (a ``MaximMemoryError`` and a ``ValueError``; the message lists them).

    Example::

        session = maxim.load.session("20260408")
        print(session.goal)
        memories = session.observe("memory")
    """
    from maxim.session import Session

    return Session.from_disk(session_id)


def sessions(*, limit: int = 20) -> "list[Session]":
    """List recent simulation sessions.

    Returns Session objects with metadata loaded from report.json,
    most recent first.

    Args:
        limit: Maximum number of sessions to return.

    Returns:
        List of Session objects.

    Example::

        for s in maxim.load.sessions(limit=5):
            print(f"{s.id}: {s.goal} ({s.turns} turns)")
    """
    from maxim.session import list_sessions

    return list_sessions(limit=limit)


def agent(
    name: str,
    *,
    base_dir: str | None = None,
    on_corrupt: str = "raise",
) -> "AgentInstance":
    """Load a persisted agent by name.

    Restores Hippocampus, NAc, ATL and SCN from the agent's persistence
    directory before returning. This function either hands back a fully
    restored agent or raises — it will not silently give you fresh state
    wearing a loaded agent's name (D17).

    Args:
        name: Agent identifier (matches the name used in ``create.agent()``).
        base_dir: Override the base agent directory (default ``~/.maxim/agents``).
        on_corrupt: What to do when a persisted file cannot be read.
            ``"raise"`` (default) aborts with a
            :class:`~maxim.exceptions.MemoryCorruptionError` naming every bad
            file. ``"fresh"`` is the explicit opt-in to start those subsystems
            empty. An unreadable Hippocampus or ATL file is copied to
            ``<name>.corrupt-<UTC timestamp>`` beside it (logged) and the agent
            saves fresh state in its place, so nothing is destroyed by the
            choice. NAc, EC and SCN files are NOT copied: their unreadable file
            is overwritten at the next save, with no copy (until #971).

    Returns:
        ``AgentInstance`` with persisted state restored.

    Raises:
        FileNotFoundError: If no persisted state found for this agent.
        ValueError: If ``on_corrupt`` is not ``"raise"`` or ``"fresh"``.
        MemoryCorruptionError: If a persisted file is unreadable and
            ``on_corrupt="raise"``.

    Example::

        agent = maxim.load.agent("scout")
        # agent.hippocampus has memories from previous sessions
        memories = agent.export_memories()

        # Accept fresh state for whatever could not be read:
        agent = maxim.load.agent("scout", on_corrupt="fresh")
    """
    if on_corrupt not in ("raise", "fresh"):
        raise ValueError(f"on_corrupt must be 'raise' or 'fresh', got {on_corrupt!r}")

    from pathlib import Path

    from maxim.runtime.agent_factory import AgentConfig, AgentFactory

    if base_dir:
        base_dir = str(Path(base_dir).expanduser())  # `~` means home (#950)
        factory = AgentFactory(base_data_dir=base_dir)
        agent_dir = Path(base_dir) / name
    else:
        factory = AgentFactory()
        from maxim.utils.paths import RUN_DIR_KINDS, data_home

        agent_dir = data_home() / RUN_DIR_KINDS["agent"] / name

    if not agent_dir.exists():
        raise FileNotFoundError(
            f"No persisted agent '{name}' found at {agent_dir}. Create one first with maxim.create.agent('{name}')."
        )

    # Create agent with persistence_dir pointing at existing state.
    # auto_load=True tells AgentFactory to restore from disk.
    config = AgentConfig(
        agent_id=name,
        persistence_dir=str(agent_dir),
        # "fresh" maps to the factory's loud-but-continue mode; the corrupt
        # file is reported as a WARNING either way.
        on_corrupt="raise" if on_corrupt == "raise" else "warn",
    )
    return factory.create_agent(config, auto_load=True)


def entity(path: str) -> Any:
    """Load a SEM entity from a YAML spec or saved JSON file.

    Supports both YAML specs (from component templates) and JSON files
    (from ``Entity.save()``).

    Args:
        path: Path to a YAML or JSON entity file.

    Returns:
        ``Entity`` object with full tree structure restored.

    Example::

        entity = maxim.load.entity("my_robot.yaml")
        body = maxim.create.embodiment(entity)

        # Or load a previously saved entity
        entity = maxim.load.entity("saved_guard.json")
    """
    if path.endswith(".json"):
        from maxim.embodiment.sem import Entity

        return Entity.load(path)
    else:
        from maxim.embodiment.spec import load_spec

        return load_spec(path)
