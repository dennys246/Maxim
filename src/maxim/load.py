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


def hippocampus(path: str) -> "Hippocampus":
    """Load a Hippocampus from a persisted JSON file.

    Args:
        path: Path to the hippocampus JSON file.

    Returns:
        ``Hippocampus`` instance with memories restored.

    Raises:
        FileNotFoundError: If the file does not exist.

    Example::

        hippo = maxim.load.hippocampus("~/.maxim/agents/scout/hippocampus.json")
        memories = hippo.recall(query="wolf", limit=5)
    """
    from pathlib import Path as _Path

    if not _Path(path).expanduser().exists():
        raise FileNotFoundError(f"Hippocampus file not found: {path}")

    from maxim.memory.hippocampus import Hippocampus

    h = Hippocampus()
    h.load(path)
    return h


def nac(path: str) -> "NAc":
    """Load a NAc from a persisted JSON file.

    Args:
        path: Path to the NAc JSON file.

    Returns:
        ``NAc`` instance with causal links restored.

    Raises:
        FileNotFoundError: If the file does not exist.

    Example::

        nac = maxim.load.nac("~/.maxim/agents/scout/nac.json")
        prediction = nac.predict("action", "ate_mushroom")
    """
    from pathlib import Path as _Path

    if not _Path(path).expanduser().exists():
        raise FileNotFoundError(f"NAc file not found: {path}")

    from maxim.decisions.nac import NAc

    n = NAc()
    # apply_decay=False: a read-only load must report disk truth — the
    # same file inspected on two different days must give the same
    # numbers, and a load→save round-trip must not compound decay.
    n.load(path, apply_decay=False)
    return n


def atl(path: str) -> "ATL":
    """Load an ATL from a persisted JSON file.

    Args:
        path: Path to the ATL JSON file.

    Returns:
        ``ATL`` instance with semantic concepts restored.

    Raises:
        FileNotFoundError: If the file does not exist.

    Example::

        atl = maxim.load.atl("~/.maxim/agents/scout/atl.json")
        concepts = atl.recall(limit=10)
    """
    from pathlib import Path as _Path

    if not _Path(path).expanduser().exists():
        raise FileNotFoundError(f"ATL file not found: {path}")

    from maxim.memory.atl import ATL

    a = ATL()
    a.load(path)
    return a


def session(session_id: str) -> "Session":
    """Load a persisted simulation session by ID.

    This is the canonical way to access past sessions.  Supports fuzzy
    matching on session_id prefix (e.g. ``"20260408"`` matches
    ``"20260408_143022"``).

    Args:
        session_id: Full or partial session ID.

    Returns:
        ``Session`` with metadata loaded from report.json.

    Raises:
        FileNotFoundError: If no matching session is found.

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
            empty; the corrupt file stays on disk until the agent next saves,
            so nothing is destroyed by the choice.

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
        factory = AgentFactory(base_data_dir=base_dir)
        agent_dir = Path(base_dir) / name
    else:
        factory = AgentFactory()
        from maxim.utils.paths import data_home

        agent_dir = data_home() / "agents" / name

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
