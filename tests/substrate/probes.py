"""Probe functions for persistence round-trip testing.

Each probe takes a dict of {component_name: loaded_instance} and returns
a JSON-serializable result. Probes MUST NOT close over module-level state —
they run in a fresh subprocess where the only state is what was deserialized.

Naming convention: probe functions start with the component they test,
e.g., hippocampus_episode_count, nac_link_count, atl_node_count.
"""

from __future__ import annotations

from typing import Any


def hippocampus_episode_count(state: dict[str, Any]) -> int:
    """Count episodes in the hippocampus."""
    h = state.get("hippocampus")
    if h is None:
        return 0
    return len(h)


def hippocampus_episode_signatures(state: dict[str, Any]) -> list[str]:
    """Return sorted list of episode content hashes for deterministic comparison."""
    h = state.get("hippocampus")
    if h is None:
        return []
    episodes = h.recall(k=1000)
    return sorted(str(getattr(e, "content", ""))[:100] for e in episodes)


def nac_link_count(state: dict[str, Any]) -> int:
    """Count total causal links in NAc."""
    n = state.get("nac")
    if n is None:
        return 0
    return sum(len(v) for v in n._links.values())


def nac_link_signatures(state: dict[str, Any]) -> list[dict[str, Any]]:
    """Return sorted list of causal link summaries."""
    n = state.get("nac")
    if n is None:
        return []
    links = []
    for sig_links in n._links.values():
        for link in sig_links:
            links.append(
                {
                    "event": getattr(link, "event_signature", ""),
                    "outcome": getattr(link, "outcome_signature", ""),
                    "confidence": round(getattr(link, "confidence", 0), 6),
                    "observations": getattr(link, "observation_count", 0),
                }
            )
    return sorted(links, key=lambda x: (x["event"], x["outcome"]))


def atl_node_count(state: dict[str, Any]) -> int:
    """Count semantic nodes in ATL."""
    a = state.get("atl")
    if a is None:
        return 0
    return len(a) if hasattr(a, "__len__") else 0


def percept_trace_entry_count(state: dict[str, Any]) -> int:
    """Count active entries in the percept trace buffer."""
    ptb = state.get("percept_trace_buffer")
    if ptb is None:
        return 0
    return len(ptb)


def combined_state_summary(state: dict[str, Any]) -> dict[str, Any]:
    """Aggregate summary of all available components — the kitchen-sink probe."""
    return {
        "hippocampus_episodes": hippocampus_episode_count(state),
        "nac_links": nac_link_count(state),
        "atl_nodes": atl_node_count(state),
        "percept_trace_entries": percept_trace_entry_count(state),
    }
