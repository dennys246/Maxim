"""Curated recall — "what Maxim remembers about you" as a provenance-filtered,
salience-ranked, readable blend.

**A general recall abstraction, NOT episode-specific.** A :class:`RecallSource`
surfaces :class:`RecalledItem`s (readable text + salience + provenance + kind)
from *any* store; :func:`curate_recall` applies the three consumer-facing rules
uniformly so every source inherits them:

1. **Provenance filter** — drop ``real=False`` items (imagined / in-fiction
   memories stay hidden; they're decaying anyway).
2. **Rank by salience, not recency** — "your rogue betrayed the party" over
   "you entered a room".
3. **Under-claim / summarize** — cap per kind, require readable text, return
   little when little is known (an empty recall is correct, not a failure).

The episodic memory join (Episode provenance+valence × the memory record's
readable text) is the first source; a conservative NAc trait source is the
second. Semantic facts, player-model stores, etc. plug in behind the same
protocol without touching the curator — that is the point of the abstraction.
The consumer (``api.recall`` → the Console MemoryView) sees only the curated
blend, never raw episodes or NAc floats (the thin-app rule).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@dataclass
class RecalledItem:
    """One curated thing Maxim remembers — the common currency across sources.

    ``text`` is always a readable summary (never node ids or raw floats).
    ``real`` is the provenance gate: ``False`` = imagined/in-fiction, dropped by
    the curator. ``salience`` is the ranking signal (higher surfaces first).
    ``kind`` routes the item into a bucket (``"story"`` / ``"trait"`` /
    ``"preference"`` / future kinds).
    """

    text: str
    kind: str
    salience: float = 0.0
    real: bool = True
    when: float | None = None
    learned_from: str | None = None


@runtime_checkable
class RecallSource(Protocol):
    """Any store that can surface consumer-shaped recalled items.

    Implementations read their OWN store and emit :class:`RecalledItem`s;
    provenance filtering, ranking and capping are the curator's job, so a source
    only needs to tag ``real`` correctly and produce readable ``text``.
    """

    def recalled_items(self, *, limit: int) -> list[RecalledItem]: ...


@dataclass
class CuratedRecall:
    """The blended result — one bucket per kind. Empty fields are honest (Maxim
    genuinely remembers little), never an error."""

    name: str | None = None
    player_model: list[str] = field(default_factory=list)
    story_memories: list[RecalledItem] = field(default_factory=list)
    preferences: list[RecalledItem] = field(default_factory=list)


def curate_recall(
    sources: "list[RecallSource]",
    *,
    name: str | None = None,
    per_kind_limit: int = 8,
    min_salience: float = 0.0,
) -> CuratedRecall:
    """Collect from every source and apply the three rules once, centrally.

    A source that raises is skipped (a broken source must not sink the whole
    recall). Items with no readable text or ``real=False`` are dropped; the rest
    are ranked by salience (NOT recency) and capped per kind.
    """
    items: list[RecalledItem] = []
    for source in sources:
        try:
            items.extend(source.recalled_items(limit=per_kind_limit * 2))
        except Exception as e:  # noqa: BLE001 - one broken source must not sink recall
            logger.warning("recall source %r failed, skipping: %s", type(source).__name__, e)

    # Rule 1 (provenance) + under-claim (require text) + threshold.
    kept = [it for it in items if it.real and it.text and it.salience >= min_salience]
    # Rule 2 (rank by salience, not recency).
    kept.sort(key=lambda it: it.salience, reverse=True)

    def _bucket(kind: str) -> list[RecalledItem]:
        return [it for it in kept if it.kind == kind][:per_kind_limit]

    return CuratedRecall(
        name=name,
        player_model=[it.text for it in _bucket("trait")],
        story_memories=_bucket("story"),
        preferences=_bucket("preference"),
    )


# ── concrete sources ─────────────────────────────────────────────────────────


class EpisodicRecallSource:
    """Story memories: non-imagined episodes (provenance + valence) joined to
    their memory records (readable text + salience).

    The join is best-effort — an episode's ``activated_nodes`` are resolved
    through ``hippocampus.get`` to whatever readable records exist; an episode
    with no resolvable text is skipped (under-claim), never emitted as node ids.
    Provenance is honored at the source: ``imagined`` episodes are never yielded.
    """

    def __init__(self, hippocampus: Any) -> None:
        self._h = hippocampus

    def recalled_items(self, *, limit: int) -> list[RecalledItem]:
        store = getattr(self._h, "_episode_store", None)
        if store is None or not hasattr(store, "all_episodes"):
            return []
        out: list[RecalledItem] = []
        for ep in store.all_episodes():
            if getattr(ep, "imagined", False):
                continue  # RULE 1 at the source: in-fiction stays hidden
            text, rec_salience = self._summarize(ep)
            if not text:
                continue  # under-claim: no readable text → skip, don't emit ids
            # Rank by the stronger of episode valence magnitude and the joined
            # record salience — both are "this mattered", vs recency.
            salience = max(abs(float(getattr(ep, "valence", 0.0) or 0.0)), rec_salience)
            out.append(RecalledItem(text=text, kind="story", salience=salience, real=True))
        return out

    def _summarize(self, ep: Any) -> tuple[str, float]:
        """Resolve an episode's nodes → readable record text + max salience.

        Returns ``("", 0.0)`` when nothing resolves (episode is skipped). Takes
        the first readable fragment, truncated — summarize, don't dump.
        """
        best_text = ""
        best_salience = 0.0
        for node in getattr(ep, "activated_nodes", ()) or ():
            try:
                rec = self._h.get(node)
            except Exception:  # noqa: BLE001
                rec = None
            if rec is None:
                continue
            text = (
                getattr(rec, "cli_input", None)
                or getattr(rec, "transcript", None)
                or getattr(getattr(rec, "context", None), "goal", None)
                or ""
            )
            if text and not best_text:
                best_text = str(text).strip()[:200]
            best_salience = max(best_salience, float(getattr(rec, "salience", 0.0) or 0.0))
        return best_text, best_salience


class NacTraitSource:
    """Player-model traits from NAc reward biases — deliberately conservative.

    NAc biases are the AGENT's learned action tendencies; turning them into
    "traits about you" is an under-claim by design. Only biases above a high
    confidence gate are surfaced, phrased generically. The gate + the phrasing
    are the tuning points the seams plan flags as open (trait vocabulary,
    confidence threshold); until they're designed this source stays quiet.
    """

    def __init__(self, nac: Any, *, min_abs_bias: float = 0.5) -> None:
        self._nac = nac
        self._gate = min_abs_bias

    def recalled_items(self, *, limit: int) -> list[RecalledItem]:
        get_biases = getattr(self._nac, "get_agent_tool_biases", None)
        agents = getattr(self._nac, "known_agent_ids", None)
        if get_biases is None or agents is None:
            return []  # under-claim: no clean read path → say nothing
        out: list[RecalledItem] = []
        try:
            agent_ids = list(agents() if callable(agents) else agents)
        except Exception:  # noqa: BLE001
            return []
        for agent_id in agent_ids:
            try:
                for tool_sig, bias in get_biases(agent_id=agent_id, top_n=limit):
                    if abs(bias) < self._gate:
                        continue  # confidence gate — under-claim below it
                    tool = tool_sig.replace("tool:", "")
                    verb = "gravitates toward" if bias > 0 else "steers clear of"
                    out.append(
                        RecalledItem(
                            text=f"{verb} {tool}",
                            kind="trait",
                            salience=abs(bias),
                            real=True,
                            learned_from="learned reward",
                        )
                    )
            except Exception:  # noqa: BLE001
                continue
        return out
