"""Roy iteration log + protocol scaffolding.

R4 of the Roy long-horizon harness build (see
``docs/plans/persona_convergence_crucible.md`` "Iteration log" + "What
we record per Roy iteration"). Every Roy iteration produces two
artifacts:

1. **Protocol runbook** at ``docs/experiments/protocols/roy_<name>.md``
   — a reproduction recipe with command lines, expected outputs, and
   divergence thresholds. Mirrors the pattern set by
   ``p2_reward_modulation_reproduction.md``.
2. **Iteration log entry** appended to the relevant plan doc
   (``persona_convergence_crucible.md`` or
   ``grounded_language_acquisition.md``) under its
   ``## Iteration log`` section.

Both are intentionally drafts: the operator reviews and edits before
committing. The generator handles the boilerplate — what command was
run, what session ids resulted, what the divergence numbers were —
and leaves the qualitative judgement calls (what we learned, what
took, what we'd change) blank with TODO markers.

Idempotence: each generated section is wrapped in HTML comment markers
(``<!-- roy-iteration:<id> -->`` / ``<!-- /roy-iteration:<id> -->``)
so re-running the generator replaces the prior block rather than
appending a duplicate. The operator's edits between the markers ARE
preserved when ``--keep-edits`` is passed; the default replaces the
whole block from the latest ``RoyIterationResult``.

Plan auto-detection by iteration-name prefix is intentionally
permissive: a misnamed iteration returns ``None`` and the CLI prompts
the operator. Add new prefixes here as new personas join the program.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Plan auto-detection
# ---------------------------------------------------------------------------


_PLAN_PREFIX_MAP: tuple[tuple[str, str], ...] = (
    # Order matters — longest prefix wins on overlap.
    ("roy-cradle-", "docs/plans/grounded_language_acquisition.md"),
    ("roy-cautious-", "docs/plans/persona_convergence_crucible.md"),
    ("roy-adversarial-", "docs/plans/persona_convergence_crucible.md"),
    ("roy-explorer-", "docs/plans/persona_convergence_crucible.md"),
    ("roy-collector-", "docs/plans/persona_convergence_crucible.md"),
    ("roy-hider-", "docs/plans/persona_convergence_crucible.md"),
    # Generic numeric/methodology iterations land on the crucible doc.
    ("roy-0-", "docs/plans/persona_convergence_crucible.md"),
    ("roy-1-", "docs/plans/persona_convergence_crucible.md"),
    ("roy-2-", "docs/plans/persona_convergence_crucible.md"),
    ("roy-3-", "docs/plans/persona_convergence_crucible.md"),
    ("roy-4-", "docs/plans/persona_convergence_crucible.md"),
    ("roy-5-", "docs/plans/persona_convergence_crucible.md"),
)


def detect_plan_path(iteration_name: str, repo_root: Path | None = None) -> Path | None:
    """Return the plan doc that should receive this iteration's log entry.

    Picks based on the iteration-name prefix. Returns ``None`` when no
    rule matches so the CLI can prompt the operator for an explicit
    plan path.
    """
    root = repo_root or _detect_repo_root()
    if root is None:
        return None
    # Sort by descending prefix length so "roy-cradle-" beats "roy-".
    rules = sorted(_PLAN_PREFIX_MAP, key=lambda kv: -len(kv[0]))
    for prefix, rel in rules:
        if iteration_name.startswith(prefix):
            return root / rel
    return None


def _detect_repo_root() -> Path | None:
    """Walk up from this file until we find a ``docs/plans`` sibling.

    Returns ``None`` when the package is installed (no source tree to
    write into). Callers degrade by asking the operator for an
    explicit path.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "docs" / "plans").is_dir():
            return parent
    return None


# ---------------------------------------------------------------------------
# Result loading
# ---------------------------------------------------------------------------


@dataclass
class LoadedResult:
    """A ``result.json`` payload loaded from
    ``~/.maxim/roy/<iteration_id>/result.json`` plus its artifact path.

    The runner persists a JSON-shaped ``RoyIterationResult`` (see
    ``simulation/roy_runner.py::_result_to_dict``). We load it back
    duck-typed so this generator stays decoupled from any future
    schema migration — adding fields to ``RoyIterationResult`` does
    NOT require touching this loader.
    """

    iteration_id: str
    payload: dict[str, Any]
    artifact_dir: Path
    result_path: Path


def load_iteration_result(
    iteration_id: str,
    artifact_root: Path | str | None = None,
) -> LoadedResult:
    """Load a persisted ``result.json`` for the named iteration.

    Parameters
    ----------
    iteration_id:
        Iteration name as it appears on disk — matches
        ``RoyIterationResult.name`` and the
        ``~/.maxim/roy/<name>/`` directory.
    artifact_root:
        Override for the iteration root (defaults to ``~/.maxim/roy``).
        Tests pass a tmp dir.
    """
    if artifact_root is None:
        from maxim.utils.paths import data_home

        base = data_home() / "roy"
    else:
        base = Path(artifact_root)
    artifact_dir = base / iteration_id
    result_path = artifact_dir / "result.json"
    if not result_path.is_file():
        raise FileNotFoundError(
            f"No Roy iteration result at {result_path}. Run `maxim roy run <spec>` to produce it first."
        )
    with result_path.open(encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Malformed result.json at {result_path}: expected mapping")
    return LoadedResult(
        iteration_id=iteration_id,
        payload=payload,
        artifact_dir=artifact_dir,
        result_path=result_path,
    )


# ---------------------------------------------------------------------------
# Marker handling — idempotent append
# ---------------------------------------------------------------------------


def _marker_open(iteration_id: str) -> str:
    return f"<!-- roy-iteration:{iteration_id} -->"


def _marker_close(iteration_id: str) -> str:
    return f"<!-- /roy-iteration:{iteration_id} -->"


def _wrap_block(iteration_id: str, body: str) -> str:
    """Wrap an iteration-log block in idempotence markers.

    The body is the full ``### Roy-...`` section — heading included.
    """
    return f"{_marker_open(iteration_id)}\n{body.rstrip()}\n{_marker_close(iteration_id)}\n"


def _find_existing_block(content: str, iteration_id: str) -> tuple[int, int] | None:
    """Return ``(start, end)`` byte offsets of an existing wrapped block,
    or ``None`` if absent.

    ``end`` points one past the close-marker line's trailing newline so
    ``content[:start] + new + content[end:]`` re-splices cleanly without
    leaving stray blank lines.
    """
    open_re = re.escape(_marker_open(iteration_id))
    close_re = re.escape(_marker_close(iteration_id))
    # DOTALL so the body can span lines; non-greedy so two iterations
    # with the same id never share a body (defensive — same id should
    # only ever appear once).
    m = re.search(open_re + r".*?" + close_re + r"\n?", content, flags=re.DOTALL)
    if m is None:
        return None
    return m.start(), m.end()


def _find_log_section_insert_point(content: str) -> int:
    """Find the byte offset where a new iteration-log block should land.

    Inserts immediately AFTER the ``## Iteration log`` heading's
    introductory paragraph (the first blank-line gap following the
    heading), BEFORE the next ``## `` section or end of file. This
    keeps existing log entries in chronological order — newest stays
    near the top of the section rather than buried at the bottom.

    Raises ``ValueError`` if the plan doc has no ``## Iteration log``
    section.
    """
    heading_match = re.search(r"^## Iteration log\b.*$", content, flags=re.MULTILINE)
    if heading_match is None:
        raise ValueError("Plan doc is missing a '## Iteration log' section")
    # Skip past the heading line itself.
    cursor = heading_match.end()
    # Skip leading blank lines + any explanatory paragraph that opens
    # the section — find the first blank line that follows non-blank
    # text. That blank line is the insertion point.
    # We don't try to detect the paragraph boundary; we just insert
    # directly under the heading after a single blank line.
    # Strip the existing newline(s) immediately after the heading.
    while cursor < len(content) and content[cursor] == "\n":
        cursor += 1
    # We're now at the first non-newline char after the heading line.
    # Insert here, with our own leading blank line + trailing newline
    # ensured by the caller. To keep newest-first ordering, we insert
    # at this point — pushing any prior parenthetical opener down. If
    # the operator dislikes the ordering they can re-arrange manually;
    # the marker comments survive moves.
    return cursor


def _find_section_end(content: str, section_start_offset: int) -> int:
    """Find the byte offset where the ``## Iteration log`` section ends.

    Returns the position of the next ``## `` heading (top-level), or
    ``len(content)`` if this is the last section.
    """
    next_match = re.search(r"^## ", content[section_start_offset:], flags=re.MULTILINE)
    if next_match is None:
        return len(content)
    return section_start_offset + next_match.start()


# ---------------------------------------------------------------------------
# Iteration log entry generation
# ---------------------------------------------------------------------------


def _humanize_iteration(iteration_id: str) -> str:
    """Render an iteration id like ``roy-1-adversarial`` as a heading
    label like ``Roy-1: Adversarial``.

    Falls back to the raw id when the convention isn't followed —
    keeps the generator usable for ad-hoc iteration names without
    forcing a naming policy on the operator.
    """
    parts = iteration_id.split("-")
    if len(parts) >= 3 and parts[0].lower() == "roy" and parts[1].isalnum():
        n = parts[1]
        rest = " ".join(p.capitalize() for p in parts[2:])
        return f"Roy-{n}: {rest}" if rest else f"Roy-{n}"
    return iteration_id


def _protocol_slug(iteration_id: str) -> str:
    """Slugify an iteration id for use in the protocol filename.

    Convention from ``persona_convergence_crucible.md``:
    ``docs/experiments/protocols/roy_<N>_<persona>.md``. We strip the
    leading ``roy-`` prefix so we don't end up with ``roy_roy-0_smoke``
    and replace remaining hyphens with underscores to match the
    underscore-separated naming the existing protocols use
    (p2_reward_modulation_reproduction.md, etc.).

    ``roy-0-smoke`` → ``0_smoke`` (filename becomes ``roy_0_smoke.md``)
    ``roy-1-adversarial`` → ``1_adversarial``
    ``roy-cradle-warmup`` → ``cradle_warmup``
    """
    body = iteration_id[4:] if iteration_id.lower().startswith("roy-") else iteration_id
    return body.replace("-", "_")


def _format_iteration_log_body(loaded: LoadedResult) -> str:
    """Render the markdown body for the iteration-log entry.

    Format follows the "Honest assessment template" in
    persona_convergence_crucible.md § Roy-1 — name what was attempted,
    record the divergence numbers, leave the qualitative findings as
    explicit TODO lines so the operator can't miss them.
    """
    payload = loaded.payload
    heading = _humanize_iteration(loaded.iteration_id)
    description = (payload.get("description") or "").strip()
    arms = payload.get("arms", {}) or {}
    pairwise = payload.get("pairwise_diffs", {}) or {}
    priming = payload.get("priming", {}) or {}

    lines: list[str] = []
    lines.append(f"### {heading}")
    lines.append("")
    if payload.get("aborted_at"):
        lines.append(f"**Status:** aborted at `{payload['aborted_at']}` — partial results below.")
        lines.append("")
    else:
        lines.append("**Status:** ran to completion; assessment pending operator review.")
        lines.append("")

    if description:
        # Stash the description as a quote so it's clearly the spec author's intent.
        for line in description.splitlines():
            lines.append(f"> {line}" if line.strip() else ">")
        lines.append("")

    # Priming summary
    priming_stages = priming.get("stages", []) or []
    priming_session = priming.get("final_session_id", "") or "<none>"
    lines.append(f"**Priming:** {len(priming_stages)} stage(s), final substrate session `{priming_session}`.")
    lines.append("")

    # Arms summary table
    lines.append("**Arms:**")
    lines.append("")
    lines.append("| Arm | Substrate | system_prompt | session_id | turns | finish_reason |")
    lines.append("|---|---|---|---|---|---|")
    for key in ("a", "b", "c"):
        arm = arms.get(key) or {}
        lines.append(
            f"| {key} "
            f"| {arm.get('substrate', '?')} "
            f"| {arm.get('system_prompt', '?')} "
            f"| `{arm.get('session_id') or '<none>'}` "
            f"| {arm.get('turns_run', 0)} "
            f"| {arm.get('finish_reason') or '<none>'}"
            f"{' ⚠ ' + arm['error'] if arm.get('error') else ''} |"
        )
    lines.append("")

    # Divergence numbers
    lines.append("**Substrate divergence (pairwise):**")
    lines.append("")
    for pair_key in ("a_vs_b", "a_vs_c", "b_vs_c"):
        diff = pairwise.get(pair_key)
        if not diff:
            lines.append(f"- **{pair_key}:** not computed")
            continue
        if not diff.get("available", True):
            lines.append(f"- **{pair_key}:** unavailable — {diff.get('reason', 'unknown')}")
            continue
        nac = diff.get("nac", {})
        ec = diff.get("ec", {})
        hipp = diff.get("hippocampus", {})
        atl = diff.get("atl", {})
        bullet = [f"- **{pair_key}:**"]
        if nac.get("available"):
            bullet.append(
                f"NAc reward_bias L2 `{nac.get('reward_bias_l2', 0.0):.4f}` · "
                f"causal-link Δ `{nac.get('causal_link_count_delta', 0):+d}`"
            )
        if ec.get("available"):
            bullet.append(f"EC nodes Δ `{ec.get('node_count_delta', 0):+d}`")
        if hipp.get("available"):
            v = hipp.get("valence", {})
            bullet.append(
                f"Hippocampus episodes Δ `{hipp.get('episode_count_delta', 0):+d}` "
                f"(valence KS `{v.get('ks_statistic', 0.0):.3f}`, "
                f"p `{v.get('ks_pvalue', 1.0):.3f}`)"
            )
        if atl.get("available"):
            bullet.append(
                f"ATL concepts Δ `{atl.get('concept_count_delta', 0):+d}` (jaccard `{atl.get('jaccard', 0.0):.3f}`)"
            )
        lines.append("  ".join(bullet))
    lines.append("")

    # Honest assessment template — required edits before commit.
    lines.append("**Honest assessment** _(operator: fill in before commit)_:")
    lines.append("")
    lines.append("- **Substrate took:** TODO — degree + evidence from divergence numbers above.")
    lines.append("- **Behavioral expression:** TODO — qualitative read of arms A/B action sequences.")
    lines.append("- **Cross-session persistence:** TODO — does arm A still express on session 2 restart?")
    lines.append("- **Generalization to novel stimuli:** TODO — held-out test scenario novel-stimuli analysis.")
    lines.append("- **What we learned:** TODO — specific findings about wires, mechanics, scenario design.")
    lines.append("- **What we'd change for next iteration:** TODO — concrete next steps.")
    lines.append("")
    lines.append(
        f"**Artifacts:** [`result.json`]({_relative_artifact_link(loaded)}) · "
        f"protocol [`roy_{_protocol_slug(loaded.iteration_id)}.md`]"
        f"(../experiments/protocols/roy_{_protocol_slug(loaded.iteration_id)}.md)"
    )
    return "\n".join(lines)


def _relative_artifact_link(loaded: LoadedResult) -> str:
    """Best-effort relative link from the plan doc to result.json.

    The artifact lives under ``~/.maxim/roy/<id>/`` which is outside
    the repo, so we link to the user-data path verbatim. Operators
    who want the artifact in-repo can copy it under
    ``docs/experiments/results/`` and edit the link.
    """
    return str(loaded.artifact_dir / "result.json")


def generate_iteration_log_entry(
    result: LoadedResult,
    plan_path: Path,
    *,
    dry_run: bool = False,
) -> str:
    """Append (or replace) an iteration-log entry in the named plan doc.

    Returns the rendered markdown block (including markers). When
    ``dry_run`` is True nothing is written to disk.

    Idempotence is enforced via HTML-comment markers around the
    generated block. Subsequent calls with the same ``iteration_id``
    replace the prior block in-place, preserving operator edits made
    OUTSIDE the marker region.
    """
    plan_path = Path(plan_path)
    if not plan_path.is_file():
        raise FileNotFoundError(f"Plan doc not found: {plan_path}")

    body = _format_iteration_log_body(result)
    wrapped = _wrap_block(result.iteration_id, body)

    if dry_run:
        return wrapped

    content = plan_path.read_text(encoding="utf-8")
    existing = _find_existing_block(content, result.iteration_id)
    if existing is not None:
        start, end = existing
        new_content = content[:start] + wrapped + content[end:]
    else:
        insert_at = _find_log_section_insert_point(content)
        # Ensure the wrapped block sits on its own paragraph: a blank
        # line above it (unless we're already at one) and a blank line
        # below before the next existing entry.
        prefix = "" if content[:insert_at].endswith("\n\n") else "\n"
        suffix = "\n" if not content[insert_at : insert_at + 1] == "\n" else ""
        new_content = content[:insert_at] + prefix + wrapped + suffix + content[insert_at:]

    if new_content != content:
        plan_path.write_text(new_content, encoding="utf-8")
    return wrapped


# ---------------------------------------------------------------------------
# Protocol runbook generation
# ---------------------------------------------------------------------------


def _format_protocol(loaded: LoadedResult) -> str:
    """Render the reproduction-protocol markdown.

    Mirrors ``docs/experiments/protocols/p2_reward_modulation_reproduction.md``:
    Status / Purpose / Prerequisites / Running / Expected output /
    What to do if it fails / Related docs. Tone is operator-facing —
    every command should be copy-pasteable as-is.
    """
    payload = loaded.payload
    iid = loaded.iteration_id
    heading = _humanize_iteration(iid)
    description = (payload.get("description") or "").strip()
    spec_path = payload.get("spec_path") or "<unknown>"
    duration = payload.get("total_duration_s") or 0.0
    arms = payload.get("arms", {}) or {}
    pairwise = payload.get("pairwise_diffs", {}) or {}
    priming = payload.get("priming", {}) or {}

    lines: list[str] = []
    lines.append(f"# {heading} — Reproduction Protocol")
    lines.append("")
    lines.append("**Status:** Draft — generated from `result.json`, awaiting operator review.")
    lines.append(
        "**Purpose:** Reproduce this Roy iteration on fresh hardware or after any change "
        "that might affect priming, action selection, or substrate persistence."
    )
    lines.append(f"**Expected runtime:** ~{duration:.0f}s wall clock (per the recorded run).")
    lines.append("")

    if description:
        lines.append("## Background")
        lines.append("")
        for line in description.splitlines():
            lines.append(line)
        lines.append("")

    # Prerequisites
    lines.append("## Prerequisites")
    lines.append("")
    lines.append("- Maxim checkout at or after the commit that produced this iteration.")
    lines.append("- `pymaxim` importable (`PYTHONPATH=src` or editable install).")
    lines.append("- A writable `~/.maxim/` (or `MAXIM_DATA_HOME=<path>` override).")
    lines.append("- The iteration spec YAML still resolvable from the recorded path:")
    lines.append("")
    lines.append(f"      {spec_path}")
    lines.append("")

    # Running
    lines.append("## Running the iteration")
    lines.append("")
    lines.append("```bash")
    lines.append(f"maxim roy run {spec_path}")
    lines.append("```")
    lines.append("")
    lines.append(
        "The runner primes arm A's substrate via the priming curriculum, runs the same "
        "held-out test scenario across all three arms, computes pairwise substrate "
        f"divergence, and writes `~/.maxim/roy/{iid}/result.json` + `summary.md`."
    )
    lines.append("")
    lines.append("Dry-run the spec first to verify schema:")
    lines.append("")
    lines.append("```bash")
    lines.append(f"maxim roy run {spec_path} --dry-run")
    lines.append("```")
    lines.append("")

    # Expected output
    lines.append("## Expected output")
    lines.append("")
    lines.append("**Arms (substrate / system_prompt → session shape):**")
    lines.append("")
    lines.append("| Arm | Substrate | system_prompt | turns | finish_reason |")
    lines.append("|---|---|---|---|---|")
    for key in ("a", "b", "c"):
        arm = arms.get(key) or {}
        lines.append(
            f"| {key} | {arm.get('substrate', '?')} | {arm.get('system_prompt', '?')} "
            f"| {arm.get('turns_run', 0)} | {arm.get('finish_reason') or '<none>'} |"
        )
    lines.append("")

    priming_stages = priming.get("stages", []) or []
    if priming_stages:
        lines.append(f"**Priming:** {len(priming_stages)} stage(s) on the curriculum:")
        lines.append("")
        for stage in priming_stages:
            lines.append(
                f"- `{stage.get('name', '?')}` "
                f"({stage.get('source_kind', '?')} `{stage.get('source_value', '?')}`) "
                f"× {stage.get('turns_run', 0)} turns"
            )
        lines.append("")

    # Divergence thresholds — derived from the recorded run as the
    # "expected" floor. Operator should tighten manually once they have
    # multiple seeds; the generator records what this single run produced.
    lines.append("**Divergence thresholds (from this run; tighten with seed variation):**")
    lines.append("")
    any_pair = False
    for pair_key in ("a_vs_b", "a_vs_c", "b_vs_c"):
        diff = pairwise.get(pair_key)
        if not diff or not diff.get("available", True):
            continue
        any_pair = True
        nac = diff.get("nac", {})
        ec = diff.get("ec", {})
        hipp = diff.get("hippocampus", {})
        atl = diff.get("atl", {})
        threshold_parts: list[str] = []
        if nac.get("available"):
            threshold_parts.append(f"NAc reward_bias L2 ≈ `{nac.get('reward_bias_l2', 0.0):.4f}`")
        if ec.get("available"):
            threshold_parts.append(f"EC nodes Δ ≈ `{ec.get('node_count_delta', 0):+d}`")
        if hipp.get("available"):
            threshold_parts.append(f"Hippocampus episodes Δ ≈ `{hipp.get('episode_count_delta', 0):+d}`")
        if atl.get("available"):
            threshold_parts.append(f"ATL jaccard ≈ `{atl.get('jaccard', 0.0):.3f}`")
        if threshold_parts:
            lines.append(f"- **{pair_key}:** " + " · ".join(threshold_parts))
    if not any_pair:
        lines.append("- _(no diffs available — every pair surfaced as `available=false`)_")
    lines.append("")

    # Failure modes
    lines.append("## What to do if it fails")
    lines.append("")
    lines.append(
        "**Priming aborts:** check the curriculum YAML's fixture paths resolve relative "
        "to the spec directory. The runner prints the failing stage name in "
        "`result.json::priming.aborted_at`."
    )
    lines.append("")
    lines.append(
        "**Arm A doesn't inherit priming substrate (a_vs_c shows ~0 divergence):** "
        "verify the priming stage wrote a real `aut_nac.json` (size > 0). The orchestrator "
        "writes substrate snapshots at session-end only — a stage that aborted before "
        "session-end leaves an empty snapshot. See "
        '[CLAUDE.md](../../../CLAUDE.md) §"Persistence uses `maxim.utils.atomic_io.atomic_write_json`".'
    )
    lines.append("")
    lines.append(
        "**Aggregate divergence numbers drift far from the table above:** the operating "
        "point depends on the spec's `aut_mode`, embodiment, and persona slugs. Re-run "
        "with `MAXIM_LOG_FILE=/tmp/roy.jsonl` and grep the JSONL for `peer_backend_call` "
        "to confirm every arm hit the expected LLM lane."
    )
    lines.append("")

    # Compare to prior
    lines.append("## What changed vs prior iterations")
    lines.append("")
    lines.append(
        "_(operator: fill in before commit. Examples: new persona slug, longer priming, "
        "different held-out fixture, encoder upgrade. Reference the prior `roy_*.md` "
        "in this directory.)_"
    )
    lines.append("")

    # Related docs
    lines.append("## Related docs")
    lines.append("")
    lines.append(
        "- [`persona_convergence_crucible.md`](../../plans/persona_convergence_crucible.md) — three-arm methodology"
    )
    lines.append(
        "- [`grounded_language_acquisition.md`](../../plans/grounded_language_acquisition.md) — Roy long-horizon context"
    )
    lines.append("- `maxim.analysis.substrate_diff` — the diff library that produced the numbers above")
    lines.append("- `maxim.simulation.roy_runner` — the iteration runner")
    lines.append("")
    lines.append(
        f"<!-- generated by `maxim roy log {iid}` — re-running replaces this file unless --keep-edits is passed -->"
    )
    return "\n".join(lines) + "\n"


def generate_protocol(
    result: LoadedResult,
    output_path: Path,
    *,
    dry_run: bool = False,
    keep_edits: bool = False,
) -> str:
    """Write the reproduction-protocol runbook.

    Returns the rendered markdown. When ``dry_run`` is True, nothing
    is written. When ``keep_edits`` is True and the target file
    already exists, raises ``FileExistsError`` instead of overwriting
    — gives the operator a chance to merge by hand.
    """
    output_path = Path(output_path)
    rendered = _format_protocol(result)
    if dry_run:
        return rendered
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and keep_edits:
        raise FileExistsError(
            f"Protocol already exists at {output_path}; pass --replace to overwrite "
            f"or omit --keep-edits to allow regeneration."
        )
    output_path.write_text(rendered, encoding="utf-8")
    return rendered


# ---------------------------------------------------------------------------
# Default output paths
# ---------------------------------------------------------------------------


def default_protocol_path(iteration_id: str, repo_root: Path | None = None) -> Path | None:
    """Default destination for the protocol runbook.

    Returns ``None`` when the repo root can't be detected (i.e. when
    running from an installed wheel). The CLI surfaces a meaningful
    error in that case.
    """
    root = repo_root or _detect_repo_root()
    if root is None:
        return None
    return root / "docs" / "experiments" / "protocols" / f"roy_{_protocol_slug(iteration_id)}.md"


__all__ = [
    "LoadedResult",
    "default_protocol_path",
    "detect_plan_path",
    "generate_iteration_log_entry",
    "generate_protocol",
    "load_iteration_result",
]
