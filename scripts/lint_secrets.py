#!/usr/bin/env python3
"""Secret scan: no credential lands on ``main``, and no email lands in experiment data (#1081, owner decisions 2026-10-08).

The repo is PUBLIC, so by the time a pull request's checks run its branch is already published. This lint does
not prevent exposure (push protection and a local run before ``git push`` are the only pre-publication layers);
it keeps the bytes off ``main``, whose history is permanent because data PRs are merge-commit only. **A finding
means: rotate the credential now.** Removing it from the branch does not unpublish it.

**What is scanned** (owner decision D1):

- the key patterns in :data:`PATTERNS` (tier ``H``) on EVERY tracked path, commit message and PR title/body;
- emails (tier ``E``) only under ``docs/experiments/data/``, never in commit messages (every commit carries a
  ``Co-Authored-By`` address). Reserved example domains (RFC 2606/6761) are never findings.

**Diff mode** (default; the ``lint`` job): every (path, blob) pair any commit in ``base..HEAD`` introduced, read
from ``git diff-tree --no-renames`` per commit (so a key added in one commit and deleted in the next still fails,
and a pure rename into ``docs/experiments/data/`` is scanned under its new path). A merge contributes only its OWN
change, what differs from every parent (``-c`` semantics): a PR is judged on the blobs it introduces, never on
``main``'s changes that a "Merge branch 'main'" commit or the synthetic PR merge brings in. Plus the final
``base..HEAD`` diff, every commit message, and on a pull request the PR title and body from
the event payload. ``base`` is ``_lint_git.base_ref``; on a push to main the range is split into the units that
landed (``_lint_git.push_units``) and each is judged against its own first parent ("on main before this
landed"). No base on a pull request or push is exit 2 (``_lint_git.must_not_skip``); locally it skips.

**Whole blobs, not added lines**, so a token split across hunks is still one token. Containers are detected by
MAGIC BYTES, not extension: gzip (multi-member), bz2, xz, zip and tar are opened, nested up to
:data:`MAX_DEPTH`, with a CUMULATIVE decompressed budget of :data:`DECOMPRESS_CAP` bytes per top-level blob
enforced by bounded streaming reads (headers are never trusted) and a member-count cap. A container the lint
cannot open (zstd, 7z, rar, corrupt, over the cap, too deep), a Git LFS pointer, a gitlink, or a text-typed blob
full of NUL bytes that is not UTF-16 FAILS under ``docs/experiments/data/`` (owner decision D4) and warns
elsewhere. UTF-16/32 text is decoded (by BOM, or by NUL position for text-typed files). JSON and JSON Lines with
escapes are also scanned DECODED, one level of JSON-in-JSON deep.

**Allowlist** ``scripts/secret_scan_allowlist.json``: ``{"_format_version": "1.0", "entries": [...]}``; each entry
is exactly ``{pattern, path, match_sha256, owner, date, ref, reason}`` and excuses one value (by the SHA-256 of
the matched token, never the token) at one exact path (``:commit-message`` and ``:pr-text`` for the two text
sources). Only entries already on the BASE act (owner decision D2: a same-PR entry would be a self-approval
switch), and the list is append-only. An entry keys on the EXACT path: renaming or moving an allowlisted file
needs a new entry for the new path, landed on ``main`` first (two PRs: the entry, then the move). Matching keys on
(path, sha256) alone; an entry's ``pattern`` is a record of what matched, so renaming a pattern never invalidates
an entry. Emails are never allowlisted (only key-tier matches consult the list, and an entry may not name an email
pattern of either copy of the table). ``--all`` reports entries that match nothing.

**The pattern table cannot shrink in the PR that needs it to:** the scan uses the union of this file's
:data:`PATTERNS` and the base copy's (parsed with ``ast.literal_eval``; an unparsable base table is exit 2). This
guards accidental narrowing only; a reviewed PR can still narrow a pattern for the PRs after it.

**Output never contains a matched value**, since CI logs on a public repo are public: ``path:line``, the pattern
name, the first 4 characters, the length and the SHA-256 (what an allowlist entry needs).

Modes: (default) diff; ``--all`` every tracked blob at HEAD (the nightly ``Secret scan (nightly)`` job; on a
schedule the diff range is empty by construction); ``--staged`` the index against HEAD, for a local run before
committing a data file. Exits: 0 clean; 1 findings or an uninspectable data blob or an edited allowlist; 2 cannot
check (no base on a PR/push, a git or ``cat-file`` failure, a malformed allowlist, an unparsable base table).

Residuals (catches forgetting, not evasion): base64/hex-encoded secrets; a token split across fields or lines; an
unprefixed key with no ``Bearer``/``token``/``key=`` context (the mesh key has no prefix; giving it one would make
it detectable everywhere); PR text that a later edit removed is not re-checked against history (the workflow's
``edited`` trigger re-runs the scan on the new text only); a deleted workflow step; a PR changing this lint's code paths rather than its table.

Stdlib only. Regression guard: tests/unit/test_lint_secrets.py.
"""

from __future__ import annotations

import argparse
import ast
import bz2
import datetime as _dt
import gzip
import hashlib
import io
import json
import lzma
import os
import re
import subprocess
import sys
import tarfile
import zipfile
import zlib
from collections.abc import Iterator
from pathlib import Path
from typing import Any, NamedTuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lint_allowance  # noqa: E402
import _lint_git  # noqa: E402
from _lint_git import GitUnavailable  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
SELF_REL = "scripts/lint_secrets.py"
ALLOWLIST_REL = "scripts/secret_scan_allowlist.json"
DATA_PREFIX = "docs/experiments/data/"
COMMIT_MSG = ":commit-message"
PR_TEXT = ":pr-text"
#: Cumulative decompressed bytes per top-level blob, across every nesting level (largest data blob today: 10 MB).
DECOMPRESS_CAP = 256 * 1024 * 1024
MAX_DEPTH = 3
MAX_MEMBERS = 10_000
_CHUNK = 1 << 20

#: (name, tier, regex). Tier H: every path. Tier E: docs/experiments/data/ only. Each regex names the token ``v``.
#: Read from the BASE copy too (``ast.literal_eval``): keep this a plain literal. No regex may match its own source.
PATTERNS = (
    ("anthropic", "H", r"(?P<v>\bsk-ant-[A-Za-z0-9_-]{20,})"),
    ("openai", "H", r"(?P<v>\bsk-(?:proj-)?[A-Za-z0-9_-]{20,})"),
    ("github", "H", r"(?P<v>\bgh[pousr]_[A-Za-z0-9]{36,}|\bgithub_pat_[A-Za-z0-9_]{22,})"),
    ("aws", "H", r"(?P<v>\b(?:AKIA|ASIA)[0-9A-Z]{16})\b"),
    ("huggingface", "H", r"(?P<v>\bhf_[A-Za-z0-9]{30,})"),
    ("slack", "H", r"(?P<v>\bxox[abposr]-[A-Za-z0-9-]{10,})"),
    ("google", "H", r"(?P<v>\bAIza[0-9A-Za-z_-]{35})(?![0-9A-Za-z_-])"),
    ("groq", "H", r"(?P<v>\bgsk_[A-Za-z0-9]{20,})"),
    ("maxim-console", "H", r"(?P<v>\bmxc_[A-Za-z0-9_-]{43})(?![A-Za-z0-9_-])"),
    ("private-key", "H", r"(?P<v>-{5}BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY(?: BLOCK)?-{5})"),
    ("auth-header", "H", r"(?i)\b(?:bearer|token)[ \t]+(?P<v>[A-Za-z0-9._~+/-]{20,}=*)"),
    (
        "key-assignment",
        "H",
        r"""(?i)(?:api[_-]?key|secret|token|password)["']?[ \t]*[:=][ \t]*["']?(?P<v>[A-Za-z0-9._~+/-]{32,})""",
    ),
    (
        "email",
        "E",
        r"(?<![A-Za-z0-9._%+\\-])(?P<v>[A-Za-z0-9._%+-]{1,64}@(?:[A-Za-z0-9-]{1,63}\.){1,8}[A-Za-z]{2,24})(?![A-Za-z0-9-])",
    ),
)

_RESERVED_DOMAINS = (b"example.com", b"example.org", b"example.net")
_RESERVED_TLDS = (b".invalid", b".test", b".example", b".localhost")
_TEXT_EXTS = {
    ".json", ".jsonl", ".ndjson", ".txt", ".md", ".csv", ".tsv", ".py", ".sh", ".out", ".log",
    ".yaml", ".yml", ".html", ".toml", ".cfg", ".ini", ".xml",
}  # fmt: skip
_JSON_EXTS = {".json", ".jsonl", ".ndjson"}
_ENTRY_KEYS = {"pattern", "path", "match_sha256", "owner", "date", "ref", "reason"}
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
ADVICE = (
    "Rotate the credential NOW: the branch is already public, and removing it from the PR does not unpublish it. "
    "If it is a false positive, land an allowlist entry in scripts/secret_scan_allowlist.json on main FIRST "
    "(an entry added in the same PR does not act), keyed on the path and the sha256 printed above. "
    "If it already reached main (this is a push run), it cannot be fixed forward: every later push is judged from "
    "the last green base and re-finds it. Rotate it FIRST, then record the offending first-parent commit of main "
    "in scripts/push_base_accepts.json ({sha, reason, owner, date}, through a merged PR; see scripts/_lint_git.py) "
    "so the push lint's base moves past it."
)


class Pattern(NamedTuple):
    name: str
    tier: str
    source: str
    rx: re.Pattern[bytes]


class Finding(NamedTuple):
    path: str  # the tracked path (or :commit-message / :pr-text): what an allowlist entry names
    where: str  # display location, with archive members and line
    pattern: str
    sha256: str
    preview: str
    length: int

    def line(self) -> str:
        sha = f"  sha256:{self.sha256}" if self.pattern != "email" else ""
        return f"{self.where}  [{self.pattern}]  {self.preview}…({self.length} chars){sha}"


class ConfigError(Exception):
    """A malformed allowlist or pattern table: exit 2."""


class _Uninspectable(Exception):
    """A blob whose content the lint cannot read: fails under data/, warns elsewhere."""


# ── the pattern table ────────────────────────────────────────────────────────


def compile_patterns(rows: Any) -> list[Pattern]:
    if not isinstance(rows, tuple | list):
        raise ConfigError("PATTERNS is not a tuple")
    out = []
    for row in rows:
        if not (isinstance(row, tuple | list) and len(row) == 3 and all(isinstance(x, str) for x in row)):
            raise ConfigError(f"PATTERNS row {row!r} is not (name, tier, regex)")
        name, tier, source = row
        if tier not in ("H", "E"):
            raise ConfigError(f"PATTERNS row {name!r}: tier {tier!r} is not H or E")
        try:
            rx = re.compile(source.encode("ascii"))
        except (re.error, UnicodeEncodeError) as exc:
            raise ConfigError(f"PATTERNS row {name!r}: {exc}") from exc
        if "v" not in rx.groupindex:
            raise ConfigError(f"PATTERNS row {name!r}: no (?P<v>...) group")
        out.append(Pattern(name, tier, source, rx))
    return out


def parse_patterns_source(text: str) -> tuple:
    """The ``PATTERNS`` literal of a copy of this file. Raises ConfigError when it is absent or not a literal."""
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        raise ConfigError(f"base copy of {SELF_REL} does not parse: {exc}") from exc
    for node in tree.body:
        targets = (
            node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
        )
        if any(isinstance(t, ast.Name) and t.id == "PATTERNS" for t in targets) and node.value is not None:
            try:
                return ast.literal_eval(node.value)
            except ValueError as exc:
                raise ConfigError(f"base copy of {SELF_REL}: PATTERNS is not a literal ({exc})") from exc
    raise ConfigError(f"base copy of {SELF_REL} has no PATTERNS literal")


def patterns_at(repo: Path, ref: str | None) -> list[Pattern]:
    """This file's patterns united with ``ref``'s copy (a pattern removed or narrowed acts only once on main)."""
    own = compile_patterns(PATTERNS)
    if ref is None:
        return own
    text = _lint_git.show(repo, ref, SELF_REL)
    if not text:
        return own  # the lint did not exist at the base
    seen = {(p.name, p.source) for p in own}
    return own + [p for p in compile_patterns(parse_patterns_source(text)) if (p.name, p.source) not in seen]


# ── the allowlist ────────────────────────────────────────────────────────────


def parse_allowlist(text: str, where: str, patterns: list[Pattern] | None = None) -> list[dict[str, str]]:
    """Entries of an allowlist file's text ("" = absent = empty). Raises ConfigError on any malformed part.

    ``patterns`` is the table in force (the base/HEAD union in diff mode). An entry's ``pattern`` is NOT checked
    against its key-tier names: matching keys on (path, sha256), so a pattern renamed on ``main`` must not turn
    every existing entry malformed and lock the lint at exit 2 (the list is append-only, so they could not be
    rewritten). It only may not name an email-tier pattern."""
    email_names = {p.name for p in (patterns if patterns is not None else compile_patterns(PATTERNS)) if p.tier == "E"}
    if not text.strip():
        return []
    try:
        data = json.loads(text)
    except ValueError as exc:
        raise ConfigError(f"{ALLOWLIST_REL} at {where}: not JSON ({exc})") from exc
    if not isinstance(data, dict) or set(data) != {"_format_version", "entries"}:
        raise ConfigError(f"{ALLOWLIST_REL} at {where}: must be exactly {{_format_version, entries}}")
    if data["_format_version"] != "1.0" or not isinstance(data["entries"], list):
        raise ConfigError(f'{ALLOWLIST_REL} at {where}: _format_version must be "1.0" and entries a list')
    for e in data["entries"]:
        ok = isinstance(e, dict) and set(e) == _ENTRY_KEYS and all(isinstance(e[k], str) and e[k].strip() for k in e)
        if not ok:
            raise ConfigError(
                f"{ALLOWLIST_REL} at {where}: an entry needs exactly {sorted(_ENTRY_KEYS)}, all non-empty"
            )
        problems = []
        if not _SHA256_RE.fullmatch(e["match_sha256"]):
            problems.append("match_sha256 is not 64 lowercase hex (store the hash, never the value)")
        try:
            _dt.date.fromisoformat(e["date"])
            if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", e["date"]):
                raise ValueError
        except ValueError:
            problems.append("date is not YYYY-MM-DD")
        if not _lint_allowance.ref_ok(e["ref"]):
            problems.append("ref is not #NNN or a github.com issue/PR URL")
        if e["pattern"] in email_names:
            problems.append(f"pattern {e['pattern']!r} is an email pattern (emails are never allowlisted)")
        if problems:
            raise ConfigError(f"{ALLOWLIST_REL} at {where}: entry for {e['path']!r}: {'; '.join(problems)}")
    return data["entries"]


# ── reading git ──────────────────────────────────────────────────────────────


class Blobs:
    """``git cat-file --batch`` as one pipe. A missing object or a short read RAISES: it never reads as clean."""

    def __init__(self, repo: Path) -> None:
        try:
            self._p = subprocess.Popen(
                ["git", "cat-file", "--batch"],
                cwd=repo,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
        except OSError as exc:
            raise GitUnavailable(f"git cat-file: {exc}") from exc

    def get(self, sha: str) -> bytes:
        assert self._p.stdin is not None and self._p.stdout is not None
        try:
            self._p.stdin.write(sha.encode("ascii") + b"\n")
            self._p.stdin.flush()
            header = self._p.stdout.readline().split()
        except OSError as exc:
            raise GitUnavailable(f"git cat-file {sha[:12]}: {exc}") from exc
        if len(header) != 3 or header[1] != b"blob":
            raise GitUnavailable(f"git cat-file {sha[:12]}: not a readable blob ({b' '.join(header)!r})")
        size = int(header[2])
        data = self._p.stdout.read(size)
        if len(data) != size or self._p.stdout.read(1) != b"\n":
            raise GitUnavailable(f"git cat-file {sha[:12]}: short read")
        return data

    def close(self) -> None:
        if self._p.stdin:
            self._p.stdin.close()
        self._p.wait(timeout=30)


def _raw_pairs(text: str) -> Iterator[tuple[str, str, str]]:
    """(mode, sha, path) of every added/modified entry in ``-z --raw`` output; deletions are skipped."""
    tokens = text.split("\0")
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.startswith(":") and i + 1 < len(tokens):
            meta = tok[1:].split()
            paired = len(meta) >= 5 and meta[4][:1] in ("R", "C")  # never with --no-renames; then the NEW path
            path = tokens[i + 2] if paired and i + 2 < len(tokens) else tokens[i + 1]
            i += 3 if paired else 2
            if len(meta) >= 5 and not meta[4].startswith("D") and set(meta[3]) != {"0"}:
                yield meta[1], meta[3], path
            continue
        i += 1


def commit_pairs(repo: Path, commit: str, parents: list[str]) -> set[tuple[str, str, str]]:
    """The (mode, blob, path) entries ``commit`` itself introduced. A merge introduced only what differs from EVERY
    parent (``git diff-tree -c`` semantics, computed as the intersection of the per-parent diffs): a pair equal to
    one parent's arrived with that parent, which is either in the range (scanned there) or already on the base.
    Diffing a merge against one parent at a time would charge a branch with ``main``'s changes since its fork, on a
    "Merge branch 'main'" commit and on the synthetic PR merge alike."""
    flags = ("-r", "-z", "--no-renames", "--no-abbrev")
    if not parents:
        return set(_raw_pairs(_lint_git.git(repo, "diff-tree", *flags, "--root", "--no-commit-id", commit)))
    common: set[tuple[str, str, str]] | None = None
    for parent in parents:
        pairs = set(_raw_pairs(_lint_git.git(repo, "diff-tree", *flags, parent, commit)))
        common = pairs if common is None else common & pairs
    return common or set()


def range_pairs(repo: Path, base: str, head: str) -> list[tuple[str, str, str]]:
    """Every (mode, blob, path) any commit in ``base..head`` introduced (a merge: its OWN change, see
    :func:`commit_pairs`), plus the final diff. ``--no-renames``: a pure move is an add, so a file moved into data/
    is scanned under its new path."""
    out: dict[tuple[str, str], str] = {}
    flags = ("-r", "-z", "--no-renames", "--no-abbrev")
    for line in _lint_git.git(repo, "rev-list", "--parents", f"{base}..{head}").splitlines():
        commit, *parents = line.split()
        for mode, sha, path in sorted(commit_pairs(repo, commit, parents)):
            out.setdefault((sha, path), mode)
    for mode, sha, path in _raw_pairs(_lint_git.git(repo, "diff", "--raw", *flags, base, head)):
        out.setdefault((sha, path), mode)
    return [(mode, sha, path) for (sha, path), mode in out.items()]


def tree_pairs(repo: Path, ref: str = "HEAD") -> list[tuple[str, str, str]]:
    out = []
    for rec in _lint_git.git(repo, "ls-tree", "-r", "-z", "--full-tree", ref).split("\0"):
        if rec:
            meta, path = rec.split("\t", 1)
            mode, _type, sha = meta.split()
            out.append((mode, sha, path))
    return out


def staged_pairs(repo: Path) -> list[tuple[str, str, str]]:
    return list(
        _raw_pairs(_lint_git.git(repo, "diff", "--cached", "--raw", "-z", "--no-renames", "--no-abbrev", "HEAD"))
    )


def commit_messages(repo: Path, base: str, head: str) -> list[tuple[str, str]]:
    out = []
    for rec in _lint_git.git(repo, "log", "-z", "--format=%H%n%B", f"{base}..{head}").split("\0"):
        if rec.strip():
            sha, _, body = rec.partition("\n")
            out.append((sha.strip(), body))
    return out


def pr_text_from_event() -> str:
    """The PR title and body of a ``pull_request`` event. Raises GitUnavailable when the payload is unreadable."""
    path = os.environ.get("GITHUB_EVENT_PATH", "")
    try:
        event = json.loads(Path(path).read_text(encoding="utf-8"))
        pr = event["pull_request"]
        return f"{pr.get('title') or ''}\n{pr.get('body') or ''}"
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise GitUnavailable(f"pull_request event payload {path!r} unreadable ({exc})") from exc


# ── scanning ─────────────────────────────────────────────────────────────────


def _placeholder(v: bytes) -> bool:
    """A documented example, not a credential: the token's last 16 characters (separators ignored) are one repeated
    character or only x/X/0/* (``AKIA`` + 16 X, a prefix + a run of x). Reviewed with the detectors."""
    tail = re.sub(rb"[-_.~+/=]", b"", v)[-16:]
    return len(set(tail)) <= 1 or not tail.strip(b"xX0*")


def _reserved_email(v: bytes) -> bool:
    domain = v.rsplit(b"@", 1)[-1].lower()
    return any(domain == d or domain.endswith(b"." + d) for d in _RESERVED_DOMAINS) or (
        domain == b"localhost" or domain.endswith(_RESERVED_TLDS)
    )


def _ext(name: str) -> str:
    return os.path.splitext(name)[1].lower()


def _kind(data: bytes) -> str | None:
    if data[:2] == b"\x1f\x8b":
        return "gzip"
    if data[:3] == b"BZh" and data[3:4] in (b"1", b"2", b"3", b"4", b"5", b"6", b"7", b"8", b"9"):
        return "bz2"
    if data[:6] == b"\xfd7zXZ\x00":
        return "xz"
    if data[:4] in (b"PK\x03\x04", b"PK\x05\x06"):
        return "zip"
    if data[257:262] == b"ustar":
        return "tar"
    if data[:4] == b"\x28\xb5\x2f\xfd":
        return "zstd"
    if data[:6] == b"7z\xbc\xaf\x27\x1c":
        return "7z"
    if data[:4] == b"Rar!":
        return "rar"
    if data.startswith(b"version https://git-lfs.github.com/spec/"):
        return "lfs-pointer"
    return None


class _Budget:
    def __init__(self, cap: int) -> None:
        self.left = cap

    def read(self, f: Any) -> bytes:
        """All of ``f``, but never more than the budget: bounded streaming reads, whatever a header claims."""
        chunks = []
        while True:
            chunk = f.read(min(_CHUNK, self.left + 1))
            if not chunk:
                return b"".join(chunks)
            self.left -= len(chunk)
            if self.left < 0:
                raise _Uninspectable(f"decompresses past the {DECOMPRESS_CAP}-byte cap")
            chunks.append(chunk)


def _decode(name: str, data: bytes) -> bytes:
    """UTF-8 bytes to scan: UTF-16/32 decoded by BOM, or (text-typed files) by where the NUL bytes sit."""
    for bom, codec in (
        (b"\xef\xbb\xbf", "utf-8-sig"),
        (b"\xff\xfe\x00\x00", "utf-32-le"),
        (b"\x00\x00\xfe\xff", "utf-32-be"),
        (b"\xff\xfe", "utf-16"),
        (b"\xfe\xff", "utf-16"),
    ):
        if data.startswith(bom):
            return data.decode(codec, errors="replace").encode("utf-8", errors="replace")
    nul = data.count(b"\0")
    if _ext(name) in _TEXT_EXTS and len(data) >= 2 and nul > len(data) // 5:
        codec = "utf-16-le" if data[1::2].count(b"\0") >= data[0::2].count(b"\0") else "utf-16-be"
        try:
            return data.decode(codec).encode("utf-8", errors="replace")
        except UnicodeDecodeError as exc:
            raise _Uninspectable(f"a text file with {nul} NUL bytes that is not UTF-16 ({exc.reason})") from exc
    return data


def _json_strings(name: str, data: bytes) -> tuple[list[str], int]:
    """Every string (keys included) of a JSON / JSON Lines payload, and of strings that are themselves JSON (one
    level), plus the count of malformed JSON Lines lines."""
    docs: list[Any] = []
    bad = 0
    try:
        docs.append(json.loads(data))
    except ValueError:
        for raw in data.splitlines():
            if raw.strip():
                try:
                    docs.append(json.loads(raw))
                except ValueError:
                    bad += 1
    out: list[str] = []
    stack: list[tuple[Any, int]] = [(d, 0) for d in docs]
    while stack:
        obj, depth = stack.pop()
        if isinstance(obj, dict):
            stack.extend((k, depth) for k in obj)
            stack.extend((v, depth) for v in obj.values())
        elif isinstance(obj, list):
            stack.extend((v, depth) for v in obj)
        elif isinstance(obj, str):
            out.append(obj)
            if depth < 1 and obj.lstrip()[:1] in ("{", "["):
                try:
                    stack.append((json.loads(obj), depth + 1))
                except ValueError:
                    pass  # an ordinary string that starts with a bracket
    return out, (bad if _ext(name) in (".jsonl", ".ndjson") else 0)


class Scan:
    """Findings, uninspectable data blobs (problems) and warnings for one run."""

    def __init__(self, patterns: list[Pattern], allowed: set[tuple[str, str]], cap: int = DECOMPRESS_CAP) -> None:
        self.patterns = patterns
        self.allowed = allowed
        self.cap = cap
        self.findings: list[Finding] = []
        self.excused: list[Finding] = []
        self.problems: list[str] = []
        self.warnings: list[str] = []
        self.blobs = 0

    def _match(self, path: str, where: str, buf: bytes, *, emails: bool, lines: bool, seen: set) -> None:
        """``seen`` holds (tier, sha, line) for every located match AND (tier, sha, None) for every value, so the
        decoded pass
        skips a value the raw pass already reported in O(1). Lines are counted incrementally (matches arrive in
        position order per pattern): linear in the buffer, whatever the number of matches."""
        for p in self.patterns:
            if p.tier == "E" and not emails:
                continue
            pos, line = 0, 1
            for m in p.rx.finditer(buf):
                v = m.group("v")
                if _reserved_email(v) if p.tier == "E" else _placeholder(v):
                    continue
                sha = hashlib.sha256(v).hexdigest()
                if lines:
                    start = m.start("v")
                    line += buf.count(b"\n", pos, start)
                    pos = start
                # The tier is in the key: an allowlisted key-tier match never hides an email-tier match of the same
                # value (delta review: a future key pattern admitting `@` would otherwise excuse an email silently).
                key = (p.tier, sha, line if lines else None)
                if key in seen:
                    continue
                seen.add(key)
                seen.add((p.tier, sha, None))
                loc = f"{where}:{line}" if lines else f"{where} (JSON-decoded)"
                f = Finding(path, loc, p.name, sha, v[:4].decode("ascii", "replace"), len(v))
                excused = p.tier == "H" and (path, sha) in self.allowed
                (self.excused if excused else self.findings).append(f)

    def text(self, path: str, where: str, data: bytes) -> None:
        """A commit message or PR text: key patterns only."""
        self._match(path, where, data, emails=False, lines=True, seen=set())

    def blob(self, path: str, data: bytes, mode: str = "100644") -> None:
        self.blobs += 1
        in_data = path.startswith(DATA_PREFIX)
        try:
            if mode == "160000":
                raise _Uninspectable("a gitlink (submodule): its content is not in this repo")
            self._walk(path, path, path, data, 0, _Budget(self.cap), in_data)
        except _Uninspectable as exc:
            msg = f"{path}: cannot inspect: {exc}"
            if in_data:
                self.problems.append(msg + " (recompress as gzip, or commit it uncompressed)")
            else:
                self.warnings.append(msg)

    def _walk(self, path: str, where: str, name: str, data: bytes, depth: int, budget: _Budget, in_data: bool) -> None:
        kind = _kind(data)
        if kind is None:
            self._leaf(path, where, name, data, in_data)
            return
        if kind in ("zstd", "7z", "rar", "lfs-pointer"):
            raise _Uninspectable(f"{where} is a {kind}, which this lint cannot open")
        if depth >= MAX_DEPTH:
            raise _Uninspectable(f"{where}: archives nested deeper than {MAX_DEPTH}")
        try:
            if kind in ("gzip", "bz2", "xz"):
                stream = io.BytesIO(data)
                opened = (
                    gzip.GzipFile(fileobj=stream)
                    if kind == "gzip"
                    else bz2.BZ2File(stream)
                    if kind == "bz2"
                    else lzma.LZMAFile(stream)
                )
                with opened as f:
                    inner = budget.read(f)
                stem = os.path.splitext(name)[0] if _ext(name) in (".gz", ".bz2", ".xz") else name
                self._walk(path, f"{where}[{kind}]", stem, inner, depth + 1, budget, in_data)
            elif kind == "zip":
                with zipfile.ZipFile(io.BytesIO(data)) as z:
                    infos = z.infolist()
                    if len(infos) > MAX_MEMBERS:
                        raise _Uninspectable(f"{where}: more than {MAX_MEMBERS} members")
                    for info in infos:
                        if not info.is_dir():
                            with z.open(info) as f:
                                inner = budget.read(f)
                            self._walk(
                                path, f"{where}!{info.filename}", info.filename, inner, depth + 1, budget, in_data
                            )
            else:  # tar
                with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as t:
                    for n, member in enumerate(t):
                        if n >= MAX_MEMBERS:
                            raise _Uninspectable(f"{where}: more than {MAX_MEMBERS} members")
                        if member.isfile():
                            f = t.extractfile(member)
                            inner = budget.read(f) if f is not None else b""
                            self._walk(path, f"{where}!{member.name}", member.name, inner, depth + 1, budget, in_data)
        except _Uninspectable:
            raise
        except (OSError, EOFError, zlib.error, lzma.LZMAError, zipfile.BadZipFile, tarfile.TarError, RuntimeError,
                NotImplementedError, ValueError) as exc:  # fmt: skip
            raise _Uninspectable(f"{where}: corrupt or unreadable {kind} ({type(exc).__name__}: {exc})") from exc

    def _leaf(self, path: str, where: str, name: str, data: bytes, in_data: bool) -> None:
        data = _decode(name, data)
        seen: set = set()
        self._match(path, where, data, emails=in_data, lines=True, seen=seen)
        # Decoding changes nothing without an escape; JSON-in-JSON needs one too.
        if b"\\" in data and (_ext(name) in _JSON_EXTS or data.lstrip()[:1] in (b"{", b"[")):
            strings, bad = _json_strings(name, data)
            if bad:
                self.warnings.append(f"{where}: {bad} malformed JSON Lines line(s), scanned raw only")
            if strings:
                buf = "\n".join(strings).encode("utf-8", errors="replace")
                self._match(path, where, buf, emails=in_data, lines=False, seen=seen)


# ── the runs ─────────────────────────────────────────────────────────────────


def scan_pairs(repo: Path, scan: Scan, pairs: list[tuple[str, str, str]]) -> None:
    blobs = Blobs(repo)
    try:
        for mode, sha, path in pairs:
            if mode == "160000":
                if path.startswith(DATA_PREFIX):
                    scan.blob(path, b"", mode)
                continue
            scan.blob(path, blobs.get(sha), mode)
    finally:
        blobs.close()


def scan_range(repo: Path, base: str, head: str, *, pr_text: str | None, cap: int = DECOMPRESS_CAP) -> Scan:
    """One diff-mode unit: patterns and allowlist from ``base``; the allowlist must be append-only base → head."""
    patterns = patterns_at(repo, base)
    base_entries = parse_allowlist(_lint_git.show(repo, base, ALLOWLIST_REL), f"base {base[:12]}", patterns)
    head_entries = parse_allowlist(_lint_git.show(repo, head, ALLOWLIST_REL), head, patterns)
    scan = Scan(patterns, {(e["path"], e["match_sha256"]) for e in base_entries}, cap)
    problem = _lint_allowance.append_only_problem(base=base_entries, head=head_entries, what="the allowlist is")
    if problem:
        scan.problems.append(f"{ALLOWLIST_REL}: {problem}")
    scan_pairs(repo, scan, range_pairs(repo, base, head))
    for sha, body in commit_messages(repo, base, head):
        scan.text(COMMIT_MSG, f"{COMMIT_MSG} {sha[:12]}", body.encode("utf-8", errors="replace"))
    if pr_text:
        scan.text(PR_TEXT, PR_TEXT, pr_text.encode("utf-8", errors="replace"))
    return scan


def scan_tree(repo: Path, pairs: list[tuple[str, str, str]], cap: int = DECOMPRESS_CAP) -> tuple[Scan, list[str]]:
    """``--all`` / ``--staged``: the allowlist at HEAD acts (on main, that is main). Also returns stale entries."""
    patterns = patterns_at(repo, None)
    entries = parse_allowlist(_lint_git.show(repo, "HEAD", ALLOWLIST_REL), "HEAD", patterns)
    scan = Scan(patterns, {(e["path"], e["match_sha256"]) for e in entries}, cap)
    scan_pairs(repo, scan, pairs)
    hit = {(f.path, f.sha256) for f in scan.excused}
    stale = [
        f"{e['path']} sha256:{e['match_sha256'][:12]}" for e in entries if (e["path"], e["match_sha256"]) not in hit
    ]
    return scan, stale


def _report(scans: list[Scan], label: str) -> int:
    findings = [f for s in scans for f in s.findings]
    problems = [p for s in scans for p in s.problems]
    for w in (w for s in scans for w in s.warnings):
        print(f"WARN: {w}")
    excused = sum(len(s.excused) for s in scans)
    print(
        f"secret scan ({label}): {sum(s.blobs for s in scans)} blob(s), {len(findings)} finding(s), "
        f"{excused} allowlisted, {len(problems)} problem(s)"
    )
    if not (findings or problems):
        return 0
    print("secret scan FAILED (no value is printed: first 4 characters, length and sha256 only):", file=sys.stderr)
    for f in findings:
        print(f"  {f.line()}", file=sys.stderr)
    for p in problems:
        print(f"  {p}", file=sys.stderr)
    if findings:
        print(ADVICE, file=sys.stderr)
    return 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--all", action="store_true", help="every tracked blob at HEAD (the nightly)")
    group.add_argument("--staged", action="store_true", help="the index against HEAD (before committing data)")
    ap.add_argument("--repo", type=Path, default=REPO_ROOT, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    repo: Path = args.repo
    try:
        if args.all or args.staged:
            pairs = tree_pairs(repo) if args.all else staged_pairs(repo)
            scan, stale = scan_tree(repo, pairs)
            for s in stale:
                print(
                    f"STALE allowlist entry (matches nothing at HEAD; the list is append-only, so reported only): {s}"
                )
            return _report([scan], "--all" if args.all else "--staged")
        event = os.environ.get("GITHUB_EVENT_NAME", "")
        try:
            base = _lint_git.base_ref(repo)
        except GitUnavailable as exc:
            if _lint_git.must_not_skip(str(exc)):
                return 2
            print(f"INFO: no base ref available; secret scan skipped ({exc}). Try --staged or --all.")
            return 0
        if event in ("schedule", "workflow_dispatch"):
            print(f"INFO: {event}: the diff range is empty by construction; the whole tree is the nightly --all job")
        if event == "push":
            scans = []
            for unit in _lint_git.push_units(repo, base):
                parent = _lint_git.git(repo, "rev-parse", f"{unit.sha}^1").strip()
                text = f"{unit.pr['title']}\n{unit.pr['body']}" if unit.pr else None
                scans.append(scan_range(repo, parent, unit.sha, pr_text=text))
            return _report(scans, f"push, {len(scans)} unit(s) since {base[:12]}")
        text = pr_text_from_event() if event == "pull_request" else None
        return _report([scan_range(repo, base, "HEAD", pr_text=text)], f"{base[:12]}..HEAD")
    except ConfigError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except GitUnavailable as exc:
        _lint_git.must_not_skip(f"git failed mid-run: {exc}")
        print(f"ERROR: secret scan could not read git ({exc}); this is not a pass", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
