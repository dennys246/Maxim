"""scripts/lint_secrets.py: the secret scan on data PRs (#1081; owner decisions 2026-10-08).

Every test builds a real temporary git repository, because the plumbing (which blobs a range introduced, which
copy of the allowlist and pattern table acts) is where a silent no-op would live. Each row of the approach note's
"how could this input lie" table has a test here.

No credential-shaped literal appears in this file (adversarial pass S2): every fake is BUILT at runtime by
:func:`fake`, so this file passes the scan it tests with an empty allowlist.
"""

from __future__ import annotations

import bz2
import gzip
import hashlib
import io
import json
import lzma
import os
import subprocess
import tarfile
import time
import zipfile
from pathlib import Path

import pytest

from scripts import lint_secrets as L
from tests.unit._push_event_helpers import fake_push, rev

DATA = "docs/experiments/data/"


def _body(n: int) -> str:
    """Mixed-case, digit-bearing filler: never a placeholder run."""
    return "".join("Q7pZk2Rt"[i % 8] for i in range(n))


def fake(kind: str) -> str:
    """A credential-shaped fake, assembled here so no literal exists in the source."""
    upper = "".join("Q7PZK2RT"[i % 8] for i in range(16))
    return {
        "anthropic": "sk-" + "ant-" + _body(24),
        "openai": "sk-" + _body(24),
        "github": "gh" + "p_" + _body(36),
        "github-pat": "github" + "_pat_" + _body(30),
        "aws": "AK" + "IA" + upper,
        "huggingface": "hf" + "_" + _body(32),
        "slack": "xo" + "xb-" + _body(16),
        "google": "AI" + "za" + _body(35),
        "groq": "gs" + "k_" + _body(24),
        "maxim-console": "mx" + "c_" + _body(43),
        "private-key": "-" * 5 + "BEGIN RSA " + "PRIVATE" + " KEY" + "-" * 5,
        "auth-header": "Authorization: " + "Bear" + "er " + _body(43),
        "key-assignment": "api" + '_key = "' + _body(40) + '"',
        "email": "jane.doe" + "@" + "acme-corp.io",
    }[kind]


def _token(kind: str) -> str:
    """The part of :func:`fake` the scanner hashes (the token itself)."""
    text = fake(kind)
    return {"auth-header": text.rsplit(" ", 1)[-1], "key-assignment": _body(40)}.get(kind, text)


def _sha(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


# ── fixture repositories ─────────────────────────────────────────────────────


def _git(repo: Path, *args: str) -> str:
    env = dict(
        os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@t"
    )
    return subprocess.run(["git", *args], cwd=repo, env=env, capture_output=True, text=True, check=True).stdout


def _write(repo: Path, rel: str, content: str | bytes) -> None:
    p = repo / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        p.write_bytes(content)
    else:
        p.write_text(content)


def _commit(repo: Path, files: dict[str, str | bytes | None] | None = None, msg: str = "change") -> str:
    for rel, content in (files or {}).items():
        if content is None:
            (repo / rel).unlink()
        else:
            _write(repo, rel, content)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "--allow-empty", "-m", msg)
    return rev(repo, "HEAD")


@pytest.fixture(autouse=True)
def _no_ci_env(monkeypatch):
    for name in ("GITHUB_EVENT_NAME", "GITHUB_EVENT_PATH"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """`main` at one base commit; HEAD on a `feature` branch cut from it."""
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q", "-b", "main")
    _git(root, "config", "commit.gpgsign", "false")
    _commit(root, {"README.md": "hello\n", "docs/notes.md": "notes\n", DATA + "keep.txt": "data\n"}, "base")
    _git(root, "checkout", "-q", "-b", "feature")
    return root


def _base(repo: Path) -> str:
    return rev(repo, "main")


def _scan(repo: Path, **kw) -> L.Scan:
    return L.scan_range(repo, _base(repo), "HEAD", pr_text=kw.pop("pr_text", None), **kw)


def _names(scan: L.Scan) -> set[str]:
    return {f.pattern for f in scan.findings}


def _allow(*entries: dict) -> str:
    return json.dumps({"_format_version": "1.0", "entries": list(entries)}, indent=2) + "\n"


def _entry(path: str, token: str, pattern: str = "openai") -> dict:
    return {
        "pattern": pattern,
        "path": path,
        "match_sha256": _sha(token),
        "owner": "dennys246",
        "date": "2026-10-08",
        "ref": "#1081",
        "reason": "deliberate test fake",
    }


# ── 1. one positive control per pattern ──────────────────────────────────────


@pytest.mark.parametrize(
    ("kind", "name"),
    [
        ("anthropic", "anthropic"),
        ("openai", "openai"),
        ("github", "github"),
        ("github-pat", "github"),
        ("aws", "aws"),
        ("huggingface", "huggingface"),
        ("slack", "slack"),
        ("google", "google"),
        ("groq", "groq"),
        ("maxim-console", "maxim-console"),
        ("private-key", "private-key"),
        ("auth-header", "auth-header"),
        ("key-assignment", "key-assignment"),
    ],
)
def test_each_key_pattern_FAILS_on_any_path(repo, kind, name):
    _commit(repo, {"src/config.txt": f"before\nline {fake(kind)} after\n"})
    scan = _scan(repo)
    assert name in _names(scan), (kind, scan.findings)
    assert any(f.where == "src/config.txt:2" for f in scan.findings)


def test_an_email_FAILS_under_data_and_passes_elsewhere(repo):
    _commit(repo, {DATA + "run/log.txt": f"from {fake('email')}\n", "docs/contact.md": f"{fake('email')}\n"})
    scan = _scan(repo)
    assert [(f.path, f.pattern) for f in scan.findings] == [(DATA + "run/log.txt", "email")]


@pytest.mark.parametrize(
    "text",
    [
        "a" + "@" + "example.com",
        "b" + "@" + "mail.example.org",
        "c" + "@" + "host.invalid",
        "d" + "@" + "box.test",
        "def f():\\n" + "@" + "pytest.mark.slow",
    ],
)
def test_reserved_domains_and_escaped_decorators_are_not_emails(repo, text):
    _commit(repo, {DATA + "x.txt": text + "\n"})
    assert _scan(repo).findings == []


@pytest.mark.parametrize("text", ["AK" + "IA" + "X" * 16, "sk-" + "x" * 30, "token " + "0" * 40])
def test_placeholders_are_not_findings(repo, text):
    _commit(repo, {"docs/guide.md": f"use {text} here\n"})
    assert _scan(repo).findings == []


def test_the_lint_and_its_tests_do_not_match_themselves():
    """S2: no fake key or PEM header literal in the regex sources, docstrings or this test file."""
    scan = L.Scan(L.compile_patterns(L.PATTERNS), set())
    for rel in ("scripts/lint_secrets.py", "tests/unit/test_lint_secrets.py"):
        scan.blob(rel, (Path(L.REPO_ROOT) / rel).read_bytes())
    assert scan.findings == []


def test_an_adversarial_4MB_string_scans_in_bounded_time():
    """S1: bounded quantifiers. The lint job's timeout is 5 minutes for every step."""
    scan = L.Scan(L.compile_patterns(L.PATTERNS), set())
    for chunk in (b"a" * 4_000_000, b"a@" * 2_000_000, b"a." * 2_000_000, b"x@a." * 1_000_000):
        started = time.monotonic()
        scan.blob(DATA + "adversarial.txt", chunk)
        assert time.monotonic() - started < 20, chunk[:4]


def test_many_matches_scan_in_linear_time():
    """Review N1: 40k distinct keys in ~9 MB of escaped JSON Lines. Per-match line counting from offset 0 and a
    linear dedup scan were each quadratic in the match count (minutes); both passes now run in seconds."""
    rows = [json.dumps({"i": i, "note": "a\nb", "k": fake("openai") + f"{i:06d}"}) for i in range(40_000)]
    data = ("\n".join(r.ljust(225) for r in rows) + "\n").encode()
    assert 8_000_000 < len(data) < 10_000_000
    scan = L.Scan(L.compile_patterns(L.PATTERNS), set())
    started = time.monotonic()
    scan.blob(DATA + "many.jsonl", data)
    elapsed = time.monotonic() - started
    assert len(scan.findings) == 40_000 and scan.findings[-1].where == DATA + "many.jsonl:40000"
    assert elapsed < 8, elapsed


# ── 2. archives ──────────────────────────────────────────────────────────────


def _zip(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in members.items():
            z.writestr(name, data)
    return buf.getvalue()


def _tar(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as t:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            t.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def _line(kind: str = "openai") -> bytes:
    return json.dumps({"turn": 1, "header": fake(kind)}).encode() + b"\n"


@pytest.mark.parametrize(
    ("rel", "blob"),
    [
        (DATA + "s/actions.jsonl.gz", gzip.compress(_line())),
        (DATA + "s/multi.jsonl.gz", gzip.compress(b'{"ok": 1}\n') + gzip.compress(_line())),
        (DATA + "s/bundle.zip", _zip({"inner/report.json": _line()})),
        (DATA + "s/bundle.tar", _tar({"actions.jsonl.gz": gzip.compress(_line())})),
        (DATA + "s/report.json", gzip.compress(_line())),  # magic, not extension
        (DATA + "s/a.jsonl.bz2", bz2.compress(_line())),
        (DATA + "s/a.jsonl.xz", lzma.compress(_line())),
        ("docs/elsewhere.gz", gzip.compress(_line())),
    ],
    ids=["gzip", "gzip-multi-member", "zip", "gz-in-tar", "gzip-named-json", "bz2", "xz", "gzip-outside-data"],
)
def test_a_key_inside_an_archive_FAILS(repo, rel, blob):
    _commit(repo, {rel: blob})
    assert "openai" in _names(_scan(repo))


def test_an_email_inside_a_gzipped_data_log_FAILS(repo):
    _commit(repo, {DATA + "s/actions.jsonl.gz": gzip.compress(_line("email"))})
    assert _names(_scan(repo)) == {"email"}


@pytest.mark.parametrize(
    ("blob", "needle"),
    [
        (gzip.compress(_line())[:-12] + b"\0" * 12, "corrupt"),
        (b"\x28\xb5\x2f\xfd" + b"\0" * 40, "zstd"),
        (b"version https://git-lfs.github.com/spec/v1\noid sha256:" + b"0" * 64 + b"\nsize 9\n", "lfs-pointer"),
        (gzip.compress(gzip.compress(gzip.compress(gzip.compress(_line())))), "nested deeper"),
    ],
    ids=["corrupt-gzip", "zstd", "lfs-pointer", "nested-4-deep"],
)
def test_an_uninspectable_data_blob_FAILS_and_only_warns_elsewhere(repo, blob, needle):
    _commit(repo, {DATA + "s/x.jsonl.gz": blob, "docs/x.jsonl.gz": blob})
    scan = _scan(repo)
    assert len(scan.problems) == 1 and DATA + "s/x.jsonl.gz" in scan.problems[0] and needle in scan.problems[0]
    assert any(w.startswith("docs/x.jsonl.gz") for w in scan.warnings)


def test_the_decompression_cap_is_cumulative_across_members_and_nesting(repo):
    """S6: two members, each under the cap, together over it; and a gzip whose header claims nothing."""
    half = b"0123456789abcdef" * 4096  # 64 KiB
    _commit(repo, {DATA + "s/two.zip": _zip({"a.txt": half, "b.txt": half}), DATA + "s/one.gz": gzip.compress(half)})
    scan = _scan(repo, cap=100 * 1024)
    assert [p.split(":")[0] for p in scan.problems] == [DATA + "s/two.zip"], scan.problems
    assert "cap" in scan.problems[0]


def test_a_gitlink_under_data_FAILS(repo):
    _git(repo, "update-index", "--add", "--cacheinfo", f"160000,{_base(repo)},{DATA}sub")
    _git(repo, "commit", "-q", "-m", "gitlink")
    scan = _scan(repo)
    assert any("gitlink" in p for p in scan.problems), scan.problems


# ── 3. encodings ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("rel", "blob"),
    [
        ("docs/a.txt", ("﻿" + fake("openai")).encode("utf-16-le")),
        (DATA + "a.txt", fake("openai").encode("utf-16-le")),  # no BOM: by where the NULs sit
        (DATA + "a.json", fake("openai").encode("utf-16-be")),
        ("docs/a.txt", ("﻿" + fake("openai")).encode("utf-32-le")),
    ],
)
def test_utf16_and_utf32_text_FAILS(repo, rel, blob):
    _commit(repo, {rel: blob})
    assert "openai" in _names(_scan(repo))


def test_a_nul_heavy_text_blob_that_is_not_utf16_FAILS_under_data(repo):
    _commit(repo, {DATA + "a.txt": b"\x00\xd8\x41\x00" * 50})  # UTF-16-LE with lone surrogates
    assert any("NUL" in p for p in _scan(repo).problems)


def test_binary_data_that_is_not_text_typed_is_scanned_raw_not_failed(repo):
    """NIT: NUL density must not fail binary audio under data/ (classified by type)."""
    _commit(repo, {DATA + "clip.wav": b"RIFF" + b"\0" * 400})
    scan = _scan(repo)
    assert scan.problems == [] and scan.findings == []


def test_a_json_escaped_key_FAILS(repo):
    escaped = fake("openai").replace("-", "\\u002d")
    _commit(repo, {DATA + "r.json": '{"header": "' + escaped + '"}\n'})
    scan = _scan(repo)
    assert "openai" in _names(scan) and scan.findings[0].where.endswith("(JSON-decoded)")


def test_a_key_in_json_inside_json_FAILS(repo):
    inner = '{"auth": "' + fake("openai").replace("-", "\\u002d") + '"}'  # JSON text holding an escape
    _commit(repo, {DATA + "r.jsonl": json.dumps({"payload": inner}) + "\n"})
    assert "openai" in _names(_scan(repo))


def test_malformed_jsonl_lines_are_reported(repo):
    _commit(repo, {DATA + "r.jsonl": '{"ok": "a\\nb"}\nnot json\n'})
    assert any("1 malformed JSON Lines line" in w for w in _scan(repo).warnings)


# ── 4. history: the range, not the tip ───────────────────────────────────────


def test_a_key_added_then_deleted_on_the_branch_FAILS(repo):
    _commit(repo, {"docs/notes.md": f"notes {fake('openai')}\n"})
    _commit(repo, {"docs/notes.md": "notes\n"})
    assert "openai" in _names(_scan(repo))


def test_a_key_only_in_a_commit_message_FAILS(repo):
    _commit(repo, {"docs/notes.md": "notes 2\n"}, msg=f"debug\n\ncurl -H '{fake('auth-header')}'")
    scan = _scan(repo)
    assert [(f.path, f.pattern) for f in scan.findings] == [(L.COMMIT_MSG, "auth-header")]


def test_commit_messages_are_not_scanned_for_emails(repo):
    _commit(repo, msg=f"change\n\nCo-Authored-By: someone <{fake('email')}>")
    assert _scan(repo).findings == []


def test_a_pure_rename_of_an_emailed_file_into_data_FAILS(repo):
    _commit(repo, {"docs/contact.md": f"{fake('email')}\n"})
    _git(repo, "checkout", "-q", "main")
    _git(repo, "merge", "-q", "--ff-only", "feature")  # the email file is already on main, outside data/
    _git(repo, "checkout", "-q", "-b", "move")
    _git(repo, "mv", "docs/contact.md", DATA + "contact.md")
    _git(repo, "commit", "-q", "-m", "move")
    assert [(f.path, f.pattern) for f in _scan(repo).findings] == [(DATA + "contact.md", "email")]


def _merge_with_own_change(repo: Path) -> None:
    """A merge of a side branch whose resolution adds docs/evil.md, a file neither parent has."""
    _git(repo, "checkout", "-q", "-b", "side")
    _commit(repo, {"docs/side.md": "side\n"})
    _git(repo, "checkout", "-q", "feature")
    _commit(repo, {"docs/feature.md": "feature\n"})
    _git(repo, "merge", "-q", "--no-commit", "side")
    _write(repo, "docs/evil.md", f"{fake('openai')}\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "merge side")


def test_a_merge_commits_own_change_is_scanned(repo):
    _merge_with_own_change(repo)
    assert "openai" in _names(_scan(repo))


def test_a_merge_commits_own_change_deleted_later_still_FAILS(repo):
    """Review S2: the final diff no longer shows evil.md, so only the merge-scanning path can see it."""
    _merge_with_own_change(repo)
    _commit(repo, {"docs/evil.md": None})
    scan = _scan(repo)
    assert [(f.path, f.pattern) for f in scan.findings] == [("docs/evil.md", "openai")]


def _main_moves_a_seeded_file(repo: Path) -> None:
    """main holds a key-bearing file from before the fork, and touches it AFTER the branch was cut."""
    _git(repo, "checkout", "-q", "main")
    _commit(repo, {"docs/seeded.md": f"seeded {fake('openai')}\n"}, "seed")
    _git(repo, "checkout", "-q", "-B", "feature", "main")
    _commit(repo, {"docs/feature.md": "feature work\n"})
    _git(repo, "checkout", "-q", "main")
    _commit(repo, {"docs/seeded.md": f"seeded {fake('openai')}\nmain edits it again\n"}, "main touches it")


def test_merging_main_into_the_branch_does_not_charge_it_with_mains_blobs(repo):
    """Review S1: "Merge branch 'main' into feature" diffed against parent 1 would show main's edit."""
    _main_moves_a_seeded_file(repo)
    _git(repo, "checkout", "-q", "feature")
    _git(repo, "merge", "-q", "--no-ff", "-m", "Merge branch 'main' into feature", "main")
    scan = _scan(repo)
    assert scan.findings == [] and scan.problems == [], scan.findings


def test_the_synthetic_pr_merge_does_not_charge_the_pr_with_mains_blobs(repo):
    """Review S1: the pull_request checkout is a merge of the PR head INTO main's tip (parent 1 = main)."""
    _main_moves_a_seeded_file(repo)
    _git(repo, "checkout", "-q", "--detach", "main")
    _git(repo, "merge", "-q", "--no-ff", "-m", "Merge feature into main", "feature")
    scan = L.scan_range(repo, L._lint_git.base_ref(repo), "HEAD", pr_text=None)
    assert scan.findings == [] and scan.problems == [], scan.findings
    _commit(repo, {"docs/after.md": f"{fake('groq')}\n"})  # positive control: the PR's own key still fails
    assert _names(L.scan_range(repo, L._lint_git.base_ref(repo), "HEAD", pr_text=None)) == {"groq"}


# ── 5. the allowlist ─────────────────────────────────────────────────────────


def _allowlisted_on_main(repo: Path, rel: str, token: str) -> None:
    _git(repo, "checkout", "-q", "main")
    _commit(repo, {L.ALLOWLIST_REL: _allow(_entry(rel, token))}, "allowlist")
    _git(repo, "checkout", "-q", "-B", "feature", "main")


def test_an_entry_on_the_base_excuses_its_exact_value_and_path(repo):
    _allowlisted_on_main(repo, "tests/test_x.py", fake("openai"))
    _commit(repo, {"tests/test_x.py": f"KEY = '{fake('openai')}'\n"})
    scan = _scan(repo)
    assert scan.findings == [] and [f.path for f in scan.excused] == ["tests/test_x.py"]


def test_the_same_entry_added_on_HEAD_only_does_not_excuse(repo):
    _commit(repo, {"tests/test_x.py": f"KEY = '{fake('openai')}'\n"})
    _commit(repo, {L.ALLOWLIST_REL: _allow(_entry("tests/test_x.py", fake("openai")))})
    assert "openai" in _names(_scan(repo))


def test_an_entry_does_not_excuse_another_value_or_a_renamed_path(repo):
    _allowlisted_on_main(repo, "tests/test_x.py", fake("openai"))
    _commit(repo, {"tests/test_y.py": f"KEY = '{fake('openai')}'\n", "tests/test_x.py": f"{fake('groq')}\n"})
    assert {(f.path, f.pattern) for f in _scan(repo).findings} == {
        ("tests/test_y.py", "openai"),
        ("tests/test_x.py", "groq"),
    }


def test_an_edited_base_entry_is_an_append_only_FAILURE(repo):
    _allowlisted_on_main(repo, "tests/test_x.py", fake("openai"))
    _commit(repo, {L.ALLOWLIST_REL: _allow({**_entry("tests/test_x.py", fake("openai")), "path": "tests/other.py"})})
    assert any("append-only" in p for p in _scan(repo).problems)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda e: e.pop("owner"),
        lambda e: e.update(reason=""),
        lambda e: e.update(ref="issue 1081"),
        lambda e: e.update(date="2026-13-40"),
        lambda e: e.update(match_sha256="sk-" + _body(24)),  # the value instead of its hash
        lambda e: e.update(pattern="email"),  # emails are never allowlisted
        lambda e: e.update(extra="x"),
    ],
)
def test_a_malformed_entry_cannot_check(repo, mutate, capsys):
    e = _entry("tests/test_x.py", fake("openai"))
    mutate(e)
    _commit(repo, {L.ALLOWLIST_REL: _allow(e)})
    assert L.main(["--repo", str(repo)]) == 2
    assert "secret_scan_allowlist.json" in capsys.readouterr().err


def test_the_committed_allowlist_is_well_formed_and_holds_no_values():
    text = (Path(L.REPO_ROOT) / L.ALLOWLIST_REL).read_text()
    entries = L.parse_allowlist(text, "working tree")
    assert all(len(e["match_sha256"]) == 64 for e in entries)


# ── 6. the pattern table cannot shrink in the same PR ─────────────────────────


def test_a_pattern_removed_on_HEAD_still_catches_through_the_base_copy(repo, monkeypatch):
    _git(repo, "checkout", "-q", "main")
    _commit(repo, {L.SELF_REL: (Path(L.REPO_ROOT) / L.SELF_REL).read_text()}, "the lint lands")
    _git(repo, "checkout", "-q", "-B", "feature", "main")
    monkeypatch.setattr(L, "PATTERNS", tuple(p for p in L.PATTERNS if p[0] != "openai"))
    _commit(repo, {"docs/notes.md": f"{fake('openai')}\n"})
    assert "openai" in _names(_scan(repo))


def test_a_renamed_pattern_does_not_lock_the_allowlist(repo, monkeypatch):
    """Review S3: base entries name "openai"; a PR (and then main) renames it. The append-only list cannot be
    rewritten, so validating `pattern` against the table would hold the lint at exit 2 for good."""
    _allowlisted_on_main(repo, "tests/test_x.py", fake("openai"))
    renamed = tuple(("openai-key", *p[1:]) if p[0] == "openai" else p for p in L.PATTERNS)
    monkeypatch.setattr(L, "PATTERNS", renamed)  # HEAD's table; the base has no lint copy here
    _commit(repo, {"tests/test_x.py": f"KEY = '{fake('openai')}'\n"})
    scan = _scan(repo)
    assert scan.findings == [] and [f.pattern for f in scan.excused] == ["openai-key"]
    assert L.main(["--repo", str(repo)]) == 0
    assert L.main(["--repo", str(repo), "--all"]) == 0


def test_an_unparsable_base_table_cannot_check(repo):
    _git(repo, "checkout", "-q", "main")
    _commit(repo, {L.SELF_REL: "PATTERNS = tuple(make())\n"}, "broken")
    _git(repo, "checkout", "-q", "-B", "feature", "main")
    _commit(repo, {"docs/notes.md": "x\n"})
    assert L.main(["--repo", str(repo)]) == 2


# ── 7. base resolution, push mode, PR text ───────────────────────────────────


def test_no_base_on_a_pull_request_is_exit_2_and_locally_a_skip(tmp_path, monkeypatch, capsys):
    root = tmp_path / "lonely"
    root.mkdir()
    _git(root, "init", "-q", "-b", "trunk")
    _git(root, "config", "commit.gpgsign", "false")
    _commit(root, {"a.txt": "a\n"})
    assert L.main(["--repo", str(root)]) == 0
    assert "skipped" in capsys.readouterr().out
    monkeypatch.setenv("GITHUB_EVENT_NAME", "pull_request")
    assert L.main(["--repo", str(root)]) == 2


def _event(tmp_path: Path, monkeypatch, body: str) -> None:
    path = tmp_path / "event.json"
    path.write_text(json.dumps({"pull_request": {"title": "data", "body": body}}))
    monkeypatch.setenv("GITHUB_EVENT_NAME", "pull_request")
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(path))


def test_a_key_in_the_pr_body_FAILS(repo, tmp_path, monkeypatch):
    _commit(repo, {"docs/notes.md": "x\n"})
    _event(tmp_path, monkeypatch, f"repro: {fake('auth-header')}")
    assert L.main(["--repo", str(repo)]) == 1


def test_an_unreadable_pr_payload_cannot_check(repo, monkeypatch):
    _commit(repo, {"docs/notes.md": "x\n"})
    monkeypatch.setenv("GITHUB_EVENT_NAME", "pull_request")
    monkeypatch.setenv("GITHUB_EVENT_PATH", "/nonexistent/event.json")
    assert L.main(["--repo", str(repo)]) == 2


def test_on_push_each_unit_reads_the_allowlist_at_its_own_first_parent(repo, monkeypatch, capsys):
    """S4: an entry that lands in the SAME push unit as the key does not excuse it; one landed before does."""
    _git(repo, "checkout", "-q", "main")
    before = rev(repo, "HEAD")
    _commit(repo, {"tests/a.py": f"{fake('openai')}\n", L.ALLOWLIST_REL: _allow(_entry("tests/a.py", fake("openai")))})
    _commit(repo, {"tests/b.py": f"{fake('groq')}\n", "tests/a.py": f"{fake('openai')}\n# touched\n"})
    fake_push(monkeypatch, repo, before=before, green={before})
    assert L.main(["--repo", str(repo)]) == 1
    err = capsys.readouterr().err
    assert "tests/a.py:1  [openai]" in err  # unit 1: the entry arrived with the key
    assert "tests/b.py:1  [groq]" in err
    assert err.count("tests/a.py") == 1  # unit 2: excused by the entry unit 1 put on main


def test_a_push_without_a_green_base_cannot_check(repo, monkeypatch):
    _git(repo, "checkout", "-q", "main")
    _commit(repo, {"docs/notes.md": "x\n"})
    fake_push(monkeypatch, repo, green=set())
    assert L.main(["--repo", str(repo)]) == 2


# ── 8. output never carries a value ──────────────────────────────────────────


@pytest.mark.parametrize("kind", ["openai", "auth-header", "aws", "email"])
def test_the_output_never_carries_the_value(repo, capsys, kind):
    rel = DATA + "r.txt" if kind == "email" else "docs/r.txt"
    _commit(repo, {rel: f"{fake(kind)}\n"})
    assert L.main(["--repo", str(repo)]) == 1
    out = capsys.readouterr()
    text = out.out + out.err
    token = _token(kind)
    assert token not in text and token[:8] not in text
    if kind != "email":
        assert _sha(token)[:12] in text
    assert "Rotate the credential NOW" in text


# ── 9. whole-tree, staged, git failures ───────────────────────────────────────


def test_all_mode_scans_the_tree_and_reports_stale_entries(repo, capsys):
    _commit(
        repo,
        {
            "tests/test_x.py": f"{fake('openai')}\n",
            "tests/test_y.py": f"{fake('groq')}\n",
            L.ALLOWLIST_REL: _allow(_entry("tests/test_x.py", fake("openai")), _entry("gone.py", fake("slack"))),
        },
    )
    assert L.main(["--repo", str(repo), "--all"]) == 1
    out = capsys.readouterr()
    assert "tests/test_y.py:1  [groq]" in out.err and "test_x.py" not in out.err
    assert "STALE allowlist entry" in out.out and "gone.py" in out.out


def test_staged_mode_scans_the_index(repo):
    _write(repo, DATA + "new.txt", f"{fake('email')}\n")
    _git(repo, "add", "-A")
    assert L.main(["--repo", str(repo), "--staged"]) == 1


def test_a_missing_object_never_reads_as_clean(repo):
    blobs = L.Blobs(repo)
    try:
        with pytest.raises(L.GitUnavailable):
            blobs.get("1" * 40)
    finally:
        blobs.close()
