"""``scripts/lint_cut_before_redact.py`` (#1801 F2): the ratchet over cuts of exception text before URL reduction.

Its planted-form tests are EXACTLY its stated sources and cuts (AF4); its blind spots are tested only as documented
non-detections, so a change in either direction is noticed.
"""

from __future__ import annotations

import importlib.util
import re
import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def _load():
    spec = importlib.util.spec_from_file_location(
        "lint_cut_before_redact_under_test", REPO / "scripts" / "lint_cut_before_redact.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.modules.pop(spec.name, None)
    return mod


lint = _load()


def _hits(src: str) -> list[str]:
    return [h.text for h in lint.hits_in_source(textwrap.dedent(src), "src/maxim/x.py")]


def _in_except(body: str) -> str:
    return "def f(proc, duration):\n    try:\n        pass\n    except Exception as e:\n" + textwrap.indent(
        body, "        "
    )


def test_branch_is_clean() -> None:
    assert lint.lint() == 0


@pytest.mark.parametrize(
    "body",
    [
        "x = str(e)[:50]",
        "x = redact_urls(str(e)[:50])",  # the slice runs first
        "x = f'{e!r}'[:50]",
        "msg = str(e)\ny = msg[:50]",
        "x = proc.stdout[-500:]",
        "x = (proc.stderr or '')[-500:]",
        # M1: format precision and other stated cutters
        "x = '{:.50}'.format(e)",
        "x = format(e, '.50')",
        "x = '%.*s' % (50, e)",
        "x = '%.50s' % e",
        "x = textwrap.shorten(str(e), 50)",
        "x = json.dumps({'k': str(e)})[:50]",
        # L2
        "x = dict(m=str(e))['m'][:50]",
        "x = f'{e!s:.50}'",
        # J3
        "if proc:\n    msg = redact_urls(str(e))\nelse:\n    msg = str(e)\ny = msg[:50]",
        "s = ''\ns += str(e)\ny = s[:50]",
        "x = [str(e)][0][:50]",
        "x = ('x %s' % e)[:50]",
        # K3
        "for line in proc.stderr:\n    y = line[:50]",
        "a, b = str(e), 1\ny = a[:50]",
        "y = (m := str(e))[:50]",
        "x = ' '.join(str(e).split())[:50]",
        "x = json.dumps({'k': str(e)})[:200]",
        # V2 / W2 / U3: split and partition results
        "head, _, _ = str(e).partition('@')",
        "x = str(e).split('@').pop(0)",
        "parts = str(e).split('@')\ny = parts[0]",
        "x = re.split('@', str(e))[0]",
        "x = str.partition(e, '@')[0]",
        "x = str(e).partition('@')[0]",
        "x = str(e).split(sep)[0]",
        # T2: a slice object
        "x = str(e)[slice(0, 50)]",
        "x = traceback.format_exc()[-500:]",
    ],
)
def test_planted_form_fails(body: str) -> None:
    assert _hits(_in_except(body)), body


@pytest.mark.parametrize(
    "body",
    [
        "x = redact_urls(str(e))[:50]",
        "msg = redact_urls(str(e))\ny = msg[:50]",
        "x = f'{duration:.1f}'",
        "x = f'{type(e).__name__}'[:50]",
        "x = url_scrubbed(str(e))[:50]",
        "x = str(e)",
        "x = e.args[0]",
        "x = f'{e.elapsed:.1f}'",  # a numeric presentation type is not a text cut
    ],
)
def test_reduced_or_uncut_form_passes(body: str) -> None:
    assert not _hits(_in_except(body)), body


@pytest.mark.parametrize(
    "src",
    [
        "def f(e):\n    return str(e)[:50]",
        "def f(exc):\n    return str(exc)[:50]",
        "def f(err):\n    return err[:50]",
        "def f(error):\n    return error[:50]",
        "def f(x: ValueError):\n    return str(x)[:50]",
    ],
)
def test_parameter_sources(src: str) -> None:
    assert _hits(src)


def test_p3_scopes_module_class_lambda_closure() -> None:
    assert _hits("try:\n    pass\nexcept Exception as e:\n    x = str(e)[:50]\n")
    assert _hits("class C:\n    try:\n        pass\n    except Exception as e:\n        x = str(e)[:50]\n")
    assert _hits("g = lambda e: str(e)[:50]\n")
    assert _hits(
        "def f():\n    try:\n        pass\n    except Exception as e:\n        def g():\n            return str(e)[:50]\n"
    )


def test_documented_blind_spots_are_not_detected() -> None:
    """Stated, not chased: an indirect callee, a cut inside a helper, an exotic cutter. Asserted as NON-detections, so a
    change is noticed."""
    assert not _hits(_in_except("cut = getattr(str(e), '__getitem__')\nx = cut(slice(0, 50))"))
    assert not _hits("def helper(text):\n    return text[:50]\n" + _in_except("x = helper(str(e))"))
    assert not _hits(_in_except("x = textwrap.wrap(str(e), 20)[1]"))
    assert not _hits(_in_except("x = itertools.islice(str(e), 50)"))
    assert not _hits(_in_except("x = '%.50s'.__mod__(e)"))


def _key(file: str = "src/maxim/x.py", scope: str = "f") -> str:
    return f"{file}::{scope}"


def test_m2_new_cut_inside_an_allowlisted_function_fails() -> None:
    found = lint.hits_in_source(_in_except("a = proc.stdout[:10]\nb = str(e)[:200]"), "src/maxim/x.py")
    allow = {_key(): {"proc.stdout[:10]": (1, "reason")}}
    errors = lint.judge(found, allow)
    assert len(errors) == 1 and "str(e)[:200]" in errors[0]


def test_p2_second_identical_hit_beyond_count_fails() -> None:
    found = lint.hits_in_source(_in_except("a = proc.stdout[:10]\nb = proc.stdout[:10]"), "src/maxim/x.py")
    errors = lint.judge(found, {_key(): {"proc.stdout[:10]": (1, "reason")}})
    assert len(errors) == 1 and "more than its allowlist count" in errors[0]
    assert lint.judge(found, {_key(): {"proc.stdout[:10]": (2, "reason")}}) == []


def test_q2_stale_entry_fails() -> None:
    found = lint.hits_in_source(_in_except("a = proc.stdout[:10]"), "src/maxim/x.py")
    errors = lint.judge(found, {_key(): {"proc.stdout[:10]": (2, "reason")}})
    assert len(errors) == 1 and "stale" in errors[0]
    assert "stale" in lint.judge([], {_key(): {"gone[:1]": (1, "r")}})[0]


def test_every_allowlist_entry_has_a_reason() -> None:
    for key, entries in lint.ALLOWLIST.items():
        assert "::" in key
        for text, (count, reason) in entries.items():
            assert count >= 1 and reason.strip(), (key, text)


# Each site this PR fixed, with its reduction deleted, is flagged again (the lint's own deletion probe).
_FIXED_SITES = [
    (
        "src/maxim/runtime/loop_planning.py",
        'redact_urls(f"an earlier approved action raised {type(exc).__name__}: {exc}")[:300]',
    ),
    ("src/maxim/models/language/router.py", "redact_urls(str(e))[:50]"),
    ("src/maxim/runtime/leader_proxy.py", 'redact_urls(result.stderr or "")[-500:]'),
    ("src/maxim/runtime/leader_proxy.py", "redact_urls(r.stderr)[-500:]"),
    ("src/maxim/runtime/leader_proxy.py", 'redact_urls(r.stdout or "")[-500:]'),
    ("src/maxim/runtime/leader_proxy.py", "redact_urls(result.stdout)[-500:]"),
    ("src/maxim/runtime/leader_proxy.py", "redact_urls(pip_result.stdout)[-500:]"),
    ("src/maxim/models/language/openai_backend.py", "redact_urls(sanitized_error)[:200]"),
    ("src/maxim/bench/recovery_time.py", "redact_urls(fix_hint)[:500]"),
    ("src/maxim/runtime/tool_dispatch.py", 'redact_urls(error or "")[:50]'),
    ("src/maxim/agents/context_pool.py", "redact_urls(error)[:100]"),
]


@pytest.mark.parametrize(("rel", "reduced"), _FIXED_SITES)
def test_fixed_site_flagged_when_its_reduction_is_deleted(rel: str, reduced: str) -> None:
    source = (REPO / rel).read_text(encoding="utf-8")
    assert reduced in source, f"{rel}: the fixed site moved; update this table"
    m = re.fullmatch(r"redact_urls\((.*)\)(\[.*\])", reduced)
    assert m
    inner = m.group(1)
    raw = (f"({inner})" if " " in inner else inner) + m.group(2)
    before = len(lint.hits_in_source(source, rel))
    after = len(lint.hits_in_source(source.replace(reduced, raw, 1), rel))
    assert after > before, (rel, raw)


def test_code_tools_flagged_when_its_reduction_is_deleted() -> None:
    rel = "src/maxim/tools/code_tools.py"
    source = (REPO / rel).read_text(encoding="utf-8")
    reduced = "stdout_text, stderr_text = redact_urls(result.stdout), redact_urls(result.stderr)"
    assert reduced in source
    raw = source.replace(reduced, "stdout_text, stderr_text = result.stdout, result.stderr", 1)
    assert len(lint.hits_in_source(raw, rel)) > len(lint.hits_in_source(source, rel))
