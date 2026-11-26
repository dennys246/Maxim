"""#1801: ``redact_urls`` with the tempered catch-all regex (docs/plans/log_url_redaction_1801_approach.md v28, §Shape
``redact_urls``), and the URL helpers' move to the stdlib-only ``utils/net.py``.

Every case set the design rounds verified on the scratch patch (rounds 15-27: ``cases4``, ``cases5``, ``af3``,
``lin`` and round 25's two-URL set) is committed here. Each owner exception (E1801-1..4) is pinned by id: a residual
is ASSERTED to leak, so a change in either direction is noticed.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path, PureWindowsPath

import pytest

from maxim.utils.net import redact_urls, url_scrubbed


# (input, exact output). Rounds 15-24 (cases4 / cases5): whitespace userinfo, quotes, backslashes, JSON escaping.
_EXACT: list[tuple[str, str]] = [
    # cases4: Z1, Y1-Y3, X1/X2, V1, W1
    ("https://u:p\r\nw@h/p", "<url>"),
    ("https://u\ns:p w@h/p", "<url>"),
    ("https://u:a b\nc@h/p", "<url>"),
    ('https://us er:p@ss"x@h/p', "<url>"),
    ("https://tok en@h/p", "<url>"),
    ("https://D\\u:p w@h/x", "<url>"),
    ("https://u:a b@c d@h/p", "<url>"),
    ("https://us er:pw@h/p", "<url>"),
    ("redis://:PW D@cache:6379/0", "<url>"),
    ("https://u:p@ss w@h/p", "<url>"),
    ('https://to"k@h/p', "https://h"),
    ("https://D\\u:pw@h", "https://h"),
    ("postgres://u:PW@db/x", "<url>"),
    ("x https:/u:PW@h/p", "x <url>"),
    ("https:\\/\\/u:p w@h\\/p", "<url>"),
    ("https://ok.example/path and later a@b", "<url>"),
    ("GET https://h failed\nmail bob@ex.com", "GET <url>"),
    ("https://plain.example/a?b=c", "https://plain.example"),
    # cases4 tail / AA1, AA2: quotes and backslashes in the userinfo
    ('https://u:p\\"SECRET@h/p', "https://h"),
    ('https://D\\u:p"SECRET@h/p', "https://h"),
    ("https://a\\b:p'SECRET@h", "https://h"),
    ("https://u:p\\<SECRET@h", "https://h"),
    ("https://D\\u s:SECRET@h", "<url>"),
    ("https://u s\\x:SECRET@h", "<url>"),
    ("https://D\\u SECRET@h", "<url>"),
    ("https://a\\b@h", "https://h"),
    # cases5: AB1, the separator forms
    ('https:\\/\\/u:p\\"SECRET@h\\/p', "<url>"),
    ('https:\\u:p\\"SECRET@h', "<url>"),
    ("https:\\/\\/u:p\\<SECRET@h", "<url>"),
    ("https://u:p\r\nSECRET@h/p", "<url>"),
    ("https://tok SECRET@h/p", "<url>"),
    ('https://us er:p@SECRET"x@h/p', "<url>"),
    ("redis://:SECRET D@cache:6379/0", "<url>"),
    ('https://to"SECRET@h/p', "https://h"),
    ("postgres://u:SECRET@db/x", "<url>"),
    ("x https:/u:SECRET@h/p", "x <url>"),
    ('{"u":"https:\\/\\/u:SECRET@h\\/p?key=K"}', '{"u":"<url>'),
    ("unix:///s.sock", "<url>"),
    ("123://u:SECRET@h", "123<url>"),
    ("http:/ prose", "http:/ prose"),
    # AC1: whitespace userinfo on the separator branch
    ("https:\\/\\/u:p SECRET@h\\/p", "<url>"),
    ('{"u":"https:\\/\\/u:p SECRET@h\\/p"}', '{"u":"<url>'),
    ("https:\\/\\/us SECRET:p@h", "<url>"),
    ("https:\\/\\/tok SECRET@h", "<url>"),
    ("https:\\u:p SECRET@h", "<url>"),
    ("ws:\\\\u:p SECRET@h", "<url>"),
    ("https:\\/\\/u:p\tSECRET@h", "<url>"),
    # AD1 / AD2 (round 22)
    ("https:\\/u:p SECRET@h", "<url>"),
    ("https:\\\\/u:p SECRET@h", "<url>"),
    ("https:\\\\\\/\\\\\\/u:p SECRET@h", "<url>"),
    (json.dumps(json.dumps("https://u:p SECRET@h")), '"\\"<url>'),
    ("https://bob@ex.com:p SECRET@h/p", "<url>"),
    ("https:\\/\\/bob@ex.com:p SECRET@h", "<url>"),
    ("https://u:p@h/x", "https://h"),
    ("https://bob@ex.com@h/p", "https://h"),
    # AE1 (round 23): an ``@`` in the user
    ("https://bob@ex.com SECRET@h/p", "<url>"),
    ("https://a@b@c SECRET@h", "<url>"),
    ("https://a@b\tSECRET@h", "<url>"),
    ("https://a@b\nSECRET@h", "<url>"),
    ("https:\\/\\/bob@ex.com SECRET@h\\/p", "<url>"),
    ('{"u":"https://bob@ex.com SECRET@h/p"}', '{"u":"<url>'),
    ("https://u@h see bob@x", "<url>"),
    # round 24: a ``/`` after the whitespace, an ``@`` with nothing after it
    ("https://tok SEC/RET@h", "<url>"),
    ("https://x-access-token ab/SECRET+ef=@github.com/o/r", "<url>"),
    ("https://tok SEC/RET:x@h", "<url>"),
    ("https://a b/SECRET:p@h", "<url>"),
    ("https:\\/\\/tok SEC\\/RET@h", "<url>"),
    ("https://a@b SECRET@ ", "<url> "),
    ("https://tok SECRET@ h", "<url> h"),
    # AF3 / AG2: ordinary text with a colon and a backslash or slash is left alone
    ("nearby:\\n\\nmore", "nearby:\\n\\nmore"),
    ("line1:\\nline2", "line1:\\nline2"),
    ('re.compile("ab:\\d+")', 're.compile("ab:\\d+")'),
    ("note:\\ x a@b", "note:\\ x a@b"),
    ("run:\\n", "run:\\n"),
    ("C:\\Users\\me", "C:\\Users\\me"),
    ("path:/Users/me", "path:/Users/me"),
    ("zzz:\\/\\/u:SECRET@db", "<url>"),
    ("postgres:\\/\\/u:SECRET@db", "<url>"),
    ("mongodb+srv:\\/\\/u:SECRET@h", "<url>"),
    # W1: every scheme on the double-slash form
    ("redis://:PW@h", "<url>"),
    ("git+ssh://u:PW@h", "<url>"),
    ("https://u:p w@h/p", "<url>"),
]


@pytest.mark.parametrize(("text", "expected"), _EXACT)
def test_redact_urls_exact(text: str, expected: str) -> None:
    assert redact_urls(text) == expected


@pytest.mark.parametrize(("text", "expected"), _EXACT)
def test_redact_urls_idempotent(text: str, expected: str) -> None:
    assert redact_urls(redact_urls(text)) == expected


@pytest.mark.parametrize(
    "path",
    [
        str(Path("/a/HTTPS://u:SECRET@h/p")),  # U2
        str(Path("/a/ftp://u:SECRET@h")),  # V6
        str(Path("/a/wss://u:SECRET@h")),
        str(Path("/a/https://u:p SECRET@h/p")),  # AC1
        str(Path("/a/redis://:SECRET@h")),  # E1801-3: a listed scheme reduces
        "/a/1https:/u:SECRET@h/p",
        str(PureWindowsPath("C:/a/https://u:SECRET@h/p")),  # U5
    ],
)
def test_path_collapsed_forms_reduce(path: str) -> None:
    out = redact_urls(path)
    assert "SECRET" not in out, out
    assert "<url>" in out


def test_postgres_backslash_form_reduces() -> None:
    """E1801-3: a LISTED scheme in the bare-backslash Windows form is recognised."""
    assert redact_urls("postgres:\\u:SECRET@h") == "<url>"


def test_e1801_3_unlisted_scheme_leaks_in_path_and_windows_forms() -> None:
    """E1801-3 (owner, AG2): an unlisted scheme is not recognised in the lone-``/`` or bare-backslash forms. Asserted,
    so a change is noticed."""
    assert "SECRET" in redact_urls(str(Path("/a/zzz://:SECRET@h")))
    assert "SECRET" in redact_urls("zzz:\\u:SECRET@h")


def test_e1801_4_url_start_inside_whitespace_userinfo_leaks_outer() -> None:
    """E1801-4 (owner, AG1): tempering ends the outer match at the inner URL's start. Asserted, so a change is noticed."""
    assert redact_urls("https://tok SECRET https://x@h") == "https://tok SECRET https://h"


_P2_FIRST = ("https://a.com/x", "https:\\/\\/a.com", "git+https://a.com")
_P2_SECOND = (
    "https://bob@ex.com:p SECRET@h/p",
    "https://u:p@ss SECRET@h/p",
    "https:\\/\\/bob@ex.com:p SECRET@h",
    "https://a@b SECRET@h",
)


@pytest.mark.parametrize("first", _P2_FIRST)
@pytest.mark.parametrize("second", _P2_SECOND)
def test_two_url_set_each_url_reduces(first: str, second: str) -> None:
    """Round 25's two-URL set (AF1): one URL's match never swallows the next URL's userinfo, at any gap length
    across the catch-all's bound."""
    assert "SECRET" not in redact_urls(second)
    for fill in range(560, 660):
        out = redact_urls(first + " " + "y" * fill + " " + second)
        assert "SECRET" not in out, (fill, out[-60:])


@pytest.mark.parametrize(
    "text",
    [
        "https://a.com https://bob@ex.com:p SECRET@h",
        "https://a.com\nhttps://u:p@ss SECRET@h",
        "https://a.com https:\\/\\/a@b SECRET@h",
    ],
)
def test_two_url_adjacent(text: str) -> None:
    assert "SECRET" not in redact_urls(text)


# E1801-1: the catch-all's two bounds, 640/641, on each branch. Each part carries a distinct marker, compared
# case-insensitively (``url_origin`` lowercases a host, AA4).
@pytest.mark.parametrize("sep", ["https://", "https:\\/\\/"])
def test_e1801_1_first_bound(sep: str) -> None:
    def run(k: int) -> str:
        return redact_urls(sep + "u" * (k - 1) + "X" + " " + "pq" + "@h/p").lower()

    at = run(640)
    assert "x" not in at and "pq" not in at
    past = run(641)
    assert "pq" in past  # past the bound the tail leaks (E1801-1), asserted so a change is noticed


@pytest.mark.parametrize("sep", ["https://", "https:\\/\\/"])
def test_e1801_1_second_bound(sep: str) -> None:
    def run(m: int) -> str:
        return redact_urls(sep + "ux" + " " + "p" * (m - 1) + "Q" + "@h/p").lower()

    at = run(640)
    assert "ux" not in at and "q" not in at
    past = run(641)
    assert "q" in past


@pytest.mark.parametrize(
    "text",
    [
        "see https://a.com and user:x@y",
        "https://h:8080 failed, mail bob@ex.com",  # X4: the host:port over-match
        "https://u:p@h/x, contact a@b",
        "https://h/x\nline2 a@b",  # across a line break
    ],
)
def test_e1801_2_overmatch_is_erased(text: str) -> None:
    """E1801-2 (fail-closed): a URL followed in bounds by whitespace and any ``@`` is erased to ``<url>``."""
    out = redact_urls(text)
    assert "<url>" in out
    assert "@" not in out


@pytest.mark.parametrize(
    ("text", "expected"),
    [("follows:\\n1.", "<url>."), ("Windows:\\n", "<url>"), ("views:/x", "<url>"), ("mysql:\\n", "<url>")],
)
def test_listed_scheme_suffix_overmatch_pinned(text: str, expected: str) -> None:
    """AH1: a word ENDING in a listed scheme counts as one (fail-closed over-match, loses text, leaks nothing)."""
    assert redact_urls(text) == expected


def test_prefilter_returns_text_unchanged_without_colon_slash() -> None:
    text = "plain text with a@b and no scheme" * 10
    assert redact_urls(text) is text or redact_urls(text) == text


def test_redact_urls_never_raises_on_non_str() -> None:
    assert redact_urls(12) == "12"  # type: ignore[arg-type]


_LINEAR_FAMILIES = {
    "a:// ": lambda n: "a:// " * (n // 5),
    "a://@\\n": lambda n: "a://@\n" * (n // 6),
    "a://x @": lambda n: "a://x @" * (n // 7),
    "a:// x@end": lambda n: "a:// x" * (n // 6) + "@",
    "https:\\/ ": lambda n: "https:\\/ " * (n // 9),
    "@ ": lambda n: "https://" + "@ " * (n // 2),
    'https:\\"': lambda n: 'https:\\"' * (n // 8),
    "long-nospace": lambda n: "https://" + "a" * n,
    "space-noat": lambda n: "https://" + "a " * (n // 2),
    "sep-q": lambda n: 'https:\\/\\/u:p\\"' * (n // 16),
    "a://: ": lambda n: "a://: " * (n // 6),
    "a://u:x x x": lambda n: "a://u:x x x " * (n // 12),
    "bs": lambda n: ("https:\\" + "\\" * 30) * (n // 38),
    "tok en": lambda n: "https://t o" * (n // 11),
    "u: a@": lambda n: "https://u:" + " a@" * (n // 3),
    "nl": lambda n: "https://u:a\n" * (n // 12),
    "/a/https:/b ": lambda n: "/a/https:/b " * (n // 12),  # AF1: the family that took 1.08 s before tempering
}


def _best_of(fn, text: str, reps: int = 3) -> float:
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn(text)
        best = min(best, time.perf_counter() - t0)
    return best


@pytest.mark.parametrize("name", sorted(_LINEAR_FAMILIES))
def test_redact_urls_is_linear(name: str) -> None:
    """AF1, Z2, AB1: every adversarial family stays linear. 4x the input may take at most ~10x the time (a quadratic
    scan takes 16x), with an absolute floor so a fast machine's noise cannot fail it."""
    make = _LINEAR_FAMILIES[name]
    small = _best_of(redact_urls, make(32_000))
    large = _best_of(redact_urls, make(128_000))
    assert large <= 10 * small + 0.05, (name, small, large)
    assert large < 2.0, (name, large)


def test_regex_matches_identically_on_this_interpreter_compiles() -> None:
    """W1/Y: the regex compiles on the running interpreter (3.10 is the floor CI runs)."""
    from maxim.utils import net

    assert net._URL_IN_TEXT.pattern.startswith("(?<![A-Za-z0-9+.-])")


# ─── url_scrubbed (moved from simulation/report.py) ───────────────────────


def test_url_scrubbed_walks_and_stringifies() -> None:
    exc = RuntimeError("GET https://u:SECRET@h/x failed")
    out = url_scrubbed({"a": [("https://u:SECRET@h/p", 1)], "e": exc, "n": None, "f": 1.5, "b": True})
    assert out == {"a": [["https://h", 1]], "e": "GET https://h failed", "n": None, "f": 1.5, "b": True}


def test_url_scrubbed_depth_cap() -> None:
    v: object = "https://u:SECRET@h/p"
    for _ in range(20):
        v = [v]
    assert "SECRET" not in json.dumps(url_scrubbed(v))


def test_url_scrubbed_keeps_numpy_numeric_scalars() -> None:
    """Code review Wire NIT: numpy NUMERIC scalars (bool, int, float, complex kinds) pass through unchanged, by exact
    type, as the record factory treats them; arrays and text scalars keep the stringify-and-reduce behaviour."""
    np = pytest.importorskip("numpy")
    scalars = [np.bool_(True), np.int64(3), np.uint8(7), np.float32(1.5), np.complex128(1j)]
    out = url_scrubbed({"v": scalars})["v"]
    assert all(type(o) is type(s) and o == s for o, s in zip(out, scalars))
    assert url_scrubbed(np.array([1, 2])) == "[1 2]"
    assert url_scrubbed(np.str_("https://u:SECRET@h/p")) == "https://h"


def test_url_scrubbed_never_imports_numpy() -> None:
    code = (
        "import sys\n"
        "from maxim.utils.net import url_scrubbed\n"
        "url_scrubbed({'o': object(), 'e': RuntimeError('https://u:p@h/x')})\n"
        "print('numpy' in sys.modules)\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60, check=True)
    assert out.stdout.strip().splitlines()[-1] == "False", out.stdout + out.stderr


def test_http_reexports_the_net_helpers() -> None:
    from maxim.utils import http, net

    assert http.redact_urls is net.redact_urls
    assert http.url_origin is net.url_origin
    assert http.url_scrubbed is net.url_scrubbed


def test_logging_modules_load_no_httpx() -> None:
    """S5: the logging path imports the helpers from the stdlib-only ``net``; none of these load httpx."""
    code = (
        "import sys\n"
        "import maxim.utils.logging, maxim.utils.structured_logging, maxim.simulation.sim_logger, maxim.tools.base\n"
        "import maxim.utils.logging as L\n"
        "L.configure_logging(0, force=True)\n"
        "print('httpx' in sys.modules)\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60, check=True)
    assert out.stdout.strip().splitlines()[-1] == "False", out.stdout + out.stderr
