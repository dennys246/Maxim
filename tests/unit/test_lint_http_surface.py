"""scripts/lint_http_surface.py -- HTTP clients only in utils/http.py; raw dials and egress SDKs pinned.

The grep it backs up (``urllib.request.urlopen``) could not see the urllib3 pool
``tools/internet_search.py::_search_duckduckgo_lite`` fetched through. Deletion probe, 2026-10-26: with
main's internet_search.py restored, ``lint()`` on the real tree exits 1 naming that import.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts import lint_http_surface as L


@pytest.mark.parametrize(
    "src",
    [
        "import urllib3\n",
        "import requests as r\n",
        "from httpx import Client\n",
        "import httpcore\n",
        "def f():\n    import aiohttp\n",
        "from http.client import HTTPConnection\n",
        "from urllib import request\n",
        "from http import client\n",
        "import urllib.request\n",
        "import urllib.parse\nurllib.request.urlopen('x')\n",  # no import of urllib.request at all
        "import importlib\nimportlib.import_module('requests')\n",
        "from maxim.utils.optional_deps import require_optional_dependency\nrequire_optional_dependency('httpx')\n",
        "__import__('urllib3.util')\n",
        "import importlib\nimportlib.import_module(name='httpx')\n",
    ],
)
def test_every_spelling_of_an_http_client_is_caught(src: str) -> None:
    assert L.http_client_uses(src), src


@pytest.mark.parametrize(
    "src",
    [
        "import http.server\n",
        "from urllib.parse import quote_plus\n",
        "import socket\nsocket.gethostname()\n",
        "BLOCKED = ['urllib3', 'requests']\n",  # strings that are not imports (sandbox_executor's blocklist)
        "from maxim.utils import http as _http\n_http.fetch_url('x')\n",
    ],
)
def test_what_is_not_an_http_client_passes(src: str) -> None:
    assert L.http_client_uses(src) == []


def test_raw_dials_count_calls_through_aliases_not_mentions() -> None:
    src = (
        "import socket as s\n"
        "from socket import create_connection as dial\n"
        "import asyncio\n"
        "def f(sock: s.socket) -> 's.socketpair':\n"
        "    '''socket.socket in a docstring'''\n"
        "    s.socket()\n"
        "    dial(('h', 1))\n"
        "    asyncio.open_connection('h', 1)\n"
        "    ctx.wrap_socket(sock)\n"
        "    loop.create_connection(p, 'h', 1)\n"
        "    s.gethostname()\n"
    )
    assert [ln for ln, _ in L.raw_dial_sites(src)] == [6, 7, 8, 9, 10]


def test_egress_sdks_are_found_by_any_import_form() -> None:
    src = (
        "from huggingface_hub import snapshot_download\nrequire_optional_dependency('anthropic')\nimport twilio.rest\n"
    )
    assert L.egress_sdks(src) == {"huggingface_hub", "anthropic", "twilio"}


def _tree(tmp_path: Path, files: dict[str, str]) -> Path:
    for rel, text in files.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return tmp_path


@pytest.fixture
def no_pins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(L, "RAW_DIAL_PINS", {"src/maxim/a.py": (1, "probe")})
    monkeypatch.setattr(L, "EGRESS_SDK_IMPORTERS", {"twilio": frozenset({"src/maxim/b.py"})})


def test_a_tree_matching_its_pins_passes(tmp_path: Path, no_pins: None) -> None:
    root = _tree(
        tmp_path,
        {
            "src/maxim/a.py": "import socket\nsocket.socket()\n",
            "src/maxim/b.py": "import twilio\n",
            "src/maxim/utils/http.py": "import httpx\n",  # the canonical module may import a client
        },
    )
    assert L.lint(root) == 0


@pytest.mark.parametrize(
    ("files", "message"),
    [
        ({"src/maxim/a.py": "import socket\nsocket.socket()\nsocket.socket()\n"}, "pinned at 1"),
        ({"src/maxim/a.py": "x = 1\n"}, "lower the pin"),
        ({"src/maxim/a.py": "import socket\nsocket.socket()\n", "src/maxim/c.py": "import twilio\n"}, "add it"),
        ({"src/maxim/a.py": "import socket\nsocket.socket()\n", "src/maxim/c.py": "import requests\n"}, "requests"),
        (
            {"src/maxim/a.py": "import socket\nsocket.socket()\n", "src/maxim/d.py": "socket.create_connection(x)\n"},
            "d.py",
        ),
    ],
)
def test_pins_are_exact_both_ways(
    tmp_path: Path, no_pins: None, capsys: pytest.CaptureFixture[str], files: dict[str, str], message: str
) -> None:
    files.setdefault("src/maxim/b.py", "import twilio\n")
    assert L.lint(_tree(tmp_path, files)) == 1
    assert message in capsys.readouterr().err


def test_a_stale_sdk_entry_fails(tmp_path: Path, no_pins: None, capsys: pytest.CaptureFixture[str]) -> None:
    assert L.lint(_tree(tmp_path, {"src/maxim/a.py": "import socket\nsocket.socket()\n"})) == 1
    assert "no longer imports twilio" in capsys.readouterr().err


def test_an_unparseable_file_fails_rather_than_being_skipped(tmp_path: Path, no_pins: None) -> None:
    assert L.lint(_tree(tmp_path, {"src/maxim/a.py": "def (:\n"})) == 2


def test_the_real_tree_passes() -> None:
    assert L.lint() == 0
