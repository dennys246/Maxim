"""``maxim serve`` — the localhost Console backend + the OpenAPI facade contract.

This package is import-light on purpose: importing ``maxim.console`` must NOT pull
FastAPI/pydantic (they live in the ``console`` extra, not core). The dep check runs
first in :func:`run_serve_subcommand`; only then is :mod:`maxim.console.server`
imported. See docs/plans/deferred/maxim_console.md § Backend.
"""

from __future__ import annotations

from typing import Sequence


def run_serve_subcommand(argv: Sequence[str]) -> int:
    """Entry point for ``maxim serve`` (dispatched from cli.py).

    Verifies the ``console`` extra is installed *before* importing the server
    (which pulls FastAPI/pydantic), so a missing extra fails loud with an
    actionable ``pip install pymaxim[console]`` hint rather than an ImportError.
    """
    from maxim.utils.optional_deps import require_optional_dependency

    require_optional_dependency("fastapi", extra="console", feature="maxim serve")
    require_optional_dependency("uvicorn", extra="console", feature="maxim serve")

    from maxim.console.server import run_serve

    return run_serve(list(argv))


__all__ = ["run_serve_subcommand"]
