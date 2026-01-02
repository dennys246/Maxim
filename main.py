from __future__ import annotations

import sys
from pathlib import Path


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent
    src_dir = repo_root / "src"
    if src_dir.is_dir():
        sys.path.insert(0, str(src_dir))

    from maxim.cli import main

    raise SystemExit(main())
