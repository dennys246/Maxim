from __future__ import annotations

import queue
from typing import Any


def put_latest(q: queue.Queue, item: Any) -> None:
    try:
        q.get_nowait()
    except queue.Empty:
        pass
    try:
        q.put_nowait(item)
    except queue.Full:
        pass
