"""#1217: the SCN is safe under concurrent registration and the readers sleep, save and enrichment use.

``SCN`` held its bins and signatures with no lock at all. ``SCN.register`` runs on the Hippocampus capture worker
(``MemoryHub._on_memory_captured_scn``, a capture callback fired after the store's write lock is released) while
sleep (``get_all_clusters`` / ``get_bin_populations``) and save (``dump``) iterate the same dicts on other threads:
"dictionary changed size during iteration", and ``BoundedBin.add``'s eviction racing another add
(``IndexError: pop index out of range``). The #1135 forced-switching pattern reproduces it on a quiet box.
"""

from __future__ import annotations

import sys
import threading
import time
import traceback

from maxim.time.scn import SCN
from maxim.time.temporal_signature import TemporalSignature


def test_concurrent_register_and_readers_under_forced_switching() -> None:
    errors: list[str] = []
    old = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        for _trial in range(3):
            scn = SCN()
            stop = threading.Event()

            def writer(t: int, scn: SCN = scn) -> None:
                try:
                    for i in range(300):
                        sig = TemporalSignature.from_timestamp(time.time() - i * 3700)
                        scn.register(f"m{t}-{i}", sig, significance=(i % 10) / 10)
                except Exception:
                    errors.append(traceback.format_exc())

            def reader(read, scn: SCN = scn, stop: threading.Event = stop) -> None:
                try:
                    while not stop.is_set():
                        read(scn)
                except Exception:
                    errors.append(traceback.format_exc())

            reads = (SCN.get_all_clusters, SCN.get_bin_populations, SCN.dump, SCN.stats)
            readers = [threading.Thread(target=reader, args=(r,)) for r in reads]
            writers = [threading.Thread(target=writer, args=(t,)) for t in range(4)]
            for t in readers + writers:
                t.start()
            for t in writers:
                t.join(timeout=60)
            stop.set()
            for t in readers:
                t.join(timeout=60)
            assert not any(t.is_alive() for t in readers + writers), "a thread is still running (deadlock?)"
    finally:
        sys.setswitchinterval(old)
    assert not errors, errors[0]
