"""A persisted store never writes over a file it did not read (#939, #971).

``StoreFileOwnership`` is mixed into every store that persists to one JSON file: Hippocampus, ATL,
NAc, EC, SCN, AngularGyrus, the cross-layer graph and the Cerebellum (#908). It lives in this leaf module, not in
``memory/store.py`` (which re-exports it), because importing the ``maxim.memory`` package runs its
``__init__``, which imports the Hippocampus, which imports NAc: NAc could not inherit from anything
inside that package. Keep this module's imports to the standard library; everything else is imported
inside the methods that need it.
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# File ownership: a store never writes over a file it did not read (#939, #971)
# ---------------------------------------------------------------------------


# The shapes BAD CONTENT raises while a loader parses a file or rebuilds a store from it: JSON errors
# (ValueError, incl. UnicodeDecodeError), missing or wrong-typed fields (KeyError, TypeError,
# AttributeError), bad indexes, sizes or divisors (IndexError, ArithmeticError: OverflowError,
# ZeroDivisionError) and pathological nesting
# (RecursionError). An allowlist on purpose (owner decision 2026-09-28, #971): anything else -- an
# OSError (the file is unreachable, not bad), or a code or environment defect (NameError, ImportError,
# AssertionError, NotImplementedError, RuntimeError, ...) -- propagates with the file untouched. A
# regression in the loader must fail loudly, not read as corruption and set every agent's memories
# aside.
UNREADABLE_STORE_ERRORS: tuple[type[Exception], ...] = (
    ValueError,
    KeyError,
    TypeError,
    AttributeError,
    IndexError,
    ArithmeticError,
    RecursionError,
)


def is_unreadable_store_error(exc: BaseException) -> bool:
    """Whether a failed load means the store FILE is unreadable (``UNREADABLE_STORE_ERRORS``) -- the one
    definition every recovery path and ``maxim.load.*`` use (#939, #971). Decided by the exception's
    TYPE: it does not know which step raised it."""
    return isinstance(exc, UNREADABLE_STORE_ERRORS)


# Copies this process made, so identical bytes met by two loaders in one construction are kept once,
# while a corruption that RECURS across runs still leaves one dated copy per run.
_COPIES_MADE_THIS_PROCESS: set[str] = set()


class StoreFileOwnership:
    """Mixin for a memory store that persists to one JSON file (Hippocampus, ATL, NAc, EC, SCN,
    AngularGyrus, the cross-layer graph, the Cerebellum; #939, #971, #908).

    A store may write a file it READ, a file that did not exist when it first wrote it (it created
    it), or a file it was explicitly told to overwrite (``save(overwrite=True)`` /
    ``allow_overwrite()``). Anything else is an existing file whose memories this instance never
    loaded, and saving would destroy them, so ``save`` raises ``StoreOverwriteRefused`` (#939: a
    ``create.*`` store, the plain constructor and ``from_config`` with the path on the config all used
    to start empty and clobber the file on the next save). Enforced at the write, so every path that
    builds a store is covered, including ones added later.

    Ownership is per instance and never released: it is not a lock. Two live stores that both read
    one file can still overwrite each other's saves, and a file deleted and recreated by another writer
    is still "owned" here. The guard stops the empty-store clobber, not concurrent writers.

    The class that mixes this in names itself in ``_store_name``. Its configured path is read from
    ``config.persistence_path`` unless it overrides ``_configured_store_path`` (SCN and the cross-layer
    graph keep theirs on the instance), and a fresh empty instance is ``type(self)(config)`` unless it
    overrides ``_fresh_store``. A store without ``dump``/``load_state`` overrides ``_reset_store_state``.
    """

    _store_name: str = "store"

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Refuse, at class definition, a store that could not empty itself on recovery: it must either
        have the snapshot contract (``dump`` + ``load_state``) the default ``_reset_store_state`` uses,
        or override ``_reset_store_state``. Otherwise the gap shows only when a corrupt file is met."""
        super().__init_subclass__(**kwargs)
        overrides_reset = cls._reset_store_state is not StoreFileOwnership._reset_store_state
        has_snapshot = callable(getattr(cls, "dump", None)) and callable(getattr(cls, "load_state", None))
        if not (overrides_reset or has_snapshot):
            raise TypeError(
                f"{cls.__name__} mixes in StoreFileOwnership without dump()/load_state(); override "
                "_reset_store_state() so recovery from an unreadable file can empty it"
            )

    def _owned_store_files(self) -> set[str]:
        return self.__dict__.setdefault("_store_owned_files", set())

    @staticmethod
    def _store_file_key(path: str) -> str:
        import os

        from maxim.utils.paths import store_file_path

        return os.path.realpath(store_file_path(path))

    def _configured_store_path(self) -> str | None:
        return getattr(getattr(self, "config", None), "persistence_path", None)

    def _default_store_path(self, path: str | None) -> str:
        chosen = path or self._configured_store_path()
        if not chosen:
            raise ValueError(f"{self._store_name}: no path given and no persistence_path configured")
        from maxim.utils.paths import store_file_path

        return store_file_path(chosen)

    def may_write(self, path: str | None = None) -> bool:
        """Whether ``save`` would write ``path`` (default: the configured one) without refusing."""
        import os

        target = self._default_store_path(path)
        return not os.path.exists(target) or self._store_file_key(target) in self._owned_store_files()

    def allow_overwrite(self, path: str | None = None) -> None:
        """Declare that this store may replace ``path`` (default: the configured one) without reading it.

        For a deliberately write-but-don't-read store (the sim NPC, a fresh start over a file the
        operator chose to discard). The file's current contents are lost on the next save.
        """
        import os

        target = self._default_store_path(path)
        self._owned_store_files().add(self._store_file_key(target))
        if os.path.exists(target):
            log.warning("%s: will overwrite %s without reading it (declared)", self._store_name, target)

    def set_aside_unreadable_file(self, path: str | None = None, *, reason: str | None = None) -> str | None:
        """Keep an unreadable store file as evidence, then let this store save in its place.

        Copies ``path`` (default: the configured one) to ``<name>.corrupt-<UTC timestamp>`` beside
        it, logs where it went, and claims the original, so a store that started empty after a
        failed load persists normally instead of being refused forever (owner decision 2026-09-28,
        #939). Returns the copy's path, or None when there is no file.

        ``reason`` replaces "could not be read" in the log, for a READABLE file set aside because its
        partner was not: an NAc beside an unreadable EC, whose biases key on EC node ids (#971).
        """
        import shutil
        import time as _time
        from pathlib import Path

        import filecmp

        target = Path(self._default_store_path(path))
        if not target.exists():
            return None
        why = reason or "could not be read"
        # A failed load can leave PART of the file in the store; "starting empty" must be true before
        # the store is allowed to save over the file.
        self._reset_store_state()
        # Two loaders can meet the same corrupt file in one construction (create_full_agent's factory,
        # then its bio stack): keep ONE copy of identical bytes -- among copies made by THIS process.
        for existing in sorted(_COPIES_MADE_THIS_PROCESS):
            existing_path = Path(existing)
            if existing_path.parent == target.parent and existing_path.name.startswith(f"{target.name}.corrupt-"):
                if existing_path.exists() and filecmp.cmp(existing_path, target, shallow=False):
                    self._claim_store_file(str(target))
                    log.warning("%s: %s %s; its copy is already at %s", self._store_name, target, why, existing)
                    return existing
        stamp = _time.strftime("%Y%m%dT%H%M%SZ", _time.gmtime())
        copy = target.with_name(f"{target.name}.corrupt-{stamp}")
        n = 1
        while copy.exists():
            copy = target.with_name(f"{target.name}.corrupt-{stamp}-{n}")
            n += 1
        shutil.copy2(target, copy)
        _COPIES_MADE_THIS_PROCESS.add(str(copy))
        self._claim_store_file(str(target))
        log.warning(
            "%s: %s %s; kept a copy at %s and starting empty -- the next save replaces it",
            self._store_name,
            target,
            why,
            copy,
        )
        return str(copy)

    def start_fresh_keeping_copy(self, path: str | None = None, *, reason: str | None = None) -> str | None:
        """Empty this store and let it save in place of ``path``, keeping the file as a copy first.

        The one recovery step every store uses (#971). Returns the copy's path, or None when there is
        no file. If the copy cannot be made, the store is emptied anyway but stops owning the file, so a
        save over it is refused and the file stays the only copy of what it held; that is logged at
        ERROR and not raised, so construction can finish.
        """
        try:
            return self.set_aside_unreadable_file(path, reason=reason)
        except OSError as e:
            target = self._default_store_path(path)
            self._reset_store_state()
            self.disown_store_file(target)
            log.error(
                "%s: could not keep a copy of %s (%s); starting empty, and saves over it are refused",
                self._store_name,
                target,
                e,
            )
            return None

    def _fresh_store(self) -> Any:
        """An empty instance configured like this one."""
        return type(self)(getattr(self, "config", None))  # type: ignore[call-arg]

    def _reset_store_state(self) -> None:
        """Empty this store completely, through its own snapshot contract: restore the dump of a fresh
        store with the same config (every surface ``load_state`` owns -- graph, indexes, stats)."""
        self.load_state(self._fresh_store().dump())  # type: ignore[attr-defined]

    def _claim_store_file(self, path: str) -> None:
        """Record that this store read, created or was told to replace ``path``."""
        self._owned_store_files().add(self._store_file_key(path))

    def disown_store_file(self, path: str | None = None) -> None:
        """Stop owning ``path`` (default: the configured one): a save over it is refused again, so the file
        stays as it is on disk (#971: an EC beside an NAc whose copy failed; a store whose load failed for
        a reason that is not bad content)."""
        self._owned_store_files().discard(self._store_file_key(self._default_store_path(path)))

    def start_fresh_in_memory(self, path: str | None = None) -> None:
        """Empty this store WITHOUT touching or claiming its file: for a load that failed for a reason
        other than bad content (an ``OSError``, a code defect), after which the caller carries on. The
        store is literally fresh, and a save over the file is refused, so the file stays as it was.
        Raises ``ValueError`` (via ``_default_store_path``) for a store with no path given or configured."""
        self._reset_store_state()
        self.disown_store_file(path)

    def _check_store_write(self, path: str, *, overwrite: bool) -> None:
        """Raise ``StoreOverwriteRefused`` unless this store may write ``path`` (expanded already)."""
        if overwrite:
            self._claim_store_file(path)
            return
        if not self.may_write(path):
            from maxim.exceptions import StoreOverwriteRefused

            raise StoreOverwriteRefused(
                f"{self._store_name}: refusing to save over {path}, which this instance never read -- "
                f"it would replace the memories stored there. Open it with maxim.load.{self._store_name}() "
                "(or the store's load()) to keep them, save to another path, or pass overwrite=True / call "
                "allow_overwrite() to replace them deliberately.",
                path=path,
                store=self._store_name,
            )
