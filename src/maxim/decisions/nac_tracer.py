"""NacTracer — real-time color-coded tracing of NAc causal learning.

Wraps NAc methods to trace event recording, outcome linking, causal
link formation, prediction, and RPE (reward prediction error) updates.

Activated by --debug nac or --debug all, or MAXIM_NAC_TRACE=1 env var.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from maxim.decisions.nac import NAc

logger = logging.getLogger(__name__)

_GREEN = "\033[32m"
_BLUE = "\033[34m"
_CYAN = "\033[36m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_MAGENTA = "\033[35m"
_DIM = "\033[90m"
_BOLD = "\033[1m"
_RESET = "\033[0m"


def _ts() -> str:
    return time.strftime("%H:%M:%S")


def _short(s: str, n: int = 30) -> str:
    if len(s) > n:
        return s[: n - 3] + "..."
    return s


class NacTracer:
    """Real-time tracer for NAc causal learning operations."""

    def __init__(self, nac: NAc) -> None:
        self._nac = nac
        self._lock = threading.Lock()
        self._observe_count = 0
        self._predict_count = 0
        self._event_count = 0

        self._wrap_observe(nac)
        self._wrap_predict(nac)
        self._wrap_record_event(nac)
        self._wrap_record_outcome(nac)

        self._print(f"{_BOLD}[NAc TRACER]{_RESET} Active — tracing events, outcomes, predictions, causal links")

    def _print(self, msg: str) -> None:
        with self._lock:
            print(msg, file=sys.stderr, flush=True)

    def _wrap_record_event(self, nac: NAc) -> None:
        original = nac.record_event

        def traced(*args: Any, **kwargs: Any) -> str:
            result = original(*args, **kwargs)
            self._event_count += 1
            sig = args[1] if len(args) > 1 else kwargs.get("event_signature", "?")
            self._print(f"{_CYAN}{_ts()} [NAc EVENT ]{_RESET} #{self._event_count} {_short(str(sig))}")
            return result

        nac.record_event = traced  # type: ignore[assignment]

    def _wrap_record_outcome(self, nac: NAc) -> None:
        original = nac.record_outcome

        def traced(*args: Any, **kwargs: Any) -> Any:
            result = original(*args, **kwargs)
            event_id = args[0] if args else kwargs.get("event_id", "?")
            valence = args[2] if len(args) > 2 else kwargs.get("outcome_valence", "?")
            color = (
                _GREEN
                if str(valence) == "Valence.POSITIVE"
                else _RED
                if str(valence) == "Valence.NEGATIVE"
                else _YELLOW
            )
            self._print(f"{color}{_ts()} [NAc OUTCOM]{_RESET} event={_short(str(event_id), 20)} valence={valence}")
            return result

        nac.record_outcome = traced  # type: ignore[assignment]

    def _wrap_observe(self, nac: NAc) -> None:
        original = nac.observe

        def traced(*args: Any, **kwargs: Any) -> Any:
            result = original(*args, **kwargs)
            self._observe_count += 1
            event_sig = args[1] if len(args) > 1 else kwargs.get("event_signature", "?")
            outcome_sig = args[3] if len(args) > 3 else kwargs.get("outcome_signature", "?")
            valence = args[4] if len(args) > 4 else kwargs.get("outcome_valence", "?")
            conf = getattr(result, "confidence", "?") if result else "?"
            obs = getattr(result, "observation_count", "?") if result else "?"
            color = _GREEN if "POSITIVE" in str(valence) else _RED if "NEGATIVE" in str(valence) else _YELLOW
            self._print(
                f"{color}{_ts()} [NAc LEARN ]{_RESET} "
                f"#{self._observe_count} "
                f"{_short(str(event_sig), 20)} → {_short(str(outcome_sig), 20)} "
                f"conf={conf} obs={obs}"
            )
            return result

        nac.observe = traced  # type: ignore[assignment]

    def _wrap_predict(self, nac: NAc) -> None:
        original = nac.predict

        def traced(*args: Any, **kwargs: Any) -> Any:
            result = original(*args, **kwargs)
            self._predict_count += 1
            event_sig = args[1] if len(args) > 1 else kwargs.get("event_signature", "?")
            if result is not None:
                predicted = getattr(result, "predicted_value", "?")
                conf = getattr(result, "confidence", "?")
                self._print(
                    f"{_BLUE}{_ts()} [NAc PRED  ]{_RESET} "
                    f"#{self._predict_count} {_short(str(event_sig))} "
                    f"→ value={predicted} conf={conf}"
                )
            else:
                self._print(
                    f"{_DIM}{_ts()} [NAc PRED  ] #{self._predict_count} {_short(str(event_sig))} → no data{_RESET}"
                )
            return result

        nac.predict = traced  # type: ignore[assignment]

    def summary(self) -> str:
        stats = self._nac.stats()
        return (
            f"NAc trace: {self._event_count} events, "
            f"{self._observe_count} observations, "
            f"{self._predict_count} predictions, "
            f"{stats.get('total_links', 0)} causal links"
        )

    def print_summary(self) -> None:
        self._print(f"\n{_BOLD}[NAc SUMMARY]{_RESET} {self.summary()}")
