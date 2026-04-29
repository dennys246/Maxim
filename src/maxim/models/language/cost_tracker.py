"""Cost tracking for cloud LLM usage (USD).

Tracks rolling hourly/daily spend, calendar-month spend, per-provider totals,
and simple EMA spend rates for long-horizon budget visibility.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from maxim.utils.logging import warn


@dataclass(frozen=True, slots=True)
class ModelPricing:
    """Actual API pricing in USD per 1M tokens.

    SHAPE-FROZEN at 1.0 (CC3). Pricing tuples are looked up by model
    name and applied directly to cost computations; an ``extra`` dict
    here would invite ad-hoc surcharges that the cost-tracker can't
    enforce uniformly. New pricing axes (e.g., per-image, per-cached-
    write tier) get added as named optional fields with defaults so
    additive updates stay non-breaking. Adding a *required* field
    post-1.0 is a major-version-bump change.
    """

    input_price: float
    output_price: float
    cached_input_price: float = 0.0


@dataclass
class SpendRate:
    """Exponential moving average of spend rate (USD/hour)."""

    value: float = 0.0
    samples: int = 0
    last_update: float = 0.0

    def update(self, amount_usd: float, now: float, alpha: float) -> None:
        if amount_usd <= 0:
            return
        elapsed = max(1.0, now - self.last_update) if self.last_update else 1.0
        rate_per_hour = amount_usd * 3600.0 / elapsed
        if self.samples <= 0:
            self.value = rate_per_hour
            self.samples = 1
            self.last_update = now
            return
        self.value = alpha * rate_per_hour + (1.0 - alpha) * self.value
        self.samples += 1
        self.last_update = now


class RollingWindow:
    """Rolling window accumulator for USD costs."""

    def __init__(self, window_seconds: float, events: list[tuple[float, float]] | None = None) -> None:
        self.window_seconds = float(window_seconds)
        self._events: list[tuple[float, float]] = list(events or [])
        self.total_usd = sum(x[1] for x in self._events)

    def add(self, amount_usd: float, timestamp: float) -> None:
        if amount_usd <= 0:
            return
        self._events.append((timestamp, amount_usd))
        self.total_usd += amount_usd
        self.prune(timestamp)

    def prune(self, now: float) -> None:
        cutoff = now - self.window_seconds
        if not self._events:
            return
        kept: list[tuple[float, float]] = []
        total = 0.0
        for ts, amt in self._events:
            if ts >= cutoff:
                kept.append((ts, amt))
                total += amt
        self._events = kept
        self.total_usd = total

    @property
    def events(self) -> list[tuple[float, float]]:
        return list(self._events)


@dataclass
class CostTrackerConfig:
    """Configuration for CostTracker."""

    state_path: str = ""  # resolved via maxim.utils.paths at runtime
    persistence_interval_s: float = 10.0
    persistence_interval_n: int = 5
    reserved_budget_ratio: float = 0.2
    min_spend_samples: int = 5
    max_events_per_window: int = 5000

    def __post_init__(self) -> None:
        if not self.state_path:
            from maxim.utils.paths import resolve_user_state

            self.state_path = str(resolve_user_state("util/cost_state.json"))


class CostTracker:
    """Tracks actual USD spend for cloud LLM usage."""

    VERSION = 1

    def __init__(
        self,
        pricing: dict[str, ModelPricing],
        config: CostTrackerConfig | None = None,
    ) -> None:
        self._pricing = pricing
        self._config = config or CostTrackerConfig()
        self._hourly = RollingWindow(3600.0)
        self._daily = RollingWindow(86400.0)
        self._monthly_total = 0.0
        self._monthly_key = self._current_month_key(time.time())
        self._lifetime_total = 0.0
        self._lifetime_requests = 0
        self._spend_rates = {
            "ema_3h": SpendRate(),
            "ema_24h": SpendRate(),
            "ema_7d": SpendRate(),
        }
        self._per_provider: dict[str, dict[str, Any]] = {}
        self._pending_writes = 0
        self._last_persist_time = 0.0
        # Session-lifetime token counters (not persisted — simulation
        # reports consume this and reset per process).
        self._session_input_tokens = 0
        self._session_output_tokens = 0
        self._load_state()

    def _current_month_key(self, ts: float) -> str:
        dt = datetime.fromtimestamp(ts)
        return f"{dt.year:04d}-{dt.month:02d}"

    def _load_state(self) -> None:
        path = self._config.state_path
        if not path or not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            self._mark_corrupt(path)
            return
        if not isinstance(data, dict) or data.get("version") != self.VERSION:
            self._mark_corrupt(path)
            return
        import logging

        from maxim.utils.format_version import check_format_version

        check_format_version(data, "cost_tracker", log=logging.getLogger(__name__))
        try:
            hourly = data.get("hourly", {})
            daily = data.get("daily", {})
            monthly = data.get("monthly", {})
            lifetime = data.get("lifetime", {})
            spend_rates = data.get("spend_rates", {})

            self._hourly = RollingWindow(
                3600.0,
                events=self._load_events(hourly.get("events", [])),
            )
            self._daily = RollingWindow(
                86400.0,
                events=self._load_events(daily.get("events", [])),
            )
            self._monthly_key = str(monthly.get("month_key") or self._current_month_key(time.time()))
            try:
                self._monthly_total = float(monthly.get("total_usd", 0.0))
            except Exception:
                self._monthly_total = 0.0
            try:
                self._lifetime_total = float(lifetime.get("total_usd", 0.0))
            except Exception:
                self._lifetime_total = 0.0
            try:
                self._lifetime_requests = int(lifetime.get("total_requests", 0))
            except Exception:
                self._lifetime_requests = 0

            self._spend_rates = {
                "ema_3h": self._load_spend_rate(spend_rates.get("ema_3h", {})),
                "ema_24h": self._load_spend_rate(spend_rates.get("ema_24h", {})),
                "ema_7d": self._load_spend_rate(spend_rates.get("ema_7d", {})),
            }
            self._per_provider = self._load_provider_state(data.get("per_provider", {}))
        except Exception:
            self._mark_corrupt(path)

    def _mark_corrupt(self, path: str) -> None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        try:
            os.rename(path, f"{path}.corrupt.{ts}")
        except Exception:
            pass
        warn("Cost state file was corrupt; starting fresh: %s", path)

    def _load_events(self, raw: Any) -> list[tuple[float, float]]:
        events: list[tuple[float, float]] = []
        if not isinstance(raw, list):
            return events
        for entry in raw:
            if not isinstance(entry, (list, tuple)) or len(entry) != 2:
                continue
            try:
                ts = float(entry[0])
                amt = float(entry[1])
            except Exception:
                continue
            events.append((ts, amt))
        return events

    def _load_spend_rate(self, raw: Any) -> SpendRate:
        if not isinstance(raw, dict):
            return SpendRate()
        try:
            return SpendRate(
                value=float(raw.get("value", 0.0)),
                samples=int(raw.get("samples", 0)),
                last_update=float(raw.get("last_update", 0.0)),
            )
        except Exception:
            return SpendRate()

    def _load_provider_state(self, raw: Any) -> dict[str, dict[str, Any]]:
        if not isinstance(raw, dict):
            return {}
        out: dict[str, dict[str, Any]] = {}
        for provider, state in raw.items():
            if not isinstance(state, dict):
                continue
            out[provider] = {
                "hourly": RollingWindow(3600.0, events=self._load_events(state.get("hourly", {}).get("events", []))),
                "daily": RollingWindow(86400.0, events=self._load_events(state.get("daily", {}).get("events", []))),
                "monthly_total": float(state.get("monthly", {}).get("total_usd", 0.0) or 0.0),
                "monthly_key": str(state.get("monthly", {}).get("month_key") or self._current_month_key(time.time())),
            }
        return out

    def _persist_state(self) -> None:
        path = self._config.state_path
        data = self.to_dict()
        from maxim.utils.atomic_io import atomic_write_json
        from maxim.utils.format_version import with_format_version

        atomic_write_json(path, with_format_version(data))
        self._pending_writes = 0
        self._last_persist_time = time.time()

    def _maybe_persist(self) -> None:
        now = time.time()
        if self._pending_writes >= self._config.persistence_interval_n:
            self._persist_state()
            return
        if (now - self._last_persist_time) >= self._config.persistence_interval_s:
            self._persist_state()

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.VERSION,
            "hourly": {
                "window_seconds": 3600,
                "events": self._hourly.events[-self._config.max_events_per_window :],
                "total_usd": round(self._hourly.total_usd, 6),
            },
            "daily": {
                "window_seconds": 86400,
                "events": self._daily.events[-self._config.max_events_per_window :],
                "total_usd": round(self._daily.total_usd, 6),
            },
            "monthly": {
                "month_key": self._monthly_key,
                "total_usd": round(self._monthly_total, 6),
            },
            "lifetime": {
                "total_usd": round(self._lifetime_total, 6),
                "total_requests": self._lifetime_requests,
            },
            "spend_rates": {
                "ema_3h": {
                    "value": round(self._spend_rates["ema_3h"].value, 6),
                    "samples": self._spend_rates["ema_3h"].samples,
                    "last_update": self._spend_rates["ema_3h"].last_update,
                },
                "ema_24h": {
                    "value": round(self._spend_rates["ema_24h"].value, 6),
                    "samples": self._spend_rates["ema_24h"].samples,
                    "last_update": self._spend_rates["ema_24h"].last_update,
                },
                "ema_7d": {
                    "value": round(self._spend_rates["ema_7d"].value, 6),
                    "samples": self._spend_rates["ema_7d"].samples,
                    "last_update": self._spend_rates["ema_7d"].last_update,
                },
            },
            "per_provider": self._serialize_per_provider(),
        }

    def _serialize_per_provider(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for provider, state in self._per_provider.items():
            hourly: RollingWindow = state["hourly"]
            daily: RollingWindow = state["daily"]
            out[provider] = {
                "hourly": {
                    "window_seconds": 3600,
                    "events": hourly.events[-self._config.max_events_per_window :],
                    "total_usd": round(hourly.total_usd, 6),
                },
                "daily": {
                    "window_seconds": 86400,
                    "events": daily.events[-self._config.max_events_per_window :],
                    "total_usd": round(daily.total_usd, 6),
                },
                "monthly": {
                    "month_key": state.get("monthly_key", self._current_month_key(time.time())),
                    "total_usd": round(float(state.get("monthly_total", 0.0) or 0.0), 6),
                },
            }
        return out

    def get_pricing(self, model: str) -> ModelPricing | None:
        if not model:
            return None
        pricing = self._pricing.get(model)
        return pricing

    def estimate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cached_input_tokens: int = 0,
        uncached_input_tokens: int = 0,
    ) -> float | None:
        pricing = self.get_pricing(model)
        if pricing is None:
            return None
        return self._calculate_cost(
            pricing,
            input_tokens,
            output_tokens,
            cached_input_tokens,
            uncached_input_tokens,
        )

    def record(
        self,
        *,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cached_input_tokens: int = 0,
        uncached_input_tokens: int = 0,
        timestamp: float | None = None,
    ) -> float:
        now = timestamp or time.time()
        pricing = self.get_pricing(model)
        if pricing is None:
            raise ValueError(f"Missing pricing for model: {model}")
        cost_usd = self._calculate_cost(
            pricing,
            input_tokens,
            output_tokens,
            cached_input_tokens,
            uncached_input_tokens,
        )
        # Track tokens even for zero-cost calls (cached prompts).
        self._session_input_tokens += max(0, int(input_tokens))
        self._session_output_tokens += max(0, int(output_tokens))

        if cost_usd <= 0:
            return 0.0

        self._hourly.add(cost_usd, now)
        self._daily.add(cost_usd, now)

        month_key = self._current_month_key(now)
        if month_key != self._monthly_key:
            self._monthly_key = month_key
            self._monthly_total = 0.0
        self._monthly_total += cost_usd

        self._lifetime_total += cost_usd
        self._lifetime_requests += 1

        self._update_spend_rates(cost_usd, now)
        self._record_provider(provider, cost_usd, now)

        self._pending_writes += 1
        self._maybe_persist()
        return cost_usd

    def _record_provider(self, provider: str, cost_usd: float, now: float) -> None:
        if not provider:
            return
        state = self._per_provider.get(provider)
        if state is None:
            state = {
                "hourly": RollingWindow(3600.0),
                "daily": RollingWindow(86400.0),
                "monthly_total": 0.0,
                "monthly_key": self._current_month_key(now),
            }
            self._per_provider[provider] = state
        hourly: RollingWindow = state["hourly"]
        daily: RollingWindow = state["daily"]
        hourly.add(cost_usd, now)
        daily.add(cost_usd, now)
        month_key = self._current_month_key(now)
        if month_key != state.get("monthly_key"):
            state["monthly_key"] = month_key
            state["monthly_total"] = 0.0
        state["monthly_total"] = float(state.get("monthly_total", 0.0)) + cost_usd

    def _update_spend_rates(self, cost_usd: float, now: float) -> None:
        self._spend_rates["ema_3h"].update(cost_usd, now, alpha=0.3)
        self._spend_rates["ema_24h"].update(cost_usd, now, alpha=0.2)
        self._spend_rates["ema_7d"].update(cost_usd, now, alpha=0.1)

    def _calculate_cost(
        self,
        pricing: ModelPricing,
        input_tokens: int,
        output_tokens: int,
        cached_input_tokens: int,
        uncached_input_tokens: int,
    ) -> float:
        cached = max(0, cached_input_tokens)
        uncached = max(0, uncached_input_tokens)
        if cached == 0 and uncached == 0:
            uncached = max(0, input_tokens)
        input_cost = (uncached * pricing.input_price + cached * pricing.cached_input_price) / 1_000_000
        output_cost = (max(0, output_tokens) * pricing.output_price) / 1_000_000
        return round(input_cost + output_cost, 6)

    def get_totals(self, now: float | None = None) -> dict[str, float]:
        now_ts = now or time.time()
        self._hourly.prune(now_ts)
        self._daily.prune(now_ts)
        return {
            "hourly": round(self._hourly.total_usd, 6),
            "daily": round(self._daily.total_usd, 6),
            "monthly": round(self._monthly_total, 6),
            "lifetime": round(self._lifetime_total, 6),
        }

    def get_per_provider_totals(self, provider: str, now: float | None = None) -> dict[str, float]:
        now_ts = now or time.time()
        state = self._per_provider.get(provider)
        if state is None:
            return {"hourly": 0.0, "daily": 0.0, "monthly": 0.0}
        hourly: RollingWindow = state["hourly"]
        daily: RollingWindow = state["daily"]
        hourly.prune(now_ts)
        daily.prune(now_ts)
        return {
            "hourly": round(hourly.total_usd, 6),
            "daily": round(daily.total_usd, 6),
            "monthly": round(float(state.get("monthly_total", 0.0) or 0.0), 6),
        }

    def get_session_tokens(self) -> dict[str, int]:
        """Return session-lifetime token counts (not persisted)."""
        return {
            "input_tokens": self._session_input_tokens,
            "output_tokens": self._session_output_tokens,
            "total_tokens": self._session_input_tokens + self._session_output_tokens,
        }

    def reset_session_tokens(self) -> None:
        """Reset session token counters (e.g. between simulation runs)."""
        self._session_input_tokens = 0
        self._session_output_tokens = 0

    def get_spend_rates(self) -> dict[str, SpendRate]:
        return {
            "ema_3h": self._spend_rates["ema_3h"],
            "ema_24h": self._spend_rates["ema_24h"],
            "ema_7d": self._spend_rates["ema_7d"],
        }

    def flush(self) -> None:
        self._persist_state()

    @property
    def config(self) -> CostTrackerConfig:
        return self._config


__all__ = [
    "ModelPricing",
    "CostTracker",
    "CostTrackerConfig",
    "SpendRate",
]
