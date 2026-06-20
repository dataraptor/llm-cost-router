"""Lightweight, in-process metrics counters (split-11 §3).

A single thread-safe :class:`Metrics` accumulator records the *money* quantities
the demo claims, from real traffic: cumulative routing requests, cumulative
``cost_usd`` (summed from the engine's own ``RouteResult.cost_usd`` — never a
separate, divergent accounting), the escalation rate, the refusal count, and
latency percentiles. Counters are process-lifetime and reset on restart.

No Prometheus dependency: ``GET /api/metrics`` serves a small JSON object.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass


@dataclass
class MetricsSnapshot:
    """An immutable read of the counters at one instant (the ``/api/metrics`` body)."""

    requests_total: int
    cost_usd_total: float
    escalation_rate: float
    refused_total: int
    latency_p50_s: float | None
    latency_p95_s: float | None
    since: float


class Metrics:
    """Thread-safe accumulator for per-route metrics.

    ``record_route`` is called once per completed single-query route with the
    engine's own reported quantities, so ``cost_usd_total`` is exactly the sum of
    routed ``cost_usd``.
    """

    # Bound the latency window so memory never grows without limit under load.
    _MAX_LATENCIES = 2048

    def __init__(self, *, now: float | None = None) -> None:
        self._lock = threading.Lock()
        self._requests = 0
        self._cost_usd = 0.0
        self._escalations = 0
        self._refused = 0
        self._latencies: deque[float] = deque(maxlen=self._MAX_LATENCIES)
        self._since = now if now is not None else time.time()

    def record_route(
        self, *, cost_usd: float, latency_s: float, escalated: bool, refused: bool
    ) -> None:
        with self._lock:
            self._requests += 1
            self._cost_usd += cost_usd
            if escalated:
                self._escalations += 1
            if refused:
                self._refused += 1
            self._latencies.append(latency_s)

    def snapshot(self) -> MetricsSnapshot:
        with self._lock:
            requests = self._requests
            escalation_rate = self._escalations / requests if requests else 0.0
            return MetricsSnapshot(
                requests_total=requests,
                cost_usd_total=self._cost_usd,
                escalation_rate=escalation_rate,
                refused_total=self._refused,
                latency_p50_s=_percentile(self._latencies, 50),
                latency_p95_s=_percentile(self._latencies, 95),
                since=self._since,
            )


def _percentile(values: deque[float], pct: int) -> float | None:
    """Nearest-rank percentile of ``values`` (``None`` when there are no samples)."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, round(pct / 100 * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]
