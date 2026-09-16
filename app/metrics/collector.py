"""Bounded, unlabelled Prometheus exposition: no session identifiers or raw text."""
from collections import defaultdict
import math


class Metrics:
    def __init__(self) -> None:
        self.counters: dict[str, float] = defaultdict(float)
        self.gauges: dict[str, float] = {}
        self.histograms: dict[str, tuple[int, float, list[int]]] = {}
        self.buckets = (1, 5, 10, 50, 100, 300, 700, 1500, 5000, 30000, 120000)

    def increment(self, name: str, value: float = 1) -> None:
        self.counters[name] += value

    def gauge(self, name: str, value: float) -> None:
        self.gauges[name] = value

    def observe(self, name: str, value: float) -> None:
        if not math.isfinite(value) or value < 0:
            return
        count, total, buckets = self.histograms.get(name, (0, 0., [0] * len(self.buckets)))
        for i, limit in enumerate(self.buckets):
            buckets[i] += value <= limit
        self.histograms[name] = (count + 1, total + value, buckets)

    def render(self) -> str:
        lines = []
        for name, value in sorted(self.counters.items()):
            lines.extend((f"# TYPE myvote_{name} counter", f"myvote_{name} {value:g}"))
        for name, value in sorted(self.gauges.items()):
            lines.extend((f"# TYPE myvote_{name} gauge", f"myvote_{name} {value:g}"))
        for name, (count, total, buckets) in sorted(self.histograms.items()):
            key = f"myvote_{name}"
            lines.append(f"# TYPE {key} histogram")
            lines.extend(f'{key}_bucket{{le="{limit}"}} {n}' for limit, n in zip(self.buckets, buckets))
            lines.extend((f'{key}_bucket{{le="+Inf"}} {count}', f"{key}_count {count}", f"{key}_sum {total:g}"))
        return "\n".join(lines) + "\n"
