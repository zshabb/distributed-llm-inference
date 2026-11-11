"""Summary statistics for benchmark results.

Percentiles use linear interpolation between order statistics (numpy's default, "type 7"), so a
P99 over 50 samples is an interpolation, not a measured request: summaries always carry ``n`` so a
reader can judge that. Confidence intervals are percentile bootstraps with a fixed seed, so the same
raw data always produces the same report.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass

import numpy as np

PERCENTILES = (50, 90, 95, 99)


@dataclass(frozen=True)
class Summary:
    n: int
    mean: float
    p50: float
    p90: float
    p95: float
    p99: float
    max: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def summarize(values: Sequence[float]) -> Summary | None:
    """Mean, P50/P90/P95/P99 and max, or None for an empty sample (never a fabricated zero)."""
    if len(values) == 0:
        return None
    a = np.asarray(values, dtype=np.float64)
    p50, p90, p95, p99 = (float(x) for x in np.percentile(a, PERCENTILES))
    return Summary(len(a), float(a.mean()), p50, p90, p95, p99, float(a.max()))


def bootstrap_ci(
    values: Sequence[float],
    stat: Callable[[np.ndarray], float] = lambda a: float(np.median(a)),
    *,
    level: float = 0.95,
    resamples: int = 2000,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile-bootstrap confidence interval of ``stat`` (default: the median)."""
    a = np.asarray(values, dtype=np.float64)
    if len(a) < 2:
        raise ValueError("a bootstrap CI needs at least 2 values")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(resamples, len(a)))
    stats = np.array([stat(a[i]) for i in idx])
    lo, hi = np.percentile(stats, [100 * (1 - level) / 2, 100 * (1 + level) / 2])
    return float(lo), float(hi)


def paired_bootstrap_ci(
    a: Sequence[float],
    b: Sequence[float],
    *,
    level: float = 0.95,
    resamples: int = 4000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Mean of ``b - a`` over paired items and its percentile-bootstrap CI (items resampled)."""
    x, y = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if x.shape != y.shape or len(x) < 2:
        raise ValueError("paired samples must have equal length >= 2")
    d = y - x
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), size=(resamples, len(d)))].mean(axis=1)
    lo, hi = np.percentile(means, [100 * (1 - level) / 2, 100 * (1 + level) / 2])
    return float(d.mean()), float(lo), float(hi)


def median_and_range(values: Sequence[float]) -> dict[str, float]:
    """Across repetitions: median, min, max (the spread a reader should see)."""
    a = np.asarray(values, dtype=np.float64)
    return {
        "median": float(np.median(a)),
        "min": float(a.min()),
        "max": float(a.max()),
        "reps": len(a),
    }


def mcnemar_exact(only_a: int, only_b: int) -> float:
    """Two-sided exact McNemar p-value from the discordant counts.

    ``only_a``: items A got right and B wrong; ``only_b``: the reverse.
    """
    n = only_a + only_b
    if n == 0:
        return 1.0
    k = min(only_a, only_b)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return float(min(1.0, 2 * tail))


def paired_proportion_diff_ci(
    only_a: int, only_b: int, n: int, z: float = 1.959964
) -> tuple[float, float]:
    """95% CI for the difference of paired proportions p_b - p_a (Wald, continuity-corrected).

    Large-sample variance of the paired difference, widened by 1/n. Adequate for the thousands of
    items used here; with few discordant pairs it is conservative. Proportion units.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    d = (only_b - only_a) / n
    var = ((only_a + only_b) - (only_b - only_a) ** 2 / n) / n**2
    half = z * math.sqrt(max(var, 0.0)) + 1 / n
    return d - half, d + half
