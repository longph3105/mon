"""Composite rankings of methods from their mean metric scores (challenge leaderboards).

A composite ranking compares methods, not images: each method is ranked on every metric (ties share the
better rank), and its score is the weighted sum of those ranks (lower is better). ``NTIRE_2025_LLIE`` holds the
weights of the NTIRE 2025 Low Light Image Enhancement challenge's "Final Rank".

The report gives only the weights ("a weighted combination of PSNR (50%), SSIM (50%), LPIPS (40%), and NIQE (20%)"),
no formula, and no scoring code is released. The weighted sum of per-metric ranks is a reconstruction: applied to
the per-metric ranks in the report's Table 1 (``weighted_rank_sum``), it reproduces the published Final Rank
order (see ``tests/metrics/test_ranking.py``). A weighted sum of raw values can't be meant: PSNR (~26 dB) would
swamp SSIM (~0.86). Ranks depend on the whole field, so a score is only comparable within one call.

Examples:
    >>> results = {
    ...     "a": {"psnr": 26.2, "ssim": 0.861, "lpips": 0.128, "niqe": 10.9},
    ...     "b": {"psnr": 26.3, "ssim": 0.858, "lpips": 0.133, "niqe": 11.8},
    ... }
    >>> [(r.name, r.score, r.rank) for r in combined_rank(results, NTIRE_2025_LLIE)]
    [('a', 2.1, 1), ('b', 2.7, 2)]

References:
    - X. Liu et al., "NTIRE 2025 Challenge on Low Light Image Enhancement: Methods and Results," CVPR Workshops 2025.
      https://arxiv.org/abs/2510.13670
"""

__all__ = [
    "NTIRE_2025_LLIE",
    "RankedMethod",
    "combined_rank",
    "competition_rank",
    "weighted_rank_sum",
]

from collections.abc import Mapping, Sequence
from typing import NamedTuple

from mon.core.utils import ReadOnlyDict
from mon.metrics.registry import METRICS

NTIRE_2025_LLIE: Mapping[str, float] = ReadOnlyDict({"psnr": 0.5, "ssim": 0.5, "lpips": 0.4, "niqe": 0.2})
"""Final Rank weights of the NTIRE 2025 LLIE challenge (Liu et al., Section 2 and Table 1)."""


class RankedMethod(NamedTuple):
    """One method's place in a composite ranking.

    Attributes:
        name: Method name (the key in ``combined_rank``'s ``results``).
        score: Weighted sum of the method's per-metric ranks (lower is better).
        rank: Final place by ``score``, 1 = best; equal scores share the better place.
        metric_ranks: The method's rank on each weighted metric.
    """

    name        : str
    score       : float
    rank        : int
    metric_ranks: dict[str, int]


def competition_rank(values: Sequence[float], *, higher_better: bool) -> list[int]:
    """Rank ``values`` 1 = best, ties sharing the better rank ("1224" ranking).

    Args:
        values: One score per method; compared exactly, so round them first to rank at a published precision.
        higher_better: Whether a larger value is better.

    Examples:
        >>> competition_rank([0.861, 0.858, 0.858, 0.863], higher_better=True)
        [2, 3, 3, 1]
    """
    if any(v != v for v in values):
        raise ValueError("values contain NaN, which has no rank")
    sign = -1.0 if higher_better else 1.0
    keys = [sign * v for v in values]
    return [1 + sum(k < key for k in keys) for key in keys]


def weighted_rank_sum(metric_ranks: Mapping[str, int], weights: Mapping[str, float]) -> float:
    """Return ``sum(weights[m] * metric_ranks[m])`` over the weighted metrics (lower is better).

    Use it to score ranks taken from a published table; ``combined_rank`` ranks raw scores first.
    The sum is rounded to 9 decimals so float noise (``0.5 * 2 + 0.4 * 7``) doesn't break ties.

    Raises:
        ValueError: If ``metric_ranks`` lacks a weighted metric.

    Examples:
        >>> weighted_rank_sum({"psnr": 2, "ssim": 2, "lpips": 7, "niqe": 11}, NTIRE_2025_LLIE)  # Table 1, NWPU-HVI
        7.0
    """
    if missing := sorted(set(weights) - set(metric_ranks)):
        raise ValueError(f"no rank for {missing}")
    return round(sum(w * metric_ranks[m] for m, w in weights.items()), 9)


def combined_rank(results: Mapping[str, Mapping[str, float]], weights: Mapping[str, float], *,
                  higher_better: Mapping[str, bool] | None = None) -> list[RankedMethod]:
    """Rank methods by the weighted sum of their per-metric ranks (lower is better), best first.

    Args:
        results: Method name -> metric name -> mean score (e.g. the ``mean`` of each run's ``metrics.yaml``);
            metrics without a weight are ignored.
        weights: Metric name -> weight, e.g. ``NTIRE_2025_LLIE``.
        higher_better: Metric name -> direction; metrics not listed take their ``METRICS`` entry's.

    Raises:
        ValueError: If ``results`` or ``weights`` is empty, or a method lacks a weighted metric.
        KeyError: If a weighted metric has no direction (not in ``higher_better`` or ``METRICS``).
    """
    if not results or not weights:
        raise ValueError("combined_rank needs at least one method and one weighted metric")
    names = list(results)
    for name in names:
        if missing := sorted(set(weights) - set(results[name])):
            raise ValueError(f"method {name!r} has no score for {missing}")
    directions = dict(higher_better or {})
    ranks      = {
        metric: competition_rank([results[n][metric] for n in names], higher_better=(
            directions[metric] if metric in directions else bool(METRICS.entry(metric).meta["higher_better"])))
        for metric in weights
    }
    per_method = [{m: ranks[m][i] for m in weights} for i in range(len(names))]
    scores     = [weighted_rank_sum(r, weights) for r in per_method]
    places     = competition_rank(scores, higher_better=False)
    ranked     = [RankedMethod(name, scores[i], places[i], per_method[i]) for i, name in enumerate(names)]
    return sorted(ranked, key=lambda r: (r.rank, names.index(r.name)))
