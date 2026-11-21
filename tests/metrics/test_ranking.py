"""Tests for ``mon.metrics.ranking``: competition ranks, weighted rank sums, and NTIRE 2025 LLIE's Table 1."""

import math

import pytest

from mon.metrics.ranking import NTIRE_2025_LLIE, RankedMethod, combined_rank, competition_rank, weighted_rank_sum

# Liu et al., NTIRE 2025 LLIE report, Table 1: per-metric ranks (PSNR, SSIM, LPIPS, NIQE) and Final Rank, top 5.
TABLE_1 = {
    "NWPU-HVI"   : ((2, 2, 7, 11), 1),
    "Imagine"    : ((1, 3, 9, 23), 2),
    "pengpeng-yu": ((4, 3, 11, 16), 3),
    "DAVIS-K"    : ((14, 1, 6, 9), 4),
    "SoloMan"    : ((5, 6, 8, 19), 5),
}
METRICS_ORDER = ("psnr", "ssim", "lpips", "niqe")


# --- competition_rank ---

@pytest.mark.parametrize(("values", "higher_better", "expected"), [
    ([26.24, 26.35, 25.85], True , [2, 1, 3]),
    ([0.128, 0.133, 0.127], False, [2, 3, 1]),
    ([0.861, 0.858, 0.858, 0.863], True, [2, 3, 3, 1]),  # Table 1 SSIM: the two 0.858 share 3rd
    ([1.0, 1.0, 1.0], False, [1, 1, 1]),
    ([], True, []),
])
def test_competition_rank(values: list[float], higher_better: bool, expected: list[int]):
    assert competition_rank(values, higher_better=higher_better) == expected


def test_competition_rank_rejects_nan():
    with pytest.raises(ValueError, match="NaN"):
        competition_rank([1.0, math.nan], higher_better=True)


# --- combined_rank ---

def test_ntire_2025_llie_weights():
    assert dict(NTIRE_2025_LLIE) == {"psnr": 0.5, "ssim": 0.5, "lpips": 0.4, "niqe": 0.2}


def test_reproduces_ntire_2025_llie_table_1():
    # The published ranks are over the whole field, so score them as given instead of re-ranking the top 5.
    scores = [weighted_rank_sum(dict(zip(METRICS_ORDER, ranks, strict=True)), NTIRE_2025_LLIE)
              for ranks, _ in TABLE_1.values()]
    assert scores == [7.0, 10.2, 11.1, 11.7, 12.5]
    assert competition_rank(scores, higher_better=False) == [final for _, final in TABLE_1.values()]


def test_weighted_rank_sum_needs_every_weighted_metric():
    with pytest.raises(ValueError, match=r"no rank for \['niqe'\]"):
        weighted_rank_sum({"psnr": 1, "ssim": 1, "lpips": 1}, NTIRE_2025_LLIE)


def test_ranks_raw_scores_with_metric_directions():
    results = {
        "NWPU-HVI"   : {"psnr": 26.24, "ssim": 0.861, "lpips": 0.128, "niqe": 10.95},
        "Imagine"    : {"psnr": 26.35, "ssim": 0.858, "lpips": 0.133, "niqe": 11.81},
        "pengpeng-yu": {"psnr": 25.85, "ssim": 0.858, "lpips": 0.134, "niqe": 11.29},
        "DAVIS-K"    : {"psnr": 25.14, "ssim": 0.863, "lpips": 0.127, "niqe": 10.58, "brisque": 1.0},
    }
    ranked = combined_rank(results, NTIRE_2025_LLIE)  # directions from METRICS; unweighted brisque ignored
    by_name = {r.name: r for r in ranked}
    assert by_name["NWPU-HVI"].metric_ranks == {"psnr": 2, "ssim": 2, "lpips": 2, "niqe": 2}
    assert by_name["pengpeng-yu"].metric_ranks == {"psnr": 3, "ssim": 3, "lpips": 4, "niqe": 3}
    assert by_name["DAVIS-K"] == RankedMethod("DAVIS-K", 3.1, 1, {"psnr": 4, "ssim": 1, "lpips": 1, "niqe": 1})
    assert [r.name for r in ranked] == ["DAVIS-K", "NWPU-HVI", "Imagine", "pengpeng-yu"]


def test_equal_scores_share_a_place_in_input_order():
    results = {"a": {"psnr": 1.0, "ssim": 2.0}, "b": {"psnr": 2.0, "ssim": 1.0}, "c": {"psnr": 0.0, "ssim": 0.0}}
    ranked  = combined_rank(results, {"psnr": 0.5, "ssim": 0.5})
    assert [(r.name, r.rank) for r in ranked] == [("a", 1), ("b", 1), ("c", 3)]


def test_errors():
    with pytest.raises(ValueError, match="at least one"):
        combined_rank({}, NTIRE_2025_LLIE)
    with pytest.raises(ValueError, match=r"'a' has no score for \['niqe'\]"):
        combined_rank({"a": {"psnr": 1.0, "ssim": 1.0, "lpips": 1.0}}, NTIRE_2025_LLIE)
    with pytest.raises(KeyError):
        combined_rank({"a": {"made_up": 1.0}}, {"made_up": 1.0})
    assert combined_rank({"a": {"made_up": 1.0}}, {"made_up": 1.0}, higher_better={"made_up": True})[0].rank == 1
