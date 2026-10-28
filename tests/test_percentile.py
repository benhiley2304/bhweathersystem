import math
from bhws_core.percentile import midrank_percentile, midrank_series, minmax_index, describe, extreme_label


def pine_midrank(a):
    # Literal port of f_midrank() in BHWS_COT_*.pine
    n = len(a); v = a[-1]
    less = sum(1 for x in a if x < v); eq = sum(1 for x in a if x == v)
    return 100.0 * (less + 0.5 * (eq - 1)) / (n - 1)


def test_matches_pine_formula_and_window():
    import random
    random.seed(7)
    xs = [random.randint(-50000, 50000) for _ in range(400)]
    assert math.isclose(midrank_percentile(xs), pine_midrank(xs[-156:]))


def test_bounds_and_ties():
    assert midrank_percentile(list(range(156))) == 100.0
    assert midrank_percentile(list(range(156, 0, -1))) == 0.0
    flat = [5.0] * 50
    assert midrank_percentile(flat) == 50.0


def test_minimum_history():
    assert midrank_percentile(list(range(19))) is None
    assert midrank_percentile(list(range(20))) is not None


def test_series_is_point_in_time():
    xs = list(range(30)) + [-1000]
    s = midrank_series(xs)
    # adding a future value must not change earlier readings
    s2 = midrank_series(xs + [10 ** 6])
    assert s == s2[:-1]


def test_method_disagreement_is_visible():
    # one outlier dominates min-max but not mid-rank
    xs = [0.0] * 150 + [1000.0] + [float(i) for i in range(1, 6)]
    d = describe(xs, basis="futures_only")
    assert d["method"] == "midrank_156w" and d["basis"] == "futures_only"
    assert d["percentile"] > 90 and d["legacy_minmax_520"] < 1


def test_labels():
    assert extreme_label(95) == "extreme high"
    assert extreme_label(10) == "extreme low"
    assert extreme_label(50) == "inside range"
    assert extreme_label(None) == "insufficient history"
