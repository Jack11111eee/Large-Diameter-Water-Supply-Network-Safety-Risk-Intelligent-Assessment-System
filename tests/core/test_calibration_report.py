"""校准对照测试（§5.3）。

守住两条禁令：不得仅凭 Brier 更低断言校准更好；
不得把校准概率与原模型 SHAP 混用。
"""

import numpy as np
import pytest

from src.evaluation.calibration_report import (
    CALIBRATION_CONCLUSION,
    MIN_BIN_N,
    compare_calibration,
    per_fold_calibration,
    reliability_bins,
)

FORBIDDEN_CLAIMS = ["校准更好", "校准更优", "已校准正确", "校准已改善"]


def _toy(n=600, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n)
    # 未校准：过度自信；校准：向基率收缩
    raw = np.clip(0.5 + 0.35 * (y - 0.5) + rng.normal(0, 0.05, n), 0.001, 0.999)
    cal = 0.5 * raw + 0.5 * float(y.mean())
    return y, cal, raw


# ---- 分箱 ----

def test_every_bin_reports_n_and_n_positive():
    y, cal, _ = _toy()
    bins = reliability_bins(y, cal)
    assert bins
    for b in bins:
        assert "n" in b and "n_positive" in b
        assert b["n_positive"] <= b["n"]


def test_empty_bins_are_kept_with_uniform_strategy():
    """等宽分箱在集中分布下会出空箱，空箱必须保留而非丢弃（§5.3）。"""
    y = np.array([0, 1] * 200)
    p = np.full(400, 0.02)
    bins = reliability_bins(y, p, strategy="uniform")
    assert len(bins) == 10
    assert any(b["n"] == 0 for b in bins)
    for b in bins:
        if b["n"] == 0:
            assert b["observed_rate"] is None and b["mean_p"] is None


def test_low_n_flag_below_threshold():
    y = np.array([0, 1] * 5)
    p = np.linspace(0.01, 0.99, 10)
    bins = reliability_bins(y, p)
    for b in bins:
        assert b["low_n"] == (b["n"] < MIN_BIN_N)


def test_unknown_strategy_rejected():
    with pytest.raises(KeyError, match="未知分箱策略"):
        reliability_bins(np.array([0, 1]), np.array([0.1, 0.9]), strategy="nope")


def test_length_mismatch_rejected():
    with pytest.raises(ValueError, match="长度不一致"):
        reliability_bins(np.array([0, 1, 0]), np.array([0.1, 0.9]))


# ---- 对照 ----

def test_compare_reports_both_arms_and_bins():
    y, cal, raw = _toy()
    r = compare_calibration(y, cal, raw)
    assert set(r["brier"]) == {"calibrated", "uncalibrated", "delta"}
    assert set(r["log_loss"]) == {"calibrated", "uncalibrated", "delta"}
    assert r["bins_calibrated"] and r["bins_uncalibrated"]
    assert r["n"] == len(y)


def test_conclusion_forbids_brier_only_claim():
    """结论必须是固定串，明写不得仅凭 Brier 断言校准更好（§5.3）。"""
    y, cal, raw = _toy()
    r = compare_calibration(y, cal, raw)
    assert r["conclusion"] == CALIBRATION_CONCLUSION
    assert "不构成校准更好的证据" in r["conclusion"]
    # 结论本身以否定形式引用该措辞；数据载荷不得出现任何肯定的校准更优断言
    payload = {k: v for k, v in r.items() if k != "conclusion"}
    assert not [w for w in FORBIDDEN_CLAIMS if w in str(payload)]


def test_compare_requires_same_pipe_set():
    y, cal, raw = _toy()
    with pytest.raises(ValueError, match="同一管段集合"):
        compare_calibration(y, cal[:-1], raw)


# ---- 逐折，非 pooled ----

def test_per_fold_is_not_pooled():
    y, cal, raw = _toy(n=500)
    folds = np.repeat([0, 1, 2, 3, 4], 100)
    r = per_fold_calibration(y, cal, raw, folds)
    assert set(r) == {0, 1, 2, 3, 4}
    assert sum(v["n"] for v in r.values()) == 500
    pooled = compare_calibration(y, cal, raw)
    # 逐折 brier 与 pooled 不是同一个数（口径必须分列）
    assert any(abs(v["brier"]["calibrated"] - pooled["brier"]["calibrated"]) > 0
               for v in r.values())


def test_identical_arms_have_zero_delta():
    y, cal, _ = _toy()
    r = compare_calibration(y, cal, cal)
    assert r["brier"]["delta"] == 0.0
    assert r["log_loss"]["delta"] == 0.0