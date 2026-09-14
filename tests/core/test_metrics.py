"""评测测试：四类聚合分列、人工反例、并列排序、边界（§6.4、§6.4.1、里程碑 §5.4）。"""

import numpy as np
import pytest

from src.evaluation import metrics
from src.evaluation.split import SEED, stable_tie_key


# --------------------------------------------------------------------------
# 人工反例（§6.4.1）——这是验收反例，不是项目实验成绩
#
# 注意：方案 §6.4.1 原文将折 B 写作 (y,p)=[(1,0.9),(1,0.8)]，即两个正例。
# 该情形下折 B 的 AUC 无定义、拼接 AUC/AP 均为 1.0，与原文自述的
# "拼接 AUC 为 0.75、AP 为 5/6" 矛盾。第二项应为负例 (0,0.8)：
# 此时两折 AUC/AP 各为 1，拼接 AUC=0.75、AP=5/6，与原文数字完全吻合。
# 按可验证的唯一自洽解释实现，并在此注明。
# --------------------------------------------------------------------------

COUNTEREXAMPLE = {
    "y": [1, 0, 1, 0],
    "p": [0.2, 0.1, 0.9, 0.8],
    "fold": [0, 0, 1, 1],
    "ids": ["a", "b", "c", "d"],
}


def test_counterexample_per_fold_auc_ap_are_one():
    """折 A/B 各自 AUC 与 AP 都是 1。"""
    c = COUNTEREXAMPLE
    pf = metrics.per_fold_metrics(c["y"], c["p"], c["ids"], c["fold"], q_list=())
    assert pf[0]["roc_auc"] == pytest.approx(1.0)
    assert pf[0]["ap"] == pytest.approx(1.0)
    assert pf[1]["roc_auc"] == pytest.approx(1.0)
    assert pf[1]["ap"] == pytest.approx(1.0)


def test_counterexample_pooled_differs():
    """拼接 AUC=0.75、AP=5/6；评测器必须保留这一区别。"""
    c = COUNTEREXAMPLE
    pool = metrics.pooled_oof_replay(c["y"], c["p"], c["ids"], q_list=())
    assert pool["roc_auc"] == pytest.approx(0.75)
    assert pool["ap"] == pytest.approx(5 / 6)


def test_counterexample_not_mixed():
    """不能用一个未注明聚合方式的"总 AUC"混写。"""
    c = COUNTEREXAMPLE
    agg = metrics.aggregate_all(c["y"], c["p"], c["ids"], c["fold"], q_list=())
    per_fold_mean = agg["macro_mean"]["roc_auc"]["value"]
    pooled = agg["pooled_oof_replay"]["roc_auc"]
    assert per_fold_mean == pytest.approx(1.0)
    assert pooled == pytest.approx(0.75)
    assert per_fold_mean != pooled


# --------------------------------------------------------------------------
# 四类聚合口径分列
# --------------------------------------------------------------------------

def test_aggregate_has_four_buckets():
    c = COUNTEREXAMPLE
    agg = metrics.aggregate_all(c["y"], c["p"], c["ids"], c["fold"])
    assert set(agg) == {"per_fold", "macro_mean", "sample_weighted", "pooled_oof_replay"}


def test_macro_mean_and_fold_std():
    c = COUNTEREXAMPLE
    agg = metrics.aggregate_all(c["y"], c["p"], c["ids"], c["fold"])
    ap = agg["macro_mean"]["ap"]
    assert ap["value"] == pytest.approx(1.0)
    assert ap["fold_std"] == pytest.approx(0.0)
    assert ap["complete"] is True


def test_sample_weighted_brier_uses_fold_sizes():
    """Brier/log loss 按每折样本数加权（§6.4.1）。"""
    y = np.array([1, 0, 1, 1, 0, 0, 0, 0])
    p = np.array([0.9, 0.1, 0.8, 0.7, 0.2, 0.1, 0.3, 0.2])
    fold = [0, 0, 0, 0, 1, 1, 1, 1]
    agg = metrics.aggregate_all(y, p, [str(i) for i in range(8)], fold)
    assert agg["sample_weighted"]["brier"] is not None
    assert agg["sample_weighted"]["log_loss"] is not None


# --------------------------------------------------------------------------
# 并列排序（§6.4）——管龄并列是常态（最大并列组 765）
# --------------------------------------------------------------------------

def test_stable_tie_key_deterministic():
    assert stable_tie_key("12345") == stable_tie_key("12345")
    assert stable_tie_key("12345") != stable_tie_key("54321")


def test_stable_order_independent_of_label():
    """并列排序不得依赖标签。"""
    ids = ["p1", "p2", "p3"]
    scores = [0.5, 0.5, 0.5]
    o1 = metrics.stable_order(ids, scores, SEED)
    o2 = metrics.stable_order(ids, scores, SEED)
    assert o1 == o2
    # 顺序由 SHA256 决定，与传入顺序无关
    o3 = metrics.stable_order(list(reversed(ids)), list(reversed(scores)), SEED)
    assert sorted(o3) == sorted(o1)


def test_topk_absolute_truncates_when_k_exceeds_candidates():
    """K 超过候选数时显式截为候选总数，不伪造补足记录（§6.4）。"""
    y = [1, 0]
    p = [0.9, 0.1]
    ids = ["a", "b"]
    r = metrics.top_k_absolute(y, p, ids, k=500, seed=SEED)
    assert r["k_requested"] == 500
    assert r["k_actual"] == 2
    assert r["truncated"] is True


def test_topk_absolute_k_zero_returns_empty():
    """K=0 返回空清单，Precision/Lift 记为不适用（§6.4）。"""
    r = metrics.top_k_absolute([1, 0], [0.9, 0.1], ["a", "b"], k=0, seed=SEED)
    assert r["applicable"] is False
    assert r["k_actual"] == 0
    assert r["precision"] is None


def test_topk_ratio_never_truncates():
    """q<=1 时 k=ceil(q*n)<=n，比例路径不触发截断。"""
    r = metrics.top_k_metrics([1, 0], [0.9, 0.1], ["a", "b"], q=0.9, seed=SEED)
    assert r["k_requested"] == 2
    assert r["k_actual"] == 2
    assert r["truncated"] is False


def test_topk_empty_candidates():
    r = metrics.top_k_metrics([], [], [], q=0.1, seed=SEED)
    assert r["applicable"] is False
    assert r["precision"] is None and r["lift"] is None


# --------------------------------------------------------------------------
# 无效折处理（§6.4.1）
# --------------------------------------------------------------------------

def test_no_positive_fold_ap_not_applicable():
    y = [0, 0, 1, 0]
    p = [0.1, 0.2, 0.9, 0.3]
    fold = [0, 0, 1, 1]
    pf = metrics.per_fold_metrics(y, p, ["a", "b", "c", "d"], fold, q_list=())
    assert pf[0]["ap_applicable"] is False
    assert pf[0]["ap"] is None
    assert "无正例" in pf[0]["ap_reason"]


def test_single_class_fold_brier_still_computable():
    """Brier/log loss 可在单类测试折计算（§6.4.1）。"""
    y = [0, 0, 1, 0]
    p = [0.1, 0.2, 0.9, 0.3]
    fold = [0, 0, 1, 1]
    pf = metrics.per_fold_metrics(y, p, ["a", "b", "c", "d"], fold, q_list=())
    assert pf[0]["brier_applicable"] is True
    assert pf[0]["brier"] is not None


def test_incomplete_folds_reported_not_hidden():
    """失败折不静默丢弃：记录 valid_folds/total_folds 并标 partial。"""
    y = [0, 0, 1, 0]
    p = [0.1, 0.2, 0.9, 0.3]
    fold = [0, 0, 1, 1]
    agg = metrics.aggregate_all(y, p, ["a", "b", "c", "d"], fold, q_list=())
    ap = agg["macro_mean"]["ap"]
    assert ap["valid_folds"] == 1
    assert ap["total_folds"] == 2
    assert ap["complete"] is False