"""成绩扁平化与协议配置测试（§13.4、§6.1、§6.2）。

`evaluation.json` 必须是契约行；折被静默丢弃必须看得见；
协议配置的数值必须与冻结常量一致。
"""

import pytest

from src.contracts import (
    EVALUATION_CONFIG_VERSION,
    MODEL_CONFIG_VERSION,
    validate_package,
)
from src.evaluation import metrics, protocol, split

ENVELOPE = {
    "schema_version": "1.0.0",
    "data_version": "dv-test",
    "run_id": "ru-test",
    "prediction_mode": "oof_replay",
    "data_kind": "real_standard",
}


def _synthetic_scores(n_folds=5):
    y = [0, 1] * 40
    p = [0.1 + 0.02 * i for i in range(len(y))]
    ids = [f"P{i:03d}" for i in range(len(y))]
    folds = [i % n_folds for i in range(len(y))]
    return metrics.aggregate_all(y, p, ids, folds, seed=20260914)


@pytest.fixture(scope="module")
def flat():
    return metrics.flatten_evaluation(_synthetic_scores(), ENVELOPE)


# ---- 契约行 ----

def test_flat_rows_pass_contract(flat):
    """扁平结果必须整体通过 evaluation 产物契约（§13.4）。"""
    validate_package("evaluation", flat)


def test_every_row_carries_the_envelope(flat):
    for r in flat:
        for k, v in ENVELOPE.items():
            assert r[k] == v


def test_per_fold_rows_cover_every_fold_and_metric(flat):
    per_fold = [r for r in flat if r["aggregation"] == "per_fold"]
    assert {r["fold"] for r in per_fold} == {0, 1, 2, 3, 4}
    assert {r["metric"] for r in per_fold} >= set(metrics.FLAT_METRICS)


def test_macro_rows_report_fold_counts(flat):
    """宏平均行必须带有效折与总折数——否则丢弃折无法察觉（§6.4.1）。"""
    macro = [r for r in flat if r["aggregation"] == "macro_mean"]
    assert macro
    for r in macro:
        assert r["total_folds"] == 5
        assert 0 <= r["valid_folds"] <= r["total_folds"]


def test_dropped_fold_is_visible():
    """回归：少一折必须体现在 valid_folds 上，不得静默。"""
    y = [0, 1] * 40
    p = [0.1 + 0.02 * i for i in range(len(y))]
    ids = [f"P{i:03d}" for i in range(len(y))]
    # 折 4 全为负例 → 该折 AP/ROC-AUC 不适用，宏平均的有效折数下降
    folds = [i % 5 for i in range(len(y))]
    scores = metrics.aggregate_all(y, p, ids, folds, seed=20260914)
    scores["per_fold"][4] = {
        "n_test": 10, "n_positive": 0, "base_rate": 0.0,
        **{f"{m}": None for m in metrics.FLAT_METRICS},
        **{f"{m}_applicable": False for m in metrics.FLAT_METRICS},
        **{f"{m}_reason": "评估集无正例" for m in metrics.FLAT_METRICS},
    }
    for metric, rec in scores["macro_mean"].items():
        if isinstance(rec, dict) and "fold_std" in rec:
            rec["valid_folds"], rec["total_folds"] = 4, 5
            rec["complete"] = False
    rows = metrics.flatten_evaluation(scores, ENVELOPE)
    mm = [r for r in rows
          if r["aggregation"] == "macro_mean" and r["metric"] == "ap"][0]
    assert mm["valid_folds"] == 4 and mm["total_folds"] == 5
    # 缺的那一折仍以不可用行出现，不消失
    missing = [r for r in rows
               if r["aggregation"] == "per_fold" and r["fold"] == 4
               and r["metric"] == "ap"][0]
    assert missing["applicable"] is False
    assert missing["reason"]


def test_pooled_rows_are_separate_from_per_fold(flat):
    pooled = [r for r in flat if r["aggregation"] == "pooled_oof_replay"]
    assert pooled
    assert all(r["fold"] is None for r in pooled)
    per_fold = [r for r in flat if r["aggregation"] == "per_fold"]
    assert all(r["fold"] is not None for r in per_fold)


def test_topk_requested_and_actual_are_preserved(flat):
    """Top-K 的请求数与实际数必须保留，截断不得伪造（§6.4）。"""
    ks = [r for r in flat if r["metric"].startswith("k_actual@")]
    assert ks and all(r["applicable"] for r in ks)
    req = [r for r in flat if r["metric"].startswith("k_requested@")]
    assert len(req) == len(ks)


def test_flat_output_has_no_nan(flat):
    for r in flat:
        if isinstance(r["value"], float):
            assert r["value"] == r["value"]


# ---- 协议配置与冻结常量一致 ----

def test_protocol_config_matches_frozen_constants():
    """配置文件是单一可读描述，数值必须与 split.py 冻结常量一致。"""
    cfg = protocol.load_protocol()
    assert cfg["seed"] == split.SEED
    assert cfg["n_outer"] == split.N_OUTER
    assert cfg["n_inner"] == split.N_INNER
    assert cfg["split_schemes"] == split.GROUP_SCHEMES


def test_protocol_version_matches_contract_constant():
    assert protocol.load_protocol()["version"] == EVALUATION_CONFIG_VERSION


def test_candidates_version_matches_contract_constant():
    from src.models import registry

    assert registry.load_candidates()["candidates_version"] == MODEL_CONFIG_VERSION


def test_config_versions_lists_release_configs():
    versions = protocol.config_versions()
    assert versions == {
        "evaluation_protocol": EVALUATION_CONFIG_VERSION,
        "candidates": MODEL_CONFIG_VERSION,
        "whitelist": versions["whitelist"],
    }
    assert versions["whitelist"]


# ---- round_id 含划分方案 ----

def test_round_id_unchanged_for_random_scheme():
    """随机方案下与历史口径逐字节一致（§6.1）。"""
    assert split.round_id_of(20260914) == "seed20260914"
    assert split.round_id_of(20260914, "random") == "seed20260914"


def test_round_id_separates_schemes():
    """随机与分组发布不得撞同一轮次 ID（§6.2）。"""
    assert split.round_id_of(20260914, "road") != split.round_id_of(20260914)
    assert split.round_id_of(20260914, "road") == "seed20260914-road"