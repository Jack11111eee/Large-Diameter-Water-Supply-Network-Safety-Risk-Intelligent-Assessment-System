"""F2/F3 消融测试（§3.2、§10 W3）。

守住三条：F2/F3 是附加消融而非默认；增益以逐折配对 delta 表述；
不得出现任何 AUC 上限断言。
"""

import pytest

from src.data import loader
from src.evaluation import ablation, split

FAST_CONFIG = {
    "candidates_version": "test_ablation",
    "selection": {"criterion": "inner_ap", "tolerance": 0.005},
    "candidates": [{
        "name": "logreg", "family": "linear", "complexity_rank": 0,
        "param_grid": [{"C": 1.0}],
    }],
}


@pytest.fixture(scope="module")
def small():
    pipes = loader.load_attributes().iloc[:1200].reset_index(drop=True)
    counts, _ = loader.make_labels(pipes, loader.load_events())
    return pipes, counts


@pytest.fixture(scope="module")
def result(small):
    pipes, counts = small
    folds = split.make_outer_folds(
        pipes["ID"].astype(str).to_numpy(), counts.gt(0).astype(int).to_numpy())
    return ablation.run_ablation(
        pipes, counts, folds=folds, candidates_config=FAST_CONFIG,
        structures={"M1": ablation.STRUCTURES["M1"],
                    "M3": ablation.STRUCTURES["M3"],
                    "M4": ablation.STRUCTURES["M4"]})


# ---- 配对 delta ----

def test_paired_delta_is_per_fold_first():
    """先算逐折差再聚合，而非两个 macro 均值相减（§10 W3）。"""
    base = {0: {"ap": 0.10}, 1: {"ap": 0.30}}
    alt = {0: {"ap": 0.20}, 1: {"ap": 0.20}}
    d = ablation.paired_delta(base, alt)
    assert d["per_fold_delta"] == {0: pytest.approx(0.10), 1: pytest.approx(-0.10)}
    assert d["mean_delta"] == pytest.approx(0.0)
    # 两个 macro 均值之差为 0 时仍能看出折间方向相反
    assert d["fold_std"] == pytest.approx(0.1414213562, abs=1e-6)


def test_paired_delta_handles_missing_and_inapplicable():
    base = {0: {"ap": 0.1, "ap_applicable": True},
            1: {"ap": None, "ap_applicable": False}}
    alt = {0: {"ap": 0.2, "ap_applicable": True},
           1: {"ap": 0.3, "ap_applicable": True}}
    d = ablation.paired_delta(base, alt, applicable_key="ap_applicable")
    assert d["n_folds"] == 1
    assert d["mean_delta"] == pytest.approx(0.1)


def test_paired_delta_empty_when_no_valid_fold():
    d = ablation.paired_delta({}, {})
    assert d["mean_delta"] is None and d["fold_std"] is None


# ---- 真实数据上的结构对照 ----

def test_structures_share_folds_and_candidates(result):
    """各结构用同一 folds、同一候选集、同一协议（§10 W3）。"""
    for key in ["M1", "M3", "M4"]:
        rec = result["structures"][key]
        assert set(rec["per_fold"]) == {0, 1, 2, 3, 4}
        assert rec["model_id_per_fold"]
        assert rec["macro_mean"]["total_folds"] == 5


def test_layers_are_additive(result):
    """F2/F3 是叠加在 F1 之上的附加层，不是替换。"""
    n = {k: result["structures"][k]["n_columns"] for k in ["M1", "M3", "M4"]}
    assert n["M1"] < n["M3"] < n["M4"]
    assert result["structures"]["M1"]["layer_spec"] == "F1_base_environment"
    assert result["structures"]["M3"]["layer_spec"].endswith("+F2_operational_snapshot")
    assert result["structures"]["M4"]["layer_spec"].endswith("+F3_topology")


def test_delta_reported_for_every_structure(result):
    for key in ["M1", "M3", "M4"]:
        d = result["structures"][key]["paired_delta_vs_M1"]
        assert d["metric"] == "ap"
        assert d["n_folds"] > 0
        assert d["fold_std"] is not None
    # M1 对自身 delta 恒为 0
    assert result["structures"]["M1"]["paired_delta_vs_M1"]["mean_delta"] == 0.0


def test_disclosures_present(result):
    """三条强制披露必须作为数据输出，不能只写在文档里（§3.2、§6.2、§10 W3）。"""
    d = result["disclosures"]
    assert "不默认为事前观测" in d["F2"]
    assert "全网结构已知" in d["F3"]
    assert "与随机 CV 接近不证明无泄漏" in d["F3"]
    assert "保留简单模型可赢的可能" in d["common"]


def test_no_auc_ceiling_claim_in_output(result):
    """不得出现任何 AUC 上限断言（数据报告 §8、方案 §5.2 已撤回）。

    披露串以否定形式提及「上限」（"不以任何 AUC 上限解释差异"），
    因此只扫描结构结果；0.80 与「天花板」全文禁止。
    """
    assert "0.80" not in str(result)
    assert "天花板" not in str(result)
    assert "上限" not in str(result["structures"])


def test_structure_selection_uses_inner_ap_only(result):
    for rec in result["structures"].values():
        for log in rec["selection_log"].values():
            assert log["criterion"] == "inner_ap"
            assert not [k for k in log if "outer" in k.lower()]