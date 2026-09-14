"""分级测试（§5.3.1）。

断言直接对照平均秩百分位公式，**不**断言夹具 decision.json 的等级
（该夹具是契约形状/展示样例，其等级为手工演示值，不遵循 §5.3.1）。
"""

import pytest

from src.contracts import GRADE_CONFIG_VERSION, ContractError, validate_package
from src.decision import grade, load_grade_config
from src.decision.errors import DecisionError

from .helpers import (
    assert_unchanged,
    deep,
    fixture,
    grade_row,
    percentile,
    prediction,
    reference,
    view,
)

CFG = load_grade_config()


# --------------------------------------------------------------------------
# 边界：80 / 95 / 99
# --------------------------------------------------------------------------

# 参考集合 100 条（0.01..1.00）。无并列时 percentile(p) = 100*below/100 = below，
# 因此取 p 落在相邻参考分之间即可精确命中整数边界。
REF_100 = [i / 100.0 for i in range(1, 101)]


@pytest.mark.parametrize("p,expected_percentile,expected", [
    (0.005, 0.0, "L1"),    # 最小
    (0.799, 79.0, "L1"),   # 恰低于 80
    (0.805, 80.0, "L2"),   # 80 边界，下含
    (0.949, 94.0, "L2"),   # 恰低于 95
    (0.955, 95.0, "L3"),   # 95 边界，下含
    (0.989, 98.0, "L3"),   # 恰低于 99
    (0.995, 99.0, "L4"),   # 99 边界，下含
    (1.000, 99.5, "L4"),   # 最大，与参考最大值并列
])
def test_grade_boundaries_80_95_99(p, expected_percentile, expected):
    """80/95/99 三个边界按 [下含, 上不含) 落档，L4 上含 100。"""
    ref = reference("r", REF_100)
    rows = grade([prediction("P", p)], ref, CFG)
    assert rows[0]["risk_percentile"] == pytest.approx(expected_percentile)
    assert rows[0]["risk_percentile"] == pytest.approx(percentile(p, REF_100))
    assert rows[0]["relative_risk_level"] == expected


def test_grade_percentile_100_upper_boundary():
    """百分位恰为 100（高于全部参考分数）落在 L4，上界含 100。"""
    scores = [i / 100.0 for i in range(1, 100)]  # 最大 0.99
    rows = grade([prediction("P", 0.995)], reference("r", scores), CFG)
    assert rows[0]["risk_percentile"] == pytest.approx(100.0)
    assert rows[0]["relative_risk_level"] == "L4"


def test_percentile_formula_matches_spec():
    """百分位定义：100*(count(r<p)+0.5*count(r==p))/n。"""
    scores = [0.1, 0.2, 0.2, 0.4, 0.9]
    ref = reference("r", scores)
    rows = grade([prediction("P", 0.2)], ref, CFG)
    # below=1, equal=2 -> 100*(1+1)/5 = 40.0
    assert rows[0]["risk_percentile"] == pytest.approx(40.0)


def test_full_precision_ties_same_level():
    """同一 R 内相同分数必须同等级；不先四舍五入显示值再分级。"""
    # 0.311 与 0.312 在两位小数下都显示 0.31，但全精度不同 -> 可以不同级；
    # 而完全相同的全精度分数必须同级。
    scores = [0.311, 0.312] + [i / 100.0 for i in range(1, 50)]
    ref = reference("r", scores)
    rows = grade([prediction("A", 0.311), prediction("B", 0.311)], ref, CFG)
    assert rows[0]["risk_percentile"] == rows[1]["risk_percentile"]
    assert rows[0]["relative_risk_level"] == rows[1]["relative_risk_level"]


def test_fixture_tie_pipes_same_level():
    """夹具 F04/F05 概率相同（p=0.31），必须同分同级。"""
    preds = fixture("predictions.json")
    ref = fixture("reference_bundle.json")
    rows = {r["pipe_id"]: r for r in grade(preds, ref, CFG)}
    f04 = rows["SYN-F04_tie_probability"]
    f05 = rows["SYN-F05_tie_probability_b"]
    assert f04["risk_percentile"] == f05["risk_percentile"]
    assert f04["relative_risk_level"] == f05["relative_risk_level"]
    # 夹具参考集合下 0.31 -> 85.71 -> L2
    assert f04["risk_percentile"] == pytest.approx(85.71428571428571)
    assert f04["relative_risk_level"] == "L2"


# --------------------------------------------------------------------------
# 参考分布不足 / 无区分度
# --------------------------------------------------------------------------

@pytest.mark.parametrize("scores", [[], [0.5]])
def test_short_reference_unavailable(scores):
    """R 为空或少于 2 条 -> unavailable 并给出原因。"""
    ref = reference("r", scores)
    rows = grade([prediction("P", 0.5)], ref, CFG)
    assert rows[0]["relative_risk_level"] == "unavailable"
    assert rows[0]["grade_status"] == "unavailable"
    assert rows[0]["risk_percentile"] is None
    assert "参考分布不足" in rows[0]["unavailable_reason"]


def test_constant_reference_unavailable():
    """所有参考分数相同 -> 无区分度 -> unavailable。"""
    ref = reference("r", [0.3, 0.3, 0.3])
    rows = grade([prediction("P", 0.3)], ref, CFG)
    assert rows[0]["relative_risk_level"] == "unavailable"
    assert "无区分度" in rows[0]["unavailable_reason"]


# --------------------------------------------------------------------------
# 无效预测
# --------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [None, float("nan"), float("inf"), -0.1, 1.5, "0.3", True])
def test_invalid_prediction_unavailable(bad):
    """缺失/非有限/越界/非数值的 p 不参与分级，该管段 unavailable。"""
    ref = reference("r", [0.1, 0.2, 0.3])
    rows = grade([prediction("P", bad)], ref, CFG)
    assert rows[0]["relative_risk_level"] == "unavailable"
    assert rows[0]["risk_percentile"] is None


def test_invalid_prediction_does_not_trigger_r01_r03():
    """无效预测不得因代填分数触发 R01/R03（§8.5 T09）。"""
    from src.decision import advise_predictive, load_advice_rules, load_scenario_config
    from src.decision.priority import prioritize

    rules = load_advice_rules()
    scenario = load_scenario_config()
    # 属性本身会触发 R01/R03（承插 + 学校），但 p 无效 -> 等级 unavailable
    views = [view("P1", joint="承插接口", facility="学校", material="铸铁管",
                  diameter_mm=600, road_type="主干道", opsst="正常", pipeage=30)]
    preds = [prediction("P1", None, ["prediction_invalid"])]
    ref = reference("r", [0.1, 0.2, 0.3])
    grades = grade(preds, ref, CFG)
    priority = prioritize(views, preds, grades, scenario, {"objective": "V", "k": 10})
    advice = advise_predictive(views, preds, grades, priority, rules)
    rule_ids = {a["rule_id"] for a in advice}
    assert "R01" not in rule_ids
    assert "R03" not in rule_ids
    assert "D01" in rule_ids


# --------------------------------------------------------------------------
# 分级不随筛选/预算改变
# --------------------------------------------------------------------------

def test_grade_invariant_under_filter_and_budget():
    """分级只依赖 R，不随筛选或预算改变（§5.3.1）。"""
    preds = fixture("predictions.json")
    ref = fixture("reference_bundle.json")
    baseline = grade(preds, ref, CFG)

    # 子集（模拟筛选后只提交部分候选）
    subset = [r for r in preds if r["pipe_id"] != "SYN-F01_normal"]
    after = grade(subset, ref, CFG)

    baseline_map = {r["pipe_id"]: r for r in baseline}
    for row in after:
        original = baseline_map[row["pipe_id"]]
        assert row["risk_percentile"] == original["risk_percentile"]
        assert row["relative_risk_level"] == original["relative_risk_level"]


def test_grade_does_not_mutate_inputs():
    preds = fixture("predictions.json")
    ref = fixture("reference_bundle.json")
    assert_unchanged(None, grade, preds, ref, CFG)


# --------------------------------------------------------------------------
# 契约与失败处理
# --------------------------------------------------------------------------

def test_grade_output_validates_as_decision_product():
    """grade 输出能投影成契约 `decision` 产物并通过校验。"""
    preds = fixture("predictions.json")
    ref = fixture("reference_bundle.json")
    rows = grade(preds, ref, CFG)
    decision_rows = [{
        "schema_version": r["schema_version"],
        "data_version": r["data_version"],
        "run_id": r["run_id"],
        "prediction_mode": r["prediction_mode"],
        "data_kind": r["data_kind"],
        "pipe_id": r["pipe_id"],
        "risk_percentile": r["risk_percentile"],
        "relative_risk_level": r["relative_risk_level"],
        "consequence_proxy": 0.5,
        "priority_value": None,
        "scenario_id": "consequence_scenario_v1",
        "evidence_refs": r["evidence_refs"],
    } for r in rows]
    validate_package("decision", decision_rows)


def test_grade_rejects_version_mismatch():
    """参考分布的 grade_config_version 不兼容 -> 拒绝接入。"""
    ref = reference("r", [0.1, 0.2])
    ref[0]["grade_config_version"] = "relative_grade_v0"
    with pytest.raises(DecisionError, match="不兼容"):
        grade([prediction("P", 0.1)], ref, CFG)


def test_grade_rejects_nonfinite_reference_scores():
    ref = reference("r", [0.1, float("nan")])
    with pytest.raises(DecisionError, match="非有限"):
        grade([prediction("P", 0.1)], ref, CFG)


def test_grade_rejects_out_of_range_reference_scores():
    ref = reference("r", [0.1, 1.5])
    with pytest.raises(DecisionError, match="超出"):
        grade([prediction("P", 0.1)], ref, CFG)


def test_grade_rejects_multi_row_reference_bundle():
    """参考分布必须恰含一行；不从子集重建。"""
    ref = reference("r", [0.1, 0.2]) + reference("r2", [0.3, 0.4])
    with pytest.raises(DecisionError, match="恰含 1 行"):
        grade([prediction("P", 0.1)], ref, CFG)


def test_grade_rejects_duplicate_pipe_id():
    preds = [prediction("P", 0.1), prediction("P", 0.2)]
    with pytest.raises(DecisionError, match="重复"):
        grade(preds, reference("r", [0.1, 0.2]), CFG)


def test_grade_config_version_is_frozen():
    assert CFG["version"] == GRADE_CONFIG_VERSION
    codes = [level["code"] for level in CFG["levels"]]
    assert codes == ["L1", "L2", "L3", "L4"]
    names = [level["display_name"] for level in CFG["levels"]]
    assert names == ["相对风险较低", "相对风险中等", "相对风险较高", "相对风险最高"]