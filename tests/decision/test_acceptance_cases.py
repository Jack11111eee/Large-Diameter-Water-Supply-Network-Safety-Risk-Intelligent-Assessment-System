"""§8.5 验收案例 T01–T10。

T02/T03/T04/T07/T08/T10 用人工构造的合成案例（夹具未覆盖）。
T05/T06/T09 直接复用夹具 F03/F02/F07。

所有案例不读原始 XLSX，不导入 src/models/ 或 src/data/。
"""

import pytest

from src.decision import (
    advise_post_event,
    advise_predictive,
    grade,
    load_advice_rules,
    load_grade_config,
    load_scenario_config,
)
from src.decision.priority import prioritize

from .helpers import (
    event_view,
    fixture,
    grade_row,
    prediction,
    reference,
    view,
)

RULES = load_advice_rules()
SCENARIO = load_scenario_config()
CFG = load_grade_config()

# 覆盖 80/95/99 边界的参考集合
REF = reference("r", [i / 100.0 for i in range(1, 101)])


def pipeline(views, preds, reference_bundle=None):
    """grade -> prioritize -> advise_predictive 的固定链路。"""
    grades = grade(preds, reference_bundle or REF, CFG)
    priority = prioritize(views, preds, grades, SCENARIO,
                          {"objective": "V", "k": 100})
    advice = advise_predictive(views, preds, grades, priority, RULES)
    return grades, priority, advice


def triggers_for(advice, pipe_id):
    return sorted({a["rule_id"] for a in advice if a["pipe_id"] == pipe_id})


# --------------------------------------------------------------------------
# T01 基础核查
# --------------------------------------------------------------------------

def test_t01_basic_verification_only_r00():
    """L1、OPSST 正常、字段完整，无历史模式 -> 只有 R00；不得写"安全"。"""
    views = [view("T01", pipeage=12, material="球墨铸铁", joint="柔性接口",
                  diameter_mm=400, road_type="次干道", facility="居民区",
                  opsst="正常", press_mpa=0.32)]
    preds = [prediction("T01", 0.05)]
    _, _, advice = pipeline(views, preds)

    assert triggers_for(advice, "T01") == ["R00"]
    record = advice[0]
    assert record["advice_mode"] == "predictive"
    assert record["as_of"] is None
    # 禁止出现"安全"类断言
    text = record["suggested_action"] + "".join(record["prohibited_inferences"])
    assert "不会爆管" in text  # 禁用项显式列入 prohibited_inferences
    assert "无需维护" in text


# --------------------------------------------------------------------------
# T02 材料接口
# --------------------------------------------------------------------------

def test_t02_material_and_joint():
    """L3、铸铁管、承插接口、OPSST 正常 -> R01；材料与接口两项证据都保留。"""
    views = [view("T02", pipeage=40, material="铸铁管", joint="承插接口",
                  diameter_mm=600, road_type="次干道", facility="居民区",
                  opsst="正常", press_mpa=0.4)]
    # p=0.96 -> 96 百分位 -> L3
    preds = [prediction("T02", 0.96)]
    grades, _, advice = pipeline(views, preds)
    assert grades[0]["relative_risk_level"] == "L3"

    rule_ids = triggers_for(advice, "T02")
    assert rule_ids == ["R01"]

    record = next(a for a in advice if a["rule_id"] == "R01")
    assert record["trigger_values"]["material_match"] is True
    assert record["trigger_values"]["joint_match"] is True
    evidence = " ".join(record["evidence_refs"])
    assert "material" in evidence and "joint" in evidence
    # 同一 action_key 只保留一条（重复动作合并）
    assert len([a for a in advice if a["action_key"] == record["action_key"]]) == 1


def test_t02_merge_keeps_all_trigger_evidence():
    """重复动作按 action_key 合并，但保留全部触发依据（§8.4）。"""
    # 两条属性视图指向同一 action_key 不可能（pipe_id 不同），此处验证合并逻辑：
    # 同一管段若既因材料又因接口触发，仍只有一条 R01，两项证据都在。
    views = [view("T02b", pipeage=40, material="铸铁管", joint="承插接口",
                  diameter_mm=600, road_type="次干道", facility="居民区",
                  opsst="正常", press_mpa=0.4)]
    preds = [prediction("T02b", 0.96)]
    _, _, advice = pipeline(views, preds)
    r01 = [a for a in advice if a["rule_id"] == "R01"]
    assert len(r01) == 1
    assert len(r01[0]["evidence_refs"]) == 2


# --------------------------------------------------------------------------
# T03 运行状态
# --------------------------------------------------------------------------

def test_t03_operation_status_with_missing_press():
    """L1、OPSST 异常、PRESS 缺失 -> R02；明确快照与待核，不捏造压力。"""
    views = [view("T03", pipeage=20, material="球墨铸铁", joint="柔性接口",
                  diameter_mm=400, road_type="支路", facility="绿地",
                  opsst="异常", press_mpa=None)]
    preds = [prediction("T03", 0.05)]
    grades, _, advice = pipeline(views, preds)
    assert grades[0]["relative_risk_level"] == "L1"

    rule_ids = triggers_for(advice, "T03")
    assert rule_ids == ["R02"]

    record = next(a for a in advice if a["rule_id"] == "R02")
    assert record["trigger_values"]["opsst"] == "异常"
    assert record["trigger_values"]["opsst_is_snapshot"] is True
    assert record["trigger_values"]["press_mpa"] is None
    # 明确 PRESS 待核，不编造压力值
    assert any("PRESS" in p for p in record["preconditions_to_check"])
    assert "不自动降压" in record["prohibited_inferences"]
    assert "不能编造压力值" in " ".join(record["prohibited_inferences"])


# --------------------------------------------------------------------------
# T04 重要设施
# --------------------------------------------------------------------------

def test_t04_critical_facility():
    """L4、医院 -> R03，先核实供水关联；不输出影响人数。"""
    views = [view("T04", pipeage=50, material="球墨铸铁", joint="柔性接口",
                  diameter_mm=800, road_type="主干道", facility="医院",
                  opsst="正常", press_mpa=0.5)]
    preds = [prediction("T04", 0.995)]  # -> L4
    grades, _, advice = pipeline(views, preds)
    assert grades[0]["relative_risk_level"] == "L4"

    rule_ids = triggers_for(advice, "T04")
    assert rule_ids == ["R03"]

    record = next(a for a in advice if a["rule_id"] == "R03")
    assert record["trigger_values"]["facility"] == "医院"
    assert "真实供水关联" in record["suggested_action"]
    assert any("人数" in p or "户数" in p for p in record["prohibited_inferences"])


# --------------------------------------------------------------------------
# T05 坐标冲突（复用夹具 F03）
# --------------------------------------------------------------------------

def test_t05_coordinate_conflict_from_fixture():
    """源节点坐标冲突、属性评分有效 -> D01；空间结果待核，属性评分不被覆盖。"""
    views = fixture("standard_attributes.json")
    preds = fixture("predictions.json")
    ref = fixture("reference_bundle.json")
    grades, _, advice = pipeline(views, preds, ref)

    pipe_id = "SYN-F03_coord_conflict"
    assert triggers_for(advice, pipe_id) == ["D01"]
    # 属性评分有效：等级仍是正常分级结果，没有被静默覆盖为 unavailable
    row = next(g for g in grades if g["pipe_id"] == pipe_id)
    assert row["grade_status"] == "graded"
    record = next(a for a in advice if a["pipe_id"] == pipe_id)
    assert "坐标" in record["suggested_action"] and "待核" in record["suggested_action"]
    assert "不把坐标冲突判断为实际断管" in record["prohibited_inferences"]


# --------------------------------------------------------------------------
# T06 未知设施（复用夹具 F02）
# --------------------------------------------------------------------------

def test_t06_unknown_facility_from_fixture():
    """FAC 未知 -> D02，展示代填标记及上下情景；C 不归零。"""
    views = fixture("standard_attributes.json")
    preds = fixture("predictions.json")
    ref = fixture("reference_bundle.json")
    _, priority, advice = pipeline(views, preds, ref)

    pipe_id = "SYN-F02_unknown_facility"
    assert triggers_for(advice, pipe_id) == ["D02"]
    record = next(a for a in advice if a["pipe_id"] == pipe_id)
    assert "facility" in record["trigger_values"]["imputed_fields"]
    assert set(record["trigger_values"]["sensitivity"]) >= {"unknown_as_0", "unknown_as_1"}
    # C 不归零
    assert priority["records"][pipe_id]["consequence_proxy"] > 0


# --------------------------------------------------------------------------
# T07 历史隔离
# --------------------------------------------------------------------------

def test_t07_history_isolation():
    """同一管段截止日前 2 条事件，开关事后模式：仅开启后出现 O01，其余逐字段不变。"""
    views = [view("T07", pipeage=45, material="铸铁管", joint="承插接口",
                  diameter_mm=600, road_type="主干道", facility="医院",
                  opsst="正常", press_mpa=0.4)]
    preds = [prediction("T07", 0.5)]
    grades, priority, predictive_advice = pipeline(views, preds)

    # 关闭事后模式：无 O01
    assert "O01" not in triggers_for(predictive_advice, "T07")

    events = event_view({"T07": 2})
    post = advise_post_event(views, events, "2024-12-31", RULES)
    o01 = [a for a in post if a["rule_id"] == "O01"]
    assert len(o01) == 1
    assert o01[0]["advice_mode"] == "post_event_review"
    assert o01[0]["trigger_values"]["pre_as_of_event_count"] == 2
    assert o01[0]["trigger_values"]["repeat_event_check"] is True
    assert o01[0]["as_of"] == "2024-12-31"

    # 预测清单逐字段不变
    grades_after = grade(preds, REF, CFG)
    assert grades_after == grades
    priority_after = prioritize(views, preds, grades_after, SCENARIO,
                                {"objective": "V", "k": 100})
    assert priority_after == priority
    advice_after = advise_predictive(views, preds, grades_after, priority_after, RULES)
    assert advice_after == predictive_advice


def test_t07_single_event_no_repeat_check():
    """≥1 触发 O01；≥2 才补充重复事件核查。"""
    views = [view("T07b", pipeage=45, material="钢管", joint="焊接接口",
                  diameter_mm=600, road_type="支路", facility="绿地",
                  opsst="正常", press_mpa=0.4)]
    post = advise_post_event(views, event_view({"T07b": 1}), "2024-12-31", RULES)
    o01 = [a for a in post if a["rule_id"] == "O01"]
    assert len(o01) == 1
    assert o01[0]["trigger_values"]["pre_as_of_event_count"] == 1
    assert o01[0]["trigger_values"]["repeat_event_check"] is False


# --------------------------------------------------------------------------
# T08 截止日期
# --------------------------------------------------------------------------

def test_t08_events_after_as_of_are_invisible():
    """只有截止日之后的事件 -> O01 不触发，不暴露未来事件内容。

    事件视图的 as_of 即请求的截止日（核心读取接口已按截止日过滤），
    因此"截止日之后"表现为截止日前事件数为 0。
    """
    views = [view("T08", pipeage=45, material="钢管", joint="焊接接口",
                  diameter_mm=600, road_type="支路", facility="绿地",
                  opsst="正常", press_mpa=0.4)]
    post = advise_post_event(views, event_view({"T08": 0}), "2024-12-31", RULES)
    assert triggers_for(post, "T08") == ["R00"]
    assert "O01" not in triggers_for(post, "T08")


def test_t08_event_list_filters_by_date():
    """事件视图给出明细时，按日期过滤；未来事件不可见。"""
    views = [view("T08b", pipeage=45, material="钢管", joint="焊接接口",
                  diameter_mm=600, road_type="支路", facility="绿地",
                  opsst="正常", press_mpa=0.4)]
    events = [{
        "schema_version": "1.0.0", "data_version": "d", "run_id": "r",
        "prediction_mode": "oof_replay", "data_kind": "synthetic_fixture",
        "pipe_id": "T08b", "event_count": 2, "as_of": "2024-12-31",
        "events": [{"date": "2024-03-01"}, {"date": "2025-02-01"}],
    }]
    post = advise_post_event(views, events, "2024-12-31", RULES)
    o01 = [a for a in post if a["rule_id"] == "O01"]
    assert len(o01) == 1
    assert o01[0]["trigger_values"]["pre_as_of_event_count"] == 1


def test_t08_as_of_mismatch_rejected():
    """事件视图 as_of 与请求不一致 -> 拒绝接入。"""
    from src.decision.errors import DecisionError
    views = [view("T08c", pipeage=45, material="钢管", joint="焊接接口",
                  diameter_mm=600, road_type="支路", facility="绿地",
                  opsst="正常", press_mpa=0.4)]
    with pytest.raises(DecisionError, match="不一致"):
        advise_post_event(views, event_view({"T08c": 1}), "2024-06-30", RULES)


# --------------------------------------------------------------------------
# T09 无效预测（复用夹具 F07）
# --------------------------------------------------------------------------

def test_t09_invalid_prediction_from_fixture():
    """p 缺失 -> D01，相对等级 unavailable；不得触发 R01/R03。"""
    views = fixture("standard_attributes.json")
    preds = fixture("predictions.json")
    ref = fixture("reference_bundle.json")
    grades, _, advice = pipeline(views, preds, ref)

    pipe_id = "SYN-F07_invalid_prediction"
    row = next(g for g in grades if g["pipe_id"] == pipe_id)
    assert row["relative_risk_level"] == "unavailable"
    assert row["risk_percentile"] is None

    rule_ids = triggers_for(advice, pipe_id)
    assert rule_ids == ["D01"]
    assert "R01" not in rule_ids and "R03" not in rule_ids


def test_t09_invalid_reference_distribution():
    """R 无效（全同分）时全部 unavailable，且不触发 R01/R03。"""
    views = [view("T09b", pipeage=60, material="铸铁管", joint="承插接口",
                  diameter_mm=600, road_type="主干道", facility="学校",
                  opsst="正常", press_mpa=0.4)]
    preds = [prediction("T09b", 0.9)]
    grades = grade(preds, reference("r", [0.5, 0.5]), CFG)
    assert grades[0]["relative_risk_level"] == "unavailable"
    priority = prioritize(views, preds, grades, SCENARIO, {"objective": "V", "k": 10})
    advice = advise_predictive(views, preds, grades, priority, RULES)
    rule_ids = triggers_for(advice, "T09b")
    assert "R01" not in rule_ids and "R03" not in rule_ids
    assert "D01" in rule_ids


# --------------------------------------------------------------------------
# T10 多条件
# --------------------------------------------------------------------------

def test_t10_multiple_conditions_coexist():
    """L4、承插接口、OPSST 偏高、学校 -> R01/R02/R03 共存，依据完整，无指令性动作。"""
    views = [view("T10", pipeage=55, material="铸铁管", joint="承插接口",
                  diameter_mm=1000, road_type="主干道", facility="学校",
                  opsst="偏高", press_mpa=0.6)]
    preds = [prediction("T10", 0.995)]  # -> L4
    grades, _, advice = pipeline(views, preds)
    assert grades[0]["relative_risk_level"] == "L4"

    rule_ids = triggers_for(advice, "T10")
    assert set(rule_ids) == {"R01", "R02", "R03"}

    # 规则按固定顺序出现：R01, R02, R03
    ordered = [a["rule_id"] for a in advice if a["pipe_id"] == "T10"]
    assert ordered == ["R01", "R02", "R03"]

    # 每条都有完整依据与禁止推断，且不含自动调压/换管/关阀指令
    for record in advice:
        assert record["evidence_refs"]
        assert record["prohibited_inferences"]
        assert record["preconditions_to_check"]
    text = " ".join(
        a["suggested_action"] + " ".join(a["prohibited_inferences"])
        for a in advice)
    assert "不自动降压" in text
    assert "自动建议更换" in text or "更换后风险" in text


def test_t10_rule_execution_order_is_fixed():
    """规则执行顺序固定为 D01、R01、R02、R03、O01、D02、R00（§8.4）。"""
    assert RULES["execution_order"] == ["D01", "R01", "R02", "R03", "O01", "D02", "R00"]


# --------------------------------------------------------------------------
# 夹具 expected_rules.json 的触发一致性
# --------------------------------------------------------------------------

def test_fixture_expected_rules_match_engine():
    """夹具 expected_rules.json 的预期触发必须被规则引擎覆盖。

    注意：expected_rules.json 只编码了 D01/D02 子集（其生成函数仅看质量标记），
    因此断言"预期触发是实际触发的子集"，并额外验证本引擎按 §8.4 正确补出的规则。
    """
    views = fixture("standard_attributes.json")
    preds = fixture("predictions.json")
    ref = fixture("reference_bundle.json")
    expected = fixture("expected_rules.json")["cases"]

    _, _, advice = pipeline(views, preds, ref)
    for row in views:
        pipe_id = row["pipe_id"]
        key = pipe_id.replace("SYN-", "")
        actual = set(triggers_for(advice, pipe_id))
        wanted = set(expected[key]["expected_triggers"])
        assert wanted <= actual, (
            f"{key}: 预期触发 {sorted(wanted)} 未被引擎覆盖，实际 {sorted(actual)}")

    # 引擎按 §8.4 补出的规则：F04/F05/F06 的 OPSST 为偏高/异常 -> R02
    assert triggers_for(advice, "SYN-F04_tie_probability") == ["R02"]
    assert triggers_for(advice, "SYN-F05_tie_probability_b") == ["R02"]
    assert triggers_for(advice, "SYN-F06_missing_explanation") == ["R02"]
    # F08 管径越界走待核路径，不触发 D02（与夹具预期一致）
    assert triggers_for(advice, "SYN-F08_diameter_out_of_range") == ["R00"]


# --------------------------------------------------------------------------
# 事后模式拒绝
# --------------------------------------------------------------------------

@pytest.mark.parametrize("bad_as_of", [None, "", "2024/12/31", "not-a-date", 20241231])
def test_post_event_requires_legal_as_of(bad_as_of):
    """未给合法截止日则拒绝开启事后模式（§8.5）。"""
    from src.decision.errors import DecisionError
    views = [view("P", pipeage=10, material="钢管", joint="焊接接口",
                  diameter_mm=400, road_type="支路", facility="绿地",
                  opsst="正常", press_mpa=0.3)]
    with pytest.raises(DecisionError):
        advise_post_event(views, event_view({"P": 1}), bad_as_of, RULES)


def test_default_as_of_constant_is_2024_12_31():
    """§8.5 默认截止日为 2024-12-31（显式常量，不隐式取系统当前日）。"""
    from src.decision.advice import DEFAULT_AS_OF
    assert DEFAULT_AS_OF == "2024-12-31"


# --------------------------------------------------------------------------
# 事后模式隔离：逐字段不变
# --------------------------------------------------------------------------

def test_post_event_mode_isolation_byte_identical():
    """开关事后模式只改变事后建议清单；p/等级/C/V/预测清单逐字段不变。"""
    views = fixture("standard_attributes.json")
    preds = fixture("predictions.json")
    ref = fixture("reference_bundle.json")

    grades_before = grade(preds, ref, CFG)
    priority_before = prioritize(views, preds, grades_before, SCENARIO,
                                 {"objective": "V", "k": 100})
    advice_before = advise_predictive(views, preds, grades_before,
                                      priority_before, RULES)

    # 开启事后模式
    events = event_view({row["pipe_id"]: 2 for row in views})
    post = advise_post_event(views, events, "2024-12-31", RULES)
    assert post  # 事后清单非空

    # 重算预测侧：逐字段相同
    grades_after = grade(preds, ref, CFG)
    priority_after = prioritize(views, preds, grades_after, SCENARIO,
                                {"objective": "V", "k": 100})
    advice_after = advise_predictive(views, preds, grades_after,
                                     priority_after, RULES)

    assert grades_after == grades_before
    assert priority_after == priority_before
    assert advice_after == advice_before

    # 事后清单与预测清单必须分开，不能合并
    assert all(a["advice_mode"] == "post_event_review" for a in post)
    assert all(a["advice_mode"] == "predictive" for a in advice_after)

# --------------------------------------------------------------------------
# 回归：全表质量标记不得让 D01 命中全部管段
# --------------------------------------------------------------------------

def test_table_wide_flags_do_not_trigger_d01_for_every_pipe():
    """semantics_unverified 是全表标记（ZDMS 语义未核，真实数据 7288 条全带），
    不是逐管段冲突。若按逐管段冲突处理，D01 会命中 100% 管段，
    恰是 D01 自身 prohibited_inferences 禁止的“为每根管段制造独立事故告警”。
    """
    views = [
        view("TW1", quality_flags=["semantics_unverified", "not_pre_event",
                                   "traffic_proxy"],
             pipeage=12, material="球墨铸铁", joint="柔性接口",
             diameter_mm=400, road_type="次干道", facility="居民区",
             opsst="正常", press_mpa=0.32),
        view("TW2", quality_flags=["semantics_unverified", "not_pre_event"],
             pipeage=20, material="球墨铸铁", joint="柔性接口",
             diameter_mm=400, road_type="次干道", facility="居民区",
             opsst="正常", press_mpa=0.32),
    ]
    preds = [prediction("TW1", 0.05), prediction("TW2", 0.05)]
    _, _, advice = pipeline(views, preds)

    # 字段完整、预测有效、等级可用、无逐管段冲突 -> 只应触发 R00
    assert triggers_for(advice, "TW1") == ["R00"]
    assert triggers_for(advice, "TW2") == ["R00"]


def test_per_pipe_coordinate_conflict_still_triggers_d01():
    """逐管段的 coordinate_conflict 仍必须触发 D01（§8.4）。"""
    views = [view("CC1", quality_flags=["coordinate_conflict"],
                  pipeage=38, material="钢管", joint="焊接接口",
                  diameter_mm=800, road_type="次干道", facility="企事业单位",
                  opsst="正常", press_mpa=0.55)]
    preds = [prediction("CC1", 0.071)]
    _, _, advice = pipeline(views, preds)
    assert "D01" in triggers_for(advice, "CC1")
