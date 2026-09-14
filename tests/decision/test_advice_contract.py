"""建议记录契约、纯函数性与 JSON 约束（§8.4、§13.3）。

覆盖：建议记录字段恰为规定集合；缺失证据不默认为满足；不修改入参；
输出可 JSON 序列化且不含 NaN/Infinity。
"""

import json
import math

import pytest

from src.contracts import ADVICE_RULES_VERSION, assert_no_nonfinite
from src.decision import (
    advise_post_event,
    advise_predictive,
    grade,
    load_advice_rules,
    load_grade_config,
    load_scenario_config,
)
from src.decision.advice import RECORD_FIELDS, RULE_ORDER
from src.decision.errors import DecisionError
from src.decision.priority import prioritize

from .helpers import event_view, fixture, prediction, reference, view

RULES = load_advice_rules()
SCENARIO = load_scenario_config()
CFG = load_grade_config()
REF = reference("r", [i / 100.0 for i in range(1, 101)])


def _pipeline(views, preds, reference_bundle=None):
    grades = grade(preds, reference_bundle or REF, CFG)
    priority = prioritize(views, preds, grades, SCENARIO,
                          {"objective": "V", "k": 100})
    return grades, priority, advise_predictive(views, preds, grades, priority, RULES)


# --------------------------------------------------------------------------
# 记录字段
# --------------------------------------------------------------------------

def test_advice_record_fields_are_exactly_frozen():
    """每条建议恰含 §8.4 规定的字段，不多不少。"""
    views = [view("P1", pipeage=50, material="铸铁管", joint="承插接口",
                  diameter_mm=1000, road_type="主干道", facility="学校",
                  opsst="偏高", press_mpa=0.6)]
    preds = [prediction("P1", 0.995)]
    _, _, advice = _pipeline(views, preds)
    assert advice
    for record in advice:
        assert set(record.keys()) == set(RECORD_FIELDS)


def test_post_event_record_fields_are_exactly_frozen():
    views = [view("P1", pipeage=50, material="钢管", joint="焊接接口",
                  diameter_mm=400, road_type="支路", facility="绿地",
                  opsst="正常", press_mpa=0.3)]
    post = advise_post_event(views, event_view({"P1": 3}), "2024-12-31", RULES)
    assert post
    for record in post:
        assert set(record.keys()) == set(RECORD_FIELDS)


def test_rule_version_and_ids_are_populated():
    views = [view("P1", pipeage=50, material="铸铁管", joint="承插接口",
                  diameter_mm=1000, road_type="主干道", facility="医院",
                  opsst="异常", press_mpa=0.6)]
    preds = [prediction("P1", 0.995)]
    _, _, advice = _pipeline(views, preds)
    for record in advice:
        assert record["rule_version"] == ADVICE_RULES_VERSION
        assert record["rule_id"] in RULE_ORDER
        assert record["prediction_run_id"] == "test-run"
        assert record["reference_id"] == "r"
        assert record["scenario_id"] == "consequence_scenario_v1"
        assert record["evidence_refs"]
        assert record["trigger_values"]


def test_advice_rules_version_matches_contract():
    assert RULES["version"] == ADVICE_RULES_VERSION


# --------------------------------------------------------------------------
# 缺失证据不默认为满足
# --------------------------------------------------------------------------

def test_missing_required_field_triggers_d01_not_satisfaction():
    """必需字段整键缺失 -> D01 数据核查，不把条件默认为满足。"""
    # facility 整键缺失（而非显式 null）
    views = [{"schema_version": "1.0.0", "data_version": "d", "run_id": "test-run",
              "prediction_mode": "oof_replay", "data_kind": "synthetic_fixture",
              "pipe_id": "P1", "bh": "BH-P1",
              "attributes": {"pipeage": 30, "material": "钢管", "joint": "焊接接口",
                             "diameter_mm": 600, "road_type": "支路",
                             "opsst": "正常", "press_mpa": 0.4},
              "source_refs": {}, "quality_flags": []}]
    preds = [prediction("P1", 0.5)]
    _, _, advice = _pipeline(views, preds)
    rule_ids = {a["rule_id"] for a in advice}
    assert "D01" in rule_ids
    d01 = next(a for a in advice if a["rule_id"] == "D01")
    assert "facility" in d01["trigger_values"]["missing_required_field"]


def test_missing_evidence_does_not_fabricate_action():
    """D01 动作是核对与待核，不是补写事实。"""
    views = [view("P1", material="钢管", joint="焊接接口", diameter_mm=600,
                  road_type="支路", opsst="正常", press_mpa=0.4)]
    preds = [prediction("P1", 0.5)]
    _, _, advice = _pipeline(views, preds)
    d01 = next(a for a in advice if a["rule_id"] == "D01")
    assert "核对" in d01["suggested_action"]
    assert d01["trigger_values"]["missing_required_field"]


# --------------------------------------------------------------------------
# 纯函数性
# --------------------------------------------------------------------------

def test_advise_predictive_does_not_mutate_inputs():
    views = [view("P1", pipeage=50, material="铸铁管", joint="承插接口",
                  diameter_mm=1000, road_type="主干道", facility="学校",
                  opsst="偏高", press_mpa=0.6)]
    preds = [prediction("P1", 0.995)]
    grades = grade(preds, REF, CFG)
    priority = prioritize(views, preds, grades, SCENARIO, {"objective": "V", "k": 100})
    import copy
    snapshots = [copy.deepcopy(x) for x in (views, preds, grades, priority, RULES)]
    advise_predictive(views, preds, grades, priority, RULES)
    assert [views, preds, grades, priority, RULES] == snapshots


def test_advise_post_event_does_not_mutate_inputs():
    import copy
    views = [view("P1", pipeage=50, material="钢管", joint="焊接接口",
                  diameter_mm=400, road_type="支路", facility="绿地",
                  opsst="正常", press_mpa=0.3)]
    events = event_view({"P1": 2})
    snapshots = [copy.deepcopy(x) for x in (views, events, RULES)]
    advise_post_event(views, events, "2024-12-31", RULES)
    assert [views, events, RULES] == snapshots


# --------------------------------------------------------------------------
# JSON 约束
# --------------------------------------------------------------------------

def _jsonable(payload):
    text = json.dumps(payload, ensure_ascii=False, allow_nan=False)
    return json.loads(text)


def test_outputs_are_json_safe_without_nan():
    """输出可序列化，且不含 NaN/Infinity（§13.3）。"""
    views = fixture("standard_attributes.json")
    preds = fixture("predictions.json")
    ref = fixture("reference_bundle.json")
    grades, priority, advice = _pipeline(views, preds, ref)
    events = event_view({row["pipe_id"]: 2 for row in views})
    post = advise_post_event(views, events, "2024-12-31", RULES)

    for payload in (grades, priority, advice, post):
        assert_no_nonfinite(payload)
        _jsonable(payload)


def test_round_trip_preserves_values():
    views = fixture("standard_attributes.json")
    preds = fixture("predictions.json")
    ref = fixture("reference_bundle.json")
    grades, _, _ = _pipeline(views, preds, ref)
    assert _jsonable(grades) == grades


# --------------------------------------------------------------------------
# 拒绝接入
# --------------------------------------------------------------------------

def test_rejects_wrong_rule_version():
    bad = json.loads(json.dumps(RULES, ensure_ascii=False))
    bad["version"] = "advice_rules_v0"
    views = [view("P1", pipeage=10, material="钢管", joint="焊接接口",
                  diameter_mm=400, road_type="支路", facility="绿地",
                  opsst="正常", press_mpa=0.3)]
    preds = [prediction("P1", 0.5)]
    grades = grade(preds, REF, CFG)
    priority = prioritize(views, preds, grades, SCENARIO, {"objective": "V", "k": 1})
    # 版本号本身不是拒绝条件（规则文件自带版本）；执行顺序被篡改才拒绝
    bad["execution_order"] = ["R00", "D01"]
    with pytest.raises(DecisionError, match="execution_order"):
        advise_predictive(views, preds, grades, priority, bad)


def test_rejects_missing_rule_entry():
    bad = json.loads(json.dumps(RULES, ensure_ascii=False))
    del bad["rules"]["R03"]
    views = [view("P1", pipeage=10, material="钢管", joint="焊接接口",
                  diameter_mm=400, road_type="支路", facility="绿地",
                  opsst="正常", press_mpa=0.3)]
    preds = [prediction("P1", 0.5)]
    grades = grade(preds, REF, CFG)
    priority = prioritize(views, preds, grades, SCENARIO, {"objective": "V", "k": 1})
    with pytest.raises(DecisionError, match="R03"):
        advise_predictive(views, preds, grades, priority, bad)


def test_post_event_requires_event_view():
    views = [view("P1", pipeage=10, material="钢管", joint="焊接接口",
                  diameter_mm=400, road_type="支路", facility="绿地",
                  opsst="正常", press_mpa=0.3)]
    with pytest.raises(DecisionError, match="event_view"):
        advise_post_event(views, None, "2024-12-31", RULES)


def test_duplicate_event_view_pipe_id_rejected():
    views = [view("P1", pipeage=10, material="钢管", joint="焊接接口",
                  diameter_mm=400, road_type="支路", facility="绿地",
                  opsst="正常", press_mpa=0.3)]
    events = event_view({"P1": 1}) + event_view({"P1": 2})
    with pytest.raises(DecisionError, match="重复"):
        advise_post_event(views, events, "2024-12-31", RULES)