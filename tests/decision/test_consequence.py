"""后果代理测试（§8.1）。

验收：最低档后果仍大于零；固定后果时 p 增大不降低 V；固定 p 时任何代理档位增大
不降低 V；未知设施不默认零后果；后果配置变化不改变 p 和模型等级。
人工边界例：默认下限下 p=0.8,S=0 得 V=0.16；p=0.01,S=1 得 V=0.01。
"""

import pytest

from src.contracts import SCENARIO_CONFIG_VERSION
from src.decision import load_scenario_config
from src.decision.consequence import (
    compute_consequence,
    compute_consequences,
    validate_scenario,
)
from src.decision.errors import DecisionError

from .helpers import deep, view, view_codes

SCENARIO = load_scenario_config()


def consequence(**attrs):
    return compute_consequence(view("P", **attrs), SCENARIO)


# --------------------------------------------------------------------------
# 公式与正值下限
# --------------------------------------------------------------------------

def test_c_min_lower_bound_strictly_positive():
    """最低档（S=0）后果仍严格为正，等于 c_min。"""
    record = consequence(diameter_mm=300, road_type="支路", facility="绿地")
    assert record["consequence_score"] == pytest.approx(0.0)
    assert record["consequence_proxy"] == pytest.approx(0.2)
    assert record["consequence_proxy"] > 0


def test_c_i_bounds():
    """c_min <= C_i <= 1。"""
    highest = consequence(diameter_mm=1600, road_type="主干道", facility="医院")
    assert highest["consequence_proxy"] == pytest.approx(1.0)
    lowest = consequence(diameter_mm=300, road_type="支路", facility="绿地")
    assert 0.2 <= lowest["consequence_proxy"] <= 1.0


def test_formula_c_min_plus_score():
    """C = c_min + (1-c_min)*S。"""
    record = consequence(diameter_mm=600, road_type="次干道", facility="居民区")
    # d=1/3, r=0.5, f=0.75 -> S=(1/3+0.5+0.75)/3
    expected_s = (1 / 3 + 0.5 + 0.75) / 3
    assert record["consequence_score"] == pytest.approx(expected_s)
    assert record["consequence_proxy"] == pytest.approx(0.2 + 0.8 * expected_s)


def test_weights_sum_to_one_and_reject_otherwise():
    bad = deep(SCENARIO)
    bad["weights"]["w_D"] = {"num": 1, "den": 2}  # 和 = 1/2+1/3+1/3 != 1
    with pytest.raises(DecisionError, match="权重和"):
        validate_scenario(bad)


@pytest.mark.parametrize("bad_c_min", [0, -0.1, 1.5])
def test_reject_invalid_c_min(bad_c_min):
    bad = deep(SCENARIO)
    bad["c_min"] = bad_c_min
    with pytest.raises(DecisionError):
        validate_scenario(bad)


def test_reject_mapping_value_out_of_unit_interval():
    bad = deep(SCENARIO)
    bad["road_map"]["主干道"] = {"num": 3, "den": 2}
    with pytest.raises(DecisionError, match="超出"):
        validate_scenario(bad)


# --------------------------------------------------------------------------
# 档位映射
# --------------------------------------------------------------------------

@pytest.mark.parametrize("diameter,expected", [
    (300, 0.0), (499, 0.0),
    (500, 1 / 3), (799, 1 / 3),
    (800, 2 / 3), (999, 2 / 3),
    (1000, 1.0), (1600, 1.0),
])
def test_diameter_map_bands(diameter, expected):
    record = consequence(diameter_mm=diameter, road_type="支路", facility="绿地")
    assert record["components"]["diameter"] == pytest.approx(expected)


@pytest.mark.parametrize("road,expected", [("支路", 0.0), ("次干道", 0.5), ("主干道", 1.0)])
def test_road_map(road, expected):
    record = consequence(diameter_mm=300, road_type=road, facility="绿地")
    assert record["components"]["road"] == pytest.approx(expected)


@pytest.mark.parametrize("facility,expected", [
    ("绿地", 0.0),
    ("商铺", 0.5), ("企事业单位", 0.5),
    ("居民区", 0.75), ("住宅小区", 0.75), ("商业中心", 0.75),
    ("学校", 1.0), ("医院", 1.0), ("交通枢纽", 1.0), ("政府机关", 1.0),
])
def test_facility_map(facility, expected):
    record = consequence(diameter_mm=300, road_type="支路", facility=facility)
    assert record["components"]["facility"] == pytest.approx(expected)


# --------------------------------------------------------------------------
# 未知/缺失/越界
# --------------------------------------------------------------------------

def test_unknown_facility_not_zero():
    """未知设施不默认零后果，走 0.5 代理并标记（§8.1、§8.5 T06）。"""
    record = consequence(diameter_mm=300, road_type="支路", facility=None)
    assert record["components"]["facility"] == pytest.approx(0.5)
    assert record["consequence_proxy"] > 0.2
    assert record["proxy_imputed"] is True
    assert "facility" in record["imputed_fields"]


def test_unknown_facility_provides_sensitivity_scenarios():
    """未知设施须同时给出 unknown-as-0 与 unknown-as-1 上下情景。"""
    record = consequence(diameter_mm=300, road_type="支路", facility=None)
    sens = record["sensitivity"]
    assert sens["unknown_as_0"]["consequence_proxy"] == pytest.approx(0.2)
    assert sens["unknown_as_1"]["consequence_proxy"] > sens["unknown_as_0"]["consequence_proxy"]


def test_diameter_out_of_range_no_extrapolation():
    """管径越界不自动外推，走待核与情景代填路径并标记。"""
    record = consequence(diameter_mm=2000, road_type="主干道", facility="交通枢纽")
    assert record["component_status"]["diameter"] == "out_of_range"
    assert record["proxy_out_of_range"] is True
    assert record["pending_verification"] is True
    # 档位取默认代理 0.5，而不是外推为 1
    assert record["components"]["diameter"] == pytest.approx(0.5)


def test_out_of_range_does_not_set_proxy_imputed():
    """越界走待核路径，不等同于"未知类别代填"，不触发 D02。

    夹具 expected_rules.json 的 F08 预期触发为空，据此区分两条路径。
    """
    record = consequence(diameter_mm=2000, road_type="主干道", facility="交通枢纽")
    assert record["proxy_imputed"] is False
    assert record["imputed_fields"] == []


def test_fixture_unknown_facility_flags_proxy_imputed():
    """夹具 F02（FAC 缺失）必须标 proxy_imputed 且 C 不归零。"""
    import json
    from pathlib import Path
    rows = json.loads(
        (Path(__file__).resolve().parent.parent / "fixtures"
         / "standard_attributes.json").read_text(encoding="utf-8"))
    f02 = next(r for r in rows if r["pipe_id"] == "SYN-F02_unknown_facility")
    record = compute_consequence(f02, SCENARIO)
    assert record["proxy_imputed"] is True
    assert record["consequence_proxy"] > 0


# --------------------------------------------------------------------------
# 单调性
# --------------------------------------------------------------------------

def test_monotone_increasing_p_never_lowers_v():
    """固定后果时，p 增大不降低 V。"""
    record = consequence(diameter_mm=600, road_type="次干道", facility="居民区")
    c = record["consequence_proxy"]
    previous = None
    for p in [0.0, 0.01, 0.1, 0.5, 0.8, 1.0]:
        v = p * c
        if previous is not None:
            assert v >= previous
        previous = v


@pytest.mark.parametrize("dimension,small,large", [
    ("diameter", 300, 1600),
    ("road", "支路", "主干道"),
    ("facility", "绿地", "医院"),
])
def test_monotone_increasing_proxy_tier_never_lowers_v(dimension, small, large):
    """固定 p 时，任何代理档位增大不降低 V。"""
    base = {"diameter_mm": 300, "road_type": "支路", "facility": "绿地"}
    low = dict(base, **{_key(dimension): small})
    high = dict(base, **{_key(dimension): large})
    v_low = 0.5 * compute_consequence(view("P", **low), SCENARIO)["consequence_proxy"]
    v_high = 0.5 * compute_consequence(view("P", **high), SCENARIO)["consequence_proxy"]
    assert v_high >= v_low


def _key(dimension):
    return {"diameter": "diameter_mm", "road": "road_type",
            "facility": "facility"}[dimension]


# --------------------------------------------------------------------------
# 边界例与配置无关性
# --------------------------------------------------------------------------

def test_manual_boundary_examples():
    """§8.1 人工边界例：默认下限下 p=0.8,S=0 -> V=0.16；p=0.01,S=1 -> V=0.01。"""
    lowest = consequence(diameter_mm=300, road_type="支路", facility="绿地")
    assert lowest["consequence_score"] == pytest.approx(0.0)
    assert 0.8 * lowest["consequence_proxy"] == pytest.approx(0.16)

    highest = consequence(diameter_mm=1600, road_type="主干道", facility="医院")
    assert highest["consequence_score"] == pytest.approx(1.0)
    assert 0.01 * highest["consequence_proxy"] == pytest.approx(0.01)


def test_scenario_change_does_not_affect_p_or_grade():
    """后果配置变化不改变 p 和模型等级（§8.1）。"""
    from src.decision import grade, load_grade_config

    scenario_alt = deep(SCENARIO)
    scenario_alt["c_min"] = 0.4
    scenario_alt["weights"]["w_D"] = {"num": 1, "den": 2}
    scenario_alt["weights"]["w_F"] = {"num": 1, "den": 4}
    scenario_alt["weights"]["w_R"] = {"num": 1, "den": 4}

    record_a = compute_consequence(view("P", diameter_mm=600, road_type="次干道",
                                        facility="居民区"), SCENARIO)
    record_b = compute_consequence(view("P", diameter_mm=600, road_type="次干道",
                                        facility="居民区"), scenario_alt)
    assert record_a["consequence_proxy"] != record_b["consequence_proxy"]
    # 后果变化不触碰分级输入：同样的 predictions + reference 得到同样的等级
    from .helpers import prediction, reference
    preds = [prediction("P", 0.31)]
    ref = reference("r", [0.1, 0.2, 0.3])
    cfg = load_grade_config()
    assert grade(preds, ref, cfg) == grade(preds, ref, cfg)


def test_accepts_both_key_spellings():
    """夹具拼写与正式列码必须得到同一后果。"""
    lower = compute_consequence(
        view("P", diameter_mm=600, road_type="次干道", facility="居民区"), SCENARIO)
    codes = compute_consequence(
        view_codes("P", GJ=600, RDTYPE="次干道", FAC="居民区"), SCENARIO)
    assert lower["consequence_proxy"] == codes["consequence_proxy"]
    assert lower["consequence_score"] == codes["consequence_score"]


def test_does_not_mutate_inputs():
    row = view("P", diameter_mm=600, road_type="次干道", facility="居民区")
    scenario = deep(SCENARIO)
    before_row, before_scenario = deep(row), deep(scenario)
    compute_consequence(row, scenario)
    assert row == before_row
    assert scenario == before_scenario


def test_batch_returns_one_record_per_pipe():
    views = [
        view("A", diameter_mm=300, road_type="支路", facility="绿地"),
        view("B", diameter_mm=1600, road_type="主干道", facility="医院"),
    ]
    records = compute_consequences(views, SCENARIO)
    assert set(records) == {"A", "B"}
    assert records["B"]["consequence_proxy"] > records["A"]["consequence_proxy"]


def test_scenario_version_frozen():
    assert SCENARIO["version"] == SCENARIO_CONFIG_VERSION
    assert SCENARIO["c_min"] == 0.2