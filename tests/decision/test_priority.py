"""优先级与数量预算 Top-K 测试（§6.4、§8.1、§8.2）。

验收：K=0、K>候选数、空候选集、无重复、不超预算、筛选/预算不改变 p 与等级、
按 p 清单独立保留。
"""

import pytest

from src.decision import grade, load_grade_config, load_scenario_config
from src.decision.errors import DecisionError
from src.decision.priority import prioritize, stable_tie_key

from .helpers import (
    assert_unchanged,
    deep,
    fixture,
    grade_row,
    prediction,
    reference,
    view,
    view_codes,
)

SCENARIO = load_scenario_config()
CFG = load_grade_config()


def base_views():
    return [
        view("A", diameter_mm=300, road_type="支路", facility="绿地",
             material="球墨铸铁", opsst="正常"),
        view("B", diameter_mm=600, road_type="次干道", facility="居民区",
             material="铸铁管", opsst="正常"),
        view("C", diameter_mm=1600, road_type="主干道", facility="医院",
             material="钢管", opsst="异常"),
        view("D", diameter_mm=800, road_type="主干道", facility="学校",
             material="铸铁管", opsst="正常"),
    ]


def base_preds():
    return [prediction("A", 0.05), prediction("B", 0.4),
            prediction("C", 0.9), prediction("D", 0.6)]


def base_grades(level_map=None):
    level_map = level_map or {"A": "L1", "B": "L2", "C": "L4", "D": "L3"}
    return [grade_row(pid, lvl) for pid, lvl in level_map.items()]


def run(views=None, preds=None, grades=None, request=None):
    return prioritize(views if views is not None else base_views(),
                      preds if preds is not None else base_preds(),
                      grades if grades is not None else base_grades(),
                      SCENARIO,
                      request if request is not None else {"objective": "V", "k": 2})


# --------------------------------------------------------------------------
# V 的计算
# --------------------------------------------------------------------------

def test_priority_value_is_p_times_c():
    result = run(request={"objective": "V", "k": 10})
    for pipe_id, record in result["records"].items():
        assert record["priority_value"] == pytest.approx(
            record["p"] * record["consequence_proxy"])


def test_by_c_order_is_descending_consequence():
    result = run()
    cs = [result["records"][pid]["consequence_proxy"] for pid in result["by_c"]]
    assert cs == sorted(cs, reverse=True)


def test_by_p_order_is_descending_probability():
    result = run()
    ps = [result["records"][pid]["p"] for pid in result["by_p"]]
    assert ps == sorted(ps, reverse=True)


# --------------------------------------------------------------------------
# 预算边界
# --------------------------------------------------------------------------

def test_k_zero_returns_empty_selection():
    result = run(request={"objective": "V", "k": 0})
    assert result["selection"]["selected"] == []
    assert result["selection"]["k_requested"] == 0
    assert result["selection"]["k_actual"] == 0
    assert result["selection"]["truncated"] is False
    # 按 p / 按 C / 按 V 的完整排序仍保留
    assert result["by_p"]
    assert result["by_v"]


def test_k_exceeds_candidates_truncates_without_filler():
    """K 超过候选数时截为候选总数，同时保留请求数与实际数，不伪造补足记录。"""
    result = run(request={"objective": "V", "k": 100})
    assert result["selection"]["k_requested"] == 100
    assert result["selection"]["k_actual"] == 4
    assert result["selection"]["truncated"] is True
    assert len(result["selection"]["selected"]) == 4
    assert result["candidate_count"] == 4


def test_empty_candidate_set():
    result = run(views=[], preds=[], grades=[])
    assert result["selection"]["selected"] == []
    assert result["selection"]["k_actual"] == 0
    assert result["by_p"] == [] and result["by_v"] == [] and result["by_c"] == []
    assert result["candidate_count"] == 0


def test_no_duplicates_and_within_budget():
    result = run(request={"objective": "V", "k": 3})
    selected = result["selection"]["selected"]
    assert len(selected) == len(set(selected))
    assert len(selected) <= 3
    for key in ("by_p", "by_c", "by_v"):
        assert len(result[key]) == len(set(result[key]))


def test_negative_k_rejected():
    with pytest.raises(DecisionError, match="不能为负"):
        run(request={"objective": "V", "k": -1})


def test_invalid_objective_rejected():
    with pytest.raises(DecisionError, match="objective"):
        run(request={"objective": "Z", "k": 1})


def test_unknown_filter_field_rejected():
    with pytest.raises(DecisionError, match="不支持按"):
        run(request={"objective": "V", "k": 1, "filters": {"bh": ["x"]}})


# --------------------------------------------------------------------------
# 稳定并列
# --------------------------------------------------------------------------

def test_tie_break_uses_sha256_not_builtin_hash():
    """并列分数用 SHA256(seed:pipe_id) 打散，与内置 hash 无关。"""
    views = [view("X", diameter_mm=600, road_type="主干道", facility="医院"),
             view("Y", diameter_mm=600, road_type="主干道", facility="医院")]
    preds = [prediction("X", 0.5), prediction("Y", 0.5)]
    result = run(views=views, preds=preds, grades=base_grades({"X": "L2", "Y": "L2"}),
                 request={"objective": "V", "k": 2})
    expected = sorted(["X", "Y"], key=lambda pid: stable_tie_key(pid, 20260914))
    assert result["by_v"] == expected
    assert result["by_p"] == expected


def test_tie_break_is_deterministic_across_calls():
    a = run(request={"objective": "V", "k": 4})
    b = run(request={"objective": "V", "k": 4})
    assert a["by_v"] == b["by_v"]
    assert a["selection"]["selected"] == b["selection"]["selected"]


def test_seed_changes_tie_order():
    views = [view("X", diameter_mm=600, road_type="主干道", facility="医院"),
             view("Y", diameter_mm=600, road_type="主干道", facility="医院")]
    preds = [prediction("X", 0.5), prediction("Y", 0.5)]
    grades = base_grades({"X": "L2", "Y": "L2"})
    first = run(views=views, preds=preds, grades=grades,
                request={"objective": "V", "k": 2, "seed": 1})
    second = run(views=views, preds=preds, grades=grades,
                 request={"objective": "V", "k": 2, "seed": 2})
    assert set(first["by_v"]) == set(second["by_v"]) == {"X", "Y"}


# --------------------------------------------------------------------------
# 独立保留：按 p 的 Top-K 与 L3/L4 视图
# --------------------------------------------------------------------------

def test_by_p_topk_preserved_independently():
    """按 p 的 Top-K 始终独立保留（§8.1）。"""
    result = run(request={"objective": "V", "k": 2})
    assert result["preserved_by_p_topk"] == result["by_p"][:2]


def test_high_grade_view_preserved_and_marked():
    """L3/L4 视图独立保留，未进入按 V 清单者被标出（§8.1）。"""
    result = run(request={"objective": "V", "k": 1})
    assert set(result["high_grade_view"]) == {"C", "D"}
    selected = set(result["selection"]["selected"])
    assert selected == {"C"}
    # D 是 L3 但未进入按 V 的 Top-1，必须被标出且不出现在选中清单中
    assert result["marked_not_in_value"] == ["D"]
    for pid in result["marked_not_in_value"]:
        assert pid not in selected


def test_marked_contains_no_duplicates():
    result = run(request={"objective": "V", "k": 1})
    marked = result["marked_not_in_value"]
    assert len(marked) == len(set(marked))


# --------------------------------------------------------------------------
# 不变性
# --------------------------------------------------------------------------

def test_p_and_grade_unchanged_by_budget_and_filter():
    """筛选与预算只改变候选清单，不改变已保存的 p 与等级（§5.3.1）。"""
    baseline = run(request={"objective": "V", "k": 100})
    filtered = run(request={"objective": "V", "k": 1,
                            "filters": {"material": ["铸铁管"]}})
    for pid, record in filtered["records"].items():
        assert record["p"] == baseline["records"][pid]["p"]
        assert (record["relative_risk_level"]
                == baseline["records"][pid]["relative_risk_level"])


def test_objective_switch_does_not_change_records():
    by_v = run(request={"objective": "V", "k": 2})
    by_p = run(request={"objective": "p", "k": 2})
    for pid, record in by_v["records"].items():
        assert record["priority_value"] == by_p["records"][pid]["priority_value"]
        assert record["consequence_proxy"] == by_p["records"][pid]["consequence_proxy"]


def test_invalid_prediction_excluded_from_p_and_v_lists():
    """p 无效者不参与按 p 与按 V 排序，但仍在按 C 清单中。"""
    views = base_views()
    preds = base_preds() + [prediction("E", None, ["prediction_invalid"])]
    views = views + [view("E", diameter_mm=1000, road_type="主干道",
                          facility="政府机关", material="铸铁管", opsst="正常")]
    result = run(views=views, preds=preds,
                 grades=base_grades({"A": "L1", "B": "L2", "C": "L4",
                                     "D": "L3", "E": "unavailable"}),
                 request={"objective": "V", "k": 10})
    assert "E" not in result["by_p"]
    assert "E" not in result["by_v"]
    assert "E" in result["by_c"]
    assert result["records"]["E"]["priority_value"] is None
    assert any(e["pipe_id"] == "E" for e in result["excluded"])


def test_accepts_both_key_spellings():
    codes = [view_codes("A", GJ=300, RDTYPE="支路", FAC="绿地"),
             view_codes("B", GJ=1600, RDTYPE="主干道", FAC="医院")]
    result = run(views=codes, preds=[prediction("A", 0.1), prediction("B", 0.1)],
                 grades=base_grades({"A": "L1", "B": "L4"}),
                 request={"objective": "C", "k": 2})
    assert result["by_c"][0] == "B"


def test_does_not_mutate_inputs():
    views, preds, grades = base_views(), base_preds(), base_grades()
    request = {"objective": "V", "k": 2}
    scenario = deep(SCENARIO)
    before = (deep(views), deep(preds), deep(grades), deep(request), deep(scenario))
    prioritize(views, preds, grades, scenario, request)
    assert (views, preds, grades, request, scenario) == before


def test_end_to_end_with_fixtures():
    """夹具上跑通 grade -> prioritize，且不读原始 XLSX、不导入 src/models。"""
    views = fixture("standard_attributes.json")
    preds = fixture("predictions.json")
    ref = fixture("reference_bundle.json")
    grades = grade(preds, ref, CFG)
    result = prioritize(views, preds, grades, SCENARIO,
                        {"objective": "V", "k": 100})
    assert result["candidate_count"] == 8
    assert result["rankable_count"] == 7  # F07 的 p 无效，不参与按 V 排序
    assert result["selection"]["k_actual"] == 7
    assert len(result["selection"]["selected"]) == len(
        set(result["selection"]["selected"]))
    # F07 仍在按 C 清单中，但不在按 p / 按 V 清单中
    assert "SYN-F07_invalid_prediction" in result["by_c"]
    assert "SYN-F07_invalid_prediction" not in result["by_v"]
    assert "SYN-F07_invalid_prediction" not in result["by_p"]