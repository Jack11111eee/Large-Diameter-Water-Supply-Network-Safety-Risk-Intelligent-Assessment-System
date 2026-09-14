"""M1 最小闭环集成测试（里程碑 §6.1、方案 §13.5、§13.6）。

M1 门禁：**真实属性 → 折外概率 → 逐折成绩 → 概率 Top-K → 页面** 全部跑通。

本测试驱动真实 M0 产物（`pipeline.build_all`）走完整链路，并断言：

1. 四类聚合成绩分列且可读（§6.4.1）；
2. 决策模块在真实 7,288 条上可运行，分级/后果/优先级来自 `src/decision/`；
3. 界面适配器消费决策结果对象，**页面与导出不重算** percentile / C / V（§13.1）；
4. 切换预算与目标不改变概率与等级（§5.3.1 筛选不变性）；
5. 事后模式不改变预测侧任何数值（§8.5）。

不读两个原始 XLSX、不导入 `src/models/` 也可运行：产物由 `pipeline.build_all`
在临时目录构建（其内部读取原始数据属核心链路，不是本测试的消费路径）。
"""

import json
from collections import Counter

import pytest

from app import adapter
from src.contracts import GRADE_CONFIG_VERSION, SCENARIO_CONFIG_VERSION
from src.decision import (
    advise_predictive,
    grade,
    load_advice_rules,
    load_grade_config,
    load_scenario_config,
    prioritize,
)
from src.integration import pipeline

N_PIPES = 7288


@pytest.fixture(scope="module")
def release(tmp_path_factory):
    """一次构建真实 M0 产物（含 OPSST/PRESS 展示字段）。"""
    out = tmp_path_factory.mktemp("m1_release")
    result = pipeline.build_all(out)
    return out, result


@pytest.fixture(scope="module")
def artifacts(release):
    out, result = release
    return {
        "dir": out,
        "result": result,
        "attrs": json.loads((out / "standard_attributes.json").read_text(encoding="utf-8")),
        "preds": json.loads((out / "predictions.json").read_text(encoding="utf-8")),
        "ref": json.loads((out / "reference_bundle.json").read_text(encoding="utf-8")),
        "eval": json.loads((out / "evaluation.json").read_text(encoding="utf-8")),
    }


# --------------------------------------------------------------------------
# 1. 真实属性 → 折外概率 → 逐折成绩
# --------------------------------------------------------------------------

def test_closure_attributes_and_probabilities_cover_all_pipes(artifacts):
    """7,288 条属性与折外概率，pipe_id 一致（§13.6 按 pipe_id 对齐）。"""
    a = {r["pipe_id"] for r in artifacts["attrs"]}
    p = {r["pipe_id"] for r in artifacts["preds"]}
    assert len(a) == len(p) == N_PIPES
    assert a == p


def test_closure_per_fold_scores_are_separated_and_complete(artifacts):
    """四类聚合分列，无失败折（§6.4.1）。"""
    ev = artifacts["eval"]
    for key in ("per_fold", "macro_mean", "sample_weighted", "pooled_oof_replay"):
        assert key in ev
    mm = ev["macro_mean"]["roc_auc"]
    assert mm["valid_folds"] == mm["total_folds"] == 5
    assert mm["complete"] is True
    assert len(ev["per_fold"]) == 5
    # pooled 与逐折分列，不混写
    assert ev["pooled_oof_replay"]["roc_auc"] != mm["value"]


def test_closure_probabilities_are_valid(artifacts):
    """概率在 [0,1]（§5.3）。"""
    for r in artifacts["preds"]:
        assert r["p"] is not None
        assert 0.0 <= r["p"] <= 1.0


# --------------------------------------------------------------------------
# 2. 概率 Top-K → 决策
# --------------------------------------------------------------------------

def test_closure_decision_runs_on_real_data(artifacts):
    """决策模块在真实 7,288 条上产出分级、后果与优先级。"""
    gc = load_grade_config()
    grades = grade(artifacts["preds"], artifacts["ref"], gc)
    assert len(grades) == N_PIPES
    dist = Counter(g["relative_risk_level"] for g in grades)
    assert set(dist) <= {"L1", "L2", "L3", "L4", "unavailable"}
    assert dist["unavailable"] == 0

    pr = prioritize(artifacts["attrs"], artifacts["preds"], grades,
                    load_scenario_config(), {"k": 100, "objective": "V"})
    sel = pr["selection"]
    assert sel["k_requested"] == 100 and sel["k_actual"] == 100
    assert len(sel["selected"]) == 100
    assert len(set(sel["selected"])) == 100  # 无重复


def test_closure_grade_matches_spec_percentile(artifacts):
    """分级用平均秩百分位，全精度分数参与（§5.3.1）。"""
    from src.decision.grading import percentile_of
    gc = load_grade_config()
    scores = artifacts["ref"][0]["scores"]
    grades = {g["pipe_id"]: g for g in
              grade(artifacts["preds"], artifacts["ref"], gc)}
    for r in artifacts["preds"][:200]:
        expected = percentile_of(r["p"], scores)
        assert grades[r["pipe_id"]]["risk_percentile"] == pytest.approx(expected)


def test_closure_high_grade_view_preserved_independently(artifacts):
    """按 p 的 Top-K 与 L3/L4 视图始终独立保留（§8.1）。"""
    gc = load_grade_config()
    grades = grade(artifacts["preds"], artifacts["ref"], gc)
    pr = prioritize(artifacts["attrs"], artifacts["preds"], grades,
                    load_scenario_config(), {"k": 50, "objective": "V"})
    assert len(pr["preserved_by_p_topk"]) == 50
    # 高概率但未进 V 清单的管段被标出
    assert isinstance(pr["marked_not_in_value"], list)


def test_closure_advice_on_real_data(artifacts):
    """建议规则在真实数据上运行，不产出禁用措辞（§8.3）。"""
    gc = load_grade_config()
    grades = grade(artifacts["preds"], artifacts["ref"], gc)
    pr = prioritize(artifacts["attrs"], artifacts["preds"], grades,
                    load_scenario_config(), {"k": 100, "objective": "V"})
    advice = advise_predictive(artifacts["attrs"], artifacts["preds"], grades,
                               pr, load_advice_rules())
    assert advice
    for a in advice:
        for bad in ("不会爆管", "无需维护", "保证安全", "立即换管", "自动调压"):
            assert bad not in a["suggested_action"]
        assert a["advice_mode"] == "predictive"


# --------------------------------------------------------------------------
# 3. 页面：消费决策结果，不重算
# --------------------------------------------------------------------------

def test_closure_page_loads_real_release(artifacts):
    """界面适配器装载真实产物，决策来自 src/decision/ 而非本地重算。"""
    pkg = adapter.load_package_set(artifacts["dir"])
    assert pkg.report.source == "real_release"
    assert pkg.report.data_kind == "real_standard"
    assert pkg.decision.source == "decision_module"
    assert len(pkg.pipes) == N_PIPES


def test_closure_page_values_equal_decision_module_output(artifacts):
    """页面展示的百分位/C/V 与决策模块逐字段一致（不重算）。"""
    gc, sc = load_grade_config(), load_scenario_config()
    grades = {g["pipe_id"]: g for g in
              grade(artifacts["preds"], artifacts["ref"], gc)}
    pr = prioritize(artifacts["attrs"], artifacts["preds"], list(grades.values()),
                    sc, {"k": N_PIPES, "objective": "V"})
    records = pr["records"]

    pkg = adapter.load_package_set(artifacts["dir"])
    rows = {r["pipe_id"]: r for r in adapter.pipe_list_rows(pkg)}
    assert len(rows) == N_PIPES

    for pid, row in list(rows.items())[:300]:
        g = grades[pid]
        assert row["相对等级"] == g["relative_risk_level"]
        assert row["百分位"] == pytest.approx(g["risk_percentile"])
        assert row["后果代理C"] == pytest.approx(records[pid]["consequence_proxy"])
        assert row["优先值V"] == pytest.approx(records[pid]["priority_value"])


def test_closure_export_uses_same_result_object(artifacts):
    """导出使用相同结果对象，不再重算（§13.4）。"""
    pkg = adapter.load_package_set(artifacts["dir"])
    rows = adapter.pipe_list_rows(pkg)
    csv_text = adapter.rows_to_csv(rows)
    lines = csv_text.strip().splitlines()
    assert len(lines) == len(rows) + 1  # 表头 + 每行
    assert "pipe_id" in lines[0]


# --------------------------------------------------------------------------
# 4. 筛选 / 预算不变性（§5.3.1）
# --------------------------------------------------------------------------

def test_closure_budget_and_objective_do_not_change_grade(artifacts):
    """改变 K 与目标只改变候选清单，不重算概率与等级（§5.3.1）。"""
    gc = load_grade_config()
    grades = {g["pipe_id"]: g for g in
              grade(artifacts["preds"], artifacts["ref"], gc)}
    sc = load_scenario_config()

    for objective in ("p", "C", "V"):
        for k in (10, 500, N_PIPES):
            pr = prioritize(artifacts["attrs"], artifacts["preds"],
                            list(grades.values()), sc,
                            {"k": k, "objective": objective})
            for pid, rec in pr["records"].items():
                g = grades[pid]
                assert rec["risk_percentile"] == pytest.approx(g["risk_percentile"])
                assert rec["relative_risk_level"] == g["relative_risk_level"]


def test_closure_grade_stable_across_repeated_calls(artifacts):
    """同一输入重复调用分级结果一致（确定性）。"""
    gc = load_grade_config()
    a = grade(artifacts["preds"], artifacts["ref"], gc)
    b = grade(artifacts["preds"], artifacts["ref"], gc)
    assert a == b


# --------------------------------------------------------------------------
# 5. 事后模式隔离（§8.5）
# --------------------------------------------------------------------------

def test_closure_post_event_does_not_change_prediction_side(artifacts):
    """事后模式只影响独立建议，不改 p/等级/C/V/预测清单（§8.5）。"""
    from src.decision import advise_post_event
    gc, sc, rules = load_grade_config(), load_scenario_config(), load_advice_rules()

    grades = grade(artifacts["preds"], artifacts["ref"], gc)
    pr = prioritize(artifacts["attrs"], artifacts["preds"], grades, sc,
                    {"k": 100, "objective": "V"})
    predictive = advise_predictive(artifacts["attrs"], artifacts["preds"],
                                   grades, pr, rules)

    before = json.dumps([grades, pr, predictive], sort_keys=True, default=str)

    events = [{
        "schema_version": "1.0.0", "data_version": "d", "run_id": "r",
        "prediction_mode": "oof_replay", "data_kind": "synthetic_fixture",
        "pipe_id": pr["selection"]["selected"][0], "event_count": 2,
        "as_of": "2024-12-31",
    }]
    post = advise_post_event(artifacts["attrs"], events, "2024-12-31", rules)
    assert post
    assert all(a["advice_mode"] == "post_event_review" for a in post)

    after = json.dumps([grades, pr, predictive], sort_keys=True, default=str)
    assert after == before


# --------------------------------------------------------------------------
# 6. 版本一致性（§13.3）
# --------------------------------------------------------------------------

def test_closure_versions_compatible(artifacts):
    """发布集合内 data_version 一致，配置版本与产物兼容（§13.3）。"""
    dv = artifacts["result"]["data_version"]
    for r in artifacts["attrs"][:50]:
        assert r["data_version"] == dv
    assert artifacts["ref"][0]["data_version"] == dv
    assert artifacts["ref"][0]["grade_config_version"] == GRADE_CONFIG_VERSION
    assert artifacts["attrs"][0]["prediction_mode"] == "oof_replay"
    assert SCENARIO_CONFIG_VERSION == "consequence_scenario_v1"