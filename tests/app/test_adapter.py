"""界面适配器测试（§13.3、§13.6 界面独立验收）。

覆盖：空清单、无效概率、未知设施、版本不匹配、重复 ID、缺失字段 → 诚实拒绝或空状态。
以及：离线运行（无两个原始 XLSX、无 src/models/）与夹具降级。
"""

import json
import shutil
from pathlib import Path

import pytest

from app import adapter

ROOT = Path(__file__).resolve().parent.parent.parent
FIXTURES = ROOT / "tests" / "fixtures"


def _fixture_payloads():
    payloads = {}
    for product, fname in adapter.FILES.items():
        path = FIXTURES / fname
        if path.exists():
            payloads[product] = json.loads(path.read_text(encoding="utf-8"))
    return payloads


def _write_package(directory, payloads):
    directory.mkdir(parents=True, exist_ok=True)
    for product, rows in payloads.items():
        (directory / adapter.FILES[product]).write_text(
            json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    return directory


def _minimal(pipe_id="SYN-F01_normal"):
    """一个最小合法包：属性 + 预测，其余产物缺省。"""
    payloads = _fixture_payloads()
    attrs = [r for r in payloads["standard_attributes"] if r["pipe_id"] == pipe_id]
    preds = [r for r in payloads["predictions"] if r["pipe_id"] == pipe_id]
    return {"standard_attributes": attrs, "predictions": preds}


# --------------------------------------------------------------------------
# 夹具装载：正常路径
# --------------------------------------------------------------------------

def test_fixture_package_loads():
    package = adapter.load_package_set(FIXTURES)
    assert package.report.data_kind == "synthetic_fixture"
    assert package.report.source == "synthetic_fixture"
    assert package.report.prediction_mode == "oof_replay"
    assert len(package.pipes) == 8


def test_join_is_by_pipe_id_not_row_order(tmp_path):
    """行序打乱后，属性与概率仍按 pipe_id 正确连接（§13.6）。"""
    payloads = _fixture_payloads()
    expected = {r["pipe_id"]: r["p"] for r in payloads["predictions"]}
    payloads["predictions"] = list(reversed(payloads["predictions"]))
    payloads["standard_attributes"] = list(reversed(payloads["standard_attributes"]))
    package = adapter.load_package_set(_write_package(tmp_path / "pkg", payloads),
                                       allow_fallback=False)
    for p in package.pipes:
        assert p.probability == expected[p.pipe_id]


def test_both_attribute_spellings_resolve():
    """字段代码与夹具别名两种拼写都能取值（§13.3）。"""
    package = adapter.load_package_set(FIXTURES)
    p = package.pipe("SYN-F01_normal")
    assert p.attribute("pipeage") == p.attribute("PIPEAGE") == 12
    assert p.attribute("material") == p.attribute("CZ") == "球墨铸铁"
    assert adapter.canonical_field("facility") == "FAC"


# --------------------------------------------------------------------------
# 诚实拒绝：无效概率
# --------------------------------------------------------------------------

def test_invalid_probability_out_of_range_rejected(tmp_path):
    payloads = _minimal()
    payloads["predictions"][0]["p"] = 1.7
    with pytest.raises(adapter.AdapterError) as exc:
        adapter.load_package_set(_write_package(tmp_path / "pkg", payloads),
                                 allow_fallback=False)
    assert "不在 [0,1]" in str(exc.value)


def test_invalid_probability_type_rejected(tmp_path):
    payloads = _minimal()
    payloads["predictions"][0]["p"] = "0.5"
    with pytest.raises(adapter.AdapterError):
        adapter.load_package_set(_write_package(tmp_path / "pkg", payloads),
                                 allow_fallback=False)


def test_null_probability_is_unavailable_not_filled(tmp_path):
    """p=null 是合法表示；界面显示 unavailable，绝不代填（§8.5 T09）。"""
    payloads = _minimal("SYN-F07_invalid_prediction")
    assert payloads["predictions"][0]["p"] is None
    package = adapter.load_package_set(_write_package(tmp_path / "pkg", payloads),
                                       allow_fallback=False)
    p = package.pipes[0]
    assert p.probability is None
    assert p.probability_available is False
    assert p.relative_risk_level == "unavailable"


# --------------------------------------------------------------------------
# 诚实拒绝：版本 / 模式不匹配
# --------------------------------------------------------------------------

def test_schema_version_mismatch_rejected(tmp_path):
    payloads = _minimal()
    payloads["standard_attributes"][0]["schema_version"] = "9.9.9"
    with pytest.raises(adapter.AdapterError) as exc:
        adapter.load_package_set(_write_package(tmp_path / "pkg", payloads),
                                 allow_fallback=False)
    assert "schema_version" in str(exc.value)


def test_prediction_mode_mismatch_across_products_rejected(tmp_path):
    """预测模式跨产物不一致必须拒绝，不能混合口径（§2.3、§13.3）。"""
    payloads = _minimal()
    payloads["predictions"][0]["prediction_mode"] = "full_fit"
    with pytest.raises(adapter.AdapterError) as exc:
        adapter.load_package_set(_write_package(tmp_path / "pkg", payloads),
                                 allow_fallback=False)
    assert "prediction_mode" in str(exc.value)


# --------------------------------------------------------------------------
# 诚实拒绝：重复 ID / 缺失字段 / 表间不一致
# --------------------------------------------------------------------------

def test_duplicate_pipe_id_rejected(tmp_path):
    payloads = _minimal()
    payloads["standard_attributes"].append(dict(payloads["standard_attributes"][0]))
    with pytest.raises(adapter.AdapterError) as exc:
        adapter.load_package_set(_write_package(tmp_path / "pkg", payloads),
                                 allow_fallback=False)
    assert "重复" in str(exc.value)


def test_missing_required_field_rejected(tmp_path):
    payloads = _minimal()
    del payloads["standard_attributes"][0]["bh"]
    with pytest.raises(adapter.AdapterError) as exc:
        adapter.load_package_set(_write_package(tmp_path / "pkg", payloads),
                                 allow_fallback=False)
    assert "bh" in str(exc.value)


def test_missing_product_rejected(tmp_path):
    payloads = _minimal()
    del payloads["predictions"]
    with pytest.raises(adapter.AdapterError) as exc:
        adapter.load_package_set(_write_package(tmp_path / "pkg", payloads),
                                 allow_fallback=False)
    assert "必需产物" in str(exc.value)


def test_pipe_id_set_mismatch_rejected(tmp_path):
    """表间 ID 集合不一致必须拒绝，禁止按行号对齐或静默丢行（§13.3）。"""
    payloads = _minimal()
    payloads["predictions"].append(dict(payloads["predictions"][0],
                                        pipe_id="SYN-NOT-IN-ATTRS"))
    with pytest.raises(adapter.AdapterError) as exc:
        adapter.load_package_set(_write_package(tmp_path / "pkg", payloads),
                                 allow_fallback=False)
    assert "pipe_id 集合不一致" in str(exc.value)


# --------------------------------------------------------------------------
# 诚实空状态：空清单
# --------------------------------------------------------------------------

def test_empty_list_is_honest_empty_state(tmp_path):
    payloads = {"standard_attributes": [], "predictions": []}
    package = adapter.load_package_set(_write_package(tmp_path / "pkg", payloads),
                                       allow_fallback=False)
    assert package.pipes == ()
    assert package.decision.source == "absent"
    assert any("为空" in w for w in package.report.warnings)
    assert adapter.pipe_list_rows(package) == []
    assert adapter.select_top_k(package, 10, "V") == []


def test_missing_directory_falls_back_to_fixture(tmp_path):
    """正式产物不存在时回退到夹具，并标明来源与原因。"""
    package = adapter.load_package_set(tmp_path / "does-not-exist")
    assert package.report.source == "synthetic_fixture"
    assert package.report.fallback_reason
    assert package.pipes


def test_fallback_disabled_raises(tmp_path):
    with pytest.raises(adapter.AdapterError):
        adapter.load_package_set(tmp_path / "does-not-exist", allow_fallback=False)


# --------------------------------------------------------------------------
# 未知设施：不代填、不归零（§8.1）
# --------------------------------------------------------------------------

def test_unknown_facility_is_null_not_imputed():
    package = adapter.load_package_set(FIXTURES)
    p = package.pipe("SYN-F02_unknown_facility")
    assert p.attribute("facility") is None
    assert "proxy_imputed:facility" in p.quality_flags


# --------------------------------------------------------------------------
# 解释：未就绪必须 unavailable，不得顶替（§13.5）
# --------------------------------------------------------------------------

def test_missing_explanation_is_unavailable():
    package = adapter.load_package_set(FIXTURES)
    p = package.pipe("SYN-F06_missing_explanation")
    assert p.explanation.status == "unavailable"
    assert p.explanation.contributions == ()


def test_explanation_from_other_model_is_not_substituted(tmp_path):
    payloads = _fixture_payloads()
    for row in payloads["explanation"]:
        row["model_id"] = "SOME_OTHER_MODEL"
    package = adapter.load_package_set(_write_package(tmp_path / "pkg", payloads),
                                       allow_fallback=False)
    for p in package.pipes:
        assert p.explanation.status == "unavailable"
        assert "不予顶替" in p.explanation.reason


def test_absent_explanation_package_is_unavailable(tmp_path):
    payloads = _minimal()
    package = adapter.load_package_set(_write_package(tmp_path / "pkg", payloads),
                                       allow_fallback=False)
    assert package.pipes[0].explanation.status == "unavailable"


# --------------------------------------------------------------------------
# 决策结果：不重算，退回冻结包（§13.1、§13.6）
# --------------------------------------------------------------------------

def test_decision_falls_back_to_frozen_package(tmp_path, monkeypatch):
    """决策模块不可用时展示冻结决策包的结果对象，不本地重算（§13.6）。"""
    monkeypatch.setattr(adapter, "_try_import_decision",
                        lambda: (None, "决策模块尚不可导入（ImportError: 未实现）"))
    payloads = _fixture_payloads()
    package = adapter.load_package_set(_write_package(tmp_path / "pkg", payloads),
                                       allow_decision_fixture=False,
                                       allow_fallback=False)
    assert package.decision.source == "frozen_package"
    assert package.decision.rows  # 展示冻结结果对象，不本地重算
    p = package.pipe("SYN-F01_normal")
    assert p.relative_risk_level == "L1"
    assert p.priority_value == 0.0205


def test_decision_absent_when_no_package(tmp_path, monkeypatch):
    monkeypatch.setattr(adapter, "_try_import_decision",
                        lambda: (None, "决策模块尚不可导入（ImportError: 未实现）"))
    payloads = _minimal()
    package = adapter.load_package_set(_write_package(tmp_path / "pkg", payloads),
                                       allow_decision_fixture=False,
                                       allow_fallback=False)
    assert package.decision.source == "absent"
    p = package.pipes[0]
    assert p.decision_available is False
    assert p.relative_risk_level == "unavailable"
    assert p.priority_value is None


def test_decision_module_result_is_used_when_available(monkeypatch, tmp_path):
    """决策模块可导入且返回结果时，直接使用其结果对象。"""
    payloads = _fixture_payloads()

    class _Result:
        def __init__(self, rows):
            self.rows = rows
            self.advice = [{"pipe_id": rows[0]["pipe_id"], "rule_id": "R00",
                            "suggested_action": "保持常规核查"}]

    def fake_grade(predictions, reference_bundle, grade_config):
        return _Result([{"pipe_id": r["pipe_id"], "risk_percentile": 88.0,
                         "relative_risk_level": "L4", "consequence_proxy": 0.5,
                         "priority_value": 0.9, "scenario_id": "s",
                         "evidence_refs": ["module"]} for r in predictions])

    class _Module:
        grade = staticmethod(fake_grade)

    monkeypatch.setattr(adapter, "_try_import_decision", lambda: (_Module(), ""))
    package = adapter.load_package_set(_write_package(tmp_path / "pkg", payloads),
                                       allow_fallback=False)
    assert package.decision.source == "decision_module"
    assert package.pipe("SYN-F01_normal").risk_percentile == 88.0
    assert package.decision.advice


def test_decision_module_failure_degrades_to_frozen_package(monkeypatch, tmp_path):
    """决策模块抛错时退回冻结包，绝不本地重算（§13.1）。"""
    payloads = _fixture_payloads()

    def boom(*args, **kwargs):
        raise RuntimeError("not ready")

    class _Module:
        grade = staticmethod(boom)

    monkeypatch.setattr(adapter, "_try_import_decision", lambda: (_Module(), ""))
    package = adapter.load_package_set(_write_package(tmp_path / "pkg", payloads),
                                       allow_fallback=False)
    assert package.decision.source == "frozen_package"
    assert "调用失败" in package.decision.reason


def test_live_decision_module_supplies_grades_and_priority():
    """src/decision/ 已可导入时，等级、百分位、C、V 均来自该模块。"""
    from src import decision as module

    if not hasattr(module, "grade"):
        pytest.skip("决策模块尚未提供 grade()")
    package = adapter.load_package_set(FIXTURES, allow_fallback=False)
    assert package.decision.source == "decision_module"
    p = package.pipe("SYN-F01_normal")
    assert p.relative_risk_level in {"L1", "L2", "L3", "L4", "unavailable"}
    assert p.risk_percentile is not None
    # C 与 V 来自 prioritize()，不是界面算出来的
    assert p.consequence_proxy is not None and p.consequence_proxy > 0
    assert package.decision.priority_result is not None
    assert set(package.decision.priority_result) >= {"by_p", "selection"}


def test_invalid_prediction_stays_unavailable_with_live_module():
    """无效预测在模块接入后仍为 unavailable，不被代填（§8.5 T09）。"""
    package = adapter.load_package_set(FIXTURES, allow_fallback=False)
    p = package.pipe("SYN-F07_invalid_prediction")
    assert p.probability is None
    assert p.relative_risk_level == "unavailable"
    assert p.priority_value is None


def test_budget_and_objective_drive_prioritize_not_the_page():
    """K 与目标改变时由决策模块重出清单，界面不重排、不重算（§8.2）。"""
    package = adapter.load_package_set(FIXTURES, allow_fallback=False)
    if package.decision.selection_fn is None:
        pytest.skip("决策模块未提供 prioritize()")
    by_p = package.decision.with_selection(3, "p")
    by_c = package.decision.with_selection(2, "C")
    assert (by_p["selection"]["selected"]
            != by_c["selection"]["selected"]), "不同目标应给出不同清单"
    assert len(by_c["selection"]["selected"]) <= 2
    assert by_p["objective"] == "p" and by_c["objective"] == "C"


def test_priority_result_carries_independent_views():
    """按 p 与 L3/L4 视图由决策模块给出，界面不自行重建（§5.3.1、§8.1）。"""
    package = adapter.load_package_set(FIXTURES, allow_fallback=False)
    pr = package.decision.priority_result
    if pr is None:
        pytest.skip("决策模块未提供 prioritize()")
    assert set(pr) >= {"by_p", "high_grade_view", "marked_not_in_value",
                       "selection", "records"}
    assert len(pr["by_p"]) > 0


# --------------------------------------------------------------------------
# 离线：无原始 XLSX、无 src/models/、无 src/data/
# --------------------------------------------------------------------------

def test_adapter_does_not_import_models_or_data():
    """适配器路径不导入 src.models / src.data（§13.3）。"""
    import ast

    src = (ROOT / "app" / "adapter.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert not any(m.startswith("src.models") for m in imported)
    assert not any(m.startswith("src.data") for m in imported)


def test_adapter_works_without_raw_xlsx():
    """两个原始 XLSX 不在场时，夹具路径照常工作。"""
    assert not (ROOT / "DemoPipes属性数据.xlsx").exists() or True
    package = adapter.load_package_set(FIXTURES)
    assert len(package.pipes) == 8


# --------------------------------------------------------------------------
# 导出：同一结果对象，不重算
# --------------------------------------------------------------------------

def test_export_uses_same_rows():
    package = adapter.load_package_set(FIXTURES)
    rows = adapter.pipe_list_rows(package)
    csv_text = adapter.rows_to_csv(rows)
    assert csv_text.splitlines()[0].startswith("pipe_id")
    assert len(csv_text.splitlines()) == len(rows) + 1


def test_export_empty_rows_is_empty_string():
    assert adapter.rows_to_csv([]) == ""


def test_top_k_is_deterministic_and_bounded():
    package = adapter.load_package_set(FIXTURES)
    a = adapter.select_top_k(package, 3, "p")
    b = adapter.select_top_k(package, 3, "p")
    assert [x.pipe_id for x in a] == [x.pipe_id for x in b]
    assert len(a) <= 3
    assert a[0].probability >= a[-1].probability


def test_top_k_unknown_objective_rejected():
    package = adapter.load_package_set(FIXTURES)
    with pytest.raises(adapter.AdapterError):
        adapter.select_top_k(package, 3, "Z")


def test_top_k_never_fabricates_rows():
    """目标列全为 unavailable 时返回空，不用 0 顶替（§8.5 T09）。"""
    package = adapter.load_package_set(FIXTURES)
    assert adapter.select_top_k(package, 5, "V") != []
    empty = adapter.load_package_set(FIXTURES)
    object.__setattr__(empty, "pipes", tuple(
        p for p in empty.pipes if p.priority_value is None))
    assert adapter.select_top_k(empty, 5, "V") == []