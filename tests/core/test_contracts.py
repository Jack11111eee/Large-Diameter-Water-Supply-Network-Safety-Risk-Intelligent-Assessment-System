"""契约测试：必需字段、唯一键、模式匹配、版本匹配、不变性（§13.3）。

对应里程碑 §5.1 验收清单。
B/C 的测试进程不访问两个原始 XLSX、不导入 src/models/ 即可运行。
"""

import json
from pathlib import Path

import pytest

from src.contracts import (
    SCHEMA_VERSION,
    ContractError,
    PRODUCTS,
    DataKind,
    PredictionMode,
    validate_package,
    assert_no_nonfinite,
)

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "fixtures"

# 产物 -> 夹具文件名
FIXTURE_FILES = {
    "standard_attributes": "standard_attributes.json",
    "geometry": "geometry.json",
    "predictions": "predictions.json",
    "explanation": "explanation.json",
    "decision": "decision.json",
    "reference_bundle": "reference_bundle.json",
}


def load(name):
    return json.loads((FIXTURE_DIR / FIXTURE_FILES[name]).read_text(encoding="utf-8"))


@pytest.mark.parametrize("product", sorted(FIXTURE_FILES))
def test_fixtures_validate(product):
    """每个夹具包通过其自身契约。"""
    validate_package(product, load(product))


@pytest.mark.parametrize("product", sorted(FIXTURE_FILES))
def test_fixture_data_kind_is_synthetic(product):
    """样例必须标 synthetic_fixture，不得冒充正式预测（§13.3）。"""
    for row in load(product):
        assert row["data_kind"] == DataKind.SYNTHETIC_FIXTURE.value


@pytest.mark.parametrize("product", sorted(FIXTURE_FILES))
def test_fixture_has_no_nonfinite(product):
    """JSON 不传 NaN/Infinity（§13.3）。"""
    assert_no_nonfinite(load(product))


def test_fixture_covers_required_boundaries():
    """覆盖 §13.3 六类边界 + 审查发现的两类人工构造边界。"""
    ids = {r["pipe_id"] for r in load("standard_attributes")}
    expected = {
        "SYN-F01_normal",
        "SYN-F02_unknown_facility",
        "SYN-F03_coord_conflict",
        "SYN-F04_tie_probability",
        "SYN-F06_missing_explanation",
        "SYN-F07_invalid_prediction",
        "SYN-F08_diameter_out_of_range",
    }
    assert expected <= ids


def test_unknown_facility_is_imputed_not_zero():
    """审查发现：真实数据 FAC 零缺失，未知设施只能人工构造；代理值非零（§8.1）。"""
    row = next(r for r in load("standard_attributes")
               if r["pipe_id"] == "SYN-F02_unknown_facility")
    assert row["attributes"]["facility"] is None
    assert "proxy_imputed:facility" in row["quality_flags"]


def test_diameter_out_of_range_is_marked():
    """审查发现：真实数据管径恰为 300–1600，越界只能人工构造（§8.1）。"""
    row = next(r for r in load("standard_attributes")
               if r["pipe_id"] == "SYN-F08_diameter_out_of_range")
    assert row["attributes"]["diameter_mm"] == 2000
    assert "diameter_out_of_range" in row["quality_flags"]


def test_coordinate_conflict_keeps_attribute_score():
    """坐标冲突不改变属性评分，空间结果待核（§8.4 D01、T05）。"""
    geom = next(r for r in load("geometry")
                if r["pipe_id"] == "SYN-F03_coord_conflict")
    assert geom["coordinate_conflict"] is True
    assert geom["crs_known"] is False
    # 属性评分仍有效
    attr = next(r for r in load("standard_attributes")
                if r["pipe_id"] == "SYN-F03_coord_conflict")
    assert attr["attributes"]["pipeage"] == 38


def test_tie_probability_same_score():
    """同分用例：两根管段概率完全相同（管龄并列是常态，最大并列组 765）。"""
    preds = {r["pipe_id"]: r["p"] for r in load("predictions")}
    assert preds["SYN-F04_tie_probability"] == preds["SYN-F05_tie_probability_b"]


def test_invalid_prediction_is_null():
    """无效预测用 null 表示，不写 NaN（§13.3、T09）。"""
    row = next(r for r in load("predictions")
               if r["pipe_id"] == "SYN-F07_invalid_prediction")
    assert row["p"] is None


def test_explanation_unavailable_not_substituted():
    """解释未就绪必须显式 unavailable，不得用别的模型顶替（§13.5）。"""
    rows = load("explanation")
    unavail = [r for r in rows if r["status"] == "unavailable"]
    assert unavail, "夹具必须包含 unavailable 用例"
    for r in unavail:
        assert r["base_value"] is None
        assert r["contributions"] == []


# --------------------------------------------------------------------------
# 反向测试：契约必须能拒绝违规输入
# --------------------------------------------------------------------------

def test_rejects_missing_required_field():
    rows = load("predictions")
    del rows[0]["model_id"]
    with pytest.raises(ContractError, match="model_id"):
        validate_package("predictions", rows)


def test_rejects_duplicate_unique_key():
    rows = load("predictions")
    rows.append(dict(rows[0]))
    with pytest.raises(ContractError, match="重复唯一键"):
        validate_package("predictions", rows)


def test_rejects_version_mismatch():
    rows = load("predictions")
    rows[0]["schema_version"] = "0.0.1"
    with pytest.raises(ContractError, match="schema_version"):
        validate_package("predictions", rows)


def test_rejects_bad_enum():
    rows = load("predictions")
    rows[0]["prediction_mode"] = "made_up_mode"
    with pytest.raises(ContractError, match="prediction_mode"):
        validate_package("predictions", rows)


def test_rejects_nonfinite():
    with pytest.raises(ContractError, match="非有限"):
        assert_no_nonfinite({"p": float("nan")})


def test_rejects_bool_as_int():
    """bool 是 int 子类，必须显式排除。"""
    rows = load("predictions")
    rows[0]["calibrated"] = True   # 合法
    validate_package("predictions", rows)
    rows[0]["fold"] = True         # 不合法
    with pytest.raises(ContractError, match="fold"):
        validate_package("predictions", rows)


def test_rejects_unknown_product():
    with pytest.raises(ContractError, match="未知产物"):
        validate_package("not_a_product", [])


def test_all_products_have_producer_and_consumers():
    """§13.4 契约表：每个产物都要有明确的生产者与消费者。"""
    for name, spec in PRODUCTS.items():
        assert spec.producer in {"A", "B", "C"}, name
        assert spec.consumers, name


def test_labels_never_in_shared_packages():
    """标准展示属性包不含 y_true/ACCID/事件（§13.4、§5.6）。"""
    for row in load("standard_attributes"):
        assert "y_true" not in row
        assert "accid" not in {k.lower() for k in row}
        assert "ACCID" not in row["attributes"]
        assert "event_count" not in row


def test_predictions_never_carry_ground_truth():
    """预测包不含 y_true（§13.4）。"""
    for row in load("predictions"):
        assert "y_true" not in row