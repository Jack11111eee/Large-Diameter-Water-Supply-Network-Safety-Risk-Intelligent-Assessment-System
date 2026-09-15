"""特征层组合与数值字段声明测试（§3.2）。

层规格用 '+' 组合（F1+F2+F3），数值/类别划分由白名单声明而非硬编码。
"""

import pytest

from src.data import loader
from src.models.baselines import make_preprocessor


@pytest.fixture(scope="module")
def pipes():
    return loader.load_attributes()


# ---- 层解析 ----

def test_single_layer_unchanged():
    assert loader.resolve_layer("F1_base_environment") == \
        loader.load_whitelist()["layers"]["F1_base_environment"]


def test_composed_layer_concatenates_in_order():
    """按声明顺序拼接并去重（§3.2）。"""
    f1 = loader.load_whitelist()["layers"]["F1_base_environment"]
    f2 = loader.load_whitelist()["layers"]["F2_operational_snapshot"]
    got = loader.resolve_layer("F1_base_environment+F2_operational_snapshot")
    assert got == f1 + f2
    assert len(got) == len(set(got))


def test_f2_f3_are_additive_on_f1():
    f1 = set(loader.resolve_layer("F1_base_environment"))
    f1f2 = set(loader.resolve_layer(
        "F1_base_environment+F2_operational_snapshot"))
    f1f2f3 = set(loader.resolve_layer(
        "F1_base_environment+F2_operational_snapshot+F3_topology"))

    assert f1 < f1f2 < f1f2f3
    assert f1f2 - f1 == set(loader.load_whitelist()["layers"]["F2_operational_snapshot"])
    assert f1f2f3 - f1f2 == set(loader.load_whitelist()["layers"]["F3_topology"])


def test_unknown_and_empty_spec_rejected():
    with pytest.raises(KeyError, match="未知特征层"):
        loader.resolve_layer("F9_nope")
    with pytest.raises(KeyError, match="未知特征层"):
        loader.resolve_layer("F1_base_environment+F9_nope")
    with pytest.raises(KeyError, match="空特征层规格"):
        loader.resolve_layer("")


def test_grouping_column_never_enters_any_structure():
    """SZDL 只作道路分组，不作预测特征（§3.2 隔离字段）。"""
    for spec in ["F1_base_environment",
                 "F1_base_environment+F2_operational_snapshot",
                 "F1_base_environment+F2_operational_snapshot+F3_topology"]:
        assert "SZDL" not in loader.resolve_layer(spec)


# ---- 数值字段声明（修正硬编码误分类）----

def test_f2_numeric_fields_are_numeric_not_onehot():
    """RENYR/REPCNT/REPCO2/INSPF 是数值，不得被当作类别独热。"""
    prep = make_preprocessor(["PIPEAGE", "RENYR", "REPCNT", "REPCO2", "INSPF", "CZ"])
    assert prep.numeric_cols == ["PIPEAGE", "RENYR", "REPCNT", "REPCO2", "INSPF"]
    assert prep.categorical_cols == ["CZ"]


def test_topology_fields_are_numeric():
    prep = make_preprocessor(["node_degree", "node_adjacency", "component_size"])
    assert prep.categorical_cols == []
    assert len(prep.numeric_cols) == 3


# ---- 配置完整性 ----

def test_every_layer_field_has_an_explanation_group():
    """每个白名单字段都有解释分组，避免 SHAP 分组出现「未分组」（§7.1）。"""
    wl = loader.load_whitelist()
    grouped = {f for fields in wl["feature_groups"].values() for f in fields}
    for layer, cols in wl["layers"].items():
        missing = [c for c in cols if c not in grouped]
        assert not missing, f"{layer} 字段缺解释分组: {missing}"


def test_numeric_declaration_matches_declared_units():
    """单位为物理量的字段必须声明为数值，否则会被静默独热（§3.2）。

    这是原硬编码数值列 bug 的回归守卫：RENYR/REPCNT/REPCO2/INSPF 单位是
    year/count，却曾被当作类别。
    """
    wl = loader.load_whitelist()
    numeric_units = {"year", "mm", "m", "m_unverified", "MPa", "m/s",
                     "count/year", "count"}
    numeric = loader.numeric_fields()
    for code, meta in wl["source_columns"].items():
        if meta["unit"] in numeric_units:
            assert code in numeric, f"{code} 单位为 {meta['unit']} 却未声明为数值"


# ---- 真实数据上的 F3 矩阵 ----

def test_f3_feature_matrix_derives_on_real_data(pipes):
    """F3 列不在原始表中，由节点 ID 派生后并入（§3.2）。"""
    m = loader.feature_matrix(pipes, "F3_topology")
    assert list(m.columns) == ["node_degree", "node_adjacency", "component_size"]
    assert len(m) == 7288

    composed = loader.feature_matrix(
        pipes, "F1_base_environment+F2_operational_snapshot+F3_topology")
    assert composed.shape == (7288, 25)
    loader.assert_no_forbidden_columns(composed.columns)


def test_missing_non_topology_column_still_rejected(pipes):
    """只有 F3 列允许派生；其他缺失列仍须报错。"""
    trimmed = pipes.drop(columns=["BURDEP"])
    with pytest.raises(KeyError, match="引用了不存在的列"):
        loader.feature_matrix(trimmed, "F1_base_environment")