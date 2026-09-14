"""审计测试：复现数据分析报告 V2.0 全部核验数值（里程碑 §1.1、§5.2）。

数值来源：数据分析报告.md §2–§6 与附录 A。
"""

import math

import pytest

from src.audit.checks import run_full_audit
from src.data import loader


@pytest.fixture(scope="module")
def audit():
    return run_full_audit()


@pytest.fixture(scope="module")
def pipes():
    return loader.load_attributes()


# ---- 文件指纹（附录 A）----

def test_file_hashes_match(audit):
    for name, v in audit["file_versions"].items():
        assert v["match"], f"{name} SHA-256 不匹配: {v}"


# ---- 键与事件（§2.1、§3.1）----

def test_row_and_key_counts(audit):
    k = audit["keys_and_events"]
    assert k["n_pipes"] == 7288
    assert k["id_unique"] and k["bh_unique"] and k["scorecard_id_unique"]
    assert k["n_events"] == 262
    assert k["events_all_linked"]


def test_label_distribution(audit):
    k = audit["keys_and_events"]
    assert k["counts"] == {0: 7037, 1: 240, 2: 11}
    assert k["n_positive"] == 251
    assert k["event_total"] == 262
    assert k["positive_rate"] == pytest.approx(0.0344402, abs=1e-6)


def test_accid_is_exact_label(audit):
    """ACCID 与 2024 事件次数完全相等（§3.2）。"""
    assert audit["keys_and_events"]["accid_equals_count"] is True
    assert audit["keys_and_events"]["scorecard_count_equals"] is True


def test_zero_share_vs_poisson(audit):
    k = audit["keys_and_events"]
    assert k["zero_share"] == pytest.approx(0.9655598, abs=1e-6)
    assert k["poisson_zero"] == pytest.approx(0.9646890, abs=1e-6)
    # 差异约 0.087 个百分点，不足以证明零膨胀
    assert abs(k["zero_share"] - k["poisson_zero"]) < 0.001


# ---- 泄漏（§3.2、§4.1）----

def test_leakage_aucs(audit):
    lk = audit["leakage"]
    assert lk["accid_is_label"] is True
    assert lk["ops_score_auc"] == pytest.approx(0.997596, abs=1e-5)
    assert lk["total_score_auc"] == pytest.approx(0.876097, abs=1e-5)
    assert lk["base_attr_score_auc"] == pytest.approx(0.795094, abs=1e-5)


def test_single_field_aucs(audit):
    a = audit["single_field_auc"]
    assert a["PIPEAGE"] == pytest.approx(0.797672, abs=1e-5)
    assert a["INSPF"] == pytest.approx(0.610271, abs=1e-5)
    assert a["REPCO2"] == pytest.approx(0.593209, abs=1e-5)
    assert a["RENYR"] == pytest.approx(0.587492, abs=1e-5)
    assert a["PRESS"] == pytest.approx(0.492605, abs=1e-5)


# ---- 管龄分箱（§4.2）----

def test_age_bins(audit):
    b = audit["age_bins"]
    assert b["[0.0, 5.0)"]["count"] == 1709
    assert b["[0.0, 5.0)"]["positive"] == 14
    assert b["[5.0, 15.0)"]["count"] == 2106
    assert b["[15.0, 25.0)"]["count"] == 1494
    assert b["[25.0, 35.0)"]["count"] == 634
    assert b["[35.0, 45.0)"]["count"] == 467
    assert b["[35.0, 45.0)"]["rate"] == pytest.approx(0.115632, abs=1e-5)
    assert b["[45.0, 55.0)"]["count"] == 817
    assert b["[45.0, 55.0)"]["rate"] == pytest.approx(0.128519, abs=1e-5)
    assert b["[55.0, 65.0)"]["count"] == 0
    assert b["[65.0, inf)"]["count"] == 61
    assert b["[65.0, inf)"]["rate"] == pytest.approx(0.196721, abs=1e-5)


def test_age_ratio_is_not_order_of_magnitude(audit):
    """2.84% → 11.56% 约 4.1 倍，并非一个数量级（§4.2）。"""
    b = audit["age_bins"]
    ratio = b["[35.0, 45.0)"]["rate"] / b["[25.0, 35.0)"]["rate"]
    assert 4.0 < ratio < 4.2


# ---- 材料（§4.2）----

def test_material(audit):
    m = audit["material"]
    assert m["球墨铸铁"]["count"] == 4751 and m["球墨铸铁"]["positive"] == 50
    assert m["铸铁管"]["count"] == 1378 and m["铸铁管"]["positive"] == 167
    assert m["钢管"]["count"] == 1080 and m["钢管"]["positive"] == 22
    assert m["自应力管"]["count"] == 72 and m["自应力管"]["positive"] == 12


# ---- 管径（§4.3）----

def test_diameter(audit):
    d = audit["diameter"]
    assert d["gj_min"] == 300 and d["gj_max"] == 1600
    assert d["dn300_count"] == 3485
    assert d["dn300_positive"] == 119
    assert d["dn300_events"] == 129
    assert d["dn800_count"] == 1427
    assert d["dn800_positive"] == 27
    assert d["dn800_rate"] == pytest.approx(0.018921, abs=1e-6)


def test_diameter_no_out_of_range():
    """审查发现：真实管径恰为 300–1600，无越界值。"""
    d = run_full_audit()["diameter"]
    assert d["gj_min"] == 300 and d["gj_max"] == 1600


# ---- 月份（§4.3）----

def test_months(audit):
    assert audit["months"] == {1: 31, 2: 16, 3: 18, 4: 18, 5: 11, 6: 20,
                               7: 31, 8: 30, 9: 24, 10: 14, 11: 22, 12: 27}


# ---- 质量（§5.3）----

def test_missing(audit):
    assert audit["missing"] == {"JSNF": 1274, "SSQY": 7286, "QYBS": 4466, "SZDL": 1}


def test_constant_columns(audit):
    c = audit["constant_columns"]
    assert c["CL"] == ["Ductile Iron"]
    assert c["QSDW"] == ["禾川水务"]
    assert set(c["constant"]) == {"CL", "QSDW"}


def test_depth_correlation(audit):
    c = audit["depth_corr"]
    assert c[0][1] == pytest.approx(0.023826, abs=1e-6)   # BURDEP-QDMS
    assert c[0][2] == pytest.approx(0.007044, abs=1e-6)   # BURDEP-ZDMS
    assert c[1][2] == pytest.approx(0.772584, abs=1e-6)   # QDMS-ZDMS


def test_flow_ratio(audit):
    f = audit["flow_ratio"]
    assert f["median"] == pytest.approx(0.2777773, abs=1e-6)
    assert 0.2775 < f["min"] < f["max"] < 0.2781


def test_score_difference(audit):
    s = audit["score_difference"]
    assert s["n_gt_015"] == 85
    assert s["max_diff"] == pytest.approx(0.2, abs=1e-6)
    assert s["risk_ge_75_attribute"] == 1506
    assert s["scorecard_levels"]["高风险"] == 21
    assert s["scorecard_levels"]["较高风险"] == 813
    assert s["scorecard_levels"]["中风险"] == 4510
    assert s["scorecard_levels"]["低风险"] == 1944


def test_traffic_road_perfect_match(audit):
    assert audit["traffic_road"]["perfect_match"] is True


# ---- 拓扑（§6）----

def test_topology_numeric_id(audit):
    t = audit["topology"]["numeric_id"]
    assert t["nodes"] == 7318 and t["edges"] == 7288
    assert t["components"] == 112 and t["cycles"] == 82
    assert t["degree2"] == 5955 and t["largest_component"] == 5652


def test_topology_other_key_orders(audit):
    t = audit["topology"]
    assert t["raw_string"]["nodes"] == 7464 and t["raw_string"]["components"] == 236
    assert t["node_code"]["nodes"] == 1081 and t["node_code"]["cycles"] == 6238


def test_coordinates(audit):
    c = audit["coordinates"]
    assert c["x_min"] == pytest.approx(324683.610, abs=1e-3)
    assert c["x_max"] == pytest.approx(330154.543, abs=1e-3)
    assert c["y_min"] == pytest.approx(349284.432, abs=1e-3)
    assert c["y_max"] == pytest.approx(355100.090, abs=1e-3)
    assert c["conflict_nonzero"] == 156
    assert c["conflict_gt_001"] == 17
    assert c["conflict_gt_1"] == 14
    assert c["max_span"] == pytest.approx(84.31, abs=0.01)


def test_inspection_dates(audit):
    d = audit["inspection_dates"]
    assert d["min"].startswith("2024-03-26")
    assert d["max"].startswith("2024-06-13")


# ---- 审查发现（里程碑 §1.2）----

def test_age_ties_are_severe(pipes):
    """审查发现：管龄仅 48 个取值，最大并列组 765。"""
    vc = pipes["PIPEAGE"].value_counts()
    assert (vc > 1).sum() == 48
    assert int(vc.max()) == 765


def test_unknown_facility_absent_in_real_data(pipes):
    """审查发现：FAC 零缺失，未知设施只能人工构造。"""
    assert int(pipes["FAC"].isna().sum()) == 0


# ---- 泄漏防护（§3.4）----

def test_forbidden_columns_rejected(pipes):
    """训练矩阵不得含禁止字段（§3.4）。"""
    for layer in ["F0_minimal_baseline", "F1_base_environment"]:
        m = loader.feature_matrix(pipes, layer)
        loader.assert_no_forbidden_columns(m.columns)


def test_forbidden_column_detected():
    with pytest.raises(ValueError, match="禁止字段"):
        loader.assert_no_forbidden_columns(["PIPEAGE", "ACCID"])


def test_whitelist_layer_has_no_forbidden():
    wl = loader.load_whitelist()
    forbidden = set(wl["forbidden_fields"])
    for layer, cols in wl["layers"].items():
        assert not (set(cols) & forbidden), f"{layer} 含禁止字段"


def test_isolated_fields_not_in_any_layer():
    """隔离字段不进入任何特征层（§3.2）。"""
    wl = loader.load_whitelist()
    isolated = set(wl["isolated_fields"])
    for layer, cols in wl["layers"].items():
        assert not (set(cols) & isolated), f"{layer} 含隔离字段"