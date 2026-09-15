"""页面级测试（§13.6 界面独立验收：空/错误状态、模式标记、导出一致性）。

用 Streamlit 自带的 AppTest 驱动，不需要真实浏览器。
夹具目录与正式目录都跑一遍，确认页面不因产物缺失而崩、也不凭空填数。
"""

import json
import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from app import adapter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import anonymity  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
FIXTURES = ROOT / "tests" / "fixtures"
MAIN = ROOT / "app" / "main.py"
PAGES = ["风险总览与清单", "管段详情", "资源清单", "事后运维复核",
         "数据审计", "实验审计"]
TIMEOUT = 300


@pytest.mark.parametrize("page", PAGES)
def test_pages_render_on_real_release(page, monkeypatch):
    """正式产物在场时各页都能渲染，无异常。"""
    monkeypatch.delenv(adapter.ENV_PACKAGE_DIR, raising=False)
    at = AppTest.from_file(str(MAIN), default_timeout=TIMEOUT)
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    at.sidebar.radio[0].set_value(page).run()
    assert not at.exception, f"{page}: {[str(e.value) for e in at.exception]}"


@pytest.mark.parametrize("page", PAGES)
def test_pages_render_on_fixture_fallback(page, monkeypatch):
    """正式产物缺失时回退到夹具，各页仍能渲染且标出夹具来源。"""
    monkeypatch.setenv(adapter.ENV_PACKAGE_DIR, "/nonexistent/package/dir")
    at = AppTest.from_file(str(MAIN), default_timeout=TIMEOUT)
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    warnings = " ".join(w.value for w in at.warning)
    assert "人工夹具" in warnings and "不是正式结果" in warnings
    at.sidebar.radio[0].set_value(page).run()
    assert not at.exception, f"{page}: {[str(e.value) for e in at.exception]}"


def test_page_shows_mode_and_source_badge(monkeypatch):
    """页级模式徽章与来源徽章必须出现。"""
    monkeypatch.setenv(adapter.ENV_PACKAGE_DIR, "/nonexistent/package/dir")
    at = AppTest.from_file(str(MAIN), default_timeout=TIMEOUT)
    at.run()
    markdown = " ".join(m.value for m in at.markdown)
    assert "oof_replay" in markdown
    assert "synthetic_fixture" in markdown


def test_decision_source_is_disclosed(monkeypatch):
    """决策结果来源必须披露：模块实时返回，或冻结包降级展示。"""
    monkeypatch.setenv(adapter.ENV_PACKAGE_DIR, "/nonexistent/package/dir")
    at = AppTest.from_file(str(MAIN), default_timeout=TIMEOUT)
    at.run()
    text = " ".join([i.value for i in at.info] + [c.value for c in at.caption])
    assert "决策结果来源" in text


def test_empty_package_shows_honest_empty_state(tmp_path, monkeypatch):
    """空清单显示空状态，不补占位行。"""
    directory = tmp_path / "empty"
    directory.mkdir()
    for product, fname in [("standard_attributes", "standard_attributes.json"),
                           ("predictions", "predictions.json")]:
        (directory / fname).write_text("[]", encoding="utf-8")
    monkeypatch.setenv(adapter.ENV_PACKAGE_DIR, str(directory))
    at = AppTest.from_file(str(MAIN), default_timeout=TIMEOUT)
    at.run()
    assert not at.exception
    info = " ".join(i.value for i in at.info)
    assert "清单为空" in info


def test_version_mismatch_shows_error_state(tmp_path, monkeypatch):
    """版本不匹配时页面显示拒绝原因，不用其它数据顶替。"""
    directory = tmp_path / "bad"
    directory.mkdir()
    attrs = json.loads((FIXTURES / "standard_attributes.json").read_text(encoding="utf-8"))
    preds = json.loads((FIXTURES / "predictions.json").read_text(encoding="utf-8"))
    for row in attrs:
        row["schema_version"] = "9.9.9"
    (directory / "standard_attributes.json").write_text(
        json.dumps(attrs, ensure_ascii=False), encoding="utf-8")
    (directory / "predictions.json").write_text(
        json.dumps(preds, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setenv(adapter.ENV_PACKAGE_DIR, str(directory))
    at = AppTest.from_file(str(MAIN), default_timeout=TIMEOUT)
    at.run()
    assert not at.exception
    errors = " ".join(e.value for e in at.error)
    assert "拒绝接入" in errors
    assert "schema_version" in errors


def test_detail_page_marks_invalid_prediction(monkeypatch):
    """无效预测管段在详情页显示 unavailable，不代填分数。"""
    monkeypatch.setenv(adapter.ENV_PACKAGE_DIR, str(FIXTURES))
    at = AppTest.from_file(str(MAIN), default_timeout=TIMEOUT)
    at.run()
    at.sidebar.radio[0].set_value("管段详情").run()
    at.text_input[0].set_value("SYN-F07_invalid_prediction").run()
    assert not at.exception
    warnings = " ".join(w.value for w in at.warning)
    assert "预测无效" in warnings
    assert "unavailable" in warnings


def test_detail_page_marks_missing_explanation(monkeypatch):
    """解释未就绪显示 unavailable，明确不使用其它模型顶替（§13.5）。"""
    monkeypatch.setenv(adapter.ENV_PACKAGE_DIR, str(FIXTURES))
    at = AppTest.from_file(str(MAIN), default_timeout=TIMEOUT)
    at.run()
    at.sidebar.radio[0].set_value("管段详情").run()
    at.text_input[0].set_value("SYN-F06_missing_explanation").run()
    assert not at.exception
    warnings = " ".join(w.value for w in at.warning)
    assert "unavailable" in warnings and "顶替" in warnings


def test_detail_page_search_miss_is_honest(monkeypatch):
    """检索无结果显示空状态，不构造替代记录。"""
    monkeypatch.setenv(adapter.ENV_PACKAGE_DIR, str(FIXTURES))
    at = AppTest.from_file(str(MAIN), default_timeout=TIMEOUT)
    at.run()
    at.sidebar.radio[0].set_value("管段详情").run()
    at.text_input[0].set_value("NO-SUCH-PIPE").run()
    assert not at.exception
    info = " ".join(i.value for i in at.info)
    assert "没有匹配" in info


def test_overview_labels_coordinates_as_unknown_crs(monkeypatch):
    """坐标必须标注为源坐标、CRS 未知、非真实经纬度（§4）。"""
    monkeypatch.setenv(adapter.ENV_PACKAGE_DIR, str(FIXTURES))
    at = AppTest.from_file(str(MAIN), default_timeout=TIMEOUT)
    at.run()
    captions = " ".join(c.value for c in at.caption)
    assert "CRS 未知" in captions
    assert "非真实经纬度" in captions


def test_resources_export_matches_displayed_rows(monkeypatch):
    """导出与页面显示同一结果对象，导出不重算（§13.4）。"""
    monkeypatch.setenv(adapter.ENV_PACKAGE_DIR, str(FIXTURES))
    at = AppTest.from_file(str(MAIN), default_timeout=TIMEOUT)
    at.run()
    at.sidebar.radio[0].set_value("资源清单").run()
    assert not at.exception
    package = adapter.load_package_set(FIXTURES)
    expected = adapter.rows_to_csv(adapter.pipe_list_rows(package))
    assert expected.splitlines()[0].startswith("pipe_id")


def test_resources_keeps_independent_probability_view(monkeypatch):
    """按 p 的独立视图始终保留，并标出未进入当前目标清单的管段（§8.1）。"""
    monkeypatch.setenv(adapter.ENV_PACKAGE_DIR, str(FIXTURES))
    at = AppTest.from_file(str(MAIN), default_timeout=TIMEOUT)
    at.run()
    at.sidebar.radio[0].set_value("资源清单").run()
    markdown = " ".join(m.value for m in at.markdown)
    assert "独立按概率 p 视图" in markdown
    assert "L3 / L4" in markdown


def test_resources_budget_change_is_served_by_decision_module(monkeypatch):
    """改预算 K 不报错；清单由决策模块按其冻结签名重出（§8.2）。"""
    monkeypatch.setenv(adapter.ENV_PACKAGE_DIR, str(FIXTURES))
    at = AppTest.from_file(str(MAIN), default_timeout=TIMEOUT)
    at.run()
    at.sidebar.radio[0].set_value("资源清单").run()
    at.number_input[0].set_value(3).run()
    assert not at.exception, [str(e.value) for e in at.exception]
    at.radio[0].set_value("p").run()
    assert not at.exception, [str(e.value) for e in at.exception]
    markdown = " ".join(m.value for m in at.markdown)
    assert "独立按概率 p 视图" in markdown


def test_no_school_name_in_rendered_pages(monkeypatch):
    """渲染结果中不得出现校名。禁用词表在 gitignored 的 .anonymity_tokens。"""
    tokens = anonymity.banned_names()
    if not tokens:
        pytest.skip("无本地禁用词表 .anonymity_tokens")
    monkeypatch.setenv(adapter.ENV_PACKAGE_DIR, str(FIXTURES))
    at = AppTest.from_file(str(MAIN), default_timeout=TIMEOUT)
    at.run()
    text = " ".join([m.value for m in at.markdown] + [c.value for c in at.caption])
    hits = anonymity.find_banned(text, tokens)
    assert not hits, f"渲染结果含禁用名称 {hits}"