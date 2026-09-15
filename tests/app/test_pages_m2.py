"""M2 审计页测试（§9、§13.6、§6.4.1）。

两条纪律：审计页只读发布清单与成绩产物，不重算、不读原始表格；
实验包与发布包不同源时**拒绝并列展示**，不用发布成绩顶替。
"""

import json
import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from app import adapter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import anonymity  # noqa: E402

# 依赖真实 7,288 条数据或完整发布构建，耗时较长（见 pytest.ini）
pytestmark = pytest.mark.slow

ROOT = Path(__file__).resolve().parent.parent.parent
MAIN = ROOT / "app" / "main.py"
VIEWS = ROOT / "app" / "views"
TIMEOUT = 300

DATA_AUDIT = "数据审计"
EXPERIMENT_AUDIT = "实验审计"
AUDIT_PAGES = (DATA_AUDIT, EXPERIMENT_AUDIT)


def _open(page):
    at = AppTest.from_file(str(MAIN), default_timeout=TIMEOUT)
    at.run()
    at.sidebar.radio[0].set_value(page).run()
    return at


def _text(at):
    """页面上的全部可读文本，含表格单元格——表格内容也是展示的一部分。"""
    parts = [el.value for el in at.caption]
    parts += [el.value for el in at.info]
    parts += [el.value for el in at.warning]
    parts += [el.value for el in at.error]
    parts += [el.value for el in at.markdown]
    for frame in at.dataframe:
        parts.append(frame.value.to_string())
    return "\n".join(str(p) for p in parts)


@pytest.fixture(scope="module")
def package():
    return adapter.load_package_set()


@pytest.fixture(scope="module")
def grouped_release(tmp_path_factory):
    """分组补充协议的发布包，用于验证分组诊断的展示（§6.2）。"""
    from src.integration import pipeline

    out = tmp_path_factory.mktemp("audit_grouped")
    pipeline.build_grouped(out)
    return out


# ---- 页面能渲染 ----

@pytest.mark.parametrize("page", AUDIT_PAGES)
def test_audit_pages_render_on_real_release(page):
    at = _open(page)
    assert not at.exception, [str(e.value) for e in at.exception]
    assert at.dataframe


@pytest.mark.parametrize("page", AUDIT_PAGES)
def test_audit_pages_render_on_fixture_fallback(page, monkeypatch):
    """夹具包缺清单与成绩：两页显示诚实空态，不崩、不填数。"""
    monkeypatch.setenv(adapter.ENV_PACKAGE_DIR, "/nonexistent/package/dir")
    at = _open(page)
    assert not at.exception, [str(e.value) for e in at.exception]


# ---- 数据审计 ----

def test_data_audit_shows_the_manifest_sections():
    text = _text(_open(DATA_AUDIT))
    for section in ("数据文件指纹", "配置版本", "划分与分组", "产物行数与指纹"):
        assert section in text


def test_data_audit_discloses_the_grouping_rule(monkeypatch, grouped_release):
    """分组协议下缺失分组值的处理与「只作分组」必须在页面可见（§6.2、§3.2）。"""
    monkeypatch.setenv(adapter.ENV_PACKAGE_DIR, str(grouped_release))
    at = _open(DATA_AUDIT)
    assert not at.exception, [str(e.value) for e in at.exception]
    text = _text(at)
    assert "缺失分组值的管段数" in text
    assert "单例组" in text
    assert "永不进训练矩阵" in text
    assert "分组协议衡量对分组迁移的敏感性" in text


def test_data_audit_reports_no_grouping_for_the_random_scheme():
    """随机主协议没有分组段：页面不伪造分组读数。"""
    text = _text(_open(DATA_AUDIT))
    assert "缺失分组值的管段数" not in text


def test_data_audit_does_not_read_raw_files():
    src = (VIEWS / "data_audit.py").read_text(encoding="utf-8")
    for reader in ("read_excel", "read_csv", "import pandas", "src.data"):
        assert reader not in src, f"data_audit.py 出现了原始数据读取 {reader!r}"


# ---- 实验审计：发布成绩 ----

def test_experiment_audit_shows_release_scores(package):
    at = _open(EXPERIMENT_AUDIT)
    text = _text(at)
    assert "发布成绩" in text
    assert "逐折成绩" in text
    assert "宏平均" in text
    assert package.report.run_id in text


def test_experiment_audit_labels_pooled_as_auxiliary():
    """pooled 单独成块并标注不替代逐折主成绩（§6.4.1）。"""
    text = _text(_open(EXPERIMENT_AUDIT))
    assert "不替代逐折主成绩" in text
    assert "全网回放" in text


def test_experiment_audit_reports_no_dropped_folds(package):
    """真实发布包没有折被丢弃，页面应如实这么说（§6.4.1）。"""
    text = _text(_open(EXPERIMENT_AUDIT))
    assert "没有折被静默丢弃" in text


# ---- 实验审计：实验包 ----

def test_experiment_audit_shows_the_experiment_bundle(package):
    if not package.experiments.available:
        pytest.skip("本地没有实验产物包")
    text = _text(_open(EXPERIMENT_AUDIT))
    assert "候选选择日志" in text
    assert "校准对照" in text
    assert "结构消融" in text
    assert package.experiments.run_id in text


def test_experiment_audit_shows_mandatory_disclosures(package):
    if not package.experiments.available:
        pytest.skip("本地没有实验产物包")
    text = _text(_open(EXPERIMENT_AUDIT))
    assert "不以任何 AUC 上限解释差异" in text
    assert "天花板" not in text


def test_experiment_audit_shows_calibration_conclusion(package):
    if not package.experiments.available:
        pytest.skip("本地没有实验产物包")
    text = _text(_open(EXPERIMENT_AUDIT))
    assert "不构成校准更好的证据" in text


def test_missing_experiment_bundle_is_honest(monkeypatch, tmp_path):
    """实验包缺失是正常状态：如实说明，不用发布成绩顶替（§13.3）。"""
    monkeypatch.setenv(adapter.ENV_EXPERIMENT_DIR, str(tmp_path / "absent"))
    at = _open(EXPERIMENT_AUDIT)
    assert not at.exception, [str(e.value) for e in at.exception]
    text = _text(at)
    assert "实验包不可用" in text
    assert "不影响发布包" in text


def test_incomplete_experiment_bundle_is_rejected(monkeypatch, tmp_path):
    """实验包不完整时整包标不可用，不按部分产物拼读。"""
    partial = tmp_path / "partial"
    partial.mkdir()
    (partial / "manifest.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv(adapter.ENV_EXPERIMENT_DIR, str(partial))
    bundle = adapter.load_experiments()
    assert bundle.available is False
    assert "不完整" in bundle.reason


def test_experiment_bundle_from_another_source_is_refused(monkeypatch, tmp_path):
    """不同源的实验包不得与发布成绩并列展示（§13.3）。"""
    other = tmp_path / "other"
    other.mkdir()
    payloads = {
        "manifest.json": {"run_id": "x", "data_version": "not-the-release",
                          "structure": "F1_base_environment", "split": {}},
        "selection_log.json": {}, "calibration_report.json": {}, "ablation.json": {},
    }
    for name, payload in payloads.items():
        (other / name).write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setenv(adapter.ENV_EXPERIMENT_DIR, str(other))

    at = _open(EXPERIMENT_AUDIT)
    assert not at.exception, [str(e.value) for e in at.exception]
    text = _text(at)
    assert "不同源" in text
    assert "拒绝并列展示" in text


def test_experiment_audit_does_not_compute_metrics():
    """审计页只展示已给出的读数，不自行计算指标（§13.1）。"""
    src = (VIEWS / "experiment_audit.py").read_text(encoding="utf-8")
    for forbidden in ("roc_auc", "average_precision", "sklearn", "import numpy"):
        assert forbidden not in src, f"experiment_audit.py 出现了指标计算 {forbidden!r}"


def test_audit_pages_have_no_absolute_paths_or_school_name():
    tokens = anonymity.banned_names()
    for name in ("data_audit", "experiment_audit"):
        text = (VIEWS / f"{name}.py").read_text(encoding="utf-8")
        assert not anonymity.find_absolute_paths(text)
        if tokens:
            assert not anonymity.find_banned(text, tokens)
