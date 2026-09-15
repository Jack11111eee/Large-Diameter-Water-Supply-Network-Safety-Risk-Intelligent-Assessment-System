"""集成测试：M0 出口判定（里程碑 §5.6、§13.6）。

判据：B/C 能在独立进程、不读两个原始 XLSX、不导入 src/models/ 的条件下，
仅凭夹具与契约测试跑通各自模块。
"""

import ast
import json
from pathlib import Path

import pytest

from src.contracts import PRODUCTS, validate_package
from src.integration import pipeline

# 依赖真实 7,288 条数据或完整发布构建，耗时较长（见 pytest.ini）
pytestmark = pytest.mark.slow

ROOT = Path(__file__).resolve().parent.parent.parent
FIXTURE_DIR = ROOT / "tests" / "fixtures"


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    out = tmp_path_factory.mktemp("release")
    result = pipeline.build_all(out)
    return out, result


# --------------------------------------------------------------------------
# 端到端发布（§13.6）
# --------------------------------------------------------------------------

def test_pipeline_produces_all_artifacts(built):
    out, _ = built
    for name in ["standard_attributes.json", "geometry.json", "predictions.json",
                 "reference_bundle.json", "manifest.json", "evaluation.json"]:
        assert (out / name).exists(), f"缺少 {name}"


def test_predictions_cover_all_pipes(built):
    out, result = built
    preds = json.loads((out / "predictions.json").read_text(encoding="utf-8"))
    assert len(preds) == 7288
    assert result["n_pipes"] == 7288


def test_published_packages_contain_no_ground_truth(built):
    """普通发布包不含 y_true（§13.4）。"""
    out, _ = built
    for name in ["standard_attributes.json", "geometry.json", "predictions.json"]:
        rows = json.loads((out / name).read_text(encoding="utf-8"))
        for r in rows:
            assert "y_true" not in r
            assert "ACCID" not in json.dumps(r)


def test_published_packages_validate(built):
    out, _ = built
    for product, fname in [("standard_attributes", "standard_attributes.json"),
                           ("geometry", "geometry.json"),
                           ("predictions", "predictions.json"),
                           ("reference_bundle", "reference_bundle.json")]:
        rows = json.loads((out / fname).read_text(encoding="utf-8"))
        validate_package(product, rows)


def test_manifest_lists_versions(built):
    """发布包列出数据、模型、参考分布、配置的版本（§13.6）。"""
    out, _ = built
    m = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert m["data_version"] and m["run_id"] and m["model_id"]
    assert set(m["artifacts"]) >= {"standard_attributes", "geometry",
                                   "predictions", "reference_bundle"}
    for name, entry in m["artifacts"].items():
        assert entry["fingerprint"] and entry["n_rows"] > 0


def test_all_artifacts_share_one_data_version(built):
    """同一发布集合内所有产物的 data_version 必须一致（§13.3）。

    回归：参考分布包曾误用 round_id（'seed20260914'）作 data_version，
    使版本校验把合法产物判为不兼容。
    """
    out, result = built
    versions = {}
    for product, fname in [("standard_attributes", "standard_attributes.json"),
                           ("geometry", "geometry.json"),
                           ("predictions", "predictions.json"),
                           ("reference_bundle", "reference_bundle.json")]:
        rows = json.loads((out / fname).read_text(encoding="utf-8"))
        versions[product] = {r["data_version"] for r in rows}
    for product, seen in versions.items():
        assert seen == {result["data_version"]}, (
            f"{product} 的 data_version={seen} 与发布版本 {result['data_version']!r} 不一致")
    m = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert m["data_version"] == result["data_version"]


def test_tables_align_by_pipe_id(built):
    """各表按 pipe_id 验证，而非按行位置拼接（§13.6）。"""
    out, _ = built
    std = json.loads((out / "standard_attributes.json").read_text(encoding="utf-8"))
    geom = json.loads((out / "geometry.json").read_text(encoding="utf-8"))
    preds = json.loads((out / "predictions.json").read_text(encoding="utf-8"))
    s = {r["pipe_id"] for r in std}
    g = {r["pipe_id"] for r in geom}
    p = {r["pipe_id"] for r in preds}
    assert s == g == p
    assert len(s) == 7288


def test_release_is_deterministic(built, tmp_path):
    """同输入两次构建产出相同指纹。"""
    out, result = built
    result2 = pipeline.build_all(tmp_path / "again")
    assert result["data_version"] == result2["data_version"]
    assert result["run_id"] == result2["run_id"]
    m1 = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    m2 = json.loads((tmp_path / "again" / "manifest.json").read_text(encoding="utf-8"))
    assert m1["artifacts"] == m2["artifacts"]


# --------------------------------------------------------------------------
# 参考分布（§5.3.1）
# --------------------------------------------------------------------------

def test_reference_bundle_has_required_fields(built):
    out, _ = built
    b = json.loads((out / "reference_bundle.json").read_text(encoding="utf-8"))[0]
    for k in ["reference_id", "model_run_id", "grade_config_version",
              "pipe_ids", "scores", "data_fingerprint", "prediction_mode"]:
        assert k in b, f"参考包缺少 {k}"
    assert len(b["scores"]) == 7288
    assert b["prediction_mode"] == "oof_replay"


def test_reference_scores_are_full_precision(built):
    """使用保存的完整精度分数，不先四舍五入（§5.3.1）。"""
    out, _ = built
    b = json.loads((out / "reference_bundle.json").read_text(encoding="utf-8"))[0]
    # 完整精度：多数分数应有多位小数
    multi = sum(1 for x in b["scores"] if len(repr(x).split(".")[-1]) > 3)
    assert multi > len(b["scores"]) * 0.9


# --------------------------------------------------------------------------
# 评测成绩分列（§6.4.1）
# --------------------------------------------------------------------------

def test_evaluation_has_four_buckets(built):
    out, result = built
    ev = result["evaluation"]
    assert set(ev) == {"per_fold", "macro_mean", "sample_weighted",
                       "pooled_oof_replay"}
    assert len(ev["per_fold"]) == 5


def test_per_fold_and_pooled_differ(built):
    """逐折与拼接成绩必须分列，不能混写（§6.4.1）。"""
    _, result = built
    ev = result["evaluation"]
    per_fold_auc = ev["macro_mean"]["roc_auc"]["value"]
    pooled_auc = ev["pooled_oof_replay"]["roc_auc"]
    assert per_fold_auc != pooled_auc


def test_no_fold_silently_dropped(built):
    _, result = built
    for metric, rec in result["evaluation"]["macro_mean"].items():
        if isinstance(rec, dict) and "total_folds" in rec:
            assert rec["valid_folds"] == rec["total_folds"], f"{metric} 有折被丢弃"
            assert rec["complete"] is True


# --------------------------------------------------------------------------
# 离线独立性（§13.3、里程碑 §5.6）
# --------------------------------------------------------------------------

def test_fixtures_readable_without_raw_data():
    """B/C 仅凭夹具即可读取，不需要两个原始 XLSX。"""
    for name in ["standard_attributes.json", "geometry.json", "predictions.json",
                 "explanation.json", "decision.json", "reference_bundle.json"]:
        rows = json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))
        assert rows


def test_fixture_consumers_do_not_import_models():
    """夹具消费者路径不导入 src/models/ 或 src/data/（§13.3）。"""
    src = (FIXTURE_DIR / "build_fixtures.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert not any(m.startswith("src.models") for m in imported)
    assert not any(m.startswith("src.data") for m in imported)


def test_contracts_module_has_no_heavy_deps():
    """契约层不依赖 sklearn/pandas，B/C 可轻量导入（§13.3）。"""
    for f in ["schema.py", "versions.py"]:
        src = (ROOT / "src" / "contracts" / f).read_text(encoding="utf-8")
        assert "import sklearn" not in src
        assert "import pandas" not in src


def test_all_product_consumers_documented():
    """§13.4 契约表：每个产物的消费者明确。"""
    for name, spec in PRODUCTS.items():
        assert spec.producer in {"A", "B", "C"}
        assert spec.consumers
