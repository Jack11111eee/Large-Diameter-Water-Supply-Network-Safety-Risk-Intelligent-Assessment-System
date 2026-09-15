"""全量拟合测试（§6.5、§2.3）。

守住两条：全量拟合在训练标签上的分数不得进入成绩表；
其参考包独立于折外回放 R，不混用。
"""

import ast
import json
from pathlib import Path

import pytest

from src.data import loader
from src.integration import fullfit, pipeline, release

FORBIDDEN_IN_FULLFIT = {"aggregate_all", "per_fold_metrics", "macro_mean"}

FAST_CONFIG = {
    "candidates_version": "test_fullfit",
    "selection": {"criterion": "inner_ap", "tolerance": 0.005},
    "candidates": [{
        "name": "logreg", "family": "linear", "complexity_rank": 0,
        "param_grid": [{"C": 1.0}],
    }],
}


@pytest.fixture(scope="module")
def small():
    pipes = loader.load_attributes().iloc[:1500].reset_index(drop=True)
    counts, _ = loader.make_labels(pipes, loader.load_events())
    return pipes, counts


@pytest.fixture(scope="module")
def fitted(small):
    pipes, counts = small
    return fullfit.fit_full(pipes, counts, candidates_config=FAST_CONFIG)


# ---- 模式与产物 ----

def test_predictions_are_marked_full_fit(fitted, small):
    pipes, _ = small
    rows = fullfit.build_full_fit_predictions(
        fitted, pipes, data_version="dv", run_id="ru")
    assert len(rows) == len(pipes)
    assert {r["prediction_mode"] for r in rows} == {"full_fit"}


def test_reference_bundle_is_separate_and_includes_training_pipes(fitted, small):
    """全量拟合 R 含训练管段，仅用于描述相对位置，不与折外回放混用（§5.3.1）。"""
    pipes, _ = small
    bundle = fullfit.build_full_fit_reference(
        fitted, pipes, data_version="dv", run_id="ru")
    assert bundle["prediction_mode"] == "full_fit"
    assert bundle["reference_id"] == "ru-fullfit"
    assert len(bundle["pipe_ids"]) == len(pipes)


def test_training_fingerprint_is_deterministic(small):
    pipes, counts = small
    a = fullfit.training_fingerprint(pipes, counts)
    b = fullfit.training_fingerprint(pipes, counts)
    assert a == b and len(a) == 16


def test_training_fingerprint_changes_with_labels(small):
    pipes, counts = small
    flipped = counts.copy()
    flipped.iloc[0] = 1 - flipped.iloc[0]
    assert fullfit.training_fingerprint(pipes, counts) != \
        fullfit.training_fingerprint(pipes, flipped)


def test_score_table_is_explicitly_absent(fitted, small):
    """成绩占位必须显式声明未计分（§6.5）。"""
    assert fullfit.NO_SCORE_TABLE["scored"] is False
    assert fullfit.NO_SCORE_TABLE["prediction_mode"] == "full_fit"
    assert "不得进入成绩表" in fullfit.NO_SCORE_TABLE["reason"]


def test_fullfit_module_cannot_reach_scoring_apis():
    """AST 守卫：全量拟合模块不得引用任何计分函数（§6.5）。"""
    src = Path(fullfit.__file__).read_text(encoding="utf-8")
    names = {n.id for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Name)}
    names |= {n.attr for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Attribute)}
    assert not (FORBIDDEN_IN_FULLFIT & names), \
        f"fullfit.py 引用了计分 API: {FORBIDDEN_IN_FULLFIT & names}"


# ---- 端到端：独立目录、无成绩表 ----

def test_build_full_fit_writes_separate_dir_without_score_table(tmp_path, monkeypatch, small):
    pipes, counts = small
    orig = fullfit.fit_full
    monkeypatch.setattr(loader, "load_attributes", lambda: pipes)
    monkeypatch.setattr(loader, "make_labels", lambda p, e: (counts, counts.gt(0)))
    monkeypatch.setattr(fullfit, "fit_full",
                        lambda *a, **k: orig(pipes, counts,
                                             candidates_config=FAST_CONFIG))

    out = tmp_path / "release_fullfit"
    result = pipeline.build_full_fit(out, structure="F1_base_environment")

    names = sorted(p.name for p in out.iterdir())
    assert names == ["full_fit_scores.json", "manifest.json",
                     "predictions.json", "reference_bundle.json"]
    assert out.name == fullfit.FULL_FIT_DIR_NAME
    assert result["training_fingerprint"]

    for path in out.iterdir():
        assert "macro_mean" not in path.read_text(encoding="utf-8")
    scores = json.loads((out / "full_fit_scores.json").read_text(encoding="utf-8"))
    assert scores["scored"] is False


# ---- run_id 覆盖模型规格（§13.3）----

def test_run_id_unchanged_for_legacy_call_shape():
    """省略 model_spec 时与 M1 口径逐字节一致。"""
    legacy = release._fingerprint(
        {"model": "B2_age_logreg", "seed": 20260914, "data": "dv"})[:16]
    assert release.run_id_of("B2_age_logreg", 20260914, "dv") == legacy


@pytest.mark.parametrize("spec", [
    {"structure": "F1_base_environment"},
    {"structure": "F1_base_environment+F2_operational_snapshot"},
    {"structure": "F1_base_environment", "mode": "full_fit"},
    {"structure": "F1_base_environment", "training_fingerprint": "abc"},
])
def test_run_id_changes_with_model_spec(spec):
    base = release.run_id_of("logreg", 20260914, "dv")
    assert release.run_id_of("logreg", 20260914, "dv", model_spec=spec) != base