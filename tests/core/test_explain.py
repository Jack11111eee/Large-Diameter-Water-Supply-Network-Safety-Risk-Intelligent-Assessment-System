"""SHAP 归因与解释产物测试（§7.1、§13.4）。

守住三条禁令：解释尺度不得写成概率百分点；
不得声称贡献加和等于最终概率；不得把归一化贡献称为事故原因占比。
"""

import numpy as np
import pandas as pd
import pytest

from src.contracts import validate_package
from src.contracts.versions import ExplainStatus, Scale
from src.data import loader
from src.evaluation import runner, split
from src.explain import artifacts
from src.explain.attribution import (
    background_sample,
    check_additivity,
    explain_model,
    explainable,
)
from src.models.baselines import B1AgeRank, LayerLogReg
from src.models.tree import HGBTClassifier

FORBIDDEN_CAUSE_WORDS = ["原因占比", "占比"]


@pytest.fixture(scope="module")
def data():
    pipes = loader.load_attributes()
    counts, _ = loader.make_labels(pipes, loader.load_events())
    return pipes, counts


@pytest.fixture(scope="module")
def subset(data):
    """小规模真实子集，让解释测试可快速跑完。"""
    pipes, counts = data
    return pipes.iloc[:1500].reset_index(drop=True), counts.iloc[:1500].reset_index(drop=True)


@pytest.fixture(scope="module")
def explained(subset):
    """真实子集上的折外预测 + 折状态 + 解释行。"""
    pipes, counts = subset
    cols = loader.resolve_layer("F1_base_environment")
    folds = split.make_outer_folds(
        pipes["ID"].astype(str).to_numpy(), counts.gt(0).astype(int).to_numpy())
    oof, state = runner.run_oof(
        pipes, counts, model_name="B2b_layer_logreg", layer_columns=cols,
        folds=folds, return_fold_state=True)
    rows = artifacts.build_explanation_rows(
        pipes, oof, state, data_version="dv-test", run_id="ru-test",
        seed=20260914, n_background=50)
    return pipes, counts, oof, state, rows


# ---- 加和核验（§7.1）----

@pytest.mark.parametrize("kind", ["tree", "linear"])
def test_additivity_holds_for_both_families(subset, kind):
    """贡献加和必须还原其所解释的原始输出。"""
    pipes, counts = subset
    y = counts.gt(0).astype(int).to_numpy()
    cols = loader.resolve_layer("F1_base_environment")
    X_tr = pipes.iloc[:1200].reset_index(drop=True)
    X_te = pipes.iloc[1200:].reset_index(drop=True)

    model = (HGBTClassifier(cols, max_leaf_nodes=15) if kind == "tree"
             else LayerLogReg(cols)).fit(X_tr, y[:1200])
    bg = background_sample(X_tr, n=50, seed=1)
    out = explain_model(model, bg, X_te)
    assert out["additivity_max_abs_error"] < 1e-6


def test_check_additivity_detects_mismatch():
    base, contrib = 0.0, np.array([[1.0, 2.0]])
    assert check_additivity(base, contrib, np.array([3.0])) < 1e-12
    assert check_additivity(base, contrib, np.array([99.0])) > 1.0


# ---- 背景样本只在训练折（§7.1）----

def test_background_sample_never_touches_test_fold():
    X_tr = pd.DataFrame({"PIPEAGE": np.arange(500, dtype=float)})
    X_te = pd.DataFrame({"PIPEAGE": np.arange(500, 600, dtype=float)})
    bg = background_sample(X_tr, n=50, seed=0)
    assert len(bg) == 50
    assert set(bg["PIPEAGE"]).issubset(set(X_tr["PIPEAGE"]))
    assert not (set(bg["PIPEAGE"]) & set(X_te["PIPEAGE"]))


def test_background_sample_is_deterministic():
    X = pd.DataFrame({"PIPEAGE": np.arange(500, dtype=float)})
    assert background_sample(X, n=40, seed=7).equals(
        background_sample(X, n=40, seed=7))


def test_explainable_flags_model_family(subset):
    pipes, counts = subset
    y = counts.gt(0).astype(int).to_numpy()
    X = pipes.iloc[:500].reset_index(drop=True)
    assert explainable(LayerLogReg(["PIPEAGE"]).fit(X, y[:500])) is True
    assert explainable(HGBTClassifier(["PIPEAGE"]).fit(X, y[:500])) is True
    assert explainable(B1AgeRank().fit(X, y[:500])) is False


# ---- 产物契约与尺度 ----

def test_explanation_rows_validate_against_contract(explained):
    rows = explained[4]
    validate_package("explanation", rows)
    assert len(rows) == 1500


def test_scale_is_log_odds_never_probability(explained):
    rows = explained[4]
    for r in rows:
        assert r["scale"] == Scale.LOG_ODDS.value
        assert r["scale"] != Scale.PROBABILITY.value


def test_attribution_sums_to_raw_score_not_final_probability(explained):
    """base + Σcontrib ≈ raw_score，但其 sigmoid 不等于最终校准概率（§7.1）。"""
    _, _, oof, _, rows = explained
    by_id = {r.pipe_id: r for r in oof.itertuples()}
    checked = 0
    for r in rows[:200]:
        if r["status"] != ExplainStatus.OK.value:
            continue
        total = r["base_value"] + sum(c["value"] for c in r["contributions"])
        # 截断后仅与省略量一起还原全量
        total += r["omitted_contribution_sum"]
        assert abs(total - r["raw_score"]) < 1e-6
        assert abs(1 / (1 + np.exp(-total)) - by_id[r["pipe_id"]].p) > 1e-9
        checked += 1
    assert checked > 0


def test_rows_flag_that_they_explain_raw_score_not_probability(explained):
    rows = explained[4]
    ok = [r for r in rows if r["status"] == ExplainStatus.OK.value]
    assert ok
    for r in ok:
        assert artifacts.NOT_FINAL_PROBABILITY_FLAG in r["quality_flags"]
        assert r["attribution_name"] == artifacts.ATTRIBUTION_NAME


def test_truncation_is_disclosed(explained):
    rows = explained[4]
    r = rows[0]
    assert len(r["contributions"]) == artifacts.DEFAULT_TOP_K
    assert r["n_contributions_total"] > artifacts.DEFAULT_TOP_K
    assert artifacts.TRUNCATED_FLAG in r["quality_flags"]
    assert r["omitted_contribution_sum"] != 0.0


def test_contributions_carry_field_and_group(explained):
    rows = explained[4]
    for c in rows[0]["contributions"]:
        assert c["field"] and c["group"] != "未分组"
        assert c["direction"] in {"positive", "negative"}


def test_no_cause_share_language(explained):
    """不得把归一化贡献称为事故原因占比（§7.1）。"""
    blob = str(explained[4])
    assert not [w for w in FORBIDDEN_CAUSE_WORDS if w in blob]


# ---- 不可解释时必须显式 unavailable（§13.5）----

def test_unavailable_rows_when_model_has_no_explainer(subset):
    """B1 无 est_/clf_，必须记 unavailable，不用别的模型顶替。"""
    pipes, counts = subset
    folds = split.make_outer_folds(
        pipes["ID"].astype(str).to_numpy(), counts.gt(0).astype(int).to_numpy())
    oof, state = runner.run_oof(pipes, counts, model_name="B1_age_rank",
                                folds=folds, return_fold_state=True)
    rows = artifacts.build_explanation_rows(
        pipes, oof, state, data_version="dv", run_id="ru", seed=20260914)
    assert len(rows) == len(pipes)
    for r in rows:
        assert r["status"] == ExplainStatus.UNAVAILABLE.value
        assert r["base_value"] is None and r["contributions"] == []
    validate_package("explanation", rows)


# ---- 特征宽度对齐（各校准折预处理不同）----

def test_alignment_handles_differing_feature_widths():
    outs = [
        {"base_value": 1.0, "feature_names": ["num:A", "cat:B=x"],
         "contributions": np.array([[1.0, 2.0]])},
        {"base_value": 3.0, "feature_names": ["num:A", "cat:B=y"],
         "contributions": np.array([[3.0, 4.0]])},
    ]
    names, mean = artifacts._align_contributions(outs)
    assert names == ["num:A", "cat:B=x", "cat:B=y"]
    assert np.allclose(mean, [[2.0, 1.0, 2.0]])
    # 对齐后仍满足加和：mean(base) + Σmean(contrib) == mean(raw)
    assert abs(np.mean([1.0, 3.0]) + mean.sum() - np.mean([4.0, 10.0])) < 1e-12