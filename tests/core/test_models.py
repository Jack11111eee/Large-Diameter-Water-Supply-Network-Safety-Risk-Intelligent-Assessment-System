"""模型与折外预测测试（§6.1、§3.4、里程碑 §5.3）。

真实数据测试较慢，标记 slow。
"""

import numpy as np
import pytest

from src.data import loader
from src.evaluation import runner, selection, split
from src.models.calibration import SigmoidCalibrator, needs_calibration


@pytest.fixture(scope="module")
def data():
    pipes = loader.load_attributes()
    counts, _ = loader.make_labels(pipes, loader.load_events())
    return pipes, counts


# --------------------------------------------------------------------------
# 划分（§6.1）
# --------------------------------------------------------------------------

def test_split_deterministic(data):
    pipes, counts = data
    t1 = split.split_table(pipes, counts)
    t2 = split.split_table(pipes, counts)
    assert split.table_fingerprint(t1) == split.table_fingerprint(t2)


def test_split_seed_and_folds(data):
    pipes, counts = data
    t = split.split_table(pipes, counts)
    assert split.SEED == 20260914
    assert split.N_OUTER == 5
    assert sorted(t.outer_fold.unique()) == [0, 1, 2, 3, 4]


def test_split_one_pipe_one_fold(data):
    """同一管段只出现在一个外层测试折（§3.4）。"""
    pipes, counts = data
    t = split.split_table(pipes, counts)
    assert t.pipe_id.is_unique
    assert len(t) == 7288


def test_split_no_degenerate_fold(data):
    """审查发现：5 折协议在当前数据上无退化折。"""
    pipes, counts = data
    t = split.split_table(pipes, counts)
    s = split.fold_summary(t)
    for f, rec in s.items():
        assert rec["n_positive"] >= 40, f"折 {f} 正例过少: {rec}"


def test_split_covers_all_pipes(data):
    pipes, counts = data
    t = split.split_table(pipes, counts)
    assert set(t.pipe_id) == set(pipes["ID"].astype(str))


# --------------------------------------------------------------------------
# 折外预测（§6.1 步骤 3）
# --------------------------------------------------------------------------

@pytest.mark.parametrize("model", ["B0_baserate", "B1_age_rank", "B2_age_logreg"])
def test_oof_unique_and_complete(data, model):
    """每轮 7,288 条唯一折外预测。"""
    pipes, counts = data
    df = runner.run_oof(pipes, counts, model_name=model)
    assert len(df) == 7288
    assert df.pipe_id.is_unique


@pytest.mark.parametrize("model", ["B0_baserate", "B1_age_rank", "B2_age_logreg"])
def test_oof_probability_in_unit_interval(data, model):
    pipes, counts = data
    df = runner.run_oof(pipes, counts, model_name=model)
    assert np.isfinite(df.p).all()
    assert (df.p >= 0).all() and (df.p <= 1).all()


def test_oof_labels_never_in_features(data):
    """折外预测表不含标签泄漏字段。"""
    pipes, counts = data
    df = runner.run_oof(pipes, counts, model_name="B2_age_logreg")
    assert "ACCID" not in df.columns
    assert "RISK" not in df.columns


def test_oof_fold_matches_split(data):
    """折外预测的折号与固定划分一致。"""
    pipes, counts = data
    t = split.split_table(pipes, counts)
    df = runner.run_oof(pipes, counts, model_name="B2_age_logreg")
    merged = df.merge(t[["pipe_id", "outer_fold"]], on="pipe_id",
                      suffixes=("", "_split"))
    assert (merged.outer_fold == merged.outer_fold_split).all()


def test_age_baseline_matches_descriptive_auc(data):
    """B1 折外 AUC 应与数据报告的描述性管龄 AUC 接近（§4.1）。

    描述性 AUC 0.797672 是全表排序；折外是嵌套协议，不应强求相等，
    但量级必须一致，且不得出现异常高分。
    """
    from src.evaluation import metrics
    pipes, counts = data
    df = runner.run_oof(pipes, counts, model_name="B1_age_rank")
    agg = metrics.aggregate_all(df.y_true, df.p, df.pipe_id, df.outer_fold, q_list=())
    auc = agg["macro_mean"]["roc_auc"]["value"]
    assert 0.75 < auc < 0.85, f"B1 折外 AUC 异常: {auc}"


def test_no_abnormal_high_score(data):
    """模型出现异常高分时要重新检查泄漏，不能只庆祝指标（§3.2）。"""
    from src.evaluation import metrics
    pipes, counts = data
    df = runner.run_oof(pipes, counts, model_name="B2_age_logreg")
    agg = metrics.aggregate_all(df.y_true, df.p, df.pipe_id, df.outer_fold, q_list=())
    auc = agg["macro_mean"]["roc_auc"]["value"]
    # 官方评分 AUC 0.876 含泄漏；纯属性模型不应达到该水平
    assert auc < 0.87, f"折外 AUC {auc} 接近泄漏水平，需检查特征"


# --------------------------------------------------------------------------
# 校准（§5.3）
# --------------------------------------------------------------------------

def test_b0_needs_no_calibration():
    assert needs_calibration("B0_baserate") is False
    assert needs_calibration("B2_age_logreg") is True


def test_sigmoid_calibrator_fits():
    rng = np.random.default_rng(0)
    s = rng.normal(size=500)
    y = (s + rng.normal(size=500) > 0.5).astype(int)
    cal = SigmoidCalibrator().fit(s, y)
    p = cal.predict(s)
    assert np.isfinite(p).all()
    assert (p >= 0).all() and (p <= 1).all()


def test_b0_output_is_constant(data):
    """B0 输出训练折基率常数。"""
    pipes, counts = data
    df = runner.run_oof(pipes, counts, model_name="B0_baserate")
    # 各折基率略有差异，但折内应为常数
    for f, g in df.groupby("outer_fold"):
        assert g.p.nunique() == 1


# --------------------------------------------------------------------------
# 泄漏防护（§3.4）
# --------------------------------------------------------------------------

def test_changing_test_fold_labels_does_not_change_predictions(data):
    """改变一个测试折的标签，不得改变该折的预测（§3.4）。

    此测试固定已保存的外层划分，不重新执行依赖标签的分层划分。
    """
    pipes, counts = data
    t = split.split_table(pipes, counts)
    fixed = {int(f): g.index.to_numpy()
             for f, g in t.groupby("outer_fold")}
    base = runner.run_oof(pipes, counts, model_name="B2_age_logreg", folds=fixed)

    # 翻转折 0 的标签，划分保持不变
    counts2 = counts.copy()
    fold0_ids = t[t.outer_fold == 0].pipe_id.astype(int).to_numpy()
    idx = pipes.index[pipes["ID"].isin(fold0_ids)]
    counts2.loc[idx] = 1 - counts2.loc[idx]

    alt = runner.run_oof(pipes, counts2, model_name="B2_age_logreg", folds=fixed)
    m = base.merge(alt, on="pipe_id", suffixes=("_a", "_b"))
    f0 = m[m.outer_fold_a == 0]
    # 折 0 的预测由其余折训练，翻转折 0 标签不应改变折 0 预测
    assert np.allclose(f0.p_a, f0.p_b, atol=1e-12)
    assert len(f0) > 0


def test_preprocessor_fit_only_on_train():
    """预处理只在训练折拟合（§3.3 规则 2）。"""
    import pandas as pd
    from src.data.preprocess import FoldPreprocessor
    Xtr = pd.DataFrame({"PIPEAGE": [1.0, 2.0, 3.0], "CZ": ["a", "b", "a"]})
    Xte = pd.DataFrame({"PIPEAGE": [999.0], "CZ": ["zzz"]})
    prep = FoldPreprocessor(["PIPEAGE"], ["CZ"], min_category_count=1)
    prep.fit(Xtr)
    # 测试折的极端值不应改变已拟合的中位数
    assert prep._medians["PIPEAGE"] == 2.0
    out = prep.transform(Xte)
    assert np.isfinite(out).all()


# --------------------------------------------------------------------------
# 树候选（§5.1）
# --------------------------------------------------------------------------

TREE_CONFIG = {
    "candidates_version": "test_tree",
    "selection": {"criterion": "inner_ap", "tolerance": 0.005},
    "candidates": [{
        "name": "hgb", "family": "tree", "complexity_rank": 1,
        "param_grid": [{"max_leaf_nodes": 15, "learning_rate": 0.06}],
    }],
}


@pytest.fixture(scope="module")
def tree_oof(data):
    pipes, counts = data
    cands = selection.candidates_from_config(
        loader.resolve_layer("F1_base_environment"), config=TREE_CONFIG)
    folds = split.make_outer_folds(
        pipes["ID"].astype(str).to_numpy(),
        counts.gt(0).astype(int).to_numpy())
    df, _ = selection.select_and_run(pipes, counts, candidates=cands, folds=folds)
    return df


def test_tree_candidate_oof_is_complete(tree_oof):
    assert len(tree_oof) == 7288
    assert tree_oof.pipe_id.is_unique
    assert np.isfinite(tree_oof.p).all()
    assert (tree_oof.p >= 0).all() and (tree_oof.p <= 1).all()


def test_tree_candidate_sanity_bound(tree_oof):
    """树候选折外 AUC 必须远离泄漏水平（§3.2）。

    0.87 是官方评分含标签的泄漏对照线，**不是**性能上限：
    数据报告 §8 已撤回「AUC 0.80 天花板」，此处不设任何上限断言。
    """
    from src.evaluation import metrics
    agg = metrics.aggregate_all(tree_oof.y_true, tree_oof.p, tree_oof.pipe_id,
                                tree_oof.outer_fold, q_list=())
    auc = agg["macro_mean"]["roc_auc"]["value"]
    assert 0.5 < auc < 0.87, f"树候选折外 AUC 异常: {auc}"


def test_tree_raw_score_is_log_odds(data):
    """树候选原始输出必须是 log-odds，不能是概率（§7.1）。"""
    from src.models.calibration import scores_fn
    from src.models.tree import HGBTClassifier
    pipes, counts = data
    y = counts.gt(0).astype(int).to_numpy()
    X = pipes.iloc[:2000].reset_index(drop=True)
    model = HGBTClassifier(["PIPEAGE", "CZ"]).fit(X, y[:2000])
    raw = scores_fn(model, X)
    assert not np.allclose(raw, model.predict_proba(X))
    assert (raw < -1).any() or (raw > 1).any()