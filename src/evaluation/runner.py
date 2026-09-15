"""折外预测回放（§6.1、§2.3）。

对每根管段展示未用其标签训练的预测。
外层测试折仅作评估；所有预处理、选择、校准都在外层训练部分内完成。
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.evaluation.split import SEED, N_INNER, inner_splits
from src.models.baselines import B0BaseRate, B1AgeRank, B2AgeLogReg
from src.models.calibration import (
    SigmoidCalibrator,
    needs_calibration,
    scores_fn,
)


def _fit_fn(model_factory):
    return lambda X, y: model_factory().fit(X, y)


def _select_within_train(X_train, y_train, candidates, seed=SEED):
    """内层 3 折选择（§6.1 步骤 1、§5.2）。

    主选择依据为内层 AP。候选差异较小时优先简单模型。
    选择只使用外层训练集内部信息。
    """
    from src.evaluation.metrics import ap, NotApplicable

    skf = StratifiedKFold(n_splits=N_INNER, shuffle=True, random_state=seed)
    scores = {name: [] for name in candidates}
    for tr, va in skf.split(np.zeros(len(y_train)), y_train):
        for name, factory in candidates.items():
            m = factory().fit(X_train.iloc[tr], y_train[tr])
            p = m.predict_proba(X_train.iloc[va])
            try:
                scores[name].append(ap(y_train[va], p))
            except NotApplicable:
                continue
    means = {n: (float(np.mean(v)) if v else -np.inf) for n, v in scores.items()}
    # 容差 0.005：差异较小时按预定义顺序优先简单模型
    best = max(means.values())
    for name in candidates:  # 预定义顺序即简单度顺序
        if means[name] >= best - 0.005:
            return name, means
    return max(means, key=means.get), means


def _fit_predict_fold(X_tr, y_tr, X_te, factory, *, calibrate, seed, groups_tr=None):
    """单外层折的拟合与预测（§6.1 步骤 2、§6.2）。

    校准折与外层训练集共享分组边界；内层退化时退回未校准对照并标记，
    绝不随机拆散分组补齐（§6.1）。
    返回 (校准概率, 原始分数, 折状态)。
    """
    state = {"models": [], "calibrators": [], "quality_flags": [], "calibrated": False}

    if not calibrate:
        model = factory().fit(X_tr, y_tr)
        state["models"] = [model]
        return model.predict_proba(X_te), scores_fn(model, X_te), state

    splits, status = inner_splits(y_tr, groups=groups_tr, seed=seed)
    if status == "degenerate":
        model = factory().fit(X_tr, y_tr)
        state["models"] = [model]
        state["quality_flags"] = ["calibration_unavailable_degenerate_groups"]
        return model.predict_proba(X_te), scores_fn(model, X_te), state

    if status == "reduced_folds":
        state["quality_flags"] = ["inner_folds_reduced"]

    models, calibrators = [], []
    for tr, cal in splits:
        mm = factory().fit(X_tr.iloc[tr], y_tr[tr])
        s = scores_fn(mm, X_tr.iloc[cal])
        cc = SigmoidCalibrator().fit(s, y_tr[cal])
        models.append(mm)
        calibrators.append(cc)

    p_cal = np.column_stack(
        [c.predict(scores_fn(m, X_te)) for m, c in zip(models, calibrators)]
    ).mean(axis=1)
    raw = np.column_stack([scores_fn(m, X_te) for m in models]).mean(axis=1)
    state["models"] = models
    state["calibrators"] = calibrators
    state["calibrated"] = True
    return p_cal, raw, state


def run_oof(pipes, counts, *, model_name="B2_age_logreg", calibrate=True,
            layer_columns=None, seed=SEED, folds=None, groups=None,
            return_fold_state=False):
    """主协议：管段级嵌套交叉验证折外预测。

    返回逐管段折外预测表（每根管段恰好一次）。

    folds: 可选，预先固定的 {fold: test_idx}。泄漏测试必须传入固定划分，
    不得重新执行依赖标签的分层划分（§3.4）。
    groups: 可选，与管段对齐的分组数组；给出时外层与内层共享分组边界（§6.2）。
    return_fold_state: 为 True 时额外返回每折已拟合的模型与校准器，
    供解释复用，避免重新拟合导致解释与已发布预测不一致（§7.1）。
    """
    from src.evaluation.split import make_outer_folds

    y = counts.gt(0).astype(int).to_numpy()
    pipe_ids = pipes["ID"].astype(str).to_numpy()
    if groups is not None:
        groups = np.asarray(groups, dtype=object)
        if len(groups) != len(pipe_ids):
            raise ValueError("groups 长度与管段数不一致")
    if folds is None:
        folds = make_outer_folds(pipe_ids, y, groups=groups, seed=seed)

    use_calibration = calibrate and needs_calibration(model_name)
    records = []
    fold_state = {}

    for f, test_idx in folds.items():
        train_idx = np.setdiff1d(np.arange(len(pipe_ids)), test_idx)
        X_tr = pipes.iloc[train_idx].reset_index(drop=True)
        y_tr = y[train_idx]
        X_te = pipes.iloc[test_idx].reset_index(drop=True)
        groups_tr = None if groups is None else groups[train_idx]

        p_cal, raw, state = _fit_predict_fold(
            X_tr, y_tr, X_te, _factory(model_name, layer_columns),
            calibrate=use_calibration, seed=seed, groups_tr=groups_tr,
        )

        if return_fold_state:
            fold_state[int(f)] = {
                "models": state["models"],
                "calibrators": state["calibrators"],
                "train_index": train_idx,
                "test_index": test_idx,
                "quality_flags": list(state["quality_flags"]),
            }

        for i, idx in enumerate(test_idx):
            records.append({
                "pipe_id": pipe_ids[idx],
                "outer_fold": int(f),
                "y_true": int(y[idx]),
                "p": float(p_cal[i]),
                "raw_score": float(raw[i]),
                "model_id": model_name,
                "calibrated": bool(state["calibrated"]),
                "round_id": f"seed{seed}",
                "quality_flags": list(state["quality_flags"]),
            })

    df = pd.DataFrame(records).sort_values("pipe_id", kind="mergesort").reset_index(drop=True)

    # 每根管段恰好一次（§6.1 步骤 3）
    if not df["pipe_id"].is_unique:
        raise ValueError("折外预测出现重复管段")
    if len(df) != len(pipe_ids):
        raise ValueError(f"折外预测数 {len(df)} != 管段数 {len(pipe_ids)}")
    if df["p"].isna().any() or not np.isfinite(df["p"]).all():
        raise ValueError("折外预测含非有限值")
    if ((df["p"] < 0) | (df["p"] > 1)).any():
        raise ValueError("折外预测概率超出 [0,1]")

    if return_fold_state:
        return df, fold_state
    return df


def _factory(model_name, layer_columns=None):
    if model_name == "B0_baserate":
        return B0BaseRate
    if model_name == "B1_age_rank":
        return B1AgeRank
    if model_name == "B2_age_logreg":
        return B2AgeLogReg
    if model_name == "B2b_layer_logreg":
        from src.models.baselines import LayerLogReg
        cols = layer_columns or ["PIPEAGE"]
        return lambda: LayerLogReg(cols)
    raise KeyError(f"未知模型 {model_name!r}")