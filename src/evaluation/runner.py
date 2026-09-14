"""折外预测回放（§6.1、§2.3）。

对每根管段展示未用其标签训练的预测。
外层测试折仅作评估；所有预处理、选择、校准都在外层训练部分内完成。
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.evaluation.split import SEED, N_INNER
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


def run_oof(pipes, counts, *, model_name="B2_age_logreg", calibrate=True,
            layer_columns=None, seed=SEED, folds=None):
    """主协议：管段级嵌套交叉验证折外预测。

    返回逐管段折外预测表（每根管段恰好一次）。

    folds: 可选，预先固定的 {fold: test_idx}。泄漏测试必须传入固定划分，
    不得重新执行依赖标签的分层划分（§3.4）。
    """
    from src.evaluation.split import make_outer_folds

    y = counts.gt(0).astype(int).to_numpy()
    pipe_ids = pipes["ID"].astype(str).to_numpy()
    if folds is None:
        folds = make_outer_folds(pipe_ids, y)

    records = []
    for f, test_idx in folds.items():
        train_idx = np.setdiff1d(np.arange(len(pipe_ids)), test_idx)
        X_tr = pipes.iloc[train_idx].reset_index(drop=True)
        y_tr = y[train_idx]
        X_te = pipes.iloc[test_idx].reset_index(drop=True)

        factory = _factory(model_name, layer_columns)

        if not calibrate or not needs_calibration(model_name):
            m = factory().fit(X_tr, y_tr)
            p_cal = m.predict_proba(X_te)
            raw = scores_fn(m, X_te)
        else:
            # 外层训练部分内 3 折校准（§6.1 步骤 2）
            skf = StratifiedKFold(n_splits=N_INNER, shuffle=True, random_state=seed)
            models, cals = [], []
            for tr, cal in skf.split(np.zeros(len(y_tr)), y_tr):
                mm = factory().fit(X_tr.iloc[tr], y_tr[tr])
                s = scores_fn(mm, X_tr.iloc[cal])
                cc = SigmoidCalibrator().fit(s, y_tr[cal])
                models.append(mm)
                cals.append(cc)
            p_cal = np.column_stack(
                [c.predict(scores_fn(m, X_te)) for m, c in zip(models, cals)]
            ).mean(axis=1)
            raw = np.column_stack(
                [scores_fn(m, X_te) for m in models]
            ).mean(axis=1)

        for i, idx in enumerate(test_idx):
            records.append({
                "pipe_id": pipe_ids[idx],
                "outer_fold": int(f),
                "y_true": int(y[idx]),
                "p": float(p_cal[i]),
                "raw_score": float(raw[i]),
                "model_id": model_name,
                "calibrated": bool(calibrate and needs_calibration(model_name)),
                "round_id": f"seed{seed}",
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