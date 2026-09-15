"""SHAP 归因（§7.1）。

解释的是模型的**原始输出**（log-odds），不是校准概率。
背景样本从相应训练折选取，绝不触碰测试折。
"""

import numpy as np

from src.models.calibration import scores_fn


def raw_scores(model, X):
    """被解释的原始输出，与 scores_fn 同尺度（§7.1）。"""
    return scores_fn(model, X)


def explainable(model):
    """是否具备可解释结构（树用 est_，线性用 clf_）。"""
    return hasattr(model, "est_") or hasattr(model, "clf_")


def background_sample(X_train, *, n=100, seed=0):
    """训练折内确定性抽样作背景（§7.1）。

    只在训练折上抽样；抽样是确定性的，保证解释可复现。
    """
    if len(X_train) <= n:
        return X_train
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(X_train), size=n, replace=False))
    return X_train.iloc[idx]


def check_additivity(base_value, contributions, raw):
    """加和核验：max|base + Σcontrib − raw|（§7.1）。

    贡献加和必须能还原其所解释的原始输出。
    """
    c = np.asarray(contributions, dtype=float)
    raw = np.asarray(raw, dtype=float)
    recovered = c.reshape(len(raw), -1).sum(axis=1) + float(base_value)
    return float(np.abs(recovered - raw).max())


def explain_model(model, X_background, X_explain):
    """对已拟合模型计算 SHAP 贡献。

    返回 {"base_value", "contributions", "feature_names", "additivity_max_abs_error"}。
    贡献在预处理后的特征空间上计算，与模型实际输入一致。
    """
    import shap

    prep = model.prep_
    bg = prep.transform(X_background)
    ex = prep.transform(X_explain)

    # 显式给足 max_samples，避免 SHAP 静默二次抽样背景导致解释不可复现。
    masker = shap.maskers.Independent(bg, max_samples=len(bg))
    if hasattr(model, "est_"):
        explainer = shap.TreeExplainer(model.est_, data=masker)
    elif hasattr(model, "clf_"):
        explainer = shap.LinearExplainer(model.clf_, masker)
    else:
        raise TypeError("模型既无 est_ 也无 clf_，无法解释")

    values = np.asarray(explainer.shap_values(ex))
    if values.ndim == 3:            # (n, p, classes) -> 取正类
        values = values[:, :, 1]
    base_value = float(np.asarray(explainer.expected_value).reshape(-1)[-1])

    raw = raw_scores(model, X_explain)
    return {
        "base_value": base_value,
        "contributions": values,
        "feature_names": list(prep.feature_names),
        "additivity_max_abs_error": check_additivity(base_value, values, raw),
    }