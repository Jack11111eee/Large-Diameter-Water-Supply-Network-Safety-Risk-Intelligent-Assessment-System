"""概率校准（§5.3）。

- 预先固定 sigmoid 为主流程，未校准版本作为对照
- 基率常数 B0 不需要校准
- 校准必须使用未参与对应基础分类器拟合的数据
- 外层测试折始终不参与校准
- 校准折保留原始正例率，不沿用类别权重
"""

import numpy as np
from sklearn.linear_model import LogisticRegression


class SigmoidCalibrator:
    """Platt 缩放。在独立校准折上拟合。"""

    def __init__(self):
        self.lr_ = LogisticRegression(C=1e6, max_iter=1000, solver="lbfgs")

    def fit(self, scores, y):
        s = np.asarray(scores, dtype=float).reshape(-1, 1)
        self.lr_.fit(s, np.asarray(y, dtype=int))
        return self

    def predict(self, scores):
        s = np.asarray(scores, dtype=float).reshape(-1, 1)
        return self.lr_.predict_proba(s)[:, 1]


def needs_calibration(model_name):
    """B0 是常数概率，不需要校准（§5.3）。"""
    return not model_name.startswith("B0")


def cross_fitted_calibration(fit_fn, scores_fn, X_train, y_train, n_splits, seed):
    """折内校准流程（§6.1 步骤 2）。

    每个基础模型仅在校准折之外拟合，再用相应校准折拟合 sigmoid。
    返回 (基础模型列表, 校准器列表) 配对集合，用于对测试折集成预测。
    """
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    models, calibrators = [], []
    for tr, cal in skf.split(np.zeros(len(y_train)), y_train):
        m = fit_fn(X_train.iloc[tr], y_train[tr])
        s_cal = scores_fn(m, X_train.iloc[cal])
        c = SigmoidCalibrator().fit(s_cal, y_train[cal])
        models.append(m)
        calibrators.append(c)
    return models, calibrators


def ensemble_predict(models, calibrators, X_test):
    """校准集成预测：各配对预测取平均。

    平均原始分数一般不等于平均校准概率的逆变换（§7.1），
    因此这里直接平均校准后的概率，并单独保存原始分数。
    """
    probs = np.column_stack([c.predict(scores_fn(m, X_test))
                             for m, c in zip(models, calibrators)])
    return probs.mean(axis=1)


def scores_fn(model, X):
    """基础模型的原始输出（log-odds 尺度，§7.1）。

    树候选显式提供 raw_scores_，必须优先判断——否则会把概率混进
    原始分数列，破坏 SHAP 加和核验与校准对照。
    """
    if hasattr(model, "raw_scores_"):
        return model.raw_scores_(X)
    if hasattr(model, "clf_"):
        Xt = model.prep_.transform(X)
        return model.clf_.decision_function(Xt)
    return model.predict_proba(X)