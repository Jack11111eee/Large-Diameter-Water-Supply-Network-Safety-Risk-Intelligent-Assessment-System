"""模型候选（§5.1 首版候选 B0/B1/B2）。

B0 训练折基率常数概率
B1 管龄原值排序（描述性参照，不冒充概率）
B2 管龄的正则化逻辑回归

首版不引入 CatBoost；候选 M1/M2 留到 M2 里程碑。
"""

import numpy as np
from sklearn.linear_model import LogisticRegression

from src.data import loader
from src.data.preprocess import FoldPreprocessor


class B0BaseRate:
    """训练折基率常数概率。AP/Brier/log loss 的基础参照。"""

    name = "B0_baserate"

    def fit(self, X, y):
        self.p_ = float(np.mean(y))
        return self

    def predict_proba(self, X):
        n = len(X)
        return np.full(n, self.p_)


class B1AgeRank:
    """管龄原值排序。描述性排序参照，不冒充概率（§5.1）。

    输出 min-max 缩放到 (0,1) 的排序分，仅用于排序指标。
    概率类指标不适用本模型。
    """

    name = "B1_age_rank"

    def fit(self, X, y):
        age = X["PIPEAGE"].to_numpy(dtype=float)
        self.lo_ = float(np.nanmin(age))
        self.hi_ = float(np.nanmax(age))
        return self

    def predict_proba(self, X):
        age = X["PIPEAGE"].to_numpy(dtype=float)
        rng = self.hi_ - self.lo_ or 1.0
        return (age - self.lo_) / rng


class B2AgeLogReg:
    """管龄的正则化逻辑回归（线性及少量样条候选中的线性版）。

    预处理在训练折拟合（§3.3 规则 2）。
    """

    name = "B2_age_logreg"

    def __init__(self, C=1.0, max_iter=1000):
        self.C = C
        self.max_iter = max_iter

    def fit(self, X, y):
        self.prep_ = FoldPreprocessor(["PIPEAGE"], [])
        Xt = self.prep_.fit_transform(X)
        self.clf_ = LogisticRegression(
            C=self.C, max_iter=self.max_iter, solver="lbfgs"
        )
        self.clf_.fit(Xt, y)
        return self

    def predict_proba(self, X):
        Xt = self.prep_.transform(X)
        return self.clf_.predict_proba(Xt)[:, 1]


def make_preprocessor(layer_columns):
    """按特征层构建预处理器（§3.2）。

    数值/类别划分由白名单 numeric_fields 声明；不再硬编码，
    否则 F2 数值字段（RENYR/REPCNT/REPCO2/INSPF）会被误作类别独热。
    """
    numeric = [c for c in layer_columns if c in loader.numeric_fields()]
    categorical = [c for c in layer_columns if c not in numeric]
    return FoldPreprocessor(numeric, categorical)


class LayerLogReg:
    """F1 正则化逻辑回归（M1 的可解释对照，此处作为年龄之外的扩展基线）。"""

    def __init__(self, columns, C=1.0, max_iter=2000):
        self.columns = list(columns)
        self.C = C
        self.max_iter = max_iter
        self.name = "B2b_layer_logreg"

    def fit(self, X, y):
        self.prep_ = make_preprocessor(self.columns)
        Xt = self.prep_.fit_transform(X)
        self.clf_ = LogisticRegression(C=self.C, max_iter=self.max_iter)
        self.clf_.fit(Xt, y)
        return self

    def predict_proba(self, X):
        Xt = self.prep_.transform(X)
        return self.clf_.predict_proba(Xt)[:, 1]