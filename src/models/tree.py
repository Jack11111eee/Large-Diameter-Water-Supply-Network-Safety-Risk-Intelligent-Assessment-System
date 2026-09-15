"""树候选（§5.1）。

方案 §5.1 记名为「F1 CatBoost」；本环境未安装 catboost，按同节
「实现优先采用 scikit-learn 和 CatBoost」改用 sklearn 的
HistGradientBoostingClassifier。该替换已在里程碑 M2 记录中说明，属有据替换。
"""

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from src.models.baselines import make_preprocessor

SEED = 20260914


class HGBTClassifier:
    """直方图梯度提升树候选。

    原始输出为 log-odds：HistGradientBoostingClassifier.decision_function
    与树 SHAP 同尺度，供 §7.1 的解释加和核验使用。
    早停只用外层训练集内部切分（§5.2 允许）。
    """

    name = "hgb"

    def __init__(self, columns, *, max_leaf_nodes=31, learning_rate=0.06,
                 max_iter=300, early_stopping=True, validation_fraction=0.15,
                 n_iter_no_change=20, l2_regularization=0.0, random_state=SEED):
        self.columns = list(columns)
        self.params = {
            "max_leaf_nodes": max_leaf_nodes,
            "learning_rate": learning_rate,
            "max_iter": max_iter,
            "early_stopping": early_stopping,
            "validation_fraction": validation_fraction,
            "n_iter_no_change": n_iter_no_change,
            "l2_regularization": l2_regularization,
            "random_state": random_state,
        }

    def fit(self, X, y):
        self.prep_ = make_preprocessor(self.columns)
        Xt = self.prep_.fit_transform(X)
        self.est_ = HistGradientBoostingClassifier(**self.params)
        self.est_.fit(Xt, y)
        return self

    def predict_proba(self, X):
        return self.est_.predict_proba(self.prep_.transform(X))[:, 1]

    def raw_scores_(self, X):
        """原始输出（log-odds），与树 SHAP 同尺度（§7.1）。"""
        return self.est_.decision_function(self.prep_.transform(X))