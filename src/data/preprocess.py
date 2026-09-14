"""折内预处理（§3.3 规则 2）。

数值缺失填补、缩放、稀有类别合并只在训练折拟合，测试折仅变换。
"""

import numpy as np
import pandas as pd


class FoldPreprocessor:
    """在训练折上拟合，在测试折上仅变换。

    不使用 sklearn Pipeline 是为了让"训练折拟合"这一步显式可见，
    并保证测试折信息绝不回流。
    """

    def __init__(self, numeric_cols, categorical_cols, min_category_count=10):
        self.numeric_cols = list(numeric_cols)
        self.categorical_cols = list(categorical_cols)
        self.min_category_count = min_category_count
        self._medians = {}
        self._categories = {}
        self._columns = []

    def fit(self, X_train):
        for c in self.numeric_cols:
            self._medians[c] = float(X_train[c].median())
        for c in self.categorical_cols:
            vc = X_train[c].value_counts()
            keep = vc[vc >= self.min_category_count].index
            # 稀有类别合并；缺失单列
            self._categories[c] = sorted(map(str, keep))
        self._columns = (
            [f"num:{c}" for c in self.numeric_cols]
            + [f"cat:{c}={v}" for c in self.categorical_cols for v in self._categories[c]]
            + [f"cat:{c}=__MISSING__" for c in self.categorical_cols]
        )
        return self

    def transform(self, X):
        n = len(X)
        out = np.zeros((n, len(self._columns)), dtype=float)
        col = 0
        for c in self.numeric_cols:
            v = pd.to_numeric(X[c], errors="coerce").to_numpy(dtype=float)
            v = np.where(np.isfinite(v), v, self._medians[c])
            out[:, col] = v
            col += 1
        for c in self.categorical_cols:
            s = X[c].astype("object").where(X[c].notna(), "__MISSING__").astype(str)
            for v in self._categories[c]:
                out[:, col] = (s == v).to_numpy(dtype=float)
                col += 1
            out[:, col] = (s == "__MISSING__").to_numpy(dtype=float)
            col += 1
        return out

    def fit_transform(self, X_train):
        return self.fit(X_train).transform(X_train)

    @property
    def feature_names(self):
        return list(self._columns)