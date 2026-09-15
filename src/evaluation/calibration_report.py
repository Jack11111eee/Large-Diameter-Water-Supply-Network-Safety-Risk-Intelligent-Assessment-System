"""校准对照与可靠性分箱（§5.3）。

- 主流程预先固定 sigmoid，未校准版本作为对照
- 校准数据不含外层测试折
- Brier/log loss 更低不构成校准更好的证据：必须同时给出可靠性分箱、
  每箱样本数与正例数
"""

import numpy as np

from src.evaluation.metrics import brier, logloss

# 每箱样本数低于此值标记 low_n，提醒分箱读数不可靠（§5.3）。
MIN_BIN_N = 20

CALIBRATION_CONCLUSION = (
    "Brier/log loss 更低不构成校准更好的证据；须同时核对可靠性分箱"
    "与每箱样本数、正例数（§5.3）。"
)


def reliability_bins(y, p, *, n_bins=10, strategy="quantile"):
    """可靠性分箱。每箱同时给出样本数与正例数；空箱保留不丢弃。

    概率分布高度集中，默认用分位数分箱——等宽 [0,1] 箱会大量为空。
    """
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    if len(y) != len(p):
        raise ValueError("y 与 p 长度不一致")
    if len(p) == 0:
        return []

    if strategy == "quantile":
        edges = np.unique(np.quantile(p, np.linspace(0.0, 1.0, n_bins + 1)))
        if len(edges) < 2:
            edges = np.array([float(p.min()), float(p.max()) + 1e-12])
    elif strategy == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    else:
        raise KeyError(f"未知分箱策略 {strategy!r}")

    out = []
    last = len(edges) - 2
    for i in range(len(edges) - 1):
        lo, hi = float(edges[i]), float(edges[i + 1])
        mask = (p >= lo) & (p <= hi) if i == last else (p >= lo) & (p < hi)
        n = int(mask.sum())
        out.append({
            "bin": i,
            "lo": lo,
            "hi": hi,
            "n": n,
            "n_positive": int(y[mask].sum()),
            "mean_p": float(p[mask].mean()) if n else None,
            "observed_rate": float(y[mask].mean()) if n else None,
            "low_n": n < MIN_BIN_N,
        })
    return out


def compare_calibration(y, p_calibrated, p_uncalibrated, *, n_bins=10):
    """校准版与未校准对照的逐项比较（§5.3）。

    未校准对照必须使用与校准版相同的数据边界。
    """
    y = np.asarray(y, dtype=int)
    pc = np.asarray(p_calibrated, dtype=float)
    pu = np.asarray(p_uncalibrated, dtype=float)
    if not (len(y) == len(pc) == len(pu)):
        raise ValueError("校准对照要求同一管段集合")

    b_cal, b_un = float(brier(y, pc)), float(brier(y, pu))
    l_cal, l_un = float(logloss(y, pc)), float(logloss(y, pu))
    return {
        "n": int(len(y)),
        "base_rate": float(y.mean()) if len(y) else None,
        "brier": {"calibrated": b_cal, "uncalibrated": b_un,
                  "delta": b_un - b_cal},
        "log_loss": {"calibrated": l_cal, "uncalibrated": l_un,
                     "delta": l_un - l_cal},
        "bins_calibrated": reliability_bins(y, pc, n_bins=n_bins),
        "bins_uncalibrated": reliability_bins(y, pu, n_bins=n_bins),
        "conclusion": CALIBRATION_CONCLUSION,
    }


def per_fold_calibration(y, p_calibrated, p_uncalibrated, fold_ids, *, n_bins=10):
    """逐折校准对照（§5.3 逐折主口径，不以 pooled 替代）。"""
    y = np.asarray(y, dtype=int)
    pc = np.asarray(p_calibrated, dtype=float)
    pu = np.asarray(p_uncalibrated, dtype=float)
    folds = np.asarray(fold_ids)
    out = {}
    for f in sorted(set(folds.tolist())):
        m = folds == f
        out[int(f)] = compare_calibration(y[m], pc[m], pu[m], n_bins=n_bins)
    return out