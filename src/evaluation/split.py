"""固定验证划分（§6.1）。

主协议：管段级嵌套交叉验证。
- 固定预注册种子 20260914
- 按管段、对 y 分层划 5 个外层折
- 保存 ID 清单，跨运行字节一致
- 外层测试折仅作评估
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent.parent.parent

SEED = 20260914
N_OUTER = 5
N_INNER = 3


def stable_tie_key(pipe_id, seed=SEED):
    """并列分数的稳定排序键（§6.4）。

    不使用可能随进程变化的内置字符串 hash。
    """
    return hashlib.sha256(f"{seed}:{pipe_id}".encode("utf-8")).hexdigest()


def make_outer_folds(pipe_ids, y):
    """生成外层折。返回 fold -> 测试集索引。"""
    skf = StratifiedKFold(n_splits=N_OUTER, shuffle=True, random_state=SEED)
    folds = {}
    for i, (_, test_idx) in enumerate(skf.split(np.asarray(pipe_ids), np.asarray(y))):
        folds[i] = np.sort(test_idx)
    return folds


def split_table(pipes, counts):
    """构建管段级划分表。

    同一管段只出现在一个外层测试折（§3.4）。
    """
    y = counts.gt(0).astype(int).to_numpy()
    pipe_ids = pipes["ID"].astype(str).to_numpy()
    folds = make_outer_folds(pipe_ids, y)

    fold_of = np.full(len(pipe_ids), -1, dtype=int)
    for f, idx in folds.items():
        fold_of[idx] = f

    if (fold_of < 0).any():
        raise ValueError("存在未分配折的管段")
    if len(set(fold_of)) != N_OUTER:
        raise ValueError("折数不完整")

    return pd.DataFrame({
        "pipe_id": pipe_ids,
        "y_true": y,
        "count_2024": counts.to_numpy(),
        "outer_fold": fold_of,
    })


def fold_summary(table):
    """逐折规模与正例数（§6.4.1 展示项）。"""
    out = {}
    for f, g in table.groupby("outer_fold"):
        out[int(f)] = {
            "n_test": int(len(g)),
            "n_positive": int(g["y_true"].sum()),
            "base_rate": float(g["y_true"].mean()),
        }
    return out


def table_fingerprint(table):
    """划分表指纹。用于跨运行一致性校验。"""
    payload = json.dumps(
        {
            "seed": SEED,
            "n_outer": N_OUTER,
            "rows": sorted(
                [str(r.pipe_id), int(r.y_true), int(r.outer_fold)]
                for r in table.itertuples()
            ),
        },
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def save_split(table, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # 按 pipe_id 排序，保证字节一致
    ordered = table.sort_values("pipe_id", kind="mergesort").reset_index(drop=True)
    ordered.to_csv(path, index=False)
    return path


def load_split(path):
    return pd.read_csv(path, dtype={"pipe_id": str})