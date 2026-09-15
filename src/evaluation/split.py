"""固定验证划分（§6.1、§6.2）。

主协议：管段级嵌套交叉验证。
- 固定预注册种子 20260914
- 按管段、对 y 分层划 5 个外层折
- 保存 ID 清单，跨运行字节一致
- 外层测试折仅作评估

补充协议（§6.2）：按道路分组的外层验证。内层训练与校准必须与外层共享
同一边界——不得外层按道路分组、内层却任意拆散同路管段。
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

ROOT = Path(__file__).resolve().parent.parent.parent

SEED = 20260914
N_OUTER = 5
N_INNER = 3

# 分组方案 -> 分组列。SZDL 仅作分组，不作预测特征（§3.2 隔离字段）。
GROUP_SCHEMES = {"random": None, "road": "SZDL"}

GROUPED_DISCLOSURE = (
    "分组协议衡量对分组迁移的敏感性；与随机 CV 接近不证明无泄漏，"
    "也不证明空间特征永远无收益（§6.2）。"
)


def stable_tie_key(pipe_id, seed=SEED):
    """并列分数的稳定排序键（§6.4）。

    不使用可能随进程变化的内置字符串 hash。
    """
    return hashlib.sha256(f"{seed}:{pipe_id}".encode("utf-8")).hexdigest()


def round_id_of(seed, scheme="random"):
    """轮次标识，含划分方案（§6.1、§6.2）。

    随机方案下与历史口径逐字节一致；分组方案带方案后缀，避免同一模型在
    两种划分下的发布撞同一 ID。
    """
    return f"seed{seed}" if scheme == "random" else f"seed{seed}-{scheme}"


def group_ids(pipes, scheme="road"):
    """生成分组 ID（§6.2）。

    缺失分组值成为单例组：既不与其他未分组管段合并，也不丢弃。
    返回 (分组数组, 诊断元数据)。
    """
    if scheme not in GROUP_SCHEMES:
        raise KeyError(f"未实现的分组方案 {scheme!r}；可用: {sorted(GROUP_SCHEMES)}")
    column = GROUP_SCHEMES[scheme]
    if column is None:
        raise ValueError(f"分组方案 {scheme!r} 不对应任何分组列")

    ids = pipes["ID"].astype(str).to_numpy()
    out = np.empty(len(pipes), dtype=object)
    n_missing = 0
    for i, v in enumerate(pipes[column].to_numpy()):
        blank = v is None or str(v).strip() == "" or (
            isinstance(v, float) and not np.isfinite(v))
        if blank:
            out[i] = f"__MISSING__:{ids[i]}"
            n_missing += 1
        else:
            out[i] = str(v)

    return out, {
        "scheme": scheme,
        "column": column,
        "n_groups": int(len(set(out))),
        "n_missing_rows": n_missing,
    }


def assert_groups_respected(groups, splits, *, label=""):
    """断言每个划分的训练集与测试集不含相同分组（§6.2）。"""
    g = np.asarray(groups, dtype=object)
    for i, (train_idx, test_idx) in enumerate(splits):
        shared = set(g[np.asarray(train_idx)]) & set(g[np.asarray(test_idx)])
        if shared:
            raise ValueError(
                f"{label}第 {i} 折训练/测试共享分组 "
                f"{sorted(shared)[:3]}（§6.2 禁止拆散同组管段）")


def make_outer_folds(pipe_ids, y, groups=None, *, n_splits=N_OUTER, seed=SEED):
    """生成外层折。返回 fold -> 测试集索引。

    groups 为 None 时走冻结的随机分层默认路径（逐字节不变）；
    否则按分组分层，同一分组不跨折（§6.2）。
    """
    y = np.asarray(y)
    if groups is None:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        folds = {}
        for i, (_, test_idx) in enumerate(skf.split(np.asarray(pipe_ids), y)):
            folds[i] = np.sort(test_idx)
        return folds

    g = np.asarray(groups, dtype=object)
    n_groups = len(set(g))
    if n_groups < 2:
        raise ValueError(f"分组数不足 2（{n_groups}），无法进行分组验证")

    k = min(n_splits, n_groups)
    sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    folds = {}
    for i, (_, test_idx) in enumerate(sgkf.split(np.zeros(len(y)), y, groups=g)):
        folds[i] = np.sort(test_idx)

    assert_groups_respected(
        g,
        [(np.setdiff1d(np.arange(len(g)), t), t) for t in folds.values()],
        label="外层",
    )
    return folds


def inner_splits(y_train, *, groups=None, n_splits=N_INNER, seed=SEED):
    """内层划分，供折内校准与候选选择共用（§6.1 步骤 1/2、§6.2）。

    返回 (splits, status)，status ∈ {"ok", "reduced_folds", "degenerate"}。
    分组协议下内层与外层共享同一边界；退化时返回空 splits，调用方须按
    预定义无效处理，**不得随机拆散分组补齐**（§6.1）。
    """
    y_train = np.asarray(y_train)
    if groups is None:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return [(tr, va) for tr, va in skf.split(np.zeros(len(y_train)), y_train)], "ok"

    g = np.asarray(groups, dtype=object)
    n_groups = len(set(g))
    for k in range(min(n_splits, n_groups), 1, -1):
        sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
        splits = [(tr, va) for tr, va in
                  sgkf.split(np.zeros(len(y_train)), y_train, groups=g)]
        if all(len(set(y_train[va])) == 2 for _, va in splits):
            assert_groups_respected(g, splits, label="内层")
            return splits, ("ok" if k == n_splits else "reduced_folds")
    return [], "degenerate"


def grouped_fold_diagnostics(groups, y, folds):
    """分组协议诊断与强制披露（§6.2）。"""
    sizes = [int(len(t)) for t in folds.values()]
    return {
        "scheme": "grouped",
        "n_groups": int(len(set(np.asarray(groups, dtype=object)))),
        "n_folds": len(folds),
        "fold_sizes": sizes,
        "fold_sizes_equal": len(set(sizes)) == 1,
        "disclosure": GROUPED_DISCLOSURE,
    }


def split_table(pipes, counts, *, groups=None):
    """构建管段级划分表。

    同一管段只出现在一个外层测试折（§3.4）。
    groups 给出时按分组划分，并附 group_id 列（§6.2）。
    """
    y = counts.gt(0).astype(int).to_numpy()
    pipe_ids = pipes["ID"].astype(str).to_numpy()
    folds = make_outer_folds(pipe_ids, y, groups=groups)

    fold_of = np.full(len(pipe_ids), -1, dtype=int)
    for f, idx in folds.items():
        fold_of[idx] = f

    if (fold_of < 0).any():
        raise ValueError("存在未分配折的管段")
    if len(set(fold_of)) != len(folds):
        raise ValueError("折数不完整")

    table = pd.DataFrame({
        "pipe_id": pipe_ids,
        "y_true": y,
        "count_2024": counts.to_numpy(),
        "outer_fold": fold_of,
    })
    if groups is not None:
        table["group_id"] = np.asarray(groups, dtype=object)
    return table


def fold_summary(table, pipes=None):
    """逐折规模与正例数（§6.4.1 展示项）。

    分组协议下折规模可能不等，仍逐折报告，不以 pooled 替代。
    pipes 给出时附分组、管龄与管材分布诊断（§6.2）。
    """
    out = {}
    for f, g in table.groupby("outer_fold"):
        row = {
            "n_test": int(len(g)),
            "n_positive": int(g["y_true"].sum()),
            "base_rate": float(g["y_true"].mean()),
        }
        if "group_id" in table.columns:
            groups_col = g["group_id"].astype(str)
            row["n_groups"] = int(groups_col.nunique())
            row["n_missing_group_rows"] = int(
                groups_col.str.startswith("__MISSING__").sum())
        if pipes is not None:
            sub = pipes.iloc[g.index]
            row["pipeage_median"] = float(sub["PIPEAGE"].median())
            row["material_counts"] = {
                str(k): int(v) for k, v in sub["CZ"].value_counts().items()
            }
        out[int(f)] = row
    return out


def table_fingerprint(table, *, scheme="random"):
    """划分表指纹。用于跨运行一致性校验。

    scheme 为 "random" 时与历史指纹逐字节一致（§6.1）。
    """
    payload = {
        "seed": SEED,
        "n_outer": N_OUTER,
        "rows": sorted(
            [str(r.pipe_id), int(r.y_true), int(r.outer_fold)]
            for r in table.itertuples()
        ),
    }
    if scheme != "random":
        payload["scheme"] = scheme
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def save_split(table, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # 按 pipe_id 排序，保证字节一致
    ordered = table.sort_values("pipe_id", kind="mergesort").reset_index(drop=True)
    ordered.to_csv(path, index=False)
    return path


def load_split(path):
    return pd.read_csv(path, dtype={"pipe_id": str})