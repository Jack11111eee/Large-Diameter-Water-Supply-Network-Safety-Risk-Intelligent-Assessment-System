"""F2/F3 附加消融与配对 delta（§3.2、§10 W3）。

F2（运行与运维快照）与 F3（无标签拓扑）都是**附加**消融，不默认为事前观测。
增益用逐折配对差值表述——先算逐折差再取宏平均与折间标准差；
两个 macro 均值之差丢掉了配对信息，不是「增益与不确定性」。
保留简单模型可赢的可能，不以任何 AUC 上限解释差异。
"""

import numpy as np

from src.data import loader
from src.evaluation import metrics, selection
from src.evaluation.split import SEED

# 结构编号沿用方案 §5.1：M1 = F1；M3 = 选定结构 + F2；M4 = 选定结构 + F2 + F3。
# X_F1F3 为 §5.1「增加 F3」字面读法的对照，多一行成本换取口径保险。
STRUCTURES = {
    "M1": "F1_base_environment",
    "M3": "F1_base_environment+F2_operational_snapshot",
    "M4": "F1_base_environment+F2_operational_snapshot+F3_topology",
    "X_F1F3": "F1_base_environment+F3_topology",
}

DISCLOSURES = {
    "F2": "F2 为运行与运维快照的附加消融，不默认为事前观测；"
          "PRESS/VELOC/OPSST 带 not_pre_event 标记，REPCNT/REPCO2 带 may_be_post_event 标记（§3.2）。",
    "F3": "F3 使用全网无标签拓扑，部署设定为「全网结构已知」；"
          "与随机 CV 接近不证明无泄漏（§6.2）。",
    "common": "增益保留简单模型可赢的可能，不以任何 AUC 上限解释差异（§10 W3）。",
}


def paired_delta(base_per_fold, alt_per_fold, *, metric="ap", applicable_key=None):
    """逐折配对差值（§10 W3）。返回差值均值与折间标准差。"""
    per_fold, deltas = {}, []
    for f, rec in base_per_fold.items():
        other = alt_per_fold.get(f)
        if other is None:
            continue
        if applicable_key and not (rec.get(applicable_key)
                                   and other.get(applicable_key)):
            continue
        a, b = rec.get(metric), other.get(metric)
        if a is None or b is None or not (np.isfinite(a) and np.isfinite(b)):
            continue
        per_fold[f] = float(b - a)
        deltas.append(b - a)

    if not deltas:
        return {"metric": metric, "n_folds": 0, "mean_delta": None,
                "fold_std": None, "per_fold_delta": {}}

    mean = float(np.mean(deltas))
    std = (float(np.sqrt(sum((d - mean) ** 2 for d in deltas) / (len(deltas) - 1)))
           if len(deltas) > 1 else 0.0)
    return {"metric": metric, "n_folds": len(deltas), "mean_delta": mean,
            "fold_std": std, "per_fold_delta": per_fold}


def run_ablation(pipes, counts, *, folds, candidates_config=None, seed=SEED,
                 groups=None, tolerance=0.005, structures=None, metric="ap"):
    """逐结构运行同一 folds、同一候选集、同一校准协议（§10 W3）。

    结构选择（线性 vs 树）在各外层训练集内部完成，不看外层成绩回选。
    """
    structures = structures or STRUCTURES
    # F3 列不在原始属性表中，进入模型前先派生一次（§3.2）。
    frame = loader.with_derived_features(pipes)
    out = {}
    for key, layer in structures.items():
        cols = loader.resolve_layer(layer)
        candidates = selection.candidates_from_config(
            cols, config=candidates_config, seed=seed)
        oof, log = selection.select_and_run(
            frame, counts, candidates=candidates, folds=folds,
            seed=seed, groups=groups, tolerance=tolerance)

        per_fold = metrics.per_fold_metrics(
            oof.y_true, oof.p, oof.pipe_id, oof.outer_fold, q_list=())
        mean, std, n_valid, n_total = metrics.macro_mean(
            per_fold, metric, applicable_key=f"{metric}_applicable")

        out[key] = {
            "structure": key,
            "layer_spec": layer,
            "n_columns": len(cols),
            "model_id_per_fold": {f: rec["chosen"] for f, rec in log.items()},
            "per_fold": per_fold,
            "macro_mean": {metric: mean, "fold_std": std,
                           "valid_folds": n_valid, "total_folds": n_total},
            "selection_log": log,
        }

    base = out.get("M1")
    for key, rec in out.items():
        rec["paired_delta_vs_M1"] = paired_delta(
            base["per_fold"] if base else {}, rec["per_fold"],
            metric=metric, applicable_key=f"{metric}_applicable")

    return {"structures": out, "disclosures": dict(DISCLOSURES), "metric": metric}