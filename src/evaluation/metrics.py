"""指标与聚合（§6.4、§6.4.1）。

严格区分四类聚合口径：
  per_fold            每个测试折单独计算
  macro_mean/fold_std 同轮各有效折的等权统计，正式主成绩
  sample_weighted     Brier/log loss 按每折样本数加权
  pooled_oof_replay   同轮折外概率拼接后计算，仅全网回放辅助

人工反例必须保留 per-fold 与 pooled 的区别（§6.4.1）。
"""

import math

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


class NotApplicable(Exception):
    """指标不适用（无正例折、缺任一类）。"""


def _check_binary(y):
    y = np.asarray(y, dtype=int)
    if y.sum() == 0:
        raise NotApplicable("评估集无正例")
    if y.sum() == len(y):
        raise NotApplicable("评估集全为正例")
    return y


def ap(y, p):
    y = _check_binary(y)
    return float(average_precision_score(y, p))


def roc_auc(y, p):
    y = _check_binary(y)
    return float(roc_auc_score(y, p))


def brier(y, p):
    """单类测试折也可计算（§6.4.1）。"""
    return float(brier_score_loss(np.asarray(y, dtype=int), np.clip(p, 1e-12, 1 - 1e-12)))


def logloss(y, p):
    return float(log_loss(np.asarray(y, dtype=int), np.clip(p, 1e-12, 1 - 1e-12),
                          labels=[0, 1]))


def stable_order(pipe_ids, scores, seed):
    """稳定并列排序（§6.4）。

    主键为分数降序，并列用 SHA256(seed:pipe_id) 打散，独立于标签。
    """
    from src.evaluation.split import stable_tie_key

    idx = list(range(len(pipe_ids)))
    idx.sort(key=lambda i: (-float(scores[i]), stable_tie_key(pipe_ids[i], seed)))
    return idx


def top_k_metrics(y, p, pipe_ids, q, seed):
    """K=ceil(q*n) 的 Precision/Recall/Lift（§6.4）。

    注意 q<=1 时 k=ceil(q*n)<=n，本路径不会触发截断。
    绝对 K 的截断场景见 top_k_absolute。
    """
    y = np.asarray(y, dtype=int)
    n = len(y)
    if n == 0:
        return {"k_requested": 0, "k_actual": 0, "precision": None,
                "recall": None, "lift": None, "applicable": False,
                "reason": "空候选集"}
    k_req = int(math.ceil(q * n))
    order = stable_order(pipe_ids, p, seed)
    k_act = min(k_req, n)
    top = [order[i] for i in range(k_act)]
    hits = int(y[top].sum())
    pos = int(y.sum())
    precision = hits / k_act if k_act else None
    recall = (hits / pos) if pos > 0 else None
    lift = (precision / (pos / n)) if (precision is not None and pos > 0) else None
    return {
        "k_requested": k_req, "k_actual": k_act,
        "precision": precision, "recall": recall, "lift": lift,
        "applicable": True, "reason": None,
        "truncated": k_act < k_req,
    }


def top_k_absolute(y, p, pipe_ids, k, seed):
    """绝对 K 的 Top-K 指标（§6.4 全网回放 K=100/500/1000）。

    K 超过候选数时显式截为候选总数，同时保留请求数量与实际数量，
    不伪造补足记录。K=0 返回空清单，Precision/Lift 记为不适用。
    """
    y = np.asarray(y, dtype=int)
    n = len(y)
    if n == 0 or k == 0:
        return {"k_requested": int(k), "k_actual": 0, "precision": None,
                "recall": None, "lift": None, "applicable": False,
                "reason": "空候选集" if n == 0 else "K=0",
                "truncated": False}
    k_act = min(int(k), n)
    order = stable_order(pipe_ids, p, seed)
    top = [order[i] for i in range(k_act)]
    hits = int(y[top].sum())
    pos = int(y.sum())
    precision = hits / k_act
    recall = (hits / pos) if pos > 0 else None
    lift = (precision / (pos / n)) if pos > 0 else None
    return {
        "k_requested": int(k), "k_actual": k_act,
        "precision": precision, "recall": recall, "lift": lift,
        "applicable": True, "reason": None,
        "truncated": k_act < int(k),
    }


# --------------------------------------------------------------------------
# 聚合
# --------------------------------------------------------------------------

def per_fold_metrics(y, p, pipe_ids, fold_ids, q_list=(0.01, 0.05, 0.10), seed=20260914):
    """逐折计算（正式成绩明细，§6.4.1）。"""
    out = {}
    for f in sorted(set(fold_ids)):
        mask = np.asarray(fold_ids) == f
        yf = np.asarray(y)[mask]
        pf = np.asarray(p)[mask]
        idf = np.asarray(pipe_ids)[mask]
        rec = {"n_test": int(mask.sum()), "n_positive": int(yf.sum()),
               "base_rate": float(yf.mean()) if mask.sum() else None}
        for name, fn in [("ap", ap), ("roc_auc", roc_auc),
                         ("brier", brier), ("log_loss", logloss)]:
            try:
                rec[name] = fn(yf, pf)
                rec[name + "_applicable"] = True
                rec[name + "_reason"] = None
            except NotApplicable as e:
                rec[name] = None
                rec[name + "_applicable"] = False
                rec[name + "_reason"] = str(e)
        for q in q_list:
            rec[f"topk@{q}"] = top_k_metrics(yf, pf, idf, q, seed)
        out[int(f)] = rec
    return out


def macro_mean(per_fold, metric, applicable_key=None):
    """等权宏平均，仅统计有效折（§6.4.1）。"""
    vals = []
    for f, rec in per_fold.items():
        if applicable_key and not rec.get(applicable_key, False):
            continue
        v = rec.get(metric)
        if v is not None and np.isfinite(v):
            vals.append(v)
    if not vals:
        return None, None, 0, len(per_fold)
    m = float(np.mean(vals))
    if len(vals) > 1:
        sd = float(np.sqrt(sum((x - m) ** 2 for x in vals) / (len(vals) - 1)))
    else:
        sd = 0.0
    return m, sd, len(vals), len(per_fold)


def sample_weighted(per_fold, metric):
    """Brier/log loss 按每折样本数加权（§6.4.1）。"""
    num, den = 0.0, 0
    for f, rec in per_fold.items():
        v = rec.get(metric)
        if v is not None and np.isfinite(v):
            num += rec["n_test"] * v
            den += rec["n_test"]
    if den == 0:
        return None
    return float(num / den)


def pooled_oof_replay(y, p, pipe_ids, q_list=(0.01, 0.05, 0.10), seed=20260914):
    """折外概率拼接后计算（仅全网回放辅助，不替代主成绩）。"""
    y = np.asarray(y)
    p = np.asarray(p)
    rec = {"n": int(len(y)), "n_positive": int(y.sum())}
    for name, fn in [("ap", ap), ("roc_auc", roc_auc),
                     ("brier", brier), ("log_loss", logloss)]:
        try:
            rec[name] = fn(y, p)
            rec[name + "_applicable"] = True
        except NotApplicable as e:
            rec[name] = None
            rec[name + "_applicable"] = False
            rec[name + "_reason"] = str(e)
    for q in q_list:
        rec[f"topk@{q}"] = top_k_metrics(y, p, pipe_ids, q, seed)
    return rec


def aggregate_all(y, p, pipe_ids, fold_ids, q_list=(0.01, 0.05, 0.10), seed=20260914):
    """产出四类聚合口径的完整成绩（§6.4.1）。"""
    pf = per_fold_metrics(y, p, pipe_ids, fold_ids, q_list, seed)
    out = {"per_fold": pf, "macro_mean": {}, "sample_weighted": {}, "pooled_oof_replay": None}
    for metric, app_key in [("ap", "ap_applicable"), ("roc_auc", "roc_auc_applicable"),
                            ("brier", "brier_applicable"), ("log_loss", "log_loss_applicable")]:
        m, sd, n_valid, n_total = macro_mean(pf, metric, app_key)
        out["macro_mean"][metric] = {
            "value": m, "fold_std": sd,
            "valid_folds": n_valid, "total_folds": n_total,
            "complete": n_valid == n_total,
        }
        if metric in ("brier", "log_loss"):
            out["sample_weighted"][metric] = sample_weighted(pf, metric)
    for q in q_list:
        vals = [rec[f"topk@{q}"]["recall"] for rec in pf.values()
                if rec[f"topk@{q}"].get("recall") is not None]
        out["macro_mean"][f"recall@{q}"] = {
            "value": float(np.mean(vals)) if vals else None,
            "valid_folds": len(vals), "total_folds": len(pf),
            "complete": len(vals) == len(pf),
        }
    out["pooled_oof_replay"] = pooled_oof_replay(y, p, pipe_ids, q_list, seed)
    return out


# --------------------------------------------------------------------------
# 扁平化：嵌套聚合 → 契约行（§13.4 `evaluation` 产物）
# --------------------------------------------------------------------------

FLAT_METRICS = ("ap", "roc_auc", "brier", "log_loss")
TOPK_FIELDS = ("k_requested", "k_actual", "precision", "recall", "lift")


def _flat_row(envelope, *, metric, aggregation, fold=None, value=None,
              applicable=True, reason=None, valid_folds=None, total_folds=None):
    row = {
        **envelope,
        "metric": metric,
        "aggregation": aggregation,
        "fold": fold,
        "value": None if value is None else float(value),
        "applicable": bool(applicable),
        "reason": reason,
    }
    if valid_folds is not None:
        row["valid_folds"] = int(valid_folds)
    if total_folds is not None:
        row["total_folds"] = int(total_folds)
    return row


def _topk_rows(envelope, rec, q_list, *, aggregation, fold=None):
    rows = []
    for q in q_list:
        tk = rec.get(f"topk@{q}")
        if tk is None:
            continue
        for field in TOPK_FIELDS:
            v = tk.get(field)
            rows.append(_flat_row(
                envelope, metric=f"{field}@{q}", aggregation=aggregation,
                fold=fold, value=v,
                applicable=bool(tk.get("applicable")) and v is not None,
                reason=tk.get("reason"),
            ))
    return rows


def flatten_evaluation(scores, envelope, *, q_list=(0.01, 0.05, 0.10)):
    """把 `aggregate_all` 的嵌套成绩摊平为契约行（§13.4）。

    每行一个 (metric, aggregation, fold) 读数；`aggregation` 取自契约枚举。
    折被静默丢弃无法表达——宏平均行带 valid_folds/total_folds，逐折行每折
    都在。pooled 单独成列，不与逐折混写（§6.4.1）。

    envelope: 包级字段（schema_version/data_version/run_id/prediction_mode/data_kind）。
    """
    rows = []

    for f, rec in sorted(scores["per_fold"].items()):
        for metric in FLAT_METRICS:
            rows.append(_flat_row(
                envelope, metric=metric, aggregation="per_fold", fold=int(f),
                value=rec.get(metric),
                applicable=rec.get(f"{metric}_applicable", False),
                reason=rec.get(f"{metric}_reason"),
            ))
        rows.extend(_topk_rows(envelope, rec, q_list,
                               aggregation="per_fold", fold=int(f)))

    for metric, rec in scores["macro_mean"].items():
        value = rec.get("value")
        rows.append(_flat_row(
            envelope, metric=metric, aggregation="macro_mean", value=value,
            applicable=value is not None,
            reason=None if value is not None else "无有效折",
            valid_folds=rec.get("valid_folds"), total_folds=rec.get("total_folds"),
        ))
        if rec.get("fold_std") is not None:
            rows.append(_flat_row(
                envelope, metric=metric, aggregation="fold_std",
                value=rec["fold_std"], applicable=True,
                valid_folds=rec.get("valid_folds"),
                total_folds=rec.get("total_folds"),
            ))

    for metric, value in scores["sample_weighted"].items():
        rows.append(_flat_row(
            envelope, metric=metric, aggregation="sample_weighted", value=value,
            applicable=value is not None,
            reason=None if value is not None else "无有效折",
        ))

    pooled = scores["pooled_oof_replay"] or {}
    for metric in ("n", "n_positive") + FLAT_METRICS:
        v = pooled.get(metric)
        rows.append(_flat_row(
            envelope, metric=metric, aggregation="pooled_oof_replay", value=v,
            applicable=v is not None,
            reason=pooled.get(f"{metric}_reason"),
        ))
    rows.extend(_topk_rows(envelope, pooled, q_list,
                           aggregation="pooled_oof_replay"))
    return rows