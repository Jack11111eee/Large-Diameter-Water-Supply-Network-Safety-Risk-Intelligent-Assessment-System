"""候选选择与逐折运行（§5.1、§5.2）。

选择只用外层训练集内部信息：主依据为内层 AP，容差内优先简单模型。
本模块**不计算任何外层指标**——「用外层成绩回选模型」（M2 具名风险）
在结构上不可达：选择日志里没有外层键，折外表每管段只一行。
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.evaluation.metrics import NotApplicable, ap
from src.evaluation.runner import _fit_predict_fold
from src.evaluation.split import N_INNER, SEED, inner_splits, round_id_of
from src.models import registry


@dataclass(frozen=True)
class Candidate:
    """一个候选（候选名 + 一组具体参数）。"""

    name: str
    family: str
    complexity_rank: int
    params: dict = field(default_factory=dict)
    factory: object = None

    @property
    def label(self):
        body = ",".join(f"{k}={v}" for k, v in sorted(self.params.items()))
        return f"{self.name}[{body}]"

    def build(self):
        return self.factory(self.params)


def candidates_from_config(layer_columns, config=None, *, seed=SEED):
    """展开预注册候选与参数网格，按 (复杂度序, 名称, 参数) 定序。"""
    cfg = config or registry.load_candidates()
    out = []
    for spec in cfg["candidates"]:
        factory = registry.factory_of(spec, layer_columns, seed=seed)
        for params in spec["param_grid"]:
            out.append(Candidate(
                name=spec["name"],
                family=spec["family"],
                complexity_rank=int(spec["complexity_rank"]),
                params=dict(params),
                factory=factory,
            ))
    return sorted(out, key=lambda c: (c.complexity_rank, c.name, c.label))


def select_within_train(X_train, y_train, candidates, *, seed=SEED, groups=None,
                        tolerance=0.005, n_splits=N_INNER):
    """内层 AP 选择（§5.2）。只用外层训练集内部信息。

    返回 (选中候选, 选择日志)。日志不含任何外层成绩。
    """
    splits, status = inner_splits(y_train, groups=groups, n_splits=n_splits,
                                  seed=seed)
    log = {
        "criterion": "inner_ap",
        "tolerance": float(tolerance),
        "tie_break": "prefer_lower_complexity_rank",
        "grouped": groups is not None,
        "status": status,
        "n_inner_folds": len(splits),
        "candidate_order": [c.label for c in candidates],
        "per_candidate": {},
    }

    if status == "degenerate":
        chosen = candidates[0]
        log["chosen"] = chosen.label
        log["note"] = "内层折退化，按预定义顺序取首个候选（§6.1）"
        return chosen, log

    scores = {c.label: [] for c in candidates}
    for tr, va in splits:
        for c in candidates:
            model = c.build().fit(X_train.iloc[tr], y_train[tr])
            try:
                scores[c.label].append(
                    float(ap(y_train[va], model.predict_proba(X_train.iloc[va]))))
            except NotApplicable:
                continue

    means = {k: (float(np.mean(v)) if v else float("-inf"))
             for k, v in scores.items()}
    best = max(means.values())
    chosen = next(
        (c for c in candidates if means[c.label] >= best - tolerance),
        max(candidates, key=lambda c: means[c.label]),
    )

    log["per_candidate"] = {
        c.label: {"per_fold_ap": scores[c.label], "mean_ap": means[c.label]}
        for c in candidates
    }
    log["chosen"] = chosen.label
    log["chosen_name"] = chosen.name
    log["chosen_params"] = dict(chosen.params)
    log["tie_break_fired"] = bool(means[chosen.label] < best)
    return chosen, log


def select_and_run(pipes, counts, *, candidates, folds, seed=SEED, groups=None,
                   tolerance=0.005, calibrate=True, split_scheme="random",
                   return_fold_state=False):
    """逐外层折独立选择候选，再出折外预测。

    返回 (折外表, 选择日志[, 折状态])。**不计算任何外层成绩**——
    成绩由调用方在选定路径上另行聚合。
    """
    y = counts.gt(0).astype(int).to_numpy()
    pipe_ids = pipes["ID"].astype(str).to_numpy()
    if groups is not None:
        groups = np.asarray(groups, dtype=object)
        if len(groups) != len(pipe_ids):
            raise ValueError("groups 长度与管段数不一致")

    records = []
    selection_log = {}
    fold_state = {}

    for f, test_idx in folds.items():
        train_idx = np.setdiff1d(np.arange(len(pipe_ids)), test_idx)
        X_tr = pipes.iloc[train_idx].reset_index(drop=True)
        y_tr = y[train_idx]
        X_te = pipes.iloc[test_idx].reset_index(drop=True)
        groups_tr = None if groups is None else groups[train_idx]

        chosen, log = select_within_train(
            X_tr, y_tr, candidates, seed=seed, groups=groups_tr,
            tolerance=tolerance)
        selection_log[int(f)] = log

        p_cal, raw, state = _fit_predict_fold(
            X_tr, y_tr, X_te, chosen.build,
            calibrate=calibrate, seed=seed, groups_tr=groups_tr)

        if return_fold_state:
            fold_state[int(f)] = {
                "models": state["models"],
                "calibrators": state["calibrators"],
                "train_index": train_idx,
                "test_index": test_idx,
                "quality_flags": list(state["quality_flags"]),
                "chosen": chosen.label,
            }

        for i, idx in enumerate(test_idx):
            records.append({
                "pipe_id": pipe_ids[idx],
                "outer_fold": int(f),
                "y_true": int(y[idx]),
                "p": float(p_cal[i]),
                "raw_score": float(raw[i]),
                "model_id": chosen.name,
                "model_params": chosen.label,
                "calibrated": bool(state["calibrated"]),
                "round_id": round_id_of(seed, split_scheme),
                "quality_flags": list(state["quality_flags"]),
            })

    df = pd.DataFrame(records).sort_values(
        "pipe_id", kind="mergesort").reset_index(drop=True)
    if not df["pipe_id"].is_unique:
        raise ValueError("折外预测出现重复管段")
    if len(df) != len(pipe_ids):
        raise ValueError(f"折外预测数 {len(df)} != 管段数 {len(pipe_ids)}")
    if df["p"].isna().any() or not np.isfinite(df["p"]).all():
        raise ValueError("折外预测含非有限值")
    if ((df["p"] < 0) | (df["p"] > 1)).any():
        raise ValueError("折外预测概率超出 [0,1]")

    if return_fold_state:
        return df, selection_log, fold_state
    return df, selection_log