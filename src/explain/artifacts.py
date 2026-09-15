"""解释产物构建（§13.4 explanation、§7.1）。

尺度固定为原始输出（log_odds）。校准集成模型分别解释各基础模型，
在**相同原始尺度**上平均贡献，命名为「基础模型平均原始分数归因」——
平均原始分数不等于平均校准概率的逆变换，不得声称贡献加和等于最终概率。
"""

import numpy as np

from src.contracts.versions import ExplainStatus, Scale, SCHEMA_VERSION
from src.data import loader
from src.explain.attribution import (
    background_sample,
    check_additivity,
    explain_model,
    explainable,
)

ATTRIBUTION_NAME = "基础模型平均原始分数归因"
NOT_FINAL_PROBABILITY_FLAG = "explains_mean_raw_score_not_final_probability"
TRUNCATED_FLAG = "contributions_truncated_to_top_k"

# 每根管段保留的贡献条数。全量 58 维 × 7288 条约 40MB，界面不可用；
# 截断后仍保留全量加和误差与省略量，供审计核对。
DEFAULT_TOP_K = 10


def _field_of(feature_name):
    """'num:PIPEAGE' / 'cat:CZ=球墨铸铁' -> 字段代码。"""
    body = feature_name.split(":", 1)[-1]
    return body.split("=", 1)[0]


def _align_contributions(outs):
    """把各基础模型的贡献对齐到特征名并集，缺失补 0（§7.1）。

    每个基础模型在各自的校准折上拟合预处理，保留类别集合可能不同，
    因此特征宽度不一致。对齐后取均值仍满足加和性：
    mean(base_k + Σ_k) = mean(raw_k)。
    """
    names = []
    for out in outs:
        for name in out["feature_names"]:
            if name not in names:
                names.append(name)

    n_rows = outs[0]["contributions"].shape[0]
    stack = np.zeros((len(outs), n_rows, len(names)), dtype=float)
    for k, out in enumerate(outs):
        pos = {name: j for j, name in enumerate(names)}
        stack[k][:, [pos[name] for name in out["feature_names"]]] = out["contributions"]
    return names, stack.mean(axis=0)


def _unavailable_row(pipe_id, model_id, *, data_version, run_id, reason):
    return {
        "schema_version": SCHEMA_VERSION,
        "data_version": data_version,
        "run_id": run_id,
        "prediction_mode": "oof_replay",
        "data_kind": "real_standard",
        "pipe_id": pipe_id,
        "model_id": model_id,
        "base_value": None,
        "contributions": [],
        "scale": Scale.LOG_ODDS.value,
        "status": ExplainStatus.UNAVAILABLE.value,
        "quality_flags": [reason],
    }


def build_explanation_rows(pipes, oof, fold_state, *, data_version, run_id,
                           seed, n_background=100, top_k=DEFAULT_TOP_K):
    """按外层折构建解释行（§7.1）。

    每折用该折训练集抽背景、解释该折测试管段；校准集成按同尺度平均贡献。
    无可解释结构时显式记 unavailable，不用别的模型顶替（§13.5）。
    """
    rows = []
    oof_by_id = {r.pipe_id: r for r in oof.itertuples()}

    for f in sorted(fold_state):
        state = fold_state[f]
        train_idx, test_idx = state["train_index"], state["test_index"]
        models = state["models"]
        X_tr = pipes.iloc[train_idx]
        X_te = pipes.iloc[test_idx]
        test_ids = [str(pipes["ID"].iloc[i]) for i in test_idx]

        if not models or not all(explainable(m) for m in models):
            rows.extend(
                _unavailable_row(pid, oof_by_id[pid].model_id,
                                 data_version=data_version, run_id=run_id,
                                 reason="explanation_pending")
                for pid in test_ids
            )
            continue

        bg = background_sample(X_tr, n=n_background, seed=seed)
        outs = [explain_model(m, bg, X_te) for m in models]

        base_value = float(np.mean([o["base_value"] for o in outs]))
        names, contributions = _align_contributions(outs)
        # 加和核验直接对着已发布的 raw_score（runner 对基础模型原始分数取均值）
        raw_mean = np.array([oof_by_id[pid].raw_score for pid in test_ids])
        additivity = check_additivity(base_value, contributions, raw_mean)

        for i, pid in enumerate(test_ids):
            row_contrib = contributions[i]
            order = np.argsort(-np.abs(row_contrib))[:top_k]
            omitted = float(row_contrib.sum() - row_contrib[order].sum())

            oof_row = oof_by_id[pid]
            flags = [NOT_FINAL_PROBABILITY_FLAG]
            if len(order) < len(names):
                flags.append(TRUNCATED_FLAG)
            flags.extend(oof_row.quality_flags or [])

            rows.append({
                "schema_version": SCHEMA_VERSION,
                "data_version": data_version,
                "run_id": run_id,
                "prediction_mode": "oof_replay",
                "data_kind": "real_standard",
                "pipe_id": pid,
                "model_id": oof_row.model_id,
                "base_value": base_value,
                "contributions": [
                    {
                        "feature": names[j],
                        "field": _field_of(names[j]),
                        "group": loader.feature_group(_field_of(names[j])),
                        "value": float(row_contrib[j]),
                        "direction": "positive" if row_contrib[j] >= 0 else "negative",
                    }
                    for j in order
                ],
                "scale": Scale.LOG_ODDS.value,
                "status": ExplainStatus.OK.value,
                "quality_flags": flags,
                # 契约外附加：审计与界面需要的全量口径
                "attribution_name": ATTRIBUTION_NAME,
                "n_base_models": len(models),
                "n_contributions_total": int(len(names)),
                "omitted_contribution_sum": omitted,
                "raw_score": float(oof_row.raw_score),
                "additivity_max_abs_error": additivity,
            })

    return rows


def verify_rows_additivity(rows, *, tol=1e-5):
    """审计：解释行截断前的加和误差必须在容差内（§7.1）。"""
    return [r["pipe_id"] for r in rows
            if r["status"] == ExplainStatus.OK.value
            and r.get("additivity_max_abs_error", 0.0) > tol]