"""概率/后果/综合优先值与数量预算 Top-K（§8.2、§8.4）。

VERSION: decision_priority_v1

    V_i = p_i * C_i   （无量纲排序代理，不是金额损失）

输入
----
- `pipe_views`：标准属性包行（含 `attributes`），两种字段拼写均可。
- `predictions`：预测包行；`p` 缺失/非有限/越界者不参与按 p 与按 V 的排序。
- `grades`：`grade()` 输出。
- `scenario`：`configs/decision/scenario_config.json` 的内容。
- `selection_request`：`{"objective": "p"|"C"|"V", "k": int, "seed": int,
  "filters": {字段: [允许值, ...]}}`；`objective` 默认 `"V"`。

输出
----
字典，含三条**分开**的清单 `by_p` / `by_c` / `by_v`、数量预算选择 `selection`
（含 `k_requested`、`k_actual`、`truncated`、`selected`），以及始终独立保留的
`preserved_by_p_topk`（按 p 的 Top-K）与 `high_grade_view`（L3/L4）。
`marked_not_in_value` 标出前述两个视图里未进入按 V 清单的管段。

并列分数用与标签独立的稳定键 `SHA256(UTF8(str(seed) + ":" + str(pipe_id)))` 排序，
不使用可能随进程变化的内置字符串 hash（§6.4）。

可运行样例
----------
    from src.decision import prioritize, load_scenario_config
    result = prioritize(views, preds, grades, load_scenario_config(),
                        {"objective": "V", "k": 2})
    result["selection"]["selected"]      # 至多 2 个 pipe_id，无重复

失败处理
--------
`k` 为负、`objective` 非法、筛选字段未知时抛 `DecisionError`。
空候选集或 `k=0` 返回空清单；`k` 超过候选数时显式截为候选总数，
同时保留请求数量与实际数量，不伪造补足记录（§6.4）。
"""

import hashlib

from .consequence import compute_consequence, validate_scenario
from .errors import DecisionError

DEFAULT_SEED = 20260914
OBJECTIVES = {"p": "p", "C": "C", "V": "V"}
HIGH_GRADES = ("L3", "L4")

# 可筛选字段（规范化后的属性键）
FILTERABLE = ("material", "road_type", "diameter_mm", "facility", "opsst")


def stable_tie_key(pipe_id, seed):
    """并列分数的稳定排序键（§6.4），独立于标签。"""
    return hashlib.sha256(f"{seed}:{pipe_id}".encode("utf-8")).hexdigest()


def _rank(records, field, seed):
    """按 field 降序、SHA256 并列打散排序。返回有序的 pipe_id 列表。"""
    return [r["pipe_id"] for r in
            sorted(records, key=lambda r: (-r[field], stable_tie_key(r["pipe_id"], seed)))]


def _objective_field(objective):
    return {"p": "p", "C": "consequence_proxy", "V": "priority_value"}[objective]


def _match_filters(attrs, filters):
    for field, allowed in filters.items():
        if field not in FILTERABLE:
            raise DecisionError(
                f"不支持按 {field!r} 筛选；可筛选字段：{list(FILTERABLE)}")
        if attrs.get(field) not in allowed:
            return False
    return True


def _validate_request(selection_request):
    if selection_request is None:
        selection_request = {}
    if not isinstance(selection_request, dict):
        raise DecisionError("selection_request 必须是 dict")

    objective = selection_request.get("objective", "V")
    if objective not in OBJECTIVES:
        raise DecisionError(
            f"objective={objective!r} 非法，必须是 {sorted(OBJECTIVES)} 之一")

    raw_k = selection_request.get("k", 0)
    if isinstance(raw_k, bool) or not isinstance(raw_k, int):
        raise DecisionError(f"k 必须是 int，得到 {type(raw_k).__name__}")
    if raw_k < 0:
        raise DecisionError(f"k={raw_k} 不能为负")

    seed = selection_request.get("seed", DEFAULT_SEED)
    filters = selection_request.get("filters") or {}
    if not isinstance(filters, dict):
        raise DecisionError("filters 必须是 dict")

    return {"objective": objective, "k": raw_k, "seed": seed, "filters": filters}


def prioritize(pipe_views, predictions, grades, scenario, selection_request=None):
    """数量预算 Top-K（§8.2）。纯函数，不修改入参。"""
    cfg = validate_scenario(scenario)
    request = _validate_request(selection_request)

    from .aliases import as_float, rows_by_pipe_id, normalize_view

    pred_rows = rows_by_pipe_id(predictions, "预测")
    grade_rows = rows_by_pipe_id(grades, "分级")

    records = []
    excluded = []
    for view in pipe_views or []:
        normalized = normalize_view(view)
        pipe_id = normalized["pipe_id"]
        attrs = normalized["attributes"]

        if not _match_filters(attrs, request["filters"]):
            continue

        consequence = compute_consequence(normalized, cfg)
        pred = pred_rows.get(pipe_id, {})
        p = as_float(pred.get("p"))
        p_valid = p is not None and 0.0 <= p <= 1.0

        grade_row = grade_rows.get(pipe_id, {})
        record = {
            "pipe_id": pipe_id,
            "consequence_proxy": consequence["consequence_proxy"],
            "consequence_score": consequence["consequence_score"],
            "relative_risk_level": grade_row.get("relative_risk_level", "unavailable"),
            "risk_percentile": grade_row.get("risk_percentile"),
            "scenario_id": consequence["scenario_id"],
            "proxy_imputed": consequence["proxy_imputed"],
            "imputed_fields": list(consequence["imputed_fields"]),
            "pending_verification": consequence["pending_verification"],
            "sensitivity": consequence["sensitivity"],
            "components": dict(consequence["components"]),
            "component_status": dict(consequence["component_status"]),
            "reference_id": grade_row.get("reference_id"),
            "prediction_run_id": grade_row.get("prediction_run_id"),
            "evidence_refs": list(grade_row.get("evidence_refs", [])),
        }

        if p_valid:
            record["p"] = p
            record["priority_value"] = p * consequence["consequence_proxy"]
        else:
            record["p"] = None
            record["priority_value"] = None
            excluded.append({
                "pipe_id": pipe_id,
                "reason": f"预测值非法（{pred.get('p')!r}）：不参与按 p 与按 V 的排序",
            })
        records.append(record)

    seed = request["seed"]
    by_p = _rank([r for r in records if r["p"] is not None], "p", seed)
    by_c = _rank(records, "consequence_proxy", seed)
    by_v = _rank([r for r in records if r["priority_value"] is not None],
                 "priority_value", seed)

    objective = request["objective"]
    field = _objective_field(objective)
    ranked = _rank([r for r in records if r[field] is not None], field, seed)

    k_requested = request["k"]
    # 候选数为筛选后的全部管段；可排序数另计（p 无效者不参与按 p/按 V 排序）
    n_candidates = len(records)
    n_rankable = len(ranked)
    k_actual = min(k_requested, n_rankable)
    selected = ranked[:k_actual]

    # 按 p 的 Top-K 与 L3/L4 视图始终独立保留，不被后果加权视图覆盖（§8.1）
    preserved_by_p = by_p[:min(k_requested, len(by_p))] if k_requested else []
    by_rank = {r["pipe_id"]: r for r in records}
    high_grade_view = [pid for pid in by_v
                       if by_rank[pid]["relative_risk_level"] in HIGH_GRADES]

    selected_set = set(selected)
    marked = [pid for pid in preserved_by_p + high_grade_view
              if pid not in selected_set]
    # 去重但保持首次出现顺序
    marked_not_in_value = list(dict.fromkeys(marked))

    return {
        "scenario_id": cfg["version"],
        "objective": objective,
        "seed": seed,
        "filters": dict(request["filters"]),
        "candidate_count": n_candidates,
        "rankable_count": n_rankable,
        "selection": {
            "objective": objective,
            "k_requested": k_requested,
            "k_actual": k_actual,
            "truncated": k_actual < k_requested,
            "selected": list(selected),
        },
        "by_p": by_p,
        "by_c": by_c,
        "by_v": by_v,
        "preserved_by_p_topk": preserved_by_p,
        "high_grade_view": high_grade_view,
        "marked_not_in_value": marked_not_in_value,
        "records": {r["pipe_id"]: r for r in records},
        "excluded": excluded,
    }