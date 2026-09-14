"""核查建议规则（§8.4）与事后运维模式（§8.5）。

VERSION: advice_rules_v1

输入
----
- `advise_predictive(pipe_views, predictions, grades, priority_result, rules)`：
  只读取预测、属性、质量标记和情景参数，**不读取目标期事件**；O01 默认关闭。
- `advise_post_event(pipe_views, event_view, as_of, rules)`：
  独立事后入口，必须给出合法 `as_of`（不隐式取系统当前日）。

输出
----
建议记录列表，每条恰含（§8.4）：

    pipe_id, rule_id, rule_version, advice_mode, as_of, trigger_values,
    evidence_refs, suggested_action, preconditions_to_check,
    prohibited_inferences, action_key, prediction_run_id, reference_id, scenario_id

规则执行顺序固定为 D01、R01、R02、R03、O01、D02、R00，不用此顺序暗示施工紧迫程度。
同一管段允许多条建议；按 `action_key` 合并重复动作并保留全部触发依据。

可运行样例
----------
    from src.decision import advise_predictive, load_advice_rules
    advice = advise_predictive(views, preds, grades, priority, load_advice_rules())
    [a["rule_id"] for a in advice]        # 按固定顺序，如 ["R02", "R00"]

失败处理
--------
- 缺失证据**不把条件默认为满足**：必需字段整键缺失 → 触发 D01 数据核查，不补写事实。
- 事后模式未给合法 `as_of`、或事件视图 `as_of` 与请求不一致 → 抛 `DecisionError` 拒绝开启。
- `advice_mode=post_event_review` 之外，O01 永不触发；截止日之后的事件不可见。
"""

from src.contracts import ADVICE_RULES_VERSION

from .aliases import as_float, normalize_view, rows_by_pipe_id
from .consequence import compute_consequence, validate_scenario
from .errors import DecisionError

MODE_PREDICTIVE = "predictive"
MODE_POST_EVENT = "post_event_review"

# §8.5：事后模式默认显式截止日；不隐式取系统当前日，必须由调用方显式传入。
DEFAULT_AS_OF = "2024-12-31"

RULE_ORDER = ("D01", "R01", "R02", "R03", "O01", "D02", "R00")

HIGH_GRADE_CODES = ("L3", "L4")
MATERIAL_MATCH = ("铸铁管", "自应力管")
JOINT_MATCH = "承插"
OPSST_MATCH = ("偏高", "异常")
FACILITY_MATCH = ("学校", "医院", "交通枢纽", "政府机关")

RECORD_FIELDS = (
    "pipe_id", "rule_id", "rule_version", "advice_mode", "as_of",
    "trigger_values", "evidence_refs", "suggested_action",
    "preconditions_to_check", "prohibited_inferences", "action_key",
    "prediction_run_id", "reference_id", "scenario_id",
)

# 属性整键缺失 → 触发 D01 数据核查（不把条件默认为满足）
REQUIRED_ATTR_KEYS = ("pipeage", "material", "joint", "diameter_mm",
                      "road_type", "facility", "opsst")

# D01 触发的字段冲突标记。只收**逐管段**的冲突：
# coordinate_conflict 来自几何包逐边标记（§4），semantic_conflict /
# field_semantic_conflict 来自单字段的语义冲突。
#
# 不收 semantics_unverified：它是**全表**标记（ZDMS 语义未核，真实数据
# 7,288 条全带，见 configs/features/whitelist.json），若按逐管段冲突处理
# 会让 D01 命中 100% 管段，恰是 D01 自身 prohibited_inferences 禁止的
# “为每根管段制造独立事故告警”。全表标记属数据集级待核，不是单管段问题。
CONFLICT_FLAGS = ("coordinate_conflict", "semantic_conflict",
                  "field_semantic_conflict")


def validate_rules(rules):
    """校验建议规则配置。返回归一化副本，不修改入参。"""
    if not isinstance(rules, dict):
        raise DecisionError("rules 必须是 dict")

    version = rules.get("version")
    if not isinstance(version, str) or not version:
        raise DecisionError("rules 缺少字符串 version")

    order = rules.get("execution_order")
    if not isinstance(order, list) or list(order) != list(RULE_ORDER):
        raise DecisionError(
            f"rules.execution_order 必须恰为 {list(RULE_ORDER)}，得到 {order!r}")

    raw_rules = rules.get("rules")
    if not isinstance(raw_rules, dict):
        raise DecisionError("rules.rules 必须是 dict")
    for rule_id in RULE_ORDER:
        if rule_id not in raw_rules:
            raise DecisionError(f"rules.rules 缺少 {rule_id}")
        entry = raw_rules[rule_id]
        if not isinstance(entry, dict):
            raise DecisionError(f"rules.rules[{rule_id}] 必须是 dict")
        for key in ("suggested_action", "preconditions_to_check",
                    "prohibited_inferences", "action_key"):
            if key not in entry:
                raise DecisionError(f"规则 {rule_id} 缺少 {key}")

    return {
        "version": version,
        "order": list(order),
        "rules": {rid: dict(raw_rules[rid]) for rid in RULE_ORDER},
        "high_grade_codes": tuple(rules.get("high_grade_codes", HIGH_GRADE_CODES)),
    }


def _record(pipe_id, rule_id, cfg, mode, as_of, trigger_values, evidence_refs,
            ctx):
    """构造一条建议记录，字段与 §8.4 的清单一致。"""
    entry = cfg["rules"][rule_id]
    return {
        "pipe_id": pipe_id,
        "rule_id": rule_id,
        "rule_version": cfg["version"],
        "advice_mode": mode,
        "as_of": as_of,
        "trigger_values": trigger_values,
        "evidence_refs": evidence_refs,
        "suggested_action": entry["suggested_action"],
        "preconditions_to_check": list(entry["preconditions_to_check"]),
        "prohibited_inferences": list(entry["prohibited_inferences"]),
        "action_key": entry["action_key"],
        "prediction_run_id": ctx.get("prediction_run_id"),
        "reference_id": ctx.get("reference_id"),
        "scenario_id": ctx.get("scenario_id"),
    }


def _evaluate(pipe_id, attrs, flags, grade_row, consequence, cfg, mode, as_of,
              event_count=None, ctx=None):
    """对一条管段按固定顺序求触发规则。返回建议记录列表。"""
    ctx = dict(ctx or {})
    records = []

    missing_keys = [k for k in REQUIRED_ATTR_KEYS if k not in attrs]
    conflict_flags = [f for f in flags if f in CONFLICT_FLAGS]

    level = grade_row.get("relative_risk_level", "unavailable")
    grade_status = grade_row.get("grade_status")
    p_value = ctx.get("p")
    prediction_invalid = bool(ctx.get("prediction_invalid"))

    material = attrs.get("material")
    joint = attrs.get("joint")
    opsst = attrs.get("opsst")
    facility = attrs.get("facility")

    material_match = material in MATERIAL_MATCH
    joint_match = isinstance(joint, str) and JOINT_MATCH in joint
    opsst_match = opsst in OPSST_MATCH
    facility_match = facility in FACILITY_MATCH
    grade_high = level in cfg["high_grade_codes"]

    # --- D01 档案核查 ---------------------------------------------------
    d01_evidence = []
    if missing_keys:
        d01_evidence.append(
            {"kind": "missing_required_field", "fields": sorted(missing_keys)})
    if prediction_invalid:
        d01_evidence.append(
            {"kind": "prediction_invalid", "value": p_value,
             "reason": grade_row.get("unavailable_reason")})
    if grade_status == "unavailable" and not prediction_invalid:
        d01_evidence.append(
            {"kind": "grade_unavailable", "value": None,
             "reason": grade_row.get("unavailable_reason")})
    if conflict_flags:
        d01_evidence.append({"kind": "field_conflict", "flags": sorted(conflict_flags)})
    if d01_evidence:
        records.append(_record(
            pipe_id, "D01", cfg, mode, as_of,
            {"missing_required_field": sorted(missing_keys) or None,
             "prediction_invalid": prediction_invalid,
             "prediction_value": p_value,
             "conflict_flags": sorted(conflict_flags) or None},
            [f"standard_attributes:{pipe_id}/attributes.{k}" for k in sorted(missing_keys)]
            + [f"predictions:{pipe_id}/p", f"quality_flags:{pipe_id}"]
            + [f"geometry:{pipe_id}/coordinate_conflict"],
            ctx))

    # --- R01 材料与接口核查 ---------------------------------------------
    if grade_high and (material_match or joint_match):
        evidence = []
        if material_match:
            evidence.append(f"standard_attributes:{pipe_id}/attributes.material={material}")
        if joint_match:
            evidence.append(f"standard_attributes:{pipe_id}/attributes.joint={joint}")
        records.append(_record(
            pipe_id, "R01", cfg, mode, as_of,
            {"relative_risk_level": level, "material": material,
             "material_match": material_match, "joint": joint,
             "joint_match": joint_match},
            evidence, ctx))

    # --- R02 运行状态核查 -----------------------------------------------
    if opsst_match:
        records.append(_record(
            pipe_id, "R02", cfg, mode, as_of,
            {"opsst": opsst, "opsst_is_snapshot": True,
             "press_mpa": attrs.get("press_mpa")},
            [f"standard_attributes:{pipe_id}/attributes.opsst={opsst}",
             f"standard_attributes:{pipe_id}/attributes.press_mpa"],
            ctx))

    # --- R03 重要设施核查 -----------------------------------------------
    if grade_high and facility_match:
        records.append(_record(
            pipe_id, "R03", cfg, mode, as_of,
            {"relative_risk_level": level, "facility": facility,
             "facility_match": True},
            [f"standard_attributes:{pipe_id}/attributes.facility={facility}"],
            ctx))

    # --- O01 历史事件复核（仅事后模式） ---------------------------------
    if mode == MODE_POST_EVENT and event_count is not None and event_count >= 1:
        records.append(_record(
            pipe_id, "O01", cfg, mode, as_of,
            {"pre_as_of_event_count": int(event_count), "as_of": as_of,
             "repeat_event_check": bool(event_count >= 2)},
            [f"event_view:{pipe_id}/event_count={int(event_count)}",
             f"event_view:{pipe_id}/as_of={as_of}"],
            ctx))

    # --- D02 后果信息待核 -----------------------------------------------
    if consequence and consequence.get("proxy_imputed"):
        records.append(_record(
            pipe_id, "D02", cfg, mode, as_of,
            {"imputed_fields": list(consequence["imputed_fields"]),
             "unknown_proxy": consequence["components"],
             "sensitivity": consequence["sensitivity"]},
            [f"standard_attributes:{pipe_id}/attributes.{f}"
             for f in consequence["imputed_fields"]]
            + [f"scenario:{consequence['scenario_id']}/unknown_proxy"],
            ctx))

    # --- R00 一般核查 ---------------------------------------------------
    if not records:
        selected = ctx.get("in_selection")
        records.append(_record(
            pipe_id, "R00", cfg, mode, as_of,
            {"triggered_rules": [], "in_selection": bool(selected),
             "selection_objective": ctx.get("selection_objective"),
             "priority_value": ctx.get("priority_value")},
            [f"advice_rules:{cfg['version']}/R00"],
            ctx))

    return records


def _merge_by_action_key(records):
    """按 (pipe_id, action_key) 合并重复动作，保留全部触发依据（§8.4）。"""
    merged = []
    index = {}
    for record in records:
        key = (record["pipe_id"], record["action_key"])
        if key not in index:
            index[key] = len(merged)
            merged.append(record)
            continue
        target = merged[index[key]]
        target["trigger_values"] = _merge_trigger_values(
            target["trigger_values"], record["trigger_values"])
        for ref in record["evidence_refs"]:
            if ref not in target["evidence_refs"]:
                target["evidence_refs"].append(ref)
    return merged


def _merge_trigger_values(left, right):
    out = dict(left)
    for key, value in right.items():
        if key not in out:
            out[key] = value
        elif out[key] != value:
            existing = out[key] if isinstance(out[key], list) else [out[key]]
            extra = value if isinstance(value, list) else [value]
            out[key] = existing + [v for v in extra if v not in existing]
    return out


def _sort_by_rule_order(records, order):
    position = {rule_id: i for i, rule_id in enumerate(order)}
    return sorted(records, key=lambda r: (r["pipe_id"],
                                          position.get(r["rule_id"], len(order))))


def advise_predictive(pipe_views, predictions, grades, priority_result, rules):
    """预测回放建议（§8.4）。不读取目标期事件，O01 关闭。纯函数，不修改入参。"""
    cfg = validate_rules(rules)
    pred_rows = rows_by_pipe_id(predictions, "预测")
    grade_rows = rows_by_pipe_id(grades, "分级")

    priority = priority_result or {}
    records_by_pipe = priority.get("records") or {}
    selected = set((priority.get("selection") or {}).get("selected") or [])
    objective = (priority.get("selection") or {}).get("objective")

    out = []
    for view in pipe_views or []:
        normalized = normalize_view(view)
        pipe_id = normalized["pipe_id"]
        attrs = normalized["attributes"]
        flags = normalized["quality_flags"]

        pred_row = pred_rows.get(pipe_id, {})
        grade_row = grade_rows.get(pipe_id, {})
        priority_record = records_by_pipe.get(pipe_id) or {}

        p_raw = pred_row.get("p")
        p_value = as_float(p_raw)
        prediction_invalid = (p_value is None or not 0.0 <= p_value <= 1.0)

        ctx = {
            "prediction_run_id": grade_row.get("prediction_run_id")
            or pred_row.get("run_id"),
            "reference_id": grade_row.get("reference_id"),
            "scenario_id": priority_record.get("scenario_id"),
            "p": p_raw,
            "prediction_invalid": prediction_invalid,
            "in_selection": pipe_id in selected,
            "selection_objective": objective,
            "priority_value": priority_record.get("priority_value"),
        }

        out.extend(_evaluate(
            pipe_id, attrs, flags, grade_row, priority_record, cfg,
            MODE_PREDICTIVE, None, None, ctx))

    return _sort_by_rule_order(_merge_by_action_key(out), cfg["order"])


def _resolve_as_of(as_of):
    if as_of is None or as_of == "":
        raise DecisionError(
            "事后运维模式必须给出显式 as_of，未给合法截止日则拒绝开启（§8.5）")
    if not isinstance(as_of, str):
        raise DecisionError(f"as_of 必须是 'YYYY-MM-DD' 字符串，得到 {as_of!r}")
    parts = as_of.split("-")
    if len(parts) != 3 or not all(p.isdigit() for p in parts):
        raise DecisionError(f"as_of={as_of!r} 格式非法，期望 'YYYY-MM-DD'")
    year, month, day = (int(p) for p in parts)
    if not (1 <= month <= 12 and 1 <= day <= 31):
        raise DecisionError(f"as_of={as_of!r} 不是合法日期")
    return as_of, (year, month, day)


def _pre_as_of_count(row, as_of, cutoff):
    """截止日之前的事件数。截止日之后的事件不可见（§8.5）。"""
    events = row.get("events")
    if isinstance(events, list):
        count = 0
        for event in events:
            date = event.get("date") if isinstance(event, dict) else event
            if not isinstance(date, str):
                raise DecisionError(f"事件记录缺少可比较的 date：{event!r}")
            _, event_key = _resolve_as_of(date)
            if event_key <= cutoff:
                count += 1
        return count

    view_as_of = row.get("as_of")
    if view_as_of != as_of:
        raise DecisionError(
            f"事件视图 as_of={view_as_of!r} 与请求 {as_of!r} 不一致，拒绝接入")
    raw = row.get("event_count")
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
        raise DecisionError(f"事件视图 event_count 非法：{raw!r}")
    return raw


def advise_post_event(pipe_views, event_view, as_of, rules):
    """事后运维建议（§8.5）。独立清单，不改变 p/等级/C/V/预测清单。纯函数。"""
    cfg = validate_rules(rules)
    resolved, cutoff = _resolve_as_of(as_of)

    if event_view is None:
        raise DecisionError("事后运维模式必须提供 event_view")

    event_rows = rows_by_pipe_id(event_view, "事件视图")
    counts = {pid: _pre_as_of_count(row, resolved, cutoff)
              for pid, row in event_rows.items()}

    out = []
    for view in pipe_views or []:
        normalized = normalize_view(view)
        pipe_id = normalized["pipe_id"]
        attrs = normalized["attributes"]
        flags = normalized["quality_flags"]

        ctx = {
            "prediction_run_id": None,
            "reference_id": None,
            "scenario_id": None,
            "p": None,
            "prediction_invalid": False,
        }
        out.extend(_evaluate(
            pipe_id, attrs, flags, {}, None, cfg,
            MODE_POST_EVENT, resolved, counts.get(pipe_id, 0), ctx))

    return _sort_by_rule_order(_merge_by_action_key(out), cfg["order"])