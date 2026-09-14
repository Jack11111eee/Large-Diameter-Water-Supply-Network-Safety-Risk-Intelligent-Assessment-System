"""后果代理（§8.1）。

VERSION: consequence_scenario_v1

    S_i = w_D * d_i + w_F * f_i + w_R * r_i
    C_i = c_min + (1 - c_min) * S_i
    w_D, w_F, w_R >= 0 且总和为 1；0 <= d, f, r <= 1；0 < c_min <= 1
    因此 c_min <= C_i <= 1，C_i 严格为正。

零档表示相对最低档，不表示爆管没有后果；正值下限避免最低档把高概率管段优先值机械归零。

输入
----
- 一条已归一化的管段视图（见 `aliases.normalize_view`）或原始属性字典。
- `scenario`：`configs/decision/scenario_config.json` 的内容。

输出
----
一条后果记录：`consequence_proxy`（C）、`consequence_score`（S）、三个分量档位、
`proxy_imputed`、`imputed_fields`、`pending_verification`、`scenario_id`、
`sensitivity`（unknown-as-0 / unknown-as-1 / c_min 0.1 / c_min 0.4 上下情景）。

可运行样例
----------
    from src.decision import load_scenario_config
    from src.decision.consequence import compute_consequence
    scenario = load_scenario_config()
    compute_consequence({"pipe_id": "P1", "attributes": {"diameter_mm": 1200,
        "road_type": "主干道", "facility": "医院"}}, scenario)
    # S = 1.0 -> C = 0.2 + 0.8*1.0 = 1.0

失败处理
--------
配置非法（权重和不为 1、下限越界、映射值超出 [0,1]、非有限数值）时抛 `DecisionError`。
管段侧的未知/缺失/越界**不**报错：走代填路径并标记，绝不自动外推。
"""

from fractions import Fraction

from src.contracts import SCENARIO_CONFIG_VERSION

from .aliases import as_float
from .errors import DecisionError

# 分量状态
OK = "ok"
UNKNOWN = "unknown"          # 值缺失或不在映射中
OUT_OF_RANGE = "out_of_range"  # 值存在但超出当前管径范围

DIMENSIONS = ("diameter", "road", "facility")


def _rational(raw, what):
    """把 {num, den} 或数值转成 Fraction。"""
    if isinstance(raw, dict):
        num = as_float(raw.get("num"))
        den = as_float(raw.get("den"))
        if num is None or den is None or den == 0:
            raise DecisionError(f"{what}: 有理数表示非法 {raw!r}")
        return Fraction(int(num), int(den))
    value = as_float(raw)
    if value is None:
        raise DecisionError(f"{what}: 期望数值，得到 {raw!r}")
    return Fraction(value)


def _unit(value, what):
    if not 0 <= value <= 1:
        raise DecisionError(f"{what}: {float(value)} 超出 [0,1]")
    return value


def validate_scenario(scenario):
    """校验情景配置。返回归一化副本，不修改入参。

    已归一化的配置带 `_validated` 标记，重复校验时直接返回，避免二次转换。
    """
    if not isinstance(scenario, dict):
        raise DecisionError("scenario 必须是 dict")
    if scenario.get("_validated") is True:
        return scenario

    version = scenario.get("version")
    if not isinstance(version, str) or not version:
        raise DecisionError("scenario 缺少字符串 version")

    c_min = _unit(_rational(scenario.get("c_min"), "c_min"), "c_min")
    if c_min <= 0:
        raise DecisionError(f"c_min={float(c_min)} 必须 > 0")

    raw_weights = scenario.get("weights")
    if not isinstance(raw_weights, dict):
        raise DecisionError("scenario.weights 必须是 dict")
    weights = {}
    for name in ("w_D", "w_F", "w_R"):
        if name not in raw_weights:
            raise DecisionError(f"scenario.weights 缺少 {name}")
        weights[name] = _unit(_rational(raw_weights[name], name), name)
    total = weights["w_D"] + weights["w_F"] + weights["w_R"]
    if total != 1:
        raise DecisionError(f"权重和必须为 1，得到 {float(total)}")

    unknown = _unit(_rational(scenario.get("unknown_proxy"), "unknown_proxy"),
                    "unknown_proxy")

    diameter_map = []
    raw_diameter = scenario.get("diameter_map")
    if not isinstance(raw_diameter, list) or not raw_diameter:
        raise DecisionError("scenario.diameter_map 必须是非空列表")
    for i, item in enumerate(raw_diameter):
        if not isinstance(item, dict):
            raise DecisionError(f"diameter_map[{i}] 必须是 dict")
        lower = as_float(item.get("lower"))
        upper = as_float(item.get("upper"))
        if lower is None or upper is None or not lower < upper:
            raise DecisionError(f"diameter_map[{i}] 区间非法")
        diameter_map.append({
            "lower": lower,
            "upper": upper,
            "lower_inclusive": bool(item.get("lower_inclusive", True)),
            "upper_inclusive": bool(item.get("upper_inclusive", False)),
            "value": _unit(_rational(item.get("value"), f"diameter_map[{i}].value"),
                           f"diameter_map[{i}].value"),
        })

    def category_map(name):
        raw = scenario.get(name)
        if not isinstance(raw, dict) or not raw:
            raise DecisionError(f"scenario.{name} 必须是非空 dict")
        return {str(k): _unit(_rational(v, f"{name}[{k!r}]"), f"{name}[{k!r}]")
                for k, v in raw.items()}

    sensitivity = scenario.get("sensitivity") or {}
    c_min_alts = [_unit(_rational(v, "c_min_alternatives"), "c_min_alternatives")
                  for v in sensitivity.get("c_min_alternatives", [])]
    unknown_alts = [_unit(_rational(v, "unknown_proxy_alternatives"),
                          "unknown_proxy_alternatives")
                    for v in sensitivity.get("unknown_proxy_alternatives", [])]

    return {
        "_validated": True,
        "version": version,
        "c_min": c_min,
        "weights": weights,
        "unknown_proxy": unknown,
        "diameter_map": diameter_map,
        "road_map": category_map("road_map"),
        "facility_map": category_map("facility_map"),
        "c_min_alternatives": c_min_alts,
        "unknown_proxy_alternatives": unknown_alts,
    }


def _diameter_proxy(raw, cfg):
    """管径档位。越界不自动外推。"""
    if raw is None:
        return cfg["unknown_proxy"], UNKNOWN
    value = as_float(raw)
    if value is None:
        return cfg["unknown_proxy"], UNKNOWN
    for band in cfg["diameter_map"]:
        lo_ok = value >= band["lower"] if band["lower_inclusive"] else value > band["lower"]
        hi_ok = value <= band["upper"] if band["upper_inclusive"] else value < band["upper"]
        if lo_ok and hi_ok:
            return band["value"], OK
    # 超出当前范围：不自动外推，走待核与情景代填路径
    return cfg["unknown_proxy"], OUT_OF_RANGE


def _category_proxy(raw, mapping, cfg):
    if raw is None or raw == "":
        return cfg["unknown_proxy"], UNKNOWN
    key = str(raw)
    if key in mapping:
        return mapping[key], OK
    return cfg["unknown_proxy"], UNKNOWN


def _score(components, cfg, overrides=None):
    """按分量档位求 S。overrides 用于上下情景。"""
    d = components["diameter"]
    f = components["facility"]
    r = components["road"]
    if overrides:
        d = overrides.get("diameter", d)
        f = overrides.get("facility", f)
        r = overrides.get("road", r)
    return (cfg["weights"]["w_D"] * d
            + cfg["weights"]["w_F"] * f
            + cfg["weights"]["w_R"] * r)


def _to_c(score, c_min):
    return c_min + (1 - c_min) * score


def compute_consequence(view, scenario):
    """一条管段的后果代理。纯函数，不修改入参。

    view 可以是 `aliases.normalize_view` 的结果，也可以是 `{pipe_id, attributes}`。
    """
    cfg = validate_scenario(scenario)

    if not isinstance(view, dict):
        raise DecisionError(f"管段视图必须是 dict，得到 {type(view).__name__}")
    pipe_id = view.get("pipe_id")
    if pipe_id is None or pipe_id == "":
        raise DecisionError("管段视图缺少 pipe_id")
    pipe_id = str(pipe_id)

    attrs = view.get("attributes")
    if attrs is None:
        attrs = view
    if not isinstance(attrs, dict):
        raise DecisionError(f"{pipe_id}: attributes 必须是 dict")

    # 夹具用可读键，正式包用列码；此处按两种拼写取值
    def pick(*keys):
        for key in keys:
            if key in attrs:
                return attrs[key]
        return None

    diameter_raw = pick("diameter_mm", "GJ")
    road_raw = pick("road_type", "RDTYPE")
    facility_raw = pick("facility", "FAC")

    d_value, d_status = _diameter_proxy(diameter_raw, cfg)
    r_value, r_status = _category_proxy(road_raw, cfg["road_map"], cfg)
    f_value, f_status = _category_proxy(facility_raw, cfg["facility_map"], cfg)

    components = {"diameter": d_value, "road": r_value, "facility": f_value}
    status = {"diameter": d_status, "road": r_status, "facility": f_status}

    score = _score(components, cfg)
    proxy = _to_c(score, cfg["c_min"])

    # 只有"未知/缺失类别"标 proxy_imputed（§8.1 表）；越界走待核路径并单列标记，
    # 否则 D02 会因越界触发，与 expected_rules.json 的 F08 预期不符。
    imputed_fields = [name for name in DIMENSIONS if status[name] == UNKNOWN]
    out_of_range_fields = [name for name in DIMENSIONS if status[name] == OUT_OF_RANGE]

    # 上下情景：所有代填/越界档位分别取 0 与 1
    sweep = {name: value for name, value in components.items()
             if status[name] in (UNKNOWN, OUT_OF_RANGE)}
    as_zero = {name: Fraction(0) for name in sweep}
    as_one = {name: Fraction(1) for name in sweep}

    sensitivity = {
        "unknown_as_0": {"consequence_score": _score(components, cfg, as_zero),
                         "consequence_proxy": _to_c(_score(components, cfg, as_zero),
                                                    cfg["c_min"])},
        "unknown_as_1": {"consequence_score": _score(components, cfg, as_one),
                         "consequence_proxy": _to_c(_score(components, cfg, as_one),
                                                    cfg["c_min"])},
    }
    for alt in cfg["c_min_alternatives"]:
        sensitivity[f"c_min_{float(alt):g}"] = {
            "consequence_score": score,
            "consequence_proxy": _to_c(score, alt),
        }

    return {
        "pipe_id": pipe_id,
        "consequence_proxy": float(proxy),
        "consequence_score": float(score),
        "components": {k: float(v) for k, v in components.items()},
        "component_status": dict(status),
        "proxy_imputed": bool(imputed_fields),
        "imputed_fields": imputed_fields,
        "proxy_out_of_range": bool(out_of_range_fields),
        "out_of_range_fields": out_of_range_fields,
        "pending_verification": bool(imputed_fields or out_of_range_fields),
        "scenario_id": cfg["version"],
        "sensitivity": {k: {kk: float(vv) for kk, vv in v.items()}
                        for k, v in sensitivity.items()},
    }


def compute_consequences(pipe_views, scenario):
    """批量后果代理，返回 pipe_id -> 记录。不修改入参。"""
    cfg = validate_scenario(scenario)
    out = {}
    for view in pipe_views or []:
        record = compute_consequence(view, cfg)
        out[record["pipe_id"]] = record
    return out