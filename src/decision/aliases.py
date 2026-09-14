"""字段别名归一化。

夹具（tests/fixtures/）在 `attributes` 内使用可读键
`pipeage`/`material`/`joint`/`diameter_mm`/`road_type`/`facility`/`opsst`/`press_mpa`；
正式发布包使用原始列码 `PIPEAGE`/`CZ`/`JOINTT`/`GJ`/`RDTYPE`/`FAC`/`OPSST`/`PRESS`。
本模块把两种拼写归一到同一组规范键，决策模块内部只认规范键。
"""

from .errors import DecisionError

# 规范键 -> 可接受的别名（正式列码在前）
ALIASES = {
    "pipeage": ("PIPEAGE", "pipeage"),
    "material": ("CZ", "material"),
    "joint": ("JOINTT", "joint"),
    "diameter_mm": ("GJ", "diameter_mm"),
    "road_type": ("RDTYPE", "road_type"),
    "facility": ("FAC", "facility"),
    "opsst": ("OPSST", "opsst"),
    "press_mpa": ("PRESS", "press_mpa"),
}

# 分级与建议必需的字段（缺失触发 D01 档案核查，不默认满足条件）
REQUIRED_FIELDS = ("pipeage", "material", "joint", "diameter_mm",
                   "road_type", "facility", "opsst")


def canonical_key(key):
    """把任意拼写映射到规范键；未知键原样返回。"""
    for canonical, names in ALIASES.items():
        if key in names:
            return canonical
    return key


def normalize_attributes(attributes):
    """把 attributes 字典的键归一为规范键。

    同一规范键出现多个拼写时，若取值不一致则拒绝（禁止静默择一）。
    不修改入参，返回新字典。
    """
    if attributes is None:
        return {}
    if not isinstance(attributes, dict):
        raise DecisionError(f"attributes 必须是 dict，得到 {type(attributes).__name__}")

    out = {}
    seen_from = {}
    for key, value in attributes.items():
        canonical = canonical_key(key)
        if canonical in out and out[canonical] != value:
            raise DecisionError(
                f"字段 {canonical} 存在冲突拼写 {seen_from[canonical]!r} 与 {key!r}"
                f"（取值不一致），拒绝接入"
            )
        out[canonical] = value
        seen_from[canonical] = key
    return out


def normalize_view(view):
    """把一条管段视图归一为 {pipe_id, attributes, quality_flags}。

    接受标准属性包行（含 `attributes` 子字典），也接受属性直接平铺的行。
    不修改入参。
    """
    if not isinstance(view, dict):
        raise DecisionError(f"管段视图必须是 dict，得到 {type(view).__name__}")

    pipe_id = view.get("pipe_id")
    if pipe_id is None or pipe_id == "":
        raise DecisionError("管段视图缺少 pipe_id")
    pipe_id = str(pipe_id)

    nested = view.get("attributes")
    if nested is not None:
        attributes = normalize_attributes(nested)
    else:
        flat = {k: v for k, v in view.items()
                if k not in ("pipe_id", "bh", "source_refs", "quality_flags",
                             "schema_version", "data_version", "run_id",
                             "prediction_mode", "data_kind")}
        attributes = normalize_attributes(flat)

    flags = view.get("quality_flags") or []
    if not isinstance(flags, list):
        raise DecisionError(f"{pipe_id}: quality_flags 必须是 list")

    return {"pipe_id": pipe_id, "attributes": attributes, "quality_flags": list(flags)}


def normalize_views(pipe_views):
    """归一化一组管段视图；重复 pipe_id 拒绝接入（§13.3）。"""
    if pipe_views is None:
        return []
    if isinstance(pipe_views, dict):
        pipe_views = list(pipe_views.values())

    out = []
    seen = set()
    for view in pipe_views:
        normalized = normalize_view(view)
        if normalized["pipe_id"] in seen:
            raise DecisionError(f"重复 pipe_id {normalized['pipe_id']!r}，拒绝按行号对齐")
        seen.add(normalized["pipe_id"])
        out.append(normalized)
    return out


def rows_by_pipe_id(rows, what="记录"):
    """把产物行列表索引成 pipe_id -> 行；重复或缺失 pipe_id 拒绝接入。"""
    if rows is None:
        return {}
    if isinstance(rows, dict):
        return {str(k): v for k, v in rows.items()}

    out = {}
    for row in rows:
        if not isinstance(row, dict):
            raise DecisionError(f"{what}行必须是 dict，得到 {type(row).__name__}")
        pipe_id = row.get("pipe_id")
        if pipe_id is None or pipe_id == "":
            raise DecisionError(f"{what}行缺少 pipe_id")
        pipe_id = str(pipe_id)
        if pipe_id in out:
            raise DecisionError(f"{what}重复 pipe_id {pipe_id!r}，拒绝接入")
        out[pipe_id] = row
    return out


def as_float(value):
    """把数值转成 float；bool 与非数值返回 None。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)