"""产物 schema 与校验。

依据 §13.3 公共约定、§13.4 输入输出契约表、§5.3 字段语义。
不引入 jsonschema 依赖：字段声明式定义，校验逻辑手写。
"""

import math
from dataclasses import dataclass, field
from typing import Any

from .versions import (
    SCHEMA_VERSION,
    DataKind,
    PredictionMode,
    Scale,
    ExplainStatus,
)


class ContractError(ValueError):
    """契约违反。接入方必须拒绝并说明原因（§13.3）。"""


# --------------------------------------------------------------------------
# 字段类型
# --------------------------------------------------------------------------

class F:
    """字段类型标记。"""

    STR = "str"
    INT = "int"
    FLOAT = "float"
    BOOL = "bool"
    LIST = "list"
    DICT = "dict"


_PY_TYPES = {
    F.STR: str,
    F.INT: int,
    F.FLOAT: (int, float),
    F.BOOL: bool,
    F.LIST: list,
    F.DICT: dict,
}

# 包级公共字段：每个产物都必须带（§13.3）。
ENVELOPE = (
    ("schema_version", F.STR, True, None),
    ("data_version", F.STR, True, None),
    ("run_id", F.STR, True, None),
    ("prediction_mode", F.STR, True, {m.value for m in PredictionMode}),
    ("data_kind", F.STR, True, {k.value for k in DataKind}),
)

# 来源追踪（§3.1）：每个特征记录来源。
SOURCE_FIELDS = (
    ("source_file", F.STR, True, None),
    ("source_sheet", F.STR, True, None),
    ("source_column", F.STR, True, None),
    ("unit", F.STR, True, None),
    ("derivation", F.STR, True, None),
    ("observed_at_status", F.STR, True, None),
    ("quality_flags", F.LIST, True, None),
)


@dataclass(frozen=True)
class Product:
    """一个产物的字段契约。"""

    name: str
    producer: str
    consumers: tuple
    # (字段名, 类型, 是否必需, 枚举集合或 None)
    fields: tuple = field(default_factory=tuple)
    # 作为唯一键的字段；None 表示行级产物无唯一键要求
    unique_key: str | None = None

    def all_fields(self):
        return ENVELOPE + self.fields


# --------------------------------------------------------------------------
# 产物注册表（§13.4）
# --------------------------------------------------------------------------

PRODUCTS = {
    "standard_attributes": Product(
        name="standard_attributes",
        producer="A",
        consumers=("B", "C"),
        fields=(
            ("pipe_id", F.STR, True, None),
            ("bh", F.STR, True, None),
            ("attributes", F.DICT, True, None),
            ("source_refs", F.DICT, True, None),
            ("quality_flags", F.LIST, True, None),
        ),
        unique_key="pipe_id",
    ),
    "geometry": Product(
        name="geometry",
        producer="A",
        consumers=("C",),
        fields=(
            ("pipe_id", F.STR, True, None),
            ("x_start", F.FLOAT, True, None),
            ("y_start", F.FLOAT, True, None),
            ("x_end", F.FLOAT, True, None),
            ("y_end", F.FLOAT, True, None),
            ("component_id", F.INT, True, None),
            ("coordinate_conflict", F.BOOL, True, None),
            # CRS 未知必须显式声明（§4）
            ("crs_known", F.BOOL, True, None),
        ),
        unique_key="pipe_id",
    ),
    "labels_and_splits": Product(
        name="labels_and_splits",
        producer="A",
        consumers=("A",),
        fields=(
            ("pipe_id", F.STR, True, None),
            ("y_true", F.INT, True, {0, 1}),
            ("outer_fold", F.INT, True, None),
            ("count_2024", F.INT, True, None),
        ),
        unique_key="pipe_id",
    ),
    "predictions": Product(
        name="predictions",
        producer="A",
        consumers=("B", "C"),
        fields=(
            ("pipe_id", F.STR, True, None),
            ("p", F.FLOAT, False, None),
            ("model_id", F.STR, True, None),
            ("round_id", F.STR, True, None),
            ("fold", F.INT, False, None),
            ("calibrated", F.BOOL, True, None),
            ("quality_flags", F.LIST, True, None),
        ),
        unique_key="pipe_id",
    ),
    "reference_bundle": Product(
        name="reference_bundle",
        producer="A",
        consumers=("B",),
        fields=(
            ("reference_id", F.STR, True, None),
            ("model_run_id", F.STR, True, None),
            ("grade_config_version", F.STR, True, None),
            ("pipe_ids", F.LIST, True, None),
            ("scores", F.LIST, True, None),
            ("data_fingerprint", F.STR, True, None),
        ),
    ),
    "explanation": Product(
        name="explanation",
        producer="A",
        consumers=("C", "B"),
        fields=(
            ("pipe_id", F.STR, True, None),
            ("model_id", F.STR, True, None),
            ("base_value", F.FLOAT, False, None),
            ("contributions", F.LIST, True, None),
            ("scale", F.STR, True, {s.value for s in Scale}),
            ("status", F.STR, True, {s.value for s in ExplainStatus}),
            ("quality_flags", F.LIST, True, None),
        ),
        unique_key="pipe_id",
    ),
    "decision": Product(
        name="decision",
        producer="B",
        consumers=("C", "A"),
        fields=(
            ("pipe_id", F.STR, True, None),
            ("risk_percentile", F.FLOAT, False, None),
            ("relative_risk_level", F.STR, True, None),
            ("consequence_proxy", F.FLOAT, False, None),
            ("priority_value", F.FLOAT, False, None),
            ("scenario_id", F.STR, True, None),
            ("evidence_refs", F.LIST, True, None),
        ),
        unique_key="pipe_id",
    ),
    "evaluation": Product(
        name="evaluation",
        producer="A",
        consumers=("C",),
        fields=(
            ("metric", F.STR, True, None),
            ("aggregation", F.STR, True,
             {"per_fold", "macro_mean", "fold_std",
              "sample_weighted", "pooled_oof_replay"}),
            ("fold", F.INT, False, None),
            ("value", F.FLOAT, False, None),
            ("applicable", F.BOOL, True, None),
            ("reason", F.STR, False, None),
            # 宏平均行的有效折计数：任何折被静默丢弃都必须看得见（§6.4.1）
            ("valid_folds", F.INT, False, None),
            ("total_folds", F.INT, False, None),
        ),
    ),
    "event_view": Product(
        name="event_view",
        producer="A",
        consumers=("B",),
        fields=(
            ("pipe_id", F.STR, True, None),
            ("event_count", F.INT, True, None),
            ("as_of", F.STR, True, None),
        ),
        unique_key="pipe_id",
    ),
}


# --------------------------------------------------------------------------
# 校验
# --------------------------------------------------------------------------

def assert_no_nonfinite(payload: Any, _path: str = "$") -> None:
    """JSON 中不传 NaN/Infinity（§13.3）。"""

    if isinstance(payload, float):
        if math.isnan(payload) or math.isinf(payload):
            raise ContractError(f"{_path}: 出现非有限数值 {payload!r}；请用 null + quality_flags")
    elif isinstance(payload, dict):
        for k, v in payload.items():
            assert_no_nonfinite(v, f"{_path}.{k}")
    elif isinstance(payload, (list, tuple)):
        for i, v in enumerate(payload):
            assert_no_nonfinite(v, f"{_path}[{i}]")


def _check_field(name, value, ftype, required, enum, path):
    if value is None:
        if required:
            raise ContractError(f"{path}.{name}: 必需字段缺失")
        return
    if ftype == F.INT:
        # bool 是 int 子类，必须排除
        if isinstance(value, bool) or not isinstance(value, int):
            raise ContractError(f"{path}.{name}: 期望 int，得到 {type(value).__name__}")
    elif ftype == F.FLOAT:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ContractError(f"{path}.{name}: 期望 float，得到 {type(value).__name__}")
    else:
        expected = _PY_TYPES[ftype]
        if not isinstance(value, expected):
            raise ContractError(f"{path}.{name}: 期望 {ftype}，得到 {type(value).__name__}")
    if enum is not None and value not in enum:
        raise ContractError(f"{path}.{name}: {value!r} 不在允许集合 {sorted(enum)}")


def validate_package(product_name: str, rows: list, *, schema_version: str = SCHEMA_VERSION) -> None:
    """校验一个产物包。不通过则抛 ContractError，接入方拒绝（§13.3）。

    rows 为记录列表；每行是一个 dict。
    """

    if product_name not in PRODUCTS:
        raise ContractError(f"未知产物类型 {product_name!r}")

    spec = PRODUCTS[product_name]
    if not isinstance(rows, list):
        raise ContractError(f"{product_name}: 包必须是记录列表")

    seen_keys = set()
    for i, row in enumerate(rows):
        path = f"{product_name}[{i}]"
        if not isinstance(row, dict):
            raise ContractError(f"{path}: 记录必须是 dict")
        assert_no_nonfinite(row, path)

        for fname, ftype, required, enum in spec.all_fields():
            _check_field(fname, row.get(fname), ftype, required, enum, path)

        # 版本不兼容必须拒绝（§13.3）
        if row.get("schema_version") != schema_version:
            raise ContractError(
                f"{path}.schema_version: 期望 {schema_version}，"
                f"得到 {row.get('schema_version')!r}"
            )

        # 唯一键
        if spec.unique_key:
            key = row.get(spec.unique_key)
            if key in seen_keys:
                raise ContractError(f"{path}.{spec.unique_key}: 重复唯一键 {key!r}")
            seen_keys.add(key)


def reject(reason: str) -> None:
    """统一拒绝入口。禁止按行号对齐或静默丢行（§13.3）。"""

    raise ContractError(reason)