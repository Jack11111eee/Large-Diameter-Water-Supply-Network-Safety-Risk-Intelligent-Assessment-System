"""界面适配器：唯一的数据装载、校验与连接入口。

依据 §13.3（冻结接口包）、§13.4（输入输出契约）、§13.6（界面独立验收）、§4（拓扑契约）。

本模块的纪律：

- **不做任何业务计算。** 百分位、后果代理 C、优先值 V、建议规则均由 `src/decision/`
  唯一实现；界面只展示其结果。本模块只做装载、校验、按 `pipe_id` 连接与视图整理。
- **不导入 Streamlit**，可脱离 Streamlit 独立测试。
- 版本、模式、唯一键不匹配时**拒绝接入并说明原因**；禁止按行号对齐或静默丢行。
- 不做任何原始数据读取：不读两个 XLSX，不导入 `src.models` / `src.data`。
"""

from __future__ import annotations

import csv
import io
import json
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from src.contracts import ContractError, validate_package

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DIR = ROOT / "outputs" / "release"
FIXTURE_DIR = ROOT / "tests" / "fixtures"
DECISION_CONFIG_DIR = ROOT / "configs" / "decision"
ENV_PACKAGE_DIR = "APP_PACKAGE_DIR"

# 契约产物 → 文件名
FILES = {
    "standard_attributes": "standard_attributes.json",
    "geometry": "geometry.json",
    "predictions": "predictions.json",
    "explanation": "explanation.json",
    "decision": "decision.json",
    "reference_bundle": "reference_bundle.json",
    "evaluation": "evaluation.json",
    "manifest": "manifest.json",
}

# 界面最小闭环所需产物；其余缺失时降级为局部不可用，不拒绝整包。
REQUIRED_PRODUCTS = ("standard_attributes", "predictions")

# 走契约校验的列表型产物
LIST_PRODUCTS = ("standard_attributes", "geometry", "predictions",
                 "explanation", "decision", "reference_bundle", "evaluation")

# 必须一致的包级字段：同一发布包的标识与口径。
# data_version 不列入：M0 核心的参考分布包以 round_id 记 data_version，
# 与其余产物合法地不同，故只记为警告并在界面显示。
CONSISTENCY_FIELDS = ("schema_version", "run_id", "prediction_mode", "data_kind")

# 属性字段双拼写别名（§13.3）：正式产物用字段代码，夹具用可读别名。
FIELD_ALIASES = {
    "pipeage": "PIPEAGE",
    "material": "CZ",
    "joint": "JOINTT",
    "diameter_mm": "GJ",
    "road_type": "RDTYPE",
    "facility": "FAC",
    "opsst": "OPSST",
    "press_mpa": "PRESS",
}
# 反向：字段代码 → 夹具别名
FIELD_CODES = {v: k for k, v in FIELD_ALIASES.items()}

LEVEL_UNAVAILABLE = "unavailable"


class AdapterError(ValueError):
    """接入拒绝。界面必须显示原因，不得静默降级或补行。"""


class PackageUnavailable(AdapterError):
    """产物**不存在**（而非存在但违反契约）。

    只有这一种失败允许回退到夹具目录；契约违反（版本、重复 ID、字段缺失、
    表间不一致）绝不回退——那会把真实问题藏在夹具后面（§13.3）。
    """


# --------------------------------------------------------------------------
# 视图对象
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class LoadReport:
    """本次装载的来源与版本标识。"""

    directory: str
    source: str                    # real_release | synthetic_fixture
    data_kind: str
    prediction_mode: str
    schema_version: str
    run_id: str
    data_version: str
    model_id: str | None
    missing_products: tuple = ()
    warnings: tuple = ()
    fallback_reason: str | None = None

    @property
    def is_fixture(self) -> bool:
        return self.source == "synthetic_fixture"

    @property
    def source_known(self) -> bool:
        return self.source in ("real_release", "synthetic_fixture")


@dataclass(frozen=True)
class ExplanationView:
    """解释视图。status 为 unavailable 时不得用别的模型顶替（§13.5）。"""

    status: str
    reason: str | None
    model_id: str | None
    scale: str | None
    base_value: float | None
    contributions: tuple


@dataclass(frozen=True)
class PipeView:
    """一条管段的展示视图：属性 + 几何 + 概率 + 解释 + 决策结果。"""

    pipe_id: str
    bh: str
    attributes: dict                 # 规范化键（字段代码）→ 值
    source_refs: dict
    quality_flags: tuple
    # 概率
    probability: float | None
    probability_available: bool
    model_id: str
    round_id: str
    fold: int | None
    calibrated: bool
    # 几何
    has_geometry: bool
    x_start: float | None
    y_start: float | None
    x_end: float | None
    y_end: float | None
    component_id: int | None
    coordinate_conflict: bool
    crs_known: bool
    # 决策结果（来自决策模块或冻结决策包，界面不重算）
    decision_available: bool
    risk_percentile: float | None
    relative_risk_level: str
    consequence_proxy: float | None
    priority_value: float | None
    scenario_id: str | None
    evidence_refs: tuple
    # 解释
    explanation: ExplanationView

    def attribute(self, key: str) -> Any:
        """按字段代码或夹具别名取值。"""
        if key in self.attributes:
            return self.attributes[key]
        return self.attributes.get(FIELD_ALIASES.get(key, key))


@dataclass(frozen=True)
class DecisionView:
    """决策结果视图。source 说明这批数值从哪来，界面据此标注。"""

    source: str          # decision_module | frozen_package | absent
    reason: str
    rows: dict           # pipe_id -> 决策行（等级、百分位、C、V）
    advice: dict         # pipe_id -> 建议记录（模块接入前为空）
    # 决策模块 prioritize() 的完整返回（含 by_p / high_grade_view /
    # marked_not_in_value 等独立视图）。界面直接展示，不重建（§5.3.1、§8.1）。
    priority_result: dict | None = None
    expected_rules: dict | None = None   # 夹具预期，仅作对照，非正式结果
    # 以不同 K / 目标重跑 prioritize() 的回调。界面改预算时调用它，
    # 而不是在页面里重排或重算（§13.1、§13.4）。
    selection_fn: object = None

    @property
    def covered_ids(self) -> frozenset:
        return frozenset(self.rows)

    def with_selection(self, k: int, objective: str) -> dict | None:
        """按给定预算与目标取决策模块的清单结果。无回调时返回 None。"""
        if self.selection_fn is None:
            return None
        return self.selection_fn(k, objective)


@dataclass(frozen=True)
class PackageSet:
    """一次装载得到的完整展示包。"""

    report: LoadReport
    pipes: tuple
    reference_bundle: dict | None
    evaluation: list | None
    manifest: dict | None
    decision: DecisionView

    @property
    def by_id(self) -> dict:
        return {p.pipe_id: p for p in self.pipes}

    def pipe(self, pipe_id: str) -> PipeView | None:
        return self.by_id.get(pipe_id)


# --------------------------------------------------------------------------
# 读取与校验
# --------------------------------------------------------------------------

def _read_json(path: Path):
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise PackageUnavailable(f"{path.name}: 读取失败：{exc}") from exc
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise AdapterError(f"{path.name}: JSON 解析失败：{exc}") from exc


def _read_payloads(directory: Path) -> tuple:
    payloads, missing = {}, []
    for product, fname in FILES.items():
        path = directory / fname
        if path.exists():
            payloads[product] = _read_json(path)
        else:
            missing.append(product)
    return payloads, missing


def _validate_products(payloads: dict) -> None:
    """契约校验：任一产物不通过则拒绝接入（§13.3）。"""
    for product in LIST_PRODUCTS:
        rows = payloads.get(product)
        if rows is None:
            continue
        if not isinstance(rows, list):
            raise AdapterError(f"{product}: 包必须是记录列表，得到 {type(rows).__name__}")
        try:
            validate_package(product, rows)
        except ContractError as exc:
            raise AdapterError(f"{product}: 契约校验未通过：{exc}") from exc


def _envelope_sources(payloads: dict) -> dict:
    """收集包级字段的实际取值，用于一致性检查。"""
    values = {}
    for product in LIST_PRODUCTS:
        rows = payloads.get(product)
        if not rows:
            continue
        for fieldname in CONSISTENCY_FIELDS + ("data_version",):
            values.setdefault(fieldname, {}).setdefault(rows[0].get(fieldname), []).append(product)
    manifest = payloads.get("manifest")
    if isinstance(manifest, dict):
        for fieldname in CONSISTENCY_FIELDS + ("data_version",):
            values.setdefault(fieldname, {}).setdefault(manifest.get(fieldname), []).append("manifest")
    return values


def _check_consistency(payloads: dict) -> tuple:
    """版本 / 模式不一致必须拒绝（§13.3）。返回警告列表。"""
    values = _envelope_sources(payloads)
    warnings = []
    for fieldname in CONSISTENCY_FIELDS:
        groups = values.get(fieldname, {})
        if len(groups) > 1:
            detail = "；".join(f"{v!r} ← {sorted(p)}" for v, p in sorted(groups.items(), key=str))
            raise AdapterError(f"{fieldname} 跨产物不一致，拒绝接入：{detail}")
    # data_version 允许不一致（参考分布包按 round_id 记），但必须显示
    groups = values.get("data_version", {})
    if len(groups) > 1:
        detail = "；".join(f"{v!r} ← {sorted(p)}" for v, p in sorted(groups.items(), key=str))
        warnings.append(f"data_version 跨产物不一致（不拒绝，仅提示）：{detail}")
    return warnings


def _single_value(values: dict, fieldname: str, default=None):
    groups = values.get(fieldname, {})
    return next(iter(groups), default)


def _check_ids(rows: list, product: str) -> list:
    ids = [r["pipe_id"] for r in rows]
    if len(ids) != len(set(ids)):
        seen, dup = set(), []
        for i in ids:
            if i in seen and i not in dup:
                dup.append(i)
            seen.add(i)
        raise AdapterError(f"{product}: pipe_id 重复，拒绝接入：{dup[:5]}")
    return ids


def _check_probability(pipe_id: str, value, flags, warnings: list):
    """概率必须为 null 或 [0,1] 内的有限数；越界拒绝，不用代填值（§13.3）。"""
    if value is None:
        if "prediction_invalid" not in flags:
            warnings.append(f"pipe_id={pipe_id}: p 为 null 但缺少 prediction_invalid 质量标记")
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AdapterError(f"predictions: pipe_id={pipe_id} 概率类型非法 {value!r}")
    p = float(value)
    if not 0.0 <= p <= 1.0:
        raise AdapterError(f"predictions: pipe_id={pipe_id} 概率 {p!r} 不在 [0,1]，拒绝接入")
    return p


def canonical_field(key: str) -> str:
    """字段代码与夹具别名统一到字段代码。"""
    return FIELD_ALIASES.get(key, key)


def canonical_attributes(raw: dict) -> dict:
    if not isinstance(raw, dict):
        raise AdapterError(f"attributes 必须是 dict，得到 {type(raw).__name__}")
    return {canonical_field(k): v for k, v in raw.items()}


# --------------------------------------------------------------------------
# 决策结果：调用决策模块，或退回到冻结决策包（不重算）
# --------------------------------------------------------------------------

def _try_import_decision():
    try:
        from src import decision as module  # noqa: PLC0415
    except Exception as exc:  # 模块尚未实现 / 依赖缺失
        return None, f"决策模块尚不可导入（{type(exc).__name__}: {exc}）"
    return module, ""


def _config_loader(module, name: str, config_dir: Path, filename: str):
    """取决策配置。

    配置归决策模块所有（§13.2 `configs/decision/`），因此优先用模块自带的加载器；
    模块未提供时再读本仓库的 configs/decision/ 下的同名文件。
    """
    loader = getattr(module, name, None)
    if loader is not None:
        try:
            config = loader()
        except Exception:
            config = None
        if isinstance(config, dict):
            return config
    path = config_dir / filename
    if path.exists():
        return _read_json(path)
    return None


def _grade_settings(module, config_dir: Path):
    """取分级配置（只装载，不解释其含义）。"""
    return _config_loader(module, "load_grade_config", config_dir, "grade_config.json")


def _as_records(result, what: str):
    """把模块返回值规范成记录列表。无法识别时返回 (None, 原因)。"""
    rows = result if isinstance(result, list) else getattr(result, "rows", None)
    if isinstance(result, dict):
        rows = result.get("rows")
    if not isinstance(rows, list) or not rows:
        return None, f"决策模块 {what} 未返回可识别的记录列表"
    if not all(isinstance(r, dict) and r.get("pipe_id") for r in rows):
        return None, f"决策模块 {what} 返回记录缺少 pipe_id"
    return rows, ""


def _advice_records(result, module):
    """取建议记录。优先用 grade/prioritize 的返回值，再退回模块的 advise_predictive()。"""
    raw = getattr(result, "advice", None)
    if raw is None and isinstance(result, dict):
        raw = result.get("advice")
    if isinstance(raw, list):
        return {a["pipe_id"]: a for a in raw
                if isinstance(a, dict) and a.get("pipe_id")}
    if isinstance(raw, dict):
        return raw
    return {}


def _call_decision_module(module, attrs_rows, predictions, reference_bundle,
                          config_dir: Path):
    """按 §13.4 的冻结签名调用决策模块。

    返回 (rows, advice, priority_result, selection_fn, reason)。
    只调用，不重算：任何一步失败都整体退回冻结决策包，绝不在界面里补算。
    """
    grade = getattr(module, "grade", None)
    if grade is None:
        return None, {}, None, None, "决策模块未提供 grade()"

    try:
        grades = grade(predictions, reference_bundle,
                       _grade_settings(module, config_dir))
    except Exception as exc:
        return None, {}, None, None, (f"决策模块 grade() 调用失败"
                                      f"（{type(exc).__name__}: {exc}）")
    grade_rows, why = _as_records(grades, "grade()")
    if grade_rows is None:
        return None, {}, None, None, why

    # 后果代理 C 与优先值 V 只能来自 prioritize()（§8.1、§13.1）
    prioritize = getattr(module, "prioritize", None)
    priority_result, priority_rows = None, []
    selection_fn = None
    if prioritize is not None:
        views = [{"pipe_id": r["pipe_id"], "attributes": r["attributes"],
                  "quality_flags": list(r.get("quality_flags") or ())}
                 for r in attrs_rows]
        scenario = _config_loader(module, "load_scenario_config",
                                  config_dir, "scenario_config.json")

        def run_prioritize(k=0, objective="V"):
            return prioritize(views, predictions, grade_rows, scenario,
                              {"k": int(k), "objective": objective})

        try:
            priority_result = run_prioritize()
        except Exception as exc:
            return None, {}, None, None, (f"决策模块 prioritize() 调用失败"
                                          f"（{type(exc).__name__}: {exc}）")
        selection_fn = run_prioritize
        records = (priority_result or {}).get("records") if isinstance(priority_result, dict) else None
        if isinstance(records, dict):
            priority_rows = list(records.values())
        elif isinstance(records, list):
            priority_rows = records

    # 合并：分级给等级与百分位，排序给 C、V 与情景版本
    merged = {r["pipe_id"]: dict(r) for r in grade_rows}
    for row in priority_rows:
        merged.setdefault(row["pipe_id"], {"pipe_id": row["pipe_id"]}).update(row)

    advice = _advice_records(grades, module)
    if not advice and priority_result is not None:
        advice = _advice_records(priority_result, module)
    return list(merged.values()), advice, priority_result, selection_fn, ""


def _load_decision_view(payloads: dict, directory: Path, attrs_rows: list,
                        predictions: list, reference_bundle,
                        *, allow_fixture: bool = True) -> DecisionView:
    module, reason = _try_import_decision()
    if module is not None:
        rows, advice, priority_result, selection_fn, why = _call_decision_module(
            module, attrs_rows, predictions, reference_bundle, DECISION_CONFIG_DIR)
        if rows is not None:
            return DecisionView(
                source="decision_module",
                reason="已接入 src/decision/，页面与导出直接使用其返回结果。",
                rows={r["pipe_id"]: r for r in rows},
                advice=advice,
                priority_result=priority_result,
                selection_fn=selection_fn,
            )
        reason = why

    # 退回到冻结决策包：优先本次装载目录，其次夹具目录（§13.5 的 M1 降级口径）
    rows = payloads.get("decision")
    origin = f"{directory.name}/decision.json"
    if rows is None and allow_fixture and FIXTURE_DIR.resolve() != directory.resolve():
        fixture_path = FIXTURE_DIR / FILES["decision"]
        if fixture_path.exists():
            rows = _read_json(fixture_path)
            origin = f"{FIXTURE_DIR.name}/decision.json"
    if rows:
        expected = None
        expected_path = FIXTURE_DIR / "expected_rules.json"
        if expected_path.exists():
            try:
                expected = _read_json(expected_path)
            except AdapterError:
                expected = None
        return DecisionView(
            source="frozen_package",
            reason=(f"{reason}；本页展示冻结决策包 {origin} 的结果对象，不在界面重算。"),
            rows={r["pipe_id"]: r for r in rows},
            advice={},
            expected_rules=expected,
        )

    return DecisionView(
        source="absent",
        reason=f"{reason}；且未找到任何决策结果包，相关字段显示 unavailable。",
        rows={},
        advice={},
    )


# --------------------------------------------------------------------------
# 连接
# --------------------------------------------------------------------------

def _explanation_view(explanation_by_id: dict, pipe_id: str, model_id: str) -> ExplanationView:
    row = explanation_by_id.get(pipe_id)
    if row is None:
        return ExplanationView(LEVEL_UNAVAILABLE, "解释包中没有该管段的记录", None, None, None, ())
    status = row.get("status")
    contributions = tuple(row.get("contributions") or ())
    if row.get("model_id") != model_id:
        # §13.5：不得用另一个模型的解释顶替（先于可用性判断，避免被状态掩盖）
        return ExplanationView(
            LEVEL_UNAVAILABLE,
            f"解释模型 {row.get('model_id')!r} 与预测模型 {model_id!r} 不一致，不予顶替",
            row.get("model_id"), row.get("scale"), None, ())
    if status != "ok":
        return ExplanationView(LEVEL_UNAVAILABLE, "解释状态为 unavailable（尚未就绪）",
                               row.get("model_id"), row.get("scale"),
                               row.get("base_value"), ())
    if not contributions:
        return ExplanationView(LEVEL_UNAVAILABLE, "解释状态为 ok 但无贡献项，按未就绪处理",
                               row.get("model_id"), row.get("scale"),
                               row.get("base_value"), ())
    return ExplanationView("ok", None, row.get("model_id"), row.get("scale"),
                           row.get("base_value"), contributions)


def _build_pipes(payloads: dict, decision: DecisionView) -> tuple:
    attrs = payloads["standard_attributes"]
    preds = {r["pipe_id"]: r for r in payloads["predictions"]}
    geom = {r["pipe_id"]: r for r in payloads.get("geometry") or []}
    expl = {r["pipe_id"]: r for r in payloads.get("explanation") or []}

    warnings = []
    pipes = []
    for row in attrs:
        pipe_id = row["pipe_id"]
        pred = preds.get(pipe_id)
        if pred is None:
            raise AdapterError(f"predictions: 缺少 pipe_id={pipe_id} 的记录，拒绝接入")
        g = geom.get(pipe_id)
        d = decision.rows.get(pipe_id)
        flags = tuple(row.get("quality_flags") or ())
        probability = _check_probability(pipe_id, pred.get("p"), pred.get("quality_flags") or [],
                                         warnings)

        pipes.append(PipeView(
            pipe_id=pipe_id,
            bh=str(row["bh"]),
            attributes=canonical_attributes(row["attributes"]),
            source_refs=row.get("source_refs") or {},
            quality_flags=flags,
            probability=probability,
            probability_available=probability is not None,
            model_id=str(pred["model_id"]),
            round_id=str(pred["round_id"]),
            fold=pred.get("fold"),
            calibrated=bool(pred["calibrated"]),
            has_geometry=g is not None,
            x_start=g.get("x_start") if g else None,
            y_start=g.get("y_start") if g else None,
            x_end=g.get("x_end") if g else None,
            y_end=g.get("y_end") if g else None,
            component_id=g.get("component_id") if g else None,
            coordinate_conflict=bool(g.get("coordinate_conflict")) if g else False,
            crs_known=bool(g.get("crs_known")) if g else False,
            decision_available=d is not None,
            risk_percentile=d.get("risk_percentile") if d else None,
            relative_risk_level=str(d["relative_risk_level"]) if d else LEVEL_UNAVAILABLE,
            consequence_proxy=d.get("consequence_proxy") if d else None,
            priority_value=d.get("priority_value") if d else None,
            scenario_id=d.get("scenario_id") if d else None,
            evidence_refs=tuple(d.get("evidence_refs") or ()) if d else (),
            explanation=_explanation_view(expl, pipe_id, str(pred["model_id"])),
        ))

    pipes.sort(key=_default_order)
    return tuple(pipes), tuple(warnings)


def _default_order(p: PipeView):
    """默认清单顺序：优先值降序 → 概率降序 → pipe_id 升序（确定性）。"""
    return (
        0 if p.priority_value is not None else 1,
        -(p.priority_value or 0.0),
        0 if p.probability is not None else 1,
        -(p.probability or 0.0),
        p.pipe_id,
    )


def _load_from(directory: Path, *, allow_decision_fixture: bool = True) -> PackageSet:
    directory = Path(directory)
    payloads, missing = _read_payloads(directory)

    absent = [p for p in REQUIRED_PRODUCTS if p not in payloads]
    if absent:
        raise PackageUnavailable(f"{directory}: 缺少必需产物 {absent}")

    _validate_products(payloads)
    warnings = list(_check_consistency(payloads))

    # 各表按 pipe_id 验证，而非按行位置拼接（§13.6）
    attrs = payloads["standard_attributes"]
    base_ids = _check_ids(attrs, "standard_attributes")
    for product in ("geometry", "predictions", "explanation", "decision"):
        rows = payloads.get(product)
        if rows is None:
            continue
        _check_ids(rows, product)

    pred_ids = [r["pipe_id"] for r in payloads["predictions"]]
    if set(pred_ids) != set(base_ids):
        raise AdapterError(
            f"predictions 与 standard_attributes 的 pipe_id 集合不一致："
            f"缺 {len(set(base_ids) - set(pred_ids))} 条，多 {len(set(pred_ids) - set(base_ids))} 条，"
            "拒绝接入（禁止按行号对齐）")
    if "geometry" in payloads:
        geom_ids = {r["pipe_id"] for r in payloads["geometry"]}
        if geom_ids != set(base_ids):
            raise AdapterError(
                f"geometry 与 standard_attributes 的 pipe_id 集合不一致："
                f"缺 {len(set(base_ids) - geom_ids)} 条，多 {len(geom_ids) - set(base_ids)} 条，拒绝接入")

    bundles = payloads.get("reference_bundle")
    if bundles is not None and len(bundles) > 1:
        raise AdapterError(f"reference_bundle: 期望单包，得到 {len(bundles)} 包，拒绝接入")
    reference_bundle = bundles[0] if bundles else None

    decision = (_load_decision_view(payloads, directory, attrs,
                                    payloads["predictions"], reference_bundle,
                                    allow_fixture=allow_decision_fixture)
                if base_ids else
                DecisionView(source="absent",
                             reason="standard_attributes 为空，不接入任何决策结果",
                             rows={}, advice={}))
    pipes, pipe_warnings = _build_pipes(payloads, decision)
    warnings.extend(pipe_warnings)

    if not pipes:
        warnings.append("standard_attributes 为空：页面显示空清单，不补任何占位行")

    values = _envelope_sources(payloads)
    data_kind = _single_value(values, "data_kind", "unknown")
    report = LoadReport(
        directory=str(directory),
        source="synthetic_fixture" if data_kind == "synthetic_fixture" else "real_release",
        data_kind=data_kind,
        prediction_mode=_single_value(values, "prediction_mode", "unknown"),
        schema_version=_single_value(values, "schema_version", "unknown"),
        run_id=_single_value(values, "run_id", "unknown"),
        data_version=_single_value(values, "data_version", "unknown"),
        model_id=(payloads.get("manifest") or {}).get("model_id")
        or (pipes[0].model_id if pipes else None),
        missing_products=tuple(missing),
        warnings=tuple(warnings),
    )
    return PackageSet(
        report=report,
        pipes=pipes,
        reference_bundle=reference_bundle,
        evaluation=payloads.get("evaluation"),
        manifest=payloads.get("manifest"),
        decision=decision,
    )


def _has_required(directory: Path) -> bool:
    return all((directory / FILES[p]).exists() for p in REQUIRED_PRODUCTS)


def load_package_set(directory=None, *, allow_fallback: bool = True,
                     allow_decision_fixture: bool = True) -> PackageSet:
    """装载一个展示包。

    默认目录为 outputs/release/；缺失或校验不通过时回退到 tests/fixtures/，
    并在 report 中标明实际来源与回退原因。两条路径都不重算任何业务量。
    """
    if directory is None:
        directory = os.environ.get(ENV_PACKAGE_DIR) or DEFAULT_DIR
    directory = Path(directory)

    try:
        return _load_from(directory, allow_decision_fixture=allow_decision_fixture)
    except PackageUnavailable as primary:
        # 只有"产物不存在"回退到夹具；契约违反绝不回退，否则真实问题被夹具掩盖
        if allow_fallback and directory.resolve() != FIXTURE_DIR.resolve() \
                and _has_required(FIXTURE_DIR):
            package = _load_from(FIXTURE_DIR, allow_decision_fixture=allow_decision_fixture)
            return replace(package, report=replace(
                package.report,
                fallback_reason=f"正式产物不可用（{primary}），已回退到冻结夹具。"))
        raise


# --------------------------------------------------------------------------
# 展示行与导出（同一结果对象，导出不再重算）
# --------------------------------------------------------------------------

def pipe_list_rows(package: PackageSet, *, levels=None, only_with_probability=False) -> list:
    """清单页使用的行。所有数值直接取自 PipeView，不做任何换算。"""
    rows = []
    for p in package.pipes:
        if levels is not None and p.relative_risk_level not in levels:
            continue
        if only_with_probability and not p.probability_available:
            continue
        rows.append({
            "pipe_id": p.pipe_id,
            "bh": p.bh,
            "p": p.probability,
            "相对等级": p.relative_risk_level,
            "百分位": p.risk_percentile,
            "后果代理C": p.consequence_proxy,
            "优先值V": p.priority_value,
            "分量": p.component_id,
            "坐标冲突": p.coordinate_conflict,
            "质量标记": "、".join(p.quality_flags),
        })
    return rows


def select_top_k(package: PackageSet, k: int, objective: str) -> list:
    """按目标列排序取前 K 条。

    这是对**已给出的展示列**做排序与截取，不计算任何新量。
    决策模块接入后，Top-K 由 prioritize() 返回，本函数退为界面排序。
    """
    key_of = {
        "p": lambda x: x.probability,
        "C": lambda x: x.consequence_proxy,
        "V": lambda x: x.priority_value,
    }
    if objective not in key_of:
        raise AdapterError(f"未知目标 {objective!r}，仅支持 p / C / V")
    get = key_of[objective]
    available = [x for x in package.pipes if get(x) is not None]
    available.sort(key=lambda x: (-get(x), x.pipe_id))
    return available[:max(0, int(k))]


def rows_to_csv(rows: list) -> str:
    """把展示行写成 CSV。传入什么就导出什么，不重新计算。"""
    if not rows:
        return ""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()