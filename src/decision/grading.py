"""相对风险等级（§5.3.1）。

VERSION: relative_grade_v1

输入
----
- `predictions`：预测包行列表（契约产物 `predictions`），含 `pipe_id`、`p`、`run_id` 等。
- `reference_bundle`：固定参考分布包行列表（契约产物 `reference_bundle`），恰一行，
  含 `reference_id`、`pipe_ids`、`scores`、`grade_config_version`。
- `grade_config`：`configs/decision/grade_config.json` 的内容。

输出
----
grade 行列表，每条含 `pipe_id`、`risk_percentile`、`relative_risk_level`、
`grade_status`、`unavailable_reason`、`evidence_refs`、`reference_id` 及包级字段。

定义（平均秩百分位，全精度分数参与计算，绝不先四舍五入）：

    percentile(p; R) = 100 * (count(r < p) + 0.5 * count(r == p)) / n

等级：L1 [0,80)、L2 [80,95)、L3 [95,99)、L4 [99,100]。
同一 R 内相同分数必然同等级。R 为空、少于 2 条或所有分数相同 → `unavailable`。
缺失、非有限或超出 [0,1] 的 p → 该管段 `unavailable`，不参与分级。

可运行样例
----------
    from src.decision import grade, load_grade_config
    rows = [{"schema_version": "1.0.0", "data_version": "d", "run_id": "r",
             "prediction_mode": "oof_replay", "data_kind": "synthetic_fixture",
             "pipe_id": "P1", "p": 0.9, "model_id": "m", "round_id": "1",
             "calibrated": True, "quality_flags": []}]
    ref = [{"schema_version": "1.0.0", "data_version": "d", "run_id": "r",
            "prediction_mode": "oof_replay", "data_kind": "synthetic_fixture",
            "reference_id": "ref-1", "model_run_id": "m",
            "grade_config_version": "relative_grade_v1",
            "pipe_ids": ["A", "B"], "scores": [0.1, 0.2],
            "data_fingerprint": "fp"}]
    grade(rows, ref, load_grade_config())   # -> [{"risk_percentile": 100.0, "relative_risk_level": "L4", ...}]

失败处理
--------
包级字段缺失、版本不兼容、参考分数非有限或超出 [0,1]、重复 pipe_id 时抛 `DecisionError`
（§13.3：拒绝接入并说明原因，禁止按行号对齐或静默丢行）。
"""

import math

from src.contracts import GRADE_CONFIG_VERSION

from .aliases import as_float, rows_by_pipe_id
from .errors import DecisionError

_ENVELOPE_KEYS = ("schema_version", "data_version", "run_id",
                  "prediction_mode", "data_kind")

UNAVAILABLE = "unavailable"


def _envelope_of(row, what):
    missing = [k for k in _ENVELOPE_KEYS if k not in row]
    if missing:
        raise DecisionError(f"{what}缺少包级字段 {missing}（§13.3），拒绝接入")
    return {k: row[k] for k in _ENVELOPE_KEYS}


def validate_grade_config(grade_config):
    """校验等级配置结构。返回归一化后的副本，不修改入参。"""
    if not isinstance(grade_config, dict):
        raise DecisionError("grade_config 必须是 dict")

    version = grade_config.get("version")
    if not isinstance(version, str) or not version:
        raise DecisionError("grade_config 缺少字符串 version")

    levels = grade_config.get("levels")
    if not isinstance(levels, list) or not levels:
        raise DecisionError("grade_config.levels 必须是非空列表")

    seen = set()
    for level in levels:
        if not isinstance(level, dict):
            raise DecisionError("grade_config.levels 每项必须是 dict")
        for key in ("code", "display_name", "lower", "upper"):
            if key not in level:
                raise DecisionError(f"等级项缺少 {key}")
        if level["code"] in seen:
            raise DecisionError(f"重复等级代码 {level['code']!r}")
        seen.add(level["code"])
        lower, upper = as_float(level["lower"]), as_float(level["upper"])
        if lower is None or upper is None or not (lower <= upper):
            raise DecisionError(f"等级 {level['code']!r} 的区间非法")

    ordered = sorted(levels, key=lambda x: float(x["lower"]))
    if float(ordered[0]["lower"]) != 0.0 or float(ordered[-1]["upper"]) != 100.0:
        raise DecisionError("等级区间必须完整覆盖 [0, 100]")

    return {
        "version": version,
        "levels": [dict(level) for level in ordered],
        "unavailable_code": grade_config.get("unavailable_code", UNAVAILABLE),
        "unavailable_display_name": grade_config.get(
            "unavailable_display_name", "参考分布不足或无区分度"),
        "min_reference_size": int(grade_config.get("min_reference_size", 2)),
    }


def assign_level(percentile, cfg):
    """按百分位落档。返回 (code, display_name)。"""
    for level in cfg["levels"]:
        lower = float(level["lower"])
        upper = float(level["upper"])
        lo_ok = percentile >= lower if level.get("lower_inclusive", True) else percentile > lower
        hi_ok = percentile <= upper if level.get("upper_inclusive", False) else percentile < upper
        if lo_ok and hi_ok:
            return level["code"], level["display_name"]
    raise DecisionError(f"百分位 {percentile!r} 未落在任何等级区间")


def percentile_of(p, reference_scores):
    """平均秩百分位（§5.3.1）。R 非空时恒有定义。"""
    n = len(reference_scores)
    below = 0
    equal = 0
    for r in reference_scores:
        if r < p:
            below += 1
        elif r == p:
            equal += 1
    return 100.0 * (below + 0.5 * equal) / n


def _reference_row(reference_bundle):
    if isinstance(reference_bundle, dict):
        return reference_bundle
    if isinstance(reference_bundle, (list, tuple)):
        if len(reference_bundle) != 1:
            raise DecisionError(
                f"参考分布包必须恰含 1 行，得到 {len(reference_bundle)} 行；"
                "不从子集重建参考分布（§5.3.1）")
        return reference_bundle[0]
    raise DecisionError(f"reference_bundle 类型非法：{type(reference_bundle).__name__}")


def _reference_state(reference_bundle, cfg):
    """校验参考分布并返回 (reference_id, scores, 不可用原因或 None)。"""
    ref = _reference_row(reference_bundle)
    _envelope_of(ref, "参考分布包")

    reference_id = ref.get("reference_id")
    if not isinstance(reference_id, str) or not reference_id:
        raise DecisionError("参考分布包缺少 reference_id")

    bundle_version = ref.get("grade_config_version")
    if bundle_version != cfg["version"]:
        raise DecisionError(
            f"参考分布 grade_config_version={bundle_version!r} 与等级配置 "
            f"{cfg['version']!r} 不兼容，拒绝接入")

    scores = ref.get("scores")
    if not isinstance(scores, list):
        raise DecisionError("参考分布包 scores 必须是列表")

    pipe_ids = ref.get("pipe_ids")
    if not isinstance(pipe_ids, list) or len(pipe_ids) != len(scores):
        raise DecisionError("参考分布包 pipe_ids 与 scores 长度不一致")

    clean = []
    for i, raw in enumerate(scores):
        value = as_float(raw)
        if value is None or not math.isfinite(value):
            raise DecisionError(f"参考分数[{i}] 非有限数值 {raw!r}，拒绝接入")
        if not 0.0 <= value <= 1.0:
            raise DecisionError(f"参考分数[{i}]={value} 超出 [0,1]，拒绝接入")
        clean.append(value)

    if len(clean) < cfg["min_reference_size"]:
        return reference_id, clean, (
            f"参考分布不足或无区分度：仅 {len(clean)} 条，少于 {cfg['min_reference_size']} 条")
    if len(set(clean)) < 2:
        return reference_id, clean, "参考分布不足或无区分度：参考分数全部相同"
    return reference_id, clean, None


def grade(predictions, reference_bundle, grade_config):
    """相对风险等级（§5.3.1）。纯函数，不修改入参。"""
    cfg = validate_grade_config(grade_config)
    pred_rows = rows_by_pipe_id(predictions, "预测")
    if not pred_rows:
        return []

    first = next(iter(pred_rows.values()))
    envelope = _envelope_of(first, "预测包")
    prediction_run_id = envelope.get("run_id")

    reference_id, ref_scores, ref_reason = _reference_state(reference_bundle, cfg)

    out = []
    for pipe_id, row in pred_rows.items():
        evidence = [f"predictions:{prediction_run_id}:{pipe_id}",
                    f"reference:{reference_id}"]

        if ref_reason is not None:
            out.append({
                **envelope,
                "pipe_id": pipe_id,
                "risk_percentile": None,
                "relative_risk_level": cfg["unavailable_code"],
                "relative_risk_display": cfg["unavailable_display_name"],
                "grade_status": UNAVAILABLE,
                "unavailable_reason": ref_reason,
                "reference_id": reference_id,
                "grade_config_version": cfg["version"],
                "prediction_run_id": prediction_run_id,
                "evidence_refs": evidence,
            })
            continue

        p = as_float(row.get("p"))
        if p is None or not math.isfinite(p) or not 0.0 <= p <= 1.0:
            out.append({
                **envelope,
                "pipe_id": pipe_id,
                "risk_percentile": None,
                "relative_risk_level": cfg["unavailable_code"],
                "relative_risk_display": cfg["unavailable_display_name"],
                "grade_status": UNAVAILABLE,
                "unavailable_reason": (
                    f"预测值非法（{row.get('p')!r}）：缺失、非有限或超出 [0,1]，"
                    "不得参与分级"),
                "reference_id": reference_id,
                "grade_config_version": cfg["version"],
                "prediction_run_id": prediction_run_id,
                "evidence_refs": evidence,
            })
            continue

        pct = percentile_of(p, ref_scores)
        code, display = assign_level(pct, cfg)
        out.append({
            **envelope,
            "pipe_id": pipe_id,
            "risk_percentile": pct,
            "relative_risk_level": code,
            "relative_risk_display": display,
            "grade_status": "graded",
            "unavailable_reason": None,
            "reference_id": reference_id,
            "grade_config_version": cfg["version"],
            "prediction_run_id": prediction_run_id,
            "evidence_refs": evidence,
        })
    return out