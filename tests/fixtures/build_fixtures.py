"""人工夹具生成器（§13.3）。

样例统一设 data_kind=synthetic_fixture，不使用真实标签伪造正式预测。
与正式产物不同目录：写入 tests/fixtures/（纳入版本控制，B/C 可离线读取）。

覆盖 §13.3 要求的六类边界 + 审查发现（§里程碑 1.2）：
真实数据 FAC 零缺失、GJ 无越界，因此未知设施与管径越界只能靠本夹具覆盖。
"""

import json
from pathlib import Path

from src.contracts import (
    SCHEMA_VERSION,
    GRADE_CONFIG_VERSION,
    SCENARIO_CONFIG_VERSION,
    ADVICE_RULES_VERSION,
    DataKind,
    PredictionMode,
    Scale,
    ExplainStatus,
)

FIXTURE_DIR = Path(__file__).resolve().parent
DATA_VERSION = "fixture-v1"
RUN_ID = "fixture-run"
FIXTURE_PREFIX = "SYN-"


def _envelope(product_name, prediction_mode=PredictionMode.OOF_REPLAY.value):
    return {
        "schema_version": SCHEMA_VERSION,
        "data_version": DATA_VERSION,
        "run_id": RUN_ID,
        "prediction_mode": prediction_mode,
        "data_kind": DataKind.SYNTHETIC_FIXTURE.value,
    }


def _qflags(*flags):
    return list(flags)


# --------------------------------------------------------------------------
# 六类边界用例定义
# --------------------------------------------------------------------------

# 每条: pipe_id, 说明, 属性, 质量标记, 概率, 等级, 解释状态
CASES = [
    {
        "id": "F01_normal",
        "note": "正常：L1、OPSST 正常、字段完整，无历史模式 → 预期只有 R00",
        "attrs": {
            "pipeage": 12, "material": "球墨铸铁", "joint": "柔性接口",
            "diameter_mm": 400, "road_type": "次干道", "facility": "居民区",
            "opsst": "正常", "press_mpa": 0.32,
        },
        "quality": [],
        "p": 0.041,
        "level": "L1",
        "explain": ExplainStatus.OK.value,
    },
    {
        "id": "F02_unknown_facility",
        "note": "未知设施：FAC 未知，代理采用参考值 0.5 → D02，C 不归零（真实数据零缺失，人工构造）",
        "attrs": {
            "pipeage": 44, "material": "铸铁管", "joint": "承插接口",
            "diameter_mm": 600, "road_type": "主干道", "facility": None,
            "opsst": "正常", "press_mpa": 0.41,
        },
        "quality": _qflags("proxy_imputed:facility"),
        "p": 0.182,
        "level": "L3",
        "explain": ExplainStatus.OK.value,
    },
    {
        "id": "F03_coord_conflict",
        "note": "坐标冲突：源节点坐标存在冲突，属性评分有效 → D01；空间结果待核",
        "attrs": {
            "pipeage": 38, "material": "钢管", "joint": "焊接接口",
            "diameter_mm": 800, "road_type": "次干道", "facility": "企事业单位",
            "opsst": "正常", "press_mpa": 0.55,
        },
        "quality": _qflags("coordinate_conflict"),
        "p": 0.071,
        "level": "L1",
        "explain": ExplainStatus.OK.value,
        "coord_conflict": True,
    },
    {
        "id": "F04_tie_probability",
        "note": "相同概率：管龄并列（真实数据最大并列组 765），分级必须同分同级",
        "attrs": {
            "pipeage": 47, "material": "铸铁管", "joint": "承插接口",
            "diameter_mm": 300, "road_type": "支路", "facility": "绿地",
            "opsst": "偏高", "press_mpa": 0.28,
        },
        "quality": [],
        "p": 0.310,
        "level": "L4",
        "explain": ExplainStatus.OK.value,
        "tie_group": ["F04_tie_probability", "F05_tie_probability_b"],
    },
    {
        "id": "F05_tie_probability_b",
        "note": "与 F04 概率完全相同，用于验证同分同级与稳定并列排序",
        "attrs": {
            "pipeage": 47, "material": "自应力管", "joint": "承插接口",
            "diameter_mm": 300, "road_type": "支路", "facility": "绿地",
            "opsst": "偏高", "press_mpa": 0.28,
        },
        "quality": [],
        "p": 0.310,
        "level": "L4",
        "explain": ExplainStatus.OK.value,
        "tie_group": ["F04_tie_probability", "F05_tie_probability_b"],
    },
    {
        "id": "F06_missing_explanation",
        "note": "缺失解释：解释未就绪必须显式 unavailable，不得用别的模型顶替（§13.5）",
        "attrs": {
            "pipeage": 61, "material": "铸铁管", "joint": "承插接口",
            "diameter_mm": 500, "road_type": "主干道", "facility": "医院",
            "opsst": "异常", "press_mpa": 0.36,
        },
        "quality": _qflags("explanation_pending"),
        "p": 0.264,
        "level": "L4",
        "explain": ExplainStatus.UNAVAILABLE.value,
    },
    {
        "id": "F07_invalid_prediction",
        "note": "无效预测：p 缺失 → D01，相对等级 unavailable；不得因代填触发 R01/R03",
        "attrs": {
            "pipeage": 70, "material": "铸铁管", "joint": "承插接口",
            "diameter_mm": 300, "road_type": "支路", "facility": "学校",
            "opsst": "正常", "press_mpa": None,
        },
        "quality": _qflags("prediction_invalid", "press_missing"),
        "p": None,
        "level": "unavailable",
        "explain": ExplainStatus.UNAVAILABLE.value,
    },
    {
        "id": "F08_diameter_out_of_range",
        "note": "管径越界：真实数据范围恰为 300–1600，无越界值，人工构造；不自动外推",
        "attrs": {
            "pipeage": 33, "material": "球墨铸铁", "joint": "柔性接口",
            "diameter_mm": 2000, "road_type": "主干道", "facility": "交通枢纽",
            "opsst": "正常", "press_mpa": 0.48,
        },
        "quality": _qflags("diameter_out_of_range"),
        "p": 0.095,
        "level": "L2",
        "explain": ExplainStatus.OK.value,
    },
]


def build_standard_attributes():
    rows = []
    for c in CASES:
        a = c["attrs"]
        rows.append({
            **_envelope("standard_attributes"),
            "pipe_id": FIXTURE_PREFIX + c["id"],
            "bh": "BH-" + c["id"],
            "attributes": dict(a),
            "source_refs": {
                "pipeage": "DemoPipes属性数据.xlsx/全部属性/PIPEAGE",
                "material": "DemoPipes属性数据.xlsx/全部属性/CZ",
                "joint": "DemoPipes属性数据.xlsx/全部属性/JOINTT",
                "diameter_mm": "DemoPipes属性数据.xlsx/全部属性/GJ",
                "road_type": "DemoPipes属性数据.xlsx/全部属性/RDTYPE",
                "facility": "DemoPipes属性数据.xlsx/全部属性/FAC",
                "opsst": "DemoPipes属性数据.xlsx/全部属性/OPSST",
                "press_mpa": "DemoPipes属性数据.xlsx/全部属性/PRESS",
            },
            "quality_flags": c["quality"],
        })
    return rows


def build_geometry():
    rows = []
    for i, c in enumerate(CASES):
        conflict = bool(c.get("coord_conflict"))
        # 冲突节点：同一点给两个不同坐标（§4）
        rows.append({
            **_envelope("geometry"),
            "pipe_id": FIXTURE_PREFIX + c["id"],
            "x_start": 324700.0 + i * 10.0,
            "y_start": 349300.0 + i * 10.0,
            "x_end": 324710.0 + i * 10.0,
            "y_end": 349310.0 + i * 10.0,
            "component_id": 0 if not conflict else 1,
            "coordinate_conflict": conflict,
            "crs_known": False,
        })
    return rows


def build_predictions():
    rows = []
    for c in CASES:
        flags = []
        if c["p"] is None:
            flags.append("prediction_invalid")
        rows.append({
            **_envelope("predictions"),
            "pipe_id": FIXTURE_PREFIX + c["id"],
            "p": c["p"],
            "model_id": "B2_logreg",
            "round_id": "round-1",
            "fold": 0,
            "calibrated": True,
            "quality_flags": flags,
        })
    return rows


def build_explanation():
    rows = []
    for c in CASES:
        status = c["explain"]
        flags = [] if status == ExplainStatus.OK.value else ["explanation_pending"]
        if status == ExplainStatus.OK.value:
            rows.append({
                **_envelope("explanation"),
                "pipe_id": FIXTURE_PREFIX + c["id"],
                "model_id": "B2_logreg",
                "base_value": -2.1,
                "contributions": [
                    {"feature": "pipeage", "value": 0.83, "direction": "positive"},
                    {"feature": "material", "value": 0.41, "direction": "positive"},
                    {"feature": "press_mpa", "value": -0.12, "direction": "negative"},
                ],
                "scale": Scale.LOG_ODDS.value,
                "status": ExplainStatus.OK.value,
                "quality_flags": flags,
            })
        else:
            rows.append({
                **_envelope("explanation"),
                "pipe_id": FIXTURE_PREFIX + c["id"],
                "model_id": "B2_logreg",
                "base_value": None,
                "contributions": [],
                "scale": Scale.LOG_ODDS.value,
                "status": ExplainStatus.UNAVAILABLE.value,
                "quality_flags": flags,
            })
    return rows


def build_decision():
    rows = []
    for c in CASES:
        rows.append({
            **_envelope("decision"),
            "pipe_id": FIXTURE_PREFIX + c["id"],
            "risk_percentile": None if c["p"] is None else 50.0,
            "relative_risk_level": c["level"],
            "consequence_proxy": 0.5,
            "priority_value": None if c["p"] is None else c["p"] * 0.5,
            "scenario_id": SCENARIO_CONFIG_VERSION,
            "evidence_refs": [f"fixture:{c['id']}"],
        })
    return rows


def build_reference_bundle():
    scores = [c["p"] for c in CASES if c["p"] is not None]
    return [{
        **_envelope("reference_bundle"),
        "reference_id": "fixture-ref-v1",
        "model_run_id": "B2_logreg",
        "grade_config_version": GRADE_CONFIG_VERSION,
        "pipe_ids": [FIXTURE_PREFIX + c["id"] for c in CASES if c["p"] is not None],
        "scores": scores,
        "data_fingerprint": "fixture-fingerprint",
    }]


def build_expected_rules():
    """夹具的预期规则触发（供 B 的 T01–T10 参考；不是正式结果）。"""
    return {
        "advice_rules_version": ADVICE_RULES_VERSION,
        "cases": {
            c["id"]: {
                "note": c["note"],
                "expected_triggers": _expected_triggers(c),
            }
            for c in CASES
        },
    }


def _expected_triggers(c):
    t = []
    q = c["quality"]
    a = c["attrs"]
    if "coordinate_conflict" in q or "prediction_invalid" in q:
        t.append("D01")
    if "proxy_imputed:facility" in q:
        t.append("D02")
    return t


BUILDERS = {
    "standard_attributes.json": build_standard_attributes,
    "geometry.json": build_geometry,
    "predictions.json": build_predictions,
    "explanation.json": build_explanation,
    "decision.json": build_decision,
    "reference_bundle.json": build_reference_bundle,
    "expected_rules.json": build_expected_rules,
}


def main():
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    for name, fn in BUILDERS.items():
        payload = fn()
        path = FIXTURE_DIR / name
        # ensure_ascii=False 保留中文；不写 NaN（契约层保证）
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print("wrote", path.relative_to(FIXTURE_DIR.parent.parent))


if __name__ == "__main__":
    main()