"""决策模块测试的共用构造器（§8.5 人工夹具）。

不读两个原始 XLSX，不导入 src/models/ 或 src/data/。
"""

import copy
import json
from pathlib import Path

from src.contracts import (
    DataKind,
    PredictionMode,
    SCHEMA_VERSION,
)

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "fixtures"

ENVELOPE = {
    "schema_version": SCHEMA_VERSION,
    "data_version": "test-v1",
    "run_id": "test-run",
    "prediction_mode": PredictionMode.OOF_REPLAY.value,
    "data_kind": DataKind.SYNTHETIC_FIXTURE.value,
}


def load(name):
    return json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))


def fixture(name):
    return load(name)


def view(pipe_id, quality_flags=None, **attributes):
    """一条标准属性包行。attributes 用夹具拼写（小写可读键）。"""
    return {
        **ENVELOPE,
        "pipe_id": pipe_id,
        "bh": "BH-" + pipe_id,
        "attributes": dict(attributes),
        "source_refs": {},
        "quality_flags": list(quality_flags or []),
    }


def view_codes(pipe_id, quality_flags=None, **attributes):
    """一条使用正式列码拼写的属性行。"""
    return {
        **ENVELOPE,
        "pipe_id": pipe_id,
        "bh": "BH-" + pipe_id,
        "attributes": dict(attributes),
        "source_refs": {},
        "quality_flags": list(quality_flags or []),
    }


def prediction(pipe_id, p, quality_flags=None):
    return {
        **ENVELOPE,
        "pipe_id": pipe_id,
        "p": p,
        "model_id": "test-model",
        "round_id": "round-1",
        "fold": 0,
        "calibrated": True,
        "quality_flags": list(quality_flags or []),
    }


def reference(reference_id, scores, pipe_ids=None):
    ids = pipe_ids or [f"REF-{i}" for i in range(len(scores))]
    return [{
        **ENVELOPE,
        "reference_id": reference_id,
        "model_run_id": "test-model",
        "grade_config_version": "relative_grade_v1",
        "pipe_ids": list(ids),
        "scores": list(scores),
        "data_fingerprint": "test-fingerprint",
    }]


def grade_row(pipe_id, level, percentile=None, status="graded", reason=None):
    """一条 grade() 形状的分级行（供 advice 的独立测试直接喂入）。"""
    return {
        **ENVELOPE,
        "pipe_id": pipe_id,
        "risk_percentile": percentile,
        "relative_risk_level": level,
        "grade_status": status,
        "unavailable_reason": reason,
        "reference_id": "test-ref",
        "grade_config_version": "relative_grade_v1",
        "prediction_run_id": "test-run",
        "evidence_refs": [f"predictions:test-run:{pipe_id}"],
    }


def event_view(rows):
    """rows: {pipe_id: count} -> 契约 event_view 行列表。"""
    return [
        {**ENVELOPE, "pipe_id": pid, "event_count": count,
         "as_of": "2024-12-31"}
        for pid, count in rows.items()
    ]


def deep(obj):
    return copy.deepcopy(obj)


def assert_unchanged(before, callable_, *args, **kwargs):
    """调用 callable_ 并断言全部入参深度不变（不修改传入表）。"""
    snapshots = [deep(arg) for arg in args]
    snapshot_kwargs = {k: deep(v) for k, v in kwargs.items()}
    result = callable_(*args, **kwargs)
    for original, arg in zip(snapshots, args):
        assert original == arg, "决策函数修改了传入参数"
    for key, original in snapshot_kwargs.items():
        assert original == kwargs[key], f"决策函数修改了传入参数 {key}"
    return result


# 分级边界用的参考集合：10 条，便于构造 80.0 的精确边界
REF_TEN = [i / 10.0 for i in range(1, 11)]


def percentile(p, scores):
    """独立重算的平均秩百分位（测试中与实现对照）。"""
    n = len(scores)
    below = sum(1 for r in scores if r < p)
    equal = sum(1 for r in scores if r == p)
    return 100.0 * (below + 0.5 * equal) / n