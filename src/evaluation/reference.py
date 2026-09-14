"""固定参考分布包（§5.3.1、§13.4）。

生产者：A。消费者：B。
B 不自行重建参考分布，也不从界面筛选子集重建。

本模块只负责**产出**参考分数集合与版本信息；
百分位计算与相对等级分级由 B 在 src/decision/ 中唯一实现（§13.1）。

折外回放：每个固定候选或训练内选择流程、每个重复轮次，
使用该轮完整的 7,288 条折外概率建立一个独立 R。
它只作事后展示分级，不参与训练、校准、模型选择或逐折 Top-K 评分。
"""

import hashlib

import numpy as np

GRADE_CONFIG_VERSION = "relative_grade_v1"


def build_reference_bundle(oof_df, *, reference_id, model_run_id,
                           data_version, grade_config_version=GRADE_CONFIG_VERSION):
    """从折外预测建立参考分布包。

    使用保存的完整精度分数，不先四舍五入（§5.3.1）。
    每份 R 保存 reference_id、prediction_mode、model_run_id、
    管段 ID 集合、数据指纹、完整精度参考分数、grade_config_version。

    data_version 必须由调用方传入真实数据版本：同一发布集合内所有产物
    的 data_version 必须一致（§13.3）。此前误用 round_id（'seed20260914'），
    使参考包与其余产物版本不一致，接入方版本校验会拒绝或误报。
    """
    p = oof_df["p"].to_numpy(dtype=float)
    pipe_ids = oof_df["pipe_id"].astype(str).tolist()

    # R 为空、少于 2 条或所有分数相同 → 无区分度，标 unavailable（§5.3.1）
    usable = not (len(p) == 0 or len(p) < 2 or bool(np.all(p == p[0])))

    fingerprint = hashlib.sha256(
        "\n".join(f"{i}:{v!r}" for i, v in zip(pipe_ids, p)).encode("utf-8")
    ).hexdigest()

    return {
        "schema_version": "1.0.0",
        "data_version": data_version,
        "run_id": model_run_id,
        "prediction_mode": "oof_replay",
        "data_kind": "real_standard",
        "reference_id": reference_id,
        "model_run_id": model_run_id,
        "grade_config_version": grade_config_version,
        "pipe_ids": pipe_ids,
        "scores": [float(x) for x in p],
        "data_fingerprint": fingerprint,
        "usable": usable,
    }