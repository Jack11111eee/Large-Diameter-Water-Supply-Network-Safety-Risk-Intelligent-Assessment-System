"""全量拟合模型（§6.5、§2.3）。

方法冻结后用全体数据拟合最终校准集成模型，保留模式与训练集指纹。
全量拟合在训练标签上的分数**不得进入成绩表**；演示默认使用折外回放。
其参考包包含训练管段，仅用于描述相对位置，不与折外回放 R 混用（§5.3.1）。
"""

import hashlib
import json

import pandas as pd

from src.contracts.versions import PredictionMode, SCHEMA_VERSION
from src.data import loader
from src.evaluation import selection
from src.evaluation.reference import build_reference_bundle
from src.evaluation.split import N_INNER, SEED
from src.models.calibration import (
    cross_fitted_calibration,
    ensemble_predict,
    scores_fn,
)

FULL_FIT_DIR_NAME = "release_fullfit"

# 全量拟合包里的成绩占位：显式声明未计分，防止训练标签分数流入成绩表（§6.5）。
NO_SCORE_TABLE = {
    "prediction_mode": PredictionMode.FULL_FIT.value,
    "scored": False,
    "reason": "全量拟合在训练标签上的分数不得进入成绩表（§6.5）",
}


def training_fingerprint(pipes, counts):
    """训练集指纹：管段 ID 与标签的确定性摘要（§6.5）。"""
    pairs = sorted(zip(pipes["ID"].astype(str).tolist(),
                       counts.astype(int).tolist()))
    return hashlib.sha256(
        json.dumps(pairs, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]


def fit_full(pipes, counts, *, structure="F1_base_environment",
             candidates_config=None, seed=SEED):
    """用全体数据拟合最终校准集成模型（§6.5）。

    候选结构在全体数据内部选择，不看任何外层成绩。
    """
    frame = loader.with_derived_features(pipes).reset_index(drop=True)
    y = counts.gt(0).astype(int).to_numpy()
    cols = loader.resolve_layer(structure)
    candidates = selection.candidates_from_config(
        cols, config=candidates_config, seed=seed)

    chosen, log = selection.select_within_train(frame, y, candidates, seed=seed)
    models, calibrators = cross_fitted_calibration(
        lambda X, yy: chosen.build().fit(X, yy),
        scores_fn,
        frame,
        y,
        N_INNER,
        seed,
    )

    return {
        "structure": structure,
        "model_id": chosen.name,
        "model_params": chosen.label,
        "models": models,
        "calibrators": calibrators,
        "prediction_mode": PredictionMode.FULL_FIT.value,
        "training_fingerprint": training_fingerprint(pipes, counts),
        "n_pipes": int(len(frame)),
        "selection_log": log,
    }


def score_snapshot(fullfit, pipes):
    """对输入快照打分（§2.3 全量拟合模式）。"""
    frame = loader.with_derived_features(pipes).reset_index(drop=True)
    return ensemble_predict(fullfit["models"], fullfit["calibrators"], frame)


def build_full_fit_predictions(fullfit, pipes, *, data_version, run_id):
    """全量拟合预测包。每行显式标 prediction_mode=full_fit。"""
    scores = score_snapshot(fullfit, pipes)
    return [
        {
            "schema_version": SCHEMA_VERSION,
            "data_version": data_version,
            "run_id": run_id,
            "prediction_mode": PredictionMode.FULL_FIT.value,
            "data_kind": "real_standard",
            "pipe_id": str(pid),
            "p": float(score),
            "model_id": fullfit["model_id"],
            "round_id": f"fullfit-{fullfit['training_fingerprint']}",
            "calibrated": True,
            "quality_flags": [],
        }
        for pid, score in zip(pipes["ID"].astype(str), scores)
    ]


def build_full_fit_reference(fullfit, pipes, *, data_version, run_id):
    """全量拟合参考包：独立 R，含训练管段，不与折外回放混用（§5.3.1）。"""
    frame = pd.DataFrame({
        "pipe_id": pipes["ID"].astype(str).to_numpy(),
        "p": score_snapshot(fullfit, pipes),
    })
    return build_reference_bundle(
        frame,
        reference_id=f"{run_id}-fullfit",
        model_run_id=run_id,
        data_version=data_version,
        prediction_mode=PredictionMode.FULL_FIT.value,
    )