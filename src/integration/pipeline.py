"""M0 可复现入口（§13.6、里程碑 §2.2）。

一次运行产出：标准属性包、几何包、预测包、参考分布包、发布清单。
默认输出到 outputs/release/（按保密要求管理，不纳入版本控制）。
"""

import argparse
import json
from pathlib import Path

from src.contracts import validate_package
from src.data import loader
from src.evaluation import reference, runner, split
from src.evaluation.metrics import aggregate_all
from src.integration import release

ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = ROOT / "outputs" / "release"

MODEL_ID = "B2_age_logreg"
SEED = 20260914


def build_all(out_dir=OUT_DIR):
    out_dir = Path(out_dir)
    pipes = loader.load_attributes()
    events = loader.load_events()
    counts, _ = loader.make_labels(pipes, events)

    pipe_ids = pipes["ID"].astype(str).tolist()
    data_version = release.data_version_of(pipe_ids)
    run_id = release.run_id_of(MODEL_ID, SEED, data_version)

    # 固定划分
    table = split.split_table(pipes, counts)
    folds = {int(f): g.index.to_numpy() for f, g in table.groupby("outer_fold")}

    # 折外预测（固定划分，不重算）
    oof = runner.run_oof(pipes, counts, model_name=MODEL_ID, folds=folds, seed=SEED)

    # 四类聚合成绩
    scores = aggregate_all(oof.y_true, oof.p, oof.pipe_id, oof.outer_fold,
                           seed=SEED)

    # 参考分布（§5.3.1）
    bundle = reference.build_reference_bundle(
        oof, reference_id=f"{run_id}-oof", model_run_id=run_id)

    # 发布包
    std = release.build_standard_attributes(pipes, data_version=data_version,
                                            run_id=run_id)
    geom = release.build_geometry(pipes, data_version=data_version, run_id=run_id)
    preds = release.build_predictions(oof, data_version=data_version, run_id=run_id)

    # 契约校验
    for name, rows in [("standard_attributes", std), ("geometry", geom),
                       ("predictions", preds)]:
        validate_package(name, rows)
    validate_package("reference_bundle", [bundle])

    # 按 pipe_id 校验（§13.6）
    release.verify_by_pipe_id(("standard_attributes", std),
                              ("geometry", geom), ("predictions", preds))

    manifest = release.build_manifest(
        {"standard_attributes": std, "geometry": geom, "predictions": preds,
         "reference_bundle": [bundle]},
        data_version=data_version, run_id=run_id, model_id=MODEL_ID, seed=SEED)

    # 落盘
    release.save_json(std, out_dir / "standard_attributes.json")
    release.save_json(geom, out_dir / "geometry.json")
    release.save_json(preds, out_dir / "predictions.json")
    release.save_json([bundle], out_dir / "reference_bundle.json")
    release.save_json(manifest, out_dir / "manifest.json")
    release.save_json(_jsonable(scores), out_dir / "evaluation.json")
    # 划分表（A 内部使用，不进入普通发布包）
    split.save_split(table, out_dir / "labels_and_splits.csv")

    return {
        "data_version": data_version,
        "run_id": run_id,
        "n_pipes": len(pipes),
        "manifest": manifest,
        "evaluation": scores,
        "reference_usable": bundle["usable"],
    }


def _jsonable(obj):
    """把 numpy 类型与不可用值转为 JSON 可写形式。"""
    import numpy as np

    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return v if np.isfinite(v) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    return obj


def main():
    ap = argparse.ArgumentParser(description="M0 发布构建")
    ap.add_argument("--out", default=str(OUT_DIR))
    args = ap.parse_args()
    result = build_all(args.out)
    summary = {
        "data_version": result["data_version"],
        "run_id": result["run_id"],
        "n_pipes": result["n_pipes"],
        "reference_usable": result["reference_usable"],
        "macro_mean_ap": result["evaluation"]["macro_mean"]["ap"],
        "macro_mean_roc_auc": result["evaluation"]["macro_mean"]["roc_auc"],
        "pooled_oof_ap": result["evaluation"]["pooled_oof_replay"]["ap"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()