"""M2 实验产物包（§5.1、§5.2、§5.3、§10 W3）。

模型流的产物（候选选择日志、校准对照、结构消融）不属于 M0 发布契约，
另存一个目录，自带清单与 run_id。

它描述的是**候选选择流程**，与发布包里冻结的 M1 模型是两条路径；两者
共用同一 data_version 与同一划分表指纹，故审计页可以核对它们是否同源，
但不得把实验包的成绩当作发布成绩。
"""

import argparse
import json
from pathlib import Path

from src.contracts import MODEL_CONFIG_VERSION
from src.data import loader
from src.evaluation import ablation, calibration_report, selection, split
from src.integration import release

ROOT = Path(__file__).resolve().parent.parent.parent
EXPERIMENT_DIR_NAME = "experiments"
EXPERIMENT_OUT_DIR = ROOT / "outputs" / EXPERIMENT_DIR_NAME

# 实验包描述的结构（§5.1 的 M1 层），与 candidates.json 的候选集配套。
EXPERIMENT_STRUCTURE = "F1_base_environment"


def build_experiments(out_dir=EXPERIMENT_OUT_DIR, *, structure=EXPERIMENT_STRUCTURE,
                      scheme="random"):
    """在冻结划分上跑候选选择、校准对照与结构消融（§5.2、§5.3、§10 W3）。

    全部使用同一 folds、同一候选集、同一校准协议；结构选择在各外层训练集
    内部完成，不看外层成绩回选。
    """
    out_dir = Path(out_dir)
    pipes = loader.load_attributes()
    events = loader.load_events()
    counts, _ = loader.make_labels(pipes, events)

    pipe_ids = pipes["ID"].astype(str).tolist()
    data_version = release.data_version_of(pipe_ids)
    groups, group_meta = (
        (None, None) if scheme == "random" else split.group_ids(pipes, scheme))

    table = split.split_table(pipes, counts, groups=groups)
    folds = {int(f): g.index.to_numpy() for f, g in table.groupby("outer_fold")}
    table_fp = split.table_fingerprint(table, scheme=scheme)

    run_id = release.run_id_of(
        "candidate_flow", split.SEED, data_version,
        model_spec={"role": "experiment", "structure": structure,
                    "split_scheme": scheme,
                    "table_fingerprint": table_fp,
                    "candidates": MODEL_CONFIG_VERSION})

    frame = loader.with_derived_features(pipes)
    cols = loader.resolve_layer(structure)
    candidates = selection.candidates_from_config(cols, seed=split.SEED)

    calibrated, log = selection.select_and_run(
        frame, counts, candidates=candidates, folds=folds, seed=split.SEED,
        groups=groups, split_scheme=scheme)
    uncalibrated, _ = selection.select_and_run(
        frame, counts, candidates=candidates, folds=folds, seed=split.SEED,
        groups=groups, calibrate=False, split_scheme=scheme)

    cal_report = calibration_report.per_fold_calibration(
        calibrated.y_true.to_numpy(), calibrated.p.to_numpy(),
        uncalibrated.p.to_numpy(), calibrated.outer_fold.to_numpy())

    abl = ablation.run_ablation(frame, counts, folds=folds, groups=groups)

    selection_log = {
        "run_id": run_id,
        "data_version": data_version,
        "structure": structure,
        "layer_spec": structure,
        "n_columns": len(cols),
        "candidates_version": MODEL_CONFIG_VERSION,
        "seed": split.SEED,
        "tolerance": 0.005,
        "criterion": "inner_ap",
        "fold_table_fingerprint": table_fp,
        "folds": {str(f): rec for f, rec in sorted(log.items())},
    }
    calibration = {
        "run_id": run_id,
        "data_version": data_version,
        "method": "sigmoid",
        "control": "uncalibrated",
        "per_fold": {str(f): rec for f, rec in sorted(cal_report.items())},
    }

    manifest = {
        "schema_version": release.SCHEMA_VERSION,
        "data_version": data_version,
        "run_id": run_id,
        "model_id": "candidate_flow",
        "seed": split.SEED,
        "prediction_mode": "oof_replay",
        "data_kind": "real_standard",
        "structure": structure,
        "split": {
            "scheme": scheme,
            "seed": split.SEED,
            "n_outer": split.N_OUTER,
            "n_inner": split.N_INNER,
            "table_fingerprint": table_fp,
        },
        "note": "本包描述候选选择流程，不是发布成绩；发布成绩见发布目录的 evaluation.json。",
        "artifacts": {},
    }
    if group_meta is not None:
        manifest["split"]["grouping"] = group_meta

    payloads = {
        "selection_log.json": selection_log,
        "calibration_report.json": calibration,
        "ablation.json": abl,
    }
    for name, payload in payloads.items():
        manifest["artifacts"][name] = {
            "n_bytes": len(json.dumps(payload, ensure_ascii=False)),
            "fingerprint": release._fingerprint(payload),
        }

    release.save_json(selection_log, out_dir / "selection_log.json")
    release.save_json(calibration, out_dir / "calibration_report.json")
    release.save_json(_jsonable(abl), out_dir / "ablation.json")
    release.save_json(manifest, out_dir / "manifest.json")

    # 发布成绩仍由发布目录给出；这里只报告实验侧的选择与增益
    return {
        "run_id": run_id,
        "data_version": data_version,
        "structure": structure,
        "chosen_per_fold": {f: rec["chosen"] for f, rec in log.items()},
        "ablation_ap": {
            k: v["macro_mean"]["ap"] for k, v in abl["structures"].items()
        },
    }


def _jsonable(obj):
    """把 numpy 标量转为 JSON 可写形式（消融结果里含 np 标量）。"""
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
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def main():
    ap = argparse.ArgumentParser(description="M2 实验产物构建")
    ap.add_argument("--out", default=str(EXPERIMENT_OUT_DIR))
    ap.add_argument("--structure", default=EXPERIMENT_STRUCTURE)
    args = ap.parse_args()
    result = build_experiments(args.out, structure=args.structure)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()