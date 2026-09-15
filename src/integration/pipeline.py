"""M0 可复现入口（§13.6、里程碑 §2.2）。

一次运行产出：标准属性包、几何包、预测包、参考分布包、发布清单。
默认输出到 outputs/release/（按保密要求管理，不纳入版本控制）。
"""

import argparse
import json
from pathlib import Path

from src.contracts import validate_package
from src.data import loader
from src.evaluation import metrics, protocol, reference, runner, split
from src.evaluation.metrics import aggregate_all
from src.integration import fullfit as fullfit_mod
from src.integration import release

ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = ROOT / "outputs" / "release"
GROUPED_DIR_NAME = "release_grouped"
GROUPED_OUT_DIR = ROOT / "outputs" / GROUPED_DIR_NAME
FULL_FIT_OUT_DIR = ROOT / "outputs" / fullfit_mod.FULL_FIT_DIR_NAME

MODEL_ID = "B2_age_logreg"
SEED = 20260914

# 事件视图的截止日（§8.5）。显式常量：发布时不隐式取系统当前日。
EVENT_AS_OF = "2024-12-31"


def build_all(out_dir=OUT_DIR, *, scheme="random"):
    """构建一个发布集合。

    scheme="random" 为冻结的随机分层主协议（§6.1），输出与历史逐字节一致；
    scheme="road" 为按道路分组的补充协议（§6.2），写入独立目录、独立 round_id。
    """
    out_dir = Path(out_dir)
    pipes = loader.load_attributes()
    events = loader.load_events()
    counts, _ = loader.make_labels(pipes, events)

    pipe_ids = pipes["ID"].astype(str).tolist()
    data_version = release.data_version_of(pipe_ids)

    groups, group_meta = (
        (None, None) if scheme == "random" else split.group_ids(pipes, scheme))

    # 固定划分
    table = split.split_table(pipes, counts, groups=groups)
    folds = {int(f): g.index.to_numpy() for f, g in table.groupby("outer_fold")}
    table_fp = split.table_fingerprint(table, scheme=scheme)

    # 划分指纹纳入 run_id：两种划分方案不得撞同一发布 ID（§6.2）。
    # 随机方案保持 M1 冻结口径逐字节不变，故不附 spec。
    run_id = release.run_id_of(
        MODEL_ID, SEED, data_version,
        model_spec=None if scheme == "random" else {
            "split_scheme": scheme, "table_fingerprint": table_fp})

    # 折外预测（固定划分，不重算）
    oof = runner.run_oof(pipes, counts, model_name=MODEL_ID, folds=folds,
                         seed=SEED, groups=groups, split_scheme=scheme)

    # 四类聚合成绩
    scores = aggregate_all(oof.y_true, oof.p, oof.pipe_id, oof.outer_fold,
                           seed=SEED)

    # 参考分布（§5.3.1）
    bundle = reference.build_reference_bundle(
        oof, reference_id=f"{run_id}-oof", model_run_id=run_id,
        data_version=data_version)

    # 发布包
    std = release.build_standard_attributes(pipes, data_version=data_version,
                                            run_id=run_id)
    geom = release.build_geometry(pipes, data_version=data_version, run_id=run_id)
    preds = release.build_predictions(oof, data_version=data_version, run_id=run_id)
    events_view = release.build_event_view(
        pipes, events, data_version=data_version, run_id=run_id,
        as_of=EVENT_AS_OF)

    # 成绩扁平化为契约行（§13.4 `evaluation` 产物）
    ev_rows = metrics.flatten_evaluation(scores, {
        "schema_version": release.SCHEMA_VERSION,
        "data_version": data_version,
        "run_id": run_id,
        "prediction_mode": "oof_replay",
        "data_kind": "real_standard",
    })

    # 契约校验
    for name, rows in [("standard_attributes", std), ("geometry", geom),
                       ("predictions", preds), ("evaluation", ev_rows),
                       ("event_view", events_view)]:
        validate_package(name, rows)
    validate_package("reference_bundle", [bundle])

    # 按 pipe_id 校验（§13.6）
    release.verify_by_pipe_id(("standard_attributes", std),
                              ("geometry", geom), ("predictions", preds))

    split_meta = {
        "scheme": scheme,
        "seed": SEED,
        "n_outer": split.N_OUTER,
        "n_inner": split.N_INNER,
        "table_fingerprint": table_fp,
        "folds": split.fold_summary(table, pipes),
    }
    if group_meta is not None:
        split_meta["grouping"] = group_meta
        split_meta["grouped_diagnostics"] = split.grouped_fold_diagnostics(
            groups, counts.gt(0).astype(int).to_numpy(), folds)

    manifest = release.build_manifest(
        {"standard_attributes": std, "geometry": geom, "predictions": preds,
         "reference_bundle": [bundle], "evaluation": ev_rows,
         "event_view": events_view},
        data_version=data_version, run_id=run_id, model_id=MODEL_ID, seed=SEED,
        configs=protocol.config_versions(),
        split=split_meta,
        data_files=dict(loader.EXPECTED_SHA256),
        training_fingerprint=fullfit_mod.training_fingerprint(pipes, counts))

    # 落盘
    release.save_json(std, out_dir / "standard_attributes.json")
    release.save_json(geom, out_dir / "geometry.json")
    release.save_json(preds, out_dir / "predictions.json")
    release.save_json([bundle], out_dir / "reference_bundle.json")
    release.save_json(manifest, out_dir / "manifest.json")
    release.save_json(ev_rows, out_dir / "evaluation.json")
    release.save_json(events_view, out_dir / "event_view.json")
    # 划分表（A 内部使用，不进入普通发布包）
    split.save_split(table, out_dir / "labels_and_splits.csv")

    return {
        "data_version": data_version,
        "run_id": run_id,
        "scheme": scheme,
        "n_pipes": len(pipes),
        "manifest": manifest,
        "evaluation": scores,
        "evaluation_rows": ev_rows,
        "reference_usable": bundle["usable"],
    }


def build_grouped(out_dir=GROUPED_OUT_DIR):
    """按道路分组的补充协议发布集合（§6.2）。

    与随机主协议各写一份独立产物：round_id 带方案后缀，两者的差异本身
    才是有效信息，不得合并成一份成绩。
    """
    return build_all(out_dir, scheme="road")


def build_full_fit(out_dir=FULL_FIT_OUT_DIR, *, structure="F1_base_environment"):
    """全量拟合发布包（§6.5）。

    独立目录、独立参考包；**不产出成绩表**——全量拟合在训练标签上的
    分数不得进入成绩表，演示默认使用折外回放。
    """
    out_dir = Path(out_dir)
    pipes = loader.load_attributes()
    counts, _ = loader.make_labels(pipes, loader.load_events())

    pipe_ids = pipes["ID"].astype(str).tolist()
    data_version = release.data_version_of(pipe_ids)
    fitted = fullfit_mod.fit_full(pipes, counts, structure=structure)
    run_id = release.run_id_of(
        fitted["model_id"], SEED, data_version,
        model_spec={
            "structure": structure,
            "mode": fullfit_mod.NO_SCORE_TABLE["prediction_mode"],
            "training_fingerprint": fitted["training_fingerprint"],
        })

    preds = fullfit_mod.build_full_fit_predictions(
        fitted, pipes, data_version=data_version, run_id=run_id)
    bundle = fullfit_mod.build_full_fit_reference(
        fitted, pipes, data_version=data_version, run_id=run_id)

    validate_package("predictions", preds)
    validate_package("reference_bundle", [bundle])

    manifest = release.build_manifest(
        {"predictions": preds, "reference_bundle": [bundle]},
        data_version=data_version, run_id=run_id,
        model_id=fitted["model_id"], seed=SEED)

    release.save_json(preds, out_dir / "predictions.json")
    release.save_json([bundle], out_dir / "reference_bundle.json")
    release.save_json(manifest, out_dir / "manifest.json")
    release.save_json(fullfit_mod.NO_SCORE_TABLE, out_dir / "full_fit_scores.json")

    return {
        "data_version": data_version,
        "run_id": run_id,
        "n_pipes": len(pipes),
        "manifest": manifest,
        "training_fingerprint": fitted["training_fingerprint"],
    }


def main():
    ap = argparse.ArgumentParser(description="M0 发布构建")
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--scheme", default="random", choices=["random", "road"],
                    help="划分方案：random 为冻结主协议，road 为道路分组补充协议")
    args = ap.parse_args()
    result = build_all(args.out, scheme=args.scheme)
    summary = {
        "data_version": result["data_version"],
        "run_id": result["run_id"],
        "scheme": result["scheme"],
        "n_pipes": result["n_pipes"],
        "reference_usable": result["reference_usable"],
        "macro_mean_ap": result["evaluation"]["macro_mean"]["ap"]["value"],
        "macro_mean_roc_auc": result["evaluation"]["macro_mean"]["roc_auc"]["value"],
        "pooled_oof_ap": result["evaluation"]["pooled_oof_replay"]["ap"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()