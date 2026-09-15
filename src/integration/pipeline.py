"""M0 可复现入口（§13.6、里程碑 §2.2）。

一次运行产出：标准属性包、几何包、预测包、参考分布包、发布清单。
默认输出到 outputs/release/（按保密要求管理，不纳入版本控制）。
"""

import argparse
import json
from pathlib import Path

from src.contracts import MODEL_CONFIG_VERSION, validate_package
from src.data import loader
from src.evaluation import metrics, protocol, reference, selection, split
from src.explain import artifacts as explain_artifacts
from src.evaluation.metrics import aggregate_all
from src.integration import fullfit as fullfit_mod
from src.integration import release

ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = ROOT / "outputs" / "release"
GROUPED_DIR_NAME = "release_grouped"
GROUPED_OUT_DIR = ROOT / "outputs" / GROUPED_DIR_NAME
FULL_FIT_OUT_DIR = ROOT / "outputs" / fullfit_mod.FULL_FIT_DIR_NAME

# 发布模型（§5.1、§5.2）。M1 冻结的是 B2_age_logreg（仅管龄，单特征）；
# M2 起换成 F1 层上的候选选择流程——逐外层折在内层 AP 上选候选，
# 容差 0.005 内优先简单模型。B2 保留为文档化的基线，不再是发布模型。
#
# 改这里 = 一次发布模型变更，必须同步三件事（tests/integration/
# test_release_model_pin.py 会因此失败，那是设计意图，不是回归）：
#   1. 重建 outputs/ 四个目录
#   2. 在 里程碑与实施计划.md 记成发布模型变更，附新旧对照成绩
#   3. 更新钉扎测试里的声明常量
RELEASE_FLOW_ID = "F1_candidate_flow"
RELEASE_STRUCTURE = "F1_base_environment"
SEED = 20260914

# 事件视图的截止日（§8.5）。显式常量：发布时不隐式取系统当前日。
EVENT_AS_OF = "2024-12-31"


def release_run_id(data_version, scheme, table_fingerprint):
    """发布 run_id。

    把结构、候选集版本、划分方案与划分指纹一并纳入：换了候选集或换了划分，
    就是另一个发布，不得复用同一 run_id（§13.3）。构建与测试共用本函数，
    避免哈希口径在两边各写一遍。

    `role` 显式区分发布包与实验包——两者跑的是同一套候选流程、同一划分，
    只有角色不同，不能让它们靠模型名字符串碰巧不同来避免撞 ID。
    """
    return release.run_id_of(
        RELEASE_FLOW_ID, SEED, data_version,
        model_spec={
            "role": "release",
            "structure": RELEASE_STRUCTURE,
            "split_scheme": scheme,
            "table_fingerprint": table_fingerprint,
            "candidates": MODEL_CONFIG_VERSION,
        })


def build_all(out_dir=OUT_DIR, *, scheme="random"):
    """构建一个发布集合。

    scheme="random" 为冻结的随机分层主协议（§6.1）；
    scheme="road" 为按道路分组的补充协议（§6.2），写入独立目录、独立 round_id。
    两种方案的产物都带划分指纹，不得互相覆盖或合并成绩。
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

    # 划分指纹纳入 run_id：换了划分就是另一个发布（§6.2、§13.3）。
    run_id = release_run_id(data_version, scheme, table_fp)

    # 候选选择流程（§5.2）：逐外层折只用该折训练集内部信息选候选，
    # 再看外层成绩。选择日志里没有外层键，回选在结构上不可达。
    frame = loader.with_derived_features(pipes)
    candidates = selection.candidates_from_config(
        loader.resolve_layer(RELEASE_STRUCTURE), seed=SEED)

    # 折外预测（固定划分，不重算）。保留每折已拟合的模型与校准器，
    # 供解释复用，避免重新拟合导致解释与已发布预测不一致（§7.1）。
    oof, selection_log, fold_state = selection.select_and_run(
        frame, counts, candidates=candidates, folds=folds, seed=SEED,
        groups=groups, split_scheme=scheme, return_fold_state=True)

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

    # 解释（§7.1）：解释的是本次发布的折外预测，尺度为 log-odds 原始分数。
    # 传入模型实际见过的派生特征帧，而非原始属性表——F3 结构的列只在帧里。
    expl_rows = explain_artifacts.build_explanation_rows(
        frame, oof, fold_state, data_version=data_version, run_id=run_id,
        seed=SEED)
    broken = explain_artifacts.verify_rows_additivity(expl_rows)
    if broken:
        # 加和核验不过就不发布解释：宁可整包 unavailable，也不发对不上的归因
        raise ValueError(f"解释加和核验未通过：{broken[:5]}（§7.1）")

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
                       ("event_view", events_view),
                       ("explanation", expl_rows)]:
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

    # 发布模型是候选选择流程：model_id 记流程标识，逐折实际选中的候选
    # 单独记在这里。predictions.json 的 model_id 是逐行的候选名——两者
    # 不同不是矛盾，是「流程」与「该折选中什么」的区别，必须都可查。
    selection_meta = {
        "flow": RELEASE_FLOW_ID,
        "structure": RELEASE_STRUCTURE,
        "criterion": "inner_ap",
        "tolerance": 0.005,
        "candidates_version": MODEL_CONFIG_VERSION,
        "n_candidates": len(candidates),
        "chosen_per_fold": {str(f): rec["chosen"]
                            for f, rec in sorted(selection_log.items())},
        "distinct_chosen": sorted({rec["chosen"]
                                   for rec in selection_log.values()}),
    }

    manifest = release.build_manifest(
        {"standard_attributes": std, "geometry": geom, "predictions": preds,
         "reference_bundle": [bundle], "evaluation": ev_rows,
         "event_view": events_view, "explanation": expl_rows},
        data_version=data_version, run_id=run_id, model_id=RELEASE_FLOW_ID,
        seed=SEED,
        configs=protocol.config_versions(),
        split=split_meta,
        data_files=dict(loader.EXPECTED_SHA256),
        training_fingerprint=fullfit_mod.training_fingerprint(pipes, counts),
        selection=selection_meta)

    # 落盘
    release.save_json(std, out_dir / "standard_attributes.json")
    release.save_json(geom, out_dir / "geometry.json")
    release.save_json(preds, out_dir / "predictions.json")
    release.save_json([bundle], out_dir / "reference_bundle.json")
    release.save_json(manifest, out_dir / "manifest.json")
    release.save_json(ev_rows, out_dir / "evaluation.json")
    release.save_json(events_view, out_dir / "event_view.json")
    release.save_json(expl_rows, out_dir / "explanation.json")
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
    ap = argparse.ArgumentParser(description="发布构建（§13.6）")
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