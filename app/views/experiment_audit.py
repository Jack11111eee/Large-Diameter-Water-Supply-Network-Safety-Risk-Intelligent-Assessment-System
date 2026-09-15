"""页面 6：实验审计（§6.4.1、§5.2、§5.3、§10 W3）。

两块，各自标明口径：

1. **发布成绩**（`evaluation.json` 契约行）——逐折为主，宏平均与折间标准差
   并列，`pooled_oof_replay` 单独成块并标注「不替代逐折主成绩」。
2. **实验包**（`outputs/experiments/`，可选）——候选选择日志、校准对照、
   结构消融。它描述候选选择流程，**不是发布成绩**；两者不同源时拒绝并列展示。
"""

import streamlit as st

from .. import components

UNKNOWN = components.UNKNOWN
POOLED_NOTE = ("`pooled_oof_replay` 是折外概率拼接后的全网回放读数，"
               "**不替代逐折主成绩**，也不与逐折混写（§6.4.1）。")


def render(package, mode):
    st.subheader("实验审计")
    st.caption("发布成绩与实验包分开呈现，各自标明口径与来源（§6.4.1）。")

    _release_scores(package)
    st.markdown("---")
    _experiments(package)


# --------------------------------------------------------------------------
# 1. 发布成绩
# --------------------------------------------------------------------------

def _release_scores(package):
    st.markdown("#### 发布成绩（正式口径）")
    rows = package.evaluation
    if not rows:
        components.empty_state(
            "本包没有成绩产物（evaluation.json）。",
            detail="界面不自行计算指标，也不以实验包的成绩顶替发布成绩（§13.4）。",
            icon="📭")
        return

    st.caption(f"`run_id` `{package.report.run_id}`　`data_version` "
               f"`{package.report.data_version}`　模型 `{package.report.model_id or UNKNOWN}`")

    _per_fold(rows)
    _macro(rows)
    _sample_weighted(rows)
    _pooled(rows)


def _group(rows, aggregation):
    return [r for r in rows if r["aggregation"] == aggregation]


def _per_fold(rows):
    per_fold = _group(rows, "per_fold")
    if not per_fold:
        return
    folds = sorted({r["fold"] for r in per_fold})
    metrics = []
    cells = {}
    for r in per_fold:
        if r["metric"] not in metrics:
            metrics.append(r["metric"])
        cells.setdefault(r["metric"], {})[r["fold"]] = r

    st.markdown("##### 逐折成绩（正式主成绩明细）")
    table = []
    for metric in metrics:
        row = {"指标": metric}
        for fold in folds:
            rec = cells[metric].get(fold)
            row[f"折 {fold}"] = (components.fmt(rec["value"])
                                if rec and rec["applicable"] else UNKNOWN)
        table.append(row)
    st.dataframe(table, width="stretch", hide_index=True)

    unusable = [r for r in per_fold if not r["applicable"]]
    if unusable:
        st.caption("不适用读数（不适用即不适用，不用 0 或空格顶替）：")
        st.dataframe([
            {"折": r["fold"], "指标": r["metric"], "原因": r["reason"] or UNKNOWN}
            for r in unusable
        ], width="stretch", hide_index=True)
    else:
        st.caption("所有逐折读数均适用，无折被丢弃。")


def _macro(rows):
    macro = _group(rows, "macro_mean")
    std = {r["metric"]: r for r in _group(rows, "fold_std")}
    if not macro:
        return
    st.markdown("##### 宏平均与折间标准差（同轮各有效折等权）")
    table = []
    for r in macro:
        valid, total = r.get("valid_folds"), r.get("total_folds")
        complete = valid == total
        table.append({
            "指标": r["metric"],
            "宏平均": components.fmt(r["value"]),
            "折间标准差": components.fmt(std[r["metric"]]["value"])
            if r["metric"] in std else UNKNOWN,
            "有效折/总折": f"{valid}/{total}" if valid is not None else UNKNOWN,
            "完整": "是" if complete else "否",
        })
    st.dataframe(table, width="stretch", hide_index=True)
    incomplete = [r["metric"] for r in macro
                  if r.get("valid_folds") != r.get("total_folds")]
    if incomplete:
        st.warning(f"以下指标有折被丢弃：{incomplete}（§6.4.1 要求逐折报告，"
                   "不得静默丢弃）", icon="⚠️")
    else:
        st.caption("每个指标的有效折数等于总折数：没有折被静默丢弃。")


def _sample_weighted(rows):
    weighted = _group(rows, "sample_weighted")
    if not weighted:
        return
    st.markdown("##### 按折样本数加权（仅 Brier / log loss）")
    st.dataframe([
        {"指标": r["metric"], "加权值": components.fmt(r["value"])}
        for r in weighted
    ], width="stretch", hide_index=True)


def _pooled(rows):
    pooled = _group(rows, "pooled_oof_replay")
    if not pooled:
        return
    st.markdown("##### 全网回放（辅助口径）")
    st.info(POOLED_NOTE, icon="ℹ️")
    st.dataframe([
        {"指标": r["metric"], "值": components.fmt(r["value"])}
        for r in pooled
    ], width="stretch", hide_index=True)


# --------------------------------------------------------------------------
# 2. 实验包
# --------------------------------------------------------------------------

def _experiments(package):
    st.markdown("#### 实验包（候选选择流程）")
    bundle = package.experiments

    if not bundle.available:
        components.empty_state(
            "实验包不可用，本节显示为空。",
            detail=f"{bundle.reason}　实验包是附加产物：它的缺失不影响发布包，"
                   "界面也不用发布成绩顶替它（§13.3）。", icon="📭")
        return

    if not bundle.same_source_as(package.report.data_version):
        components.error_state(
            "实验包与发布包不同源，拒绝并列展示。",
            detail=f"实验包 data_version={bundle.data_version!r}、"
                   f"发布包 data_version={package.report.data_version!r}。"
                   "两份产物的划分或数据可能不同，并排会误导（§13.3）。")
        return

    st.caption(f"实验包 `run_id` `{bundle.run_id}`　结构 `{bundle.structure}`　"
               f"划分指纹 `{(bundle.manifest or {}).get('split', {}).get('table_fingerprint', UNKNOWN)}`"
               "　—— 与发布包同源，但描述的是候选选择流程，不是发布成绩。")
    if (bundle.manifest or {}).get("note"):
        st.caption(bundle.manifest["note"])

    _selection_log(bundle.selection_log)
    _calibration(bundle.calibration_report)
    _ablation(bundle.ablation)


def _selection_log(log):
    st.markdown("##### 候选选择日志（内层 AP，§5.2）")
    if not log:
        components.empty_state("实验包没有选择日志。", icon="📭")
        return
    folds = log.get("folds") or {}
    st.dataframe([
        {"折": fold, "选中候选": rec.get("chosen"),
         "内层折数": rec.get("n_inner_folds"), "状态": rec.get("status")}
        for fold, rec in sorted(folds.items(), key=lambda kv: int(kv[0]))
    ], width="stretch", hide_index=True)

    st.caption(f"选择依据 `{log.get('criterion')}`，容差 `{log.get('tolerance')}`，"
               f"并列时 `{log.get('tie_break')}`；只用外层训练集内部信息，"
               "日志不含任何外层成绩（§5.2）。")

    detail = []
    for fold, rec in sorted(folds.items(), key=lambda kv: int(kv[0])):
        for label, cand in (rec.get("per_candidate") or {}).items():
            detail.append({
                "折": fold, "候选": label,
                "内层均值 AP": components.fmt(cand.get("mean_ap"), 5),
                "选中": "✓" if label == rec.get("chosen") else "",
            })
    if detail:
        st.dataframe(detail, width="stretch", hide_index=True)


def _calibration(report):
    st.markdown("##### 校准对照（逐折，§5.3）")
    if not report:
        components.empty_state("实验包没有校准对照。", icon="📭")
        return
    st.caption(f"主流程 `{report.get('method')}`，对照 `{report.get('control')}`；"
               "校准数据不含外层测试折。")
    rows = []
    for fold, rec in sorted((report.get("per_fold") or {}).items(),
                            key=lambda kv: int(kv[0])):
        brier, ll = rec.get("brier") or {}, rec.get("log_loss") or {}
        rows.append({
            "折": fold, "n": rec.get("n"), "基率": components.fmt(rec.get("base_rate")),
            "Brier 校准": components.fmt(brier.get("calibrated")),
            "Brier 未校准": components.fmt(brier.get("uncalibrated")),
            "log loss 校准": components.fmt(ll.get("calibrated")),
            "log loss 未校准": components.fmt(ll.get("uncalibrated")),
        })
    st.dataframe(rows, width="stretch", hide_index=True)

    first = next(iter((report.get("per_fold") or {}).values()), None)
    if first:
        st.caption(first.get("conclusion") or "")
        _bins("可靠性分箱（校准版）", first.get("bins_calibrated") or [])
        _bins("可靠性分箱（未校准对照）", first.get("bins_uncalibrated") or [])


def _bins(title, bins):
    if not bins:
        return
    st.markdown(f"###### {title}")
    st.dataframe([
        {"箱": b["bin"], "下界": components.fmt(b["lo"], 4),
         "上界": components.fmt(b["hi"], 4), "样本数": b["n"],
         "正例数": b["n_positive"], "平均预测": components.fmt(b["mean_p"], 4),
         "实际率": components.fmt(b["observed_rate"], 4),
         "样本偏少": "是" if b["low_n"] else ""}
        for b in bins
    ], width="stretch", hide_index=True)


def _ablation(abl):
    st.markdown("##### 结构消融（逐折配对 delta，§10 W3）")
    if not abl:
        components.empty_state("实验包没有消融结果。", icon="📭")
        return

    structures = abl.get("structures") or {}
    metric = abl.get("metric", "ap")
    rows = []
    for key, rec in structures.items():
        macro = rec.get("macro_mean") or {}
        delta = rec.get("paired_delta_vs_M1") or {}
        rows.append({
            "结构": key,
            "层规格": rec.get("layer_spec"),
            "列数": rec.get("n_columns"),
            f"{metric} 宏平均": components.fmt(macro.get(metric), 5),
            "折间标准差": components.fmt(macro.get("fold_std"), 5),
            "有效折/总折": f"{macro.get('valid_folds')}/{macro.get('total_folds')}",
            "配对 delta vs M1": components.fmt(delta.get("mean_delta"), 5),
            "delta 折间标准差": components.fmt(delta.get("fold_std"), 5),
        })
    st.dataframe(rows, width="stretch", hide_index=True)
    st.caption("增益用逐折配对差值表述：先算逐折差再取宏平均与折间标准差。"
               "两个宏均值之差丢掉了配对信息，不是「增益与不确定性」。")

    per_fold = {}
    for key, rec in structures.items():
        for fold, chosen in (rec.get("model_id_per_fold") or {}).items():
            per_fold.setdefault(key, {})[fold] = chosen
    if per_fold:
        st.caption("逐折选中的候选（各结构内部选择，不看外层成绩）：")
        folds = sorted({f for rec in per_fold.values() for f in rec})
        st.dataframe([
            {"结构": key, **{f"折 {f}": rec.get(f, UNKNOWN) for f in folds}}
            for key, rec in per_fold.items()
        ], width="stretch", hide_index=True)

    disclosures = abl.get("disclosures") or {}
    st.markdown("###### 强制披露")
    for key in ("F2", "F3", "common"):
        if disclosures.get(key):
            st.warning(disclosures[key], icon="⚠️")