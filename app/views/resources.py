"""页面 3：资源清单（§9 第 3 项、§8.1、§8.2）。

数量预算 K、目标切换（p / C / V）、独立按 p 视图、L3/L4 视图、参数版本与导出。
排序与截取只作用于**已给出的展示列**；不计算百分位、C、V，也不重跑建议规则。
导出使用与页面相同的行对象，导出时不重算。
"""

import streamlit as st

from .. import adapter, components

OBJECTIVES = {
    "p": "仅按概率 p",
    "C": "仅按后果代理 C",
    "V": "按优先值 V（p × C）",
}


def render(package, mode):
    st.subheader("资源清单")
    st.caption("给定最多处理 K 根，按明确目标选 Top-K，输出「优先核查/巡检候选清单」，"
               "不是施工排程，也不保证避免事故（§8.2）。")

    if not package.pipes:
        components.empty_state("清单为空，无法生成资源清单。",
                               detail=f"目录：{package.report.directory}。", icon="📭")
        return

    left, right = st.columns([1, 3])
    with left:
        k = st.number_input("预算 K（根数）", min_value=0,
                            max_value=len(package.pipes), value=min(50, len(package.pipes)),
                            step=1, key="resource_k")
        objective = st.radio("目标", options=list(OBJECTIVES),
                             format_func=lambda x: OBJECTIVES[x], key="resource_objective")
        st.caption("预测无效的管段由决策模块排除并列出原因，界面不另行过滤。")
        st.markdown("**参数版本**")
        st.dataframe([
            {"项": "情景版本 scenario_id",
             "值": package.pipes[0].scenario_id or components.UNKNOWN},
            {"项": "预测模式", "值": package.report.prediction_mode},
            {"项": "数据版本 data_version", "值": package.report.data_version},
            {"项": "run_id", "值": package.report.run_id},
            {"项": "模型 model_id", "值": package.report.model_id or components.UNKNOWN},
            {"项": "决策结果来源", "值": package.decision.source},
        ], width="stretch", hide_index=True)

    with right:
        _render_lists(package, int(k), objective)


def _render_lists(package, k, objective):
    pr = package.decision.priority_result
    if isinstance(pr, dict):
        # 预算或目标改变时，请决策模块按其冻结签名重出清单；界面不重排
        if (pr.get("objective") != objective
                or (pr.get("selection") or {}).get("k_requested") != k):
            rerun = package.decision.with_selection(k, objective)
            if isinstance(rerun, dict):
                pr = rerun
        _render_module_result(package, pr, k, objective)
        return

    selected = _select(package, k, objective)

    st.markdown(f"#### 当前目标 Top-{k}：{OBJECTIVES[objective]}")
    if not selected:
        components.empty_state(
            "当前目标下没有可用记录。",
            detail="目标列全部为 unavailable（例如该包缺少决策结果），"
                   "界面不以 0 顶替，也不改变目标口径。", icon="📭")
    else:
        st.caption("排序为确定性排序：目标值降序 → pipe_id 升序；同值不合并、不随机。")
        st.dataframe(_rows(selected), width="stretch", hide_index=True)
        st.download_button(
            "导出当前清单（CSV，同一结果对象）",
            data=adapter.rows_to_csv(_rows(selected)),
            file_name=f"resource_top{k}_{objective}.csv", mime="text/csv",
            key="resource_download")

    st.markdown("---")
    _independent_probability_view(package, selected)
    _high_level_view(package, selected)


def _render_module_result(package, pr, k, objective):
    """决策模块已接入：直接展示 prioritize() 返回的清单与独立视图（§5.3.1、§8.1）。"""
    selection = pr.get("selection") or {}
    ids = selection.get("selected") or []

    st.markdown(f"#### 当前目标 Top-{k}：{OBJECTIVES.get(objective, objective)}")
    st.caption(f"清单由决策模块 `prioritize()` 返回（目标 `{pr.get('objective')}`、"
               f"情景 `{pr.get('scenario_id')}`、随机种子 `{pr.get('seed')}`）；"
               "界面不重排、不重算。")
    if not ids:
        components.empty_state(
            "当前目标下决策模块没有选出管段。",
            detail=f"候选 {pr.get('candidate_count')} 条，可排序 {pr.get('rankable_count')} 条；"
                   "界面不补占位行。", icon="📭")
    else:
        rows = _module_rows(ids, package)
        st.dataframe(rows, width="stretch", hide_index=True)
        st.download_button(
            "导出当前清单（CSV，同一结果对象）",
            data=adapter.rows_to_csv(rows),
            file_name=f"resource_top{k}_{objective}.csv", mime="text/csv",
            key="resource_download")

    excluded = pr.get("excluded") or []
    if excluded:
        st.caption("被排除、不参与排序的管段（原因来自决策模块）：")
        st.dataframe([{"pipe_id": e.get("pipe_id"), "原因": e.get("reason")}
                      for e in excluded[:50]], width="stretch", hide_index=True)

    st.markdown("---")
    _module_view("独立按概率 p 视图（始终保留，不被后果加权覆盖）",
                 pr.get("by_p") or [], package, "在按p清单中")
    marked = pr.get("marked_not_in_value") or []
    if marked:
        st.caption(f"其中 {len(marked)} 条进入按 p 清单但未进入按 V 清单；"
                   "后果加权视图不覆盖按 p 视图（§8.1）。")
    _module_view("L3 / L4 管段视图", pr.get("high_grade_view") or [], package, None)


def _module_view(title, ids, package, mark_column):
    st.markdown(f"#### {title}")
    if not ids:
        components.empty_state("决策模块返回的该视图为空。", icon="📭")
        return
    rows = _module_rows(ids, package)
    if mark_column:
        for row in rows:
            row[mark_column] = True
    st.dataframe(rows[:200], width="stretch", hide_index=True)
    if len(ids) > 200:
        st.caption(f"共 {len(ids)} 条，显示前 200 条。")


def _module_rows(ids, package):
    by_id = package.by_id
    rows = []
    for pipe_id in ids:
        p = by_id.get(pipe_id)
        if p is None:
            # 决策模块返回了本包没有的 ID：如实标出，不静默丢弃
            rows.append({"pipe_id": pipe_id, "bh": None, "p": None,
                         "相对等级": "不在本包中", "后果代理C": None,
                         "优先值V": None, "质量标记": ""})
            continue
        rows.append({
            "pipe_id": p.pipe_id,
            "bh": p.bh,
            "p": p.probability,
            "相对等级": p.relative_risk_level,
            "百分位": p.risk_percentile,
            "后果代理C": p.consequence_proxy,
            "优先值V": p.priority_value,
            "质量标记": "、".join(p.quality_flags),
        })
    return rows


def _select(package, k, objective):
    """冻结包降级路径的排序：只对已给出的展示列排序截取，不计算新量。"""
    rows = adapter.select_top_k(package, k, objective)
    return [p for p in rows if p.probability_available]


def _independent_probability_view(package, selected):
    """按 p 的视图始终独立保留，并标出未进入当前目标清单的管段（§8.1）。"""
    st.markdown("#### 独立按概率 p 视图（始终保留，不被后果加权覆盖）")
    top_p = [p for p in package.pipes if p.probability_available]
    top_p.sort(key=lambda x: (-x.probability, x.pipe_id))
    top_p = top_p[:50]
    if not top_p:
        components.empty_state("没有可用的概率记录。", icon="📭")
        return
    chosen = {p.pipe_id for p in selected}
    rows = []
    for p in top_p:
        row = _rows([p])[0]
        row["在按p清单中"] = True
        row["在当前目标清单中"] = p.pipe_id in chosen
        rows.append(row)
    st.dataframe(rows, width="stretch", hide_index=True)
    missing = [r for r in rows if not r["在当前目标清单中"]]
    if missing:
        st.caption(f"其中 {len(missing)} 条进入按 p 清单但未进入当前目标清单；"
                   "后果加权视图不覆盖按 p 视图（§8.1）。")
    st.caption("仅按概率的 Top-K 与 L3/L4 视图始终独立保留。")


def _high_level_view(package, selected):
    st.markdown("#### L3 / L4 管段视图")
    highs = [p for p in package.pipes if p.relative_risk_level in ("L3", "L4")]
    if not highs:
        components.empty_state("当前包中没有 L3/L4 管段，或等级全部为 unavailable。", icon="📭")
        return
    chosen = {p.pipe_id for p in selected}
    rows = []
    for p in highs:
        row = _rows([p])[0]
        row["在当前目标清单中"] = p.pipe_id in chosen
        rows.append(row)
    st.dataframe(rows[:200], width="stretch", hide_index=True)
    st.caption(f"L3/L4 共 {len(highs):,} 条（显示前 200 条）。"
               "等级由决策模块给出，不随筛选或预算改变（§8.4）。")


def _rows(pipes):
    return [{
        "pipe_id": p.pipe_id,
        "bh": p.bh,
        "p": p.probability,
        "相对等级": p.relative_risk_level,
        "后果代理C": p.consequence_proxy,
        "优先值V": p.priority_value,
        "质量标记": "、".join(p.quality_flags),
    } for p in pipes]