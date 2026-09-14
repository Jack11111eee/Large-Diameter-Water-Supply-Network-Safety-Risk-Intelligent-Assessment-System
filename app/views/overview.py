"""页面 1：风险总览与清单（§9 第 1 项）。

概率、相对等级、百分位、后果代理、优先值全部来自适配器；本页不重算任何一项。
"""

import streamlit as st

from .. import adapter, components, sketch, theme


def render(package, mode):
    st.subheader("风险总览与清单")
    st.caption("记录概率、相对等级、固定百分位、本地坐标示意、分量与坐标冲突标记。"
               "本页所有数值由上游产物与决策模块给出，界面不重算。")

    if not package.pipes:
        components.empty_state(
            "清单为空：装载到的 standard_attributes 没有任何管段记录。",
            detail=f"目录：{package.report.directory}。界面不补占位行。")
        return
    _render_body(package, mode)


def _render_body(package, mode):
    pipes = package.pipes
    summary = sketch.component_summary(pipes)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("管段总数", f"{len(pipes):,}")
    c2.metric("有概率记录", f"{sum(1 for p in pipes if p.probability_available):,}")
    c3.metric("坐标冲突边", f"{summary['n_conflict']:,}")
    c4.metric("连通分量数", f"{summary['n_components']:,}")
    st.caption(f"几何覆盖 {summary['n_with_geometry']:,} / {len(pipes):,} 条；"
               "分量仅按端点 ID 统计，**无数据不能推断分量断连就是实际供水故障**（§4）。")

    # ---------------- 图 ----------------
    st.markdown("#### 本地管网示意")
    left, right = st.columns([3, 1])
    with right:
        layout_choice = st.radio(
            "布局", options=["coordinate", "topology"],
            format_func=lambda x: "源坐标直线示意" if x == "coordinate" else "纯拓扑布局",
            key="overview_layout")
        component_filter = st.multiselect(
            "按分量筛选（0 = 不过滤）", options=sorted({p.component_id for p in pipes
                                                        if p.component_id is not None}),
            default=[], key="overview_components")
        conflict_only = st.checkbox("只看坐标冲突边", key="overview_conflict_only")
        level_filter = st.multiselect(
            "按相对等级筛选", options=list(theme.LEVEL_ORDER) + [theme.LEVEL_UNAVAILABLE],
            default=[], key="overview_levels")

    shown = _filter(pipes, component_filter, conflict_only, level_filter)
    with left:
        if not shown:
            components.empty_state("当前筛选条件下没有管段。", icon="🔍")
        elif not any(p.has_geometry for p in shown):
            components.empty_state(
                "当前筛选结果没有几何记录，无法绘图。",
                detail="几何包缺失或筛选后为空；清单表仍然可用。", icon="🗺️")
        else:
            with st.spinner("正在计算纯拓扑布局…" if layout_choice == "topology" else None):
                figure, layout_note = sketch.build_figure(shown, mode, layout=layout_choice)
            st.plotly_chart(figure, width="stretch")
            if layout_note:
                st.info(layout_note, icon="ℹ️")
            st.caption(sketch.SKETCH_CAPTION)
            if layout_choice == "topology" and not layout_note:
                st.caption(sketch.TOPOLOGY_CAPTION)
            st.caption(f"当前图中显示 {len(shown):,} 条管段；"
                       f"颜色为相对等级的有序阶梯（L1 浅 → L4 深），"
                       f"菱形标记表示节点坐标冲突。")

    # ---------------- 清单 ----------------
    st.markdown("#### 风险清单")
    st.caption("按优先值 V 降序排列；V 与百分位来自决策结果，界面只做排序与筛选。")
    rows = adapter.pipe_list_rows(package)
    rows = [r for r in rows if r["pipe_id"] in {p.pipe_id for p in shown}]
    if not rows:
        components.empty_state("当前筛选条件下清单为空。", icon="📭")
    else:
        st.dataframe(
            rows, width="stretch", hide_index=True,
            column_config={
                "p": st.column_config.NumberColumn("概率 p", format="%.4f"),
                "百分位": st.column_config.NumberColumn("百分位", format="%.1f"),
                "后果代理C": st.column_config.NumberColumn("后果代理 C", format="%.4f"),
                "优先值V": st.column_config.NumberColumn("优先值 V", format="%.4f"),
                "坐标冲突": st.column_config.CheckboxColumn("坐标冲突"),
            })
        st.caption(f"共 {len(rows):,} 行。`unavailable` 表示该字段在上游产物中缺失或不可用，"
                   "界面不以 0 或占位值顶替。")

    st.markdown("##### 分量分布（前 5）")
    st.dataframe([{"分量 ID": cid, "管段数": n} for cid, n in summary["largest"]],
                 width="stretch", hide_index=True)


def _filter(pipes, component_ids, conflict_only, levels):
    out = []
    for p in pipes:
        if component_ids and p.component_id not in component_ids:
            continue
        if conflict_only and not p.coordinate_conflict:
            continue
        if levels and p.relative_risk_level not in levels:
            continue
        out.append(p)
    return out