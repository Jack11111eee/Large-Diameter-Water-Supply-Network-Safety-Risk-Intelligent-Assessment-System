"""页面共用组件：徽章、空/错误状态与属性展示。

所有函数只负责**呈现**适配器已给出的字段；不计算百分位、后果、优先值或建议规则。
"""

import streamlit as st

from . import theme

MODE_LABELS = {
    "oof_replay": "oof_replay（折外回放，可用于评测）",
    "full_fit": "full_fit（全量拟合，不能用原训练标签证明泛化）",
    "scenario": "scenario（假设情景，不能宣称真实收益）",
}
SOURCE_LABELS = {
    "real_release": "real_standard（正式产物）",
    "synthetic_fixture": "synthetic_fixture（人工夹具，非正式结果）",
}
DECISION_SOURCE_LABELS = {
    "decision_module": "决策模块 src/decision/ 实时返回",
    "frozen_package": "冻结决策包（决策模块未接入时的降级展示）",
    "absent": "无决策结果，相关字段 unavailable",
}

UNKNOWN = "unavailable"


def fmt(value, digits=4):
    """数值格式化。None / 非有限值一律显示 unavailable，不用 0 或空格顶替。"""
    if value is None:
        return UNKNOWN
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, (int, float)):
        if isinstance(value, float) and value != value:  # NaN
            return UNKNOWN
        if isinstance(value, int):
            return str(value)
        return f"{value:.{digits}f}"
    return str(value)


def badges(package):
    """页级模式徽章与数据来源徽章。"""
    report = package.report
    st.markdown(
        f"`模式` **{MODE_LABELS.get(report.prediction_mode, report.prediction_mode)}**　"
        f"`来源` **{SOURCE_LABELS.get(report.source, report.source)}**　"
        f"`run_id` `{report.run_id}`　`data_version` `{report.data_version}`　"
        f"`schema` `{report.schema_version}`"
    )
    if report.is_fixture:
        st.warning(
            "当前展示的是人工夹具（data_kind=synthetic_fixture），"
            "**其中的数字不是正式结果**，不得用于正式报告或成绩引用（§13.3）。",
            icon="⚠️",
        )
    if report.fallback_reason:
        st.info(f"已回退到夹具目录：{report.fallback_reason}", icon="ℹ️")
    # 决策结果来源始终披露：模块实时返回，或冻结包降级展示（§13.6）
    source_label = DECISION_SOURCE_LABELS.get(package.decision.source,
                                              package.decision.source)
    if package.decision.source == "decision_module":
        st.caption(f"决策结果来源：{source_label}。")
    else:
        st.info(f"决策结果来源：{source_label}。{package.decision.reason}", icon="ℹ️")
    for note in report.warnings:
        st.caption(f"提示：{note}")


def empty_state(message, *, detail=None, icon="📭"):
    """诚实的空状态：说明为空，不补占位行。

    只负责说明，不终止脚本——调用方据此 return 或跳过对应的渲染块。
    """
    st.info(f"{icon} {message}" + (f"\n\n{detail}" if detail else ""))


def error_state(message, *, detail=None):
    """诚实的错误状态：说明拒绝原因，不用其它数据顶替。"""
    st.error(f"无法展示该内容：{message}" + (f"\n\n{detail}" if detail else ""))


def quality_flag_text(flags):
    if not flags:
        return "无"
    return "、".join(theme.QUALITY_FLAG_LABELS.get(f, f) for f in flags)


def attribute_rows(p, keys=None):
    """属性表行：字段代码、展示值、来源与单位。取值来自源包，不做换算。"""
    items = keys if keys is not None else sorted(p.attributes)
    rows = []
    for key in items:
        value = p.attribute(key)
        ref = p.source_refs.get(key)
        if isinstance(ref, dict):
            unit = ref.get("unit", UNKNOWN)
            source = f"{ref.get('source_file', '')}/{ref.get('source_sheet', '')}/{ref.get('source_column', '')}"
            extra = "、".join(ref.get("quality_flags") or ())
        elif isinstance(ref, str):
            unit, source, extra = UNKNOWN, ref, ""
        else:
            unit, source, extra = UNKNOWN, UNKNOWN, ""
        rows.append({
            "字段": key,
            "展示值": UNKNOWN if value is None else str(value),
            "单位": unit,
            "来源": source,
            "来源质量标记": extra or "无",
        })
    return rows


def explanation_panel(p, mode):
    """解释面板。unavailable 时明确说明，不用别的模型顶替（§13.5）。"""
    exp = p.explanation
    if exp.status != "ok":
        st.warning(f"解释 unavailable：{exp.reason or '未就绪'}。"
                   "本页不使用其它模型的解释顶替。", icon="🚫")
        return
    st.caption(f"解释模型 `{exp.model_id}`　尺度 `{exp.scale}`　基准值 `{fmt(exp.base_value)}`")
    if exp.scale == "log_odds":
        st.caption("尺度为 log-odds 原始分数贡献，**不是概率百分点**（§7.1）。")
    names = [c.get("feature") for c in exp.contributions]
    values = [c.get("value") for c in exp.contributions]
    fig = _contribution_figure(names, values, mode)
    st.plotly_chart(fig, width="stretch")
    st.dataframe(
        [{"特征": c.get("feature"), "贡献值": fmt(c.get("value")),
          "方向": c.get("direction") or UNKNOWN} for c in exp.contributions],
        width="stretch", hide_index=True,
    )
    st.caption("归因描述模型使用了哪些信息，不确认腐蚀、冻胀、水锤等机制真实发生（§7.1）。")


def _contribution_figure(names, values, mode):
    """贡献条：正负分色（发散：暖=推高、冷=压低，中点为 0）。"""
    import plotly.graph_objects as go

    ink = theme.TEXT.get(mode, theme.TEXT["light"])
    colors = [theme.STATUS["critical"] if (v or 0) >= 0 else "#2a78d6" for v in values]
    fig = go.Figure(go.Bar(
        x=values, y=names, orientation="h",
        marker={"color": colors, "line": {"width": 0}},
        text=[fmt(v, 3) for v in values], textposition="outside",
        textfont={"color": ink["secondary"]},
        hovertemplate="%{y}: %{x:.4f}<extra></extra>",
    ))
    layout = theme.plotly_layout(mode, title="", height=180 + 34 * max(1, len(names)))
    layout["xaxis"].update({"title": {"text": "原始尺度贡献值（非概率百分点）"},
                            "scaleanchor": None, "scaleratio": None})
    layout["yaxis"].update({"title": {"text": ""}, "autorange": "reversed",
                            "scaleanchor": None, "scaleratio": None})
    layout["showlegend"] = False
    layout["bargap"] = 0.45
    fig.update_layout(**layout)
    return fig