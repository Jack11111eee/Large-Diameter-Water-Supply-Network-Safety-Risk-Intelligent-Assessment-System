"""页面 2：管段详情（§9 第 2 项、§7.2 单管段报告）。

用语纪律（§7.2）：不自动判定"立即换管"，不输出数据中没有的裂缝、漏点、
影响用户数或未来具体爆管日期。建议文本来自决策模块；模块未接入时显示
冻结决策包中的规则证据，不自行生成建议。
"""

import streamlit as st

from .. import adapter, components, sketch


def render(package, mode):
    st.subheader("管段详情")
    st.caption("按 pipe_id 或 BH 检索；展示源字段与质量标记、模型口径、解释、"
               "后果代理、优先值、规则触发依据与建议动作。")

    if not package.pipes:
        components.empty_state(
            "清单为空，无法检索管段。",
            detail=f"目录：{package.report.directory}。界面不补占位行。")
        return

    key = st.text_input("pipe_id 或 BH", value="", placeholder="例如 237191 或 SGD02473408")
    matches = _search(package.pipes, key)
    if key.strip() and not matches:
        components.empty_state(f"没有匹配 {key!r} 的管段。",
                               detail="请核对 pipe_id 或 BH；界面不构造替代记录。",
                               icon="🔍")
        return
    if not matches:
        st.info("请输入 pipe_id 或 BH 进行检索。也可以直接选择前若干条高优先值管段。", icon="🔎")
        matches = adapter.select_top_k(package, 20, "V") or list(package.pipes[:20])

    labels = {f"{p.pipe_id}｜BH {p.bh}｜{p.relative_risk_level}": p for p in matches[:200]}
    choice = st.selectbox("检索结果", options=list(labels), key="detail_choice")
    _render_pipe(labels[choice], package, mode)


def _search(pipes, key):
    key = key.strip()
    if not key:
        return []
    lowered = key.lower()
    return [p for p in pipes if p.pipe_id == key or p.bh == key
            or lowered in p.pipe_id.lower() or lowered in p.bh.lower()]


def _render_pipe(p, package, mode):
    st.markdown(f"### `{p.pipe_id}`　BH `{p.bh}`")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("概率 p", components.fmt(p.probability))
    c2.metric("相对等级", p.relative_risk_level)
    c3.metric("百分位", components.fmt(p.risk_percentile, 1))
    c4.metric("优先值 V", components.fmt(p.priority_value))
    st.caption(f"模型口径 `{p.model_id}`　轮次 `{p.round_id}`　折 `{components.fmt(p.fold)}`　"
               f"校准 `{components.fmt(p.calibrated)}`　"
               f"预测模式 `{package.report.prediction_mode}`")

    if not p.probability_available:
        st.warning("该管段预测无效（p 缺失或非有限），概率与百分位显示 unavailable，"
                   "界面不代填分数（§8.5 T09）。", icon="⚠️")

    tabs = st.tabs(["源字段与质量", "解释", "后果与优先", "规则与建议", "几何"])

    with tabs[0]:
        st.markdown("**质量标记**")
        st.write(components.quality_flag_text(p.quality_flags))
        st.markdown("**源字段**")
        st.dataframe(components.attribute_rows(p), width="stretch", hide_index=True)
        st.caption("字段值原样展示，不做单位换算。`unavailable` 表示源包中该字段为 null。")

    with tabs[1]:
        components.explanation_panel(p, mode)

    with tabs[2]:
        _consequence_panel(p)

    with tabs[3]:
        _advice_panel(p, package)

    with tabs[4]:
        _geometry_panel(p, mode)


def _consequence_panel(p):
    if not p.decision_available:
        components.empty_state(
            "该管段没有决策结果记录，后果代理与优先值 unavailable。",
            detail="界面不重算 C 或 V。", icon="📭")
        return
    st.dataframe([
        {"项": "后果代理 C", "值": components.fmt(p.consequence_proxy),
         "说明": "带正值下限的无量纲代理，不是金额损失"},
        {"项": "优先值 V", "值": components.fmt(p.priority_value),
         "说明": "V = p × C 的无量纲排序代理，不是金额损失"},
        {"项": "情景版本 scenario_id", "值": p.scenario_id or components.UNKNOWN,
         "说明": "改变配置产生新情景版本，不覆盖旧清单"},
        {"项": "证据引用", "值": "、".join(p.evidence_refs) or components.UNKNOWN,
         "说明": "指向源表字段或模型产物"},
    ], width="stretch", hide_index=True)
    st.caption("C 与 V 由决策模块唯一实现，本页只展示；未接入时展示冻结决策包的结果对象。")


def _advice_panel(p, package):
    advice = package.decision.advice.get(p.pipe_id)
    if advice:
        st.markdown("**建议记录（来自决策模块）**")
        st.dataframe([{"字段": k, "值": _render_value(v)} for k, v in advice.items()],
                     width="stretch", hide_index=True)
        return

    st.info("决策模块尚未接入，本页不自行生成建议文本。"
            "以下为该管段在冻结决策包中的规则触发依据（不是正式建议）。", icon="ℹ️")
    expected = (package.decision.expected_rules or {}).get("cases", {})
    case = expected.get(p.pipe_id) or expected.get(p.pipe_id.replace("SYN-", ""))
    if not case:
        st.caption("冻结决策包中没有该管段的规则参考记录。")
    else:
        triggers = case.get("expected_triggers") or []
        st.write(f"预期触发规则：{'、'.join(triggers) if triggers else 'R00（一般核查）'}")
        st.caption(case.get("note", ""))
    st.caption("规则证据与模型归因分开呈现：R01 因材料和相对等级触发，"
               "不等于材料一定是正向贡献（§8.4）。")


def _render_value(v):
    if isinstance(v, (list, tuple)):
        return "、".join(str(x) for x in v) or components.UNKNOWN
    if isinstance(v, dict):
        return "；".join(f"{k}={v2}" for k, v2 in v.items()) or components.UNKNOWN
    return components.UNKNOWN if v is None else str(v)


def _geometry_panel(p, mode):
    if not p.has_geometry:
        components.empty_state("该管段没有几何记录，无法绘制。",
                               detail="属性评分不受影响，仍可单独展示（§8.4 D01）。", icon="🗺️")
        return
    st.dataframe([
        {"项": "起点 X", "值": components.fmt(p.x_start, 3)},
        {"项": "起点 Y", "值": components.fmt(p.y_start, 3)},
        {"项": "终点 X", "值": components.fmt(p.x_end, 3)},
        {"项": "终点 Y", "值": components.fmt(p.y_end, 3)},
        {"项": "分量 ID", "值": components.fmt(p.component_id)},
        {"项": "坐标冲突", "值": components.fmt(p.coordinate_conflict)},
        {"项": "CRS 已知", "值": components.fmt(p.crs_known)},
    ], width="stretch", hide_index=True)
    if p.coordinate_conflict:
        st.warning("该管段端点存在节点坐标冲突，保留原始端点、不自动吸附。"
                   "空间结果待核；属性评分仍然有效（§8.4 D01）。", icon="⚠️")
    if not p.crs_known:
        st.caption(sketch.SKETCH_CAPTION)
    figure, _ = sketch.build_figure([p], mode, layout="coordinate", highlight=p)
    st.plotly_chart(figure, width="stretch")