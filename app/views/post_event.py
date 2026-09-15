"""页面 4：事后运维复核（§9 第 4 项、§8.5）。

默认关闭。开启后以**显式 as_of** 调用决策模块的 `advise_post_event()`：
事后建议是独立清单，不改变 p、等级、C、V，也不改变预测侧清单。
界面不隐式取系统当前日，不重排、不重算、不补占位行。
"""

import streamlit as st

from .. import adapter, components

MAX_ROWS = 200


def render(package, mode):
    st.subheader("事后运维复核")
    st.caption("用于复核既有事件记录，默认关闭。开启后只读截止日之前的记录，"
               "与预测侧清单相互独立（§8.5）。")

    enabled = st.toggle("开启事后运维复核", value=False, key="post_event_enabled")
    if not enabled:
        components.empty_state(
            "事后运维复核默认关闭。",
            detail="该模式读取截止日之前的历史事件。开启后不改变概率、相对等级、"
                   "后果代理 C、优先值 V 或预测清单（§8.5）。", icon="🔒")
        return

    if package.decision.post_event_fn is None:
        components.empty_state(
            "决策模块未提供事后入口，无法开启。",
            detail=f"{package.decision.reason}。界面不自行实现事后规则。", icon="🚫")
        return

    if not package.event_view:
        components.empty_state(
            "本包没有事件视图（event_view）。",
            detail="事后复核需要事件视图；界面不读取原始爆管记录顶替，"
                   "也不以 0 事件填充（§13.3）。", icon="📭")
        return

    as_of = st.text_input(
        "截止日 as_of（YYYY-MM-DD）",
        value=_default_as_of(package),
        key="post_event_as_of",
        help="不隐式取系统当前日；截止日之后的事件不可见（§8.5）。")

    try:
        rows = package.decision.post_event(as_of)
    except Exception as exc:
        components.error_state(
            "决策模块拒绝开启事后模式。",
            detail=f"{type(exc).__name__}: {exc}\n\n界面如实显示拒绝原因，"
                   "不换用其它截止日、不自行过滤事件。")
        return

    _render_rows(package, rows or [], as_of)


def _default_as_of(package):
    """默认截止日取本包事件视图自述的 as_of；缺省时才用适配器常量。"""
    first = package.event_view[0] if package.event_view else None
    if isinstance(first, dict) and first.get("as_of"):
        return str(first["as_of"])
    return adapter.DEFAULT_AS_OF


def _render_rows(package, rows, as_of):
    st.markdown(f"#### 截止日 {as_of} 的事后建议")
    st.caption(f"由决策模块 `advise_post_event()` 返回，共 {len(rows)} 条；"
               "界面不重排、不合并、不补占位行。")

    if not rows:
        components.empty_state(
            "该截止日下没有触发任何事后规则。",
            detail="没有触发不等于没有风险，也不表示无需核查（§8.5）。", icon="📭")
        return

    by_pipe = {}
    for row in rows:
        by_pipe.setdefault(row.get("pipe_id"), []).append(row)

    summary = []
    for pipe_id, items in by_pipe.items():
        p = package.by_id.get(pipe_id)
        summary.append({
            "pipe_id": pipe_id,
            "bh": p.bh if p else components.UNKNOWN,
            "相对等级": p.relative_risk_level if p else components.UNKNOWN,
            "规则": "、".join(str(i.get("rule_id")) for i in items),
            "历史事件数": _event_count(items),
            "建议动作": "；".join(str(i.get("suggested_action")) for i in items),
        })
    summary.sort(key=lambda r: (-(r["历史事件数"] or 0), r["pipe_id"]))
    st.dataframe(summary[:MAX_ROWS], width="stretch", hide_index=True)
    if len(summary) > MAX_ROWS:
        st.caption(f"共 {len(summary)} 条管段，显示前 {MAX_ROWS} 条。")

    st.markdown("#### 触发依据与前置条件")
    st.dataframe(_evidence_rows(rows[:MAX_ROWS]), width="stretch", hide_index=True)
    st.caption("建议文本与前置条件来自决策模块的规则配置；界面不生成建议、"
               "不评估紧迫程度，也不推断失效机理（§8.4）。")


def _event_count(items):
    for item in items:
        values = item.get("trigger_values") or {}
        if "pre_as_of_event_count" in values:
            return values["pre_as_of_event_count"]
    return None


def _evidence_rows(rows):
    out = []
    for row in rows:
        values = row.get("trigger_values") or {}
        out.append({
            "pipe_id": row.get("pipe_id"),
            "规则": row.get("rule_id"),
            "模式": row.get("advice_mode"),
            "as_of": row.get("as_of"),
            "触发值": "、".join(f"{k}={v}" for k, v in values.items()),
            "依据": "；".join(str(x) for x in (row.get("evidence_refs") or ())),
            "前置条件": "；".join(str(x) for x in (row.get("preconditions_to_check") or ())),
            "禁止推断": "；".join(str(x) for x in (row.get("prohibited_inferences") or ())),
        })
    return out