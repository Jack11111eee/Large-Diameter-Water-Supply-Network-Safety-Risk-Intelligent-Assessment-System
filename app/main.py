"""Streamlit 入口（§9）。启动：`streamlit run app/main.py`。

本文件只负责：装载包 → 显示页级徽章 → 分发到各页面。
不包含任何业务计算；所有数值来自 `app.adapter` 的视图对象。
"""

import sys
from pathlib import Path

import streamlit as st

# 允许 `streamlit run app/main.py` 时以仓库根为导入根
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import adapter, components, theme  # noqa: E402
from app.views import (  # noqa: E402
    data_audit,
    detail,
    experiment_audit,
    overview,
    post_event,
    resources,
)

PAGES = {
    "风险总览与清单": overview.render,
    "管段详情": detail.render,
    "资源清单": resources.render,
    "事后运维复核": post_event.render,
    "数据审计": data_audit.render,
    "实验审计": experiment_audit.render,
}


def main():
    st.set_page_config(page_title="大口径供水管网安全风险智能评估与决策",
                       page_icon="🛠", layout="wide")
    st.title("大口径供水管网安全风险智能评估与决策")
    st.caption("界面范围：风险总览与清单、管段详情、资源清单、事后运维复核、"
               "数据审计、实验审计。")

    try:
        package = adapter.load_package_set()
    except adapter.AdapterError as exc:
        components.error_state(
            "数据包未通过契约校验，已拒绝接入。",
            detail=f"{exc}\n\n界面不按行号对齐、不静默丢行、不以其它数据顶替（§13.3）。")
        st.stop()

    components.badges(package)

    mode = theme.mode_of(getattr(st.context.theme, "type", None))
    choice = st.sidebar.radio("页面", options=list(PAGES), key="page_choice")
    st.sidebar.caption(f"装载目录：`{package.report.directory}`")
    st.sidebar.caption(f"管段数：{len(package.pipes):,}")

    PAGES[choice](package, mode)


main()