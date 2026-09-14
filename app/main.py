"""Streamlit 入口（§9）。启动：`streamlit run app/main.py`。

本文件只负责：装载包 → 显示页级徽章 → 分发到三个页面。
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
from app.views import detail, overview, resources  # noqa: E402

PAGES = {
    "风险总览与清单": overview.render,
    "管段详情": detail.render,
    "资源清单": resources.render,
}


def main():
    st.set_page_config(page_title="大口径供水管网安全风险智能评估与决策",
                       page_icon="🛠", layout="wide")
    st.title("大口径供水管网安全风险智能评估与决策")
    st.caption("M1 界面范围：风险总览与清单、管段详情、资源清单。"
               "数据与实验审计页留待后续里程碑。")

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