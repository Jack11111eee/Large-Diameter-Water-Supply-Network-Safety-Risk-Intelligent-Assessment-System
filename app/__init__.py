"""界面包：Streamlit 单体应用（§9、§13.2）。

启动：`streamlit run app/main.py`

模块边界（§13.1）：

- `adapter.py`　唯一的装载、校验与按 `pipe_id` 连接入口，不导入 Streamlit。
- `theme.py`　　颜色与版式令牌。
- `sketch.py`　 本地坐标/拓扑示意图，只做呈现。
- `components.py` 共用展示组件。
- `views/`　　　三个 M1 页面。

**本包不含任何业务计算。** 百分位、后果代理 C、优先值 V、建议规则均在
`src/decision/` 唯一实现；界面只展示其结果。`tests/app/` 用 AST 扫描守住这条边界。
"""

__all__ = ["adapter", "components", "sketch", "theme"]