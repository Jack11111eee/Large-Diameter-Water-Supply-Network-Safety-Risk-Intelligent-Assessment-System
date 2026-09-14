"""本地管网示意图（§4、§9）。

**只做数据可视化，不做业务计算。** 颜色、布局与标记都只是呈现；等级、百分位、
后果、优先值一律取自适配器传入的视图对象。

坐标纪律（§4）：

- 源坐标单位未知、CRS 未知，因此只画**直线示意**，不叠加真实底图，
  不输出米制半径、真实管长、供水范围或最近阀门距离。
- 坐标冲突边保留原始端点，并单独标记；不跨节点自动吸附或合并。
- 纯拓扑布局仅按端点坐标重合推断连接关系用于布局，明确标注这是展示用近似。
"""

import networkx as nx
import plotly.graph_objects as go

from . import theme

MARKER_SIZE = 9          # >= 8px（marks-and-anatomy）
LINE_WIDTH = 2

SKETCH_CAPTION = (
    "源坐标，CRS 未知，非真实经纬度。本图为直线示意，不叠加底图，"
    "不代表真实管长、供水范围或阀门距离。"
)
TOPOLOGY_CAPTION = (
    "纯拓扑布局：仅按端点坐标重合推断连接关系用于排布，不合并节点、"
    "不改变任何评分，也不代表真实拓扑或供水连通性。"
)
# spring 布局的节点预算与迭代数：在此规模内约 3–4 秒可收敛。
# 超过则退回坐标示意并如实提示（不是静默切换口径）。
TOPOLOGY_NODE_BUDGET = 1500
TOPOLOGY_ITERATIONS = 30


def _midpoints(pipes):
    ids, xs, ys, levels = [], [], [], []
    for p in pipes:
        if not p.has_geometry:
            continue
        ids.append(p.pipe_id)
        xs.append((p.x_start + p.x_end) / 2.0)
        ys.append((p.y_start + p.y_end) / 2.0)
        levels.append(p.relative_risk_level)
    return ids, xs, ys, levels


def topology_positions(pipes):
    """纯拓扑布局：端点坐标重合 → 连接；对节点图做 spring 布局，取边中点。

    仅用于展示排布。不合并节点，不回写任何属性或评分。
    节点数超过 TOPOLOGY_NODE_BUDGET 时返回 None，由调用方退回坐标示意并如实提示，
    不静默换一套口径。
    """
    graph = nx.Graph()
    for p in pipes:
        if not p.has_geometry:
            continue
        a = (p.x_start, p.y_start)
        b = (p.x_end, p.y_end)
        graph.add_edge(a, b, pipe_id=p.pipe_id)
    if graph.number_of_nodes() == 0:
        return {}
    if graph.number_of_nodes() > TOPOLOGY_NODE_BUDGET:
        return None
    pos = nx.spring_layout(graph, seed=20260914, iterations=TOPOLOGY_ITERATIONS)
    return {data["pipe_id"]: ((pos[a][0] + pos[b][0]) / 2.0,
                              (pos[a][1] + pos[b][1]) / 2.0)
            for a, b, data in graph.edges(data=True)}


def build_figure(pipes, mode, *, layout="coordinate", highlight=None):
    """构建示意图。layout 为 coordinate（源坐标直线示意）或 topology（纯拓扑布局）。

    拓扑布局超出节点预算时自动退回坐标示意，并返回 `(figure, note)` 说明原因；
    note 为 None 表示按请求的布局绘制。
    """
    note = None
    if layout == "topology":
        positions = topology_positions(pipes)
        if positions is None:
            note = (f"当前筛选结果超出纯拓扑布局的节点预算"
                    f"（{TOPOLOGY_NODE_BUDGET} 个节点），已按源坐标直线示意绘制；"
                    "请先按分量或等级缩小范围再切换布局。")
            layout = "coordinate"
    return _build(pipes, mode, layout, highlight), note


def _build(pipes, mode, layout, highlight):
    ink = theme.TEXT.get(mode, theme.TEXT["light"])
    colors = theme.level_colors(mode)
    fig = go.Figure()

    if layout == "topology":
        positions = topology_positions(pipes)
        shown = [p for p in pipes if p.pipe_id in positions]
        centers = {pid: positions[pid] for pid in positions}
        coords = {p.pipe_id: (centers[p.pipe_id], centers[p.pipe_id]) for p in shown}
    else:
        shown = [p for p in pipes if p.has_geometry]
        coords = {p.pipe_id: ((p.x_start, p.y_start), (p.x_end, p.y_end)) for p in shown}

    # 每条管段画成独立线段；同一等级合并为一条 trace，段间用 None 断开。
    for level in theme.LEVEL_ORDER + (theme.LEVEL_UNAVAILABLE,):
        xs, ys, texts = [], [], []
        for p in shown:
            if p.relative_risk_level != level:
                continue
            (x0, y0), (x1, y1) = coords[p.pipe_id]
            xs.extend([x0, x1, None])
            ys.extend([y0, y1, None])
            label = _hover_text(p)
            texts.extend([label, label, None])
        if not xs:
            continue
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines", name=_level_name(level),
            line={"color": colors[level], "width": LINE_WIDTH},
            text=texts, hoverinfo="text", opacity=0.55 if layout == "topology" else 0.9,
        ))

    # 坐标冲突：形状通道（菱形），不与等级色相冲突
    conflict = [p for p in shown if p.coordinate_conflict]
    if conflict:
        cx, cy, ct = [], [], []
        for p in conflict:
            (x0, y0), (x1, y1) = coords[p.pipe_id]
            cx.append((x0 + x1) / 2.0)
            cy.append((y0 + y1) / 2.0)
            ct.append(_hover_text(p) + "｜节点坐标冲突，空间结果待核")
        fig.add_trace(go.Scatter(
            x=cx, y=cy, mode="markers", name="坐标冲突",
            marker={"symbol": "diamond", "size": MARKER_SIZE + 1,
                    "color": "rgba(0,0,0,0)", "line": {"color": ink["primary"], "width": 2}},
            text=ct, hoverinfo="text",
        ))

    if highlight is not None:
        p = highlight
        (x0, y0), (x1, y1) = coords.get(p.pipe_id, ((None, None), (None, None)))
        if x0 is not None:
            fig.add_trace(go.Scatter(
                x=[x0, x1], y=[y0, y1], mode="lines", name="当前管段",
                line={"color": theme.STATUS["critical"], "width": 4},
                text=[_hover_text(p), _hover_text(p)], hoverinfo="text",
            ))

    title = "本地管网示意（纯拓扑布局）" if layout == "topology" else "本地管网示意（源坐标直线）"
    fig.update_layout(**theme.plotly_layout(mode, title=title))
    if layout == "topology":
        # 拓扑布局没有坐标含义，去掉坐标轴文字与等比例锚定
        fig.update_xaxes(title_text="拓扑布局 X（无坐标含义）", showticklabels=False,
                         showgrid=False, scaleanchor=None)
        fig.update_yaxes(title_text="拓扑布局 Y（无坐标含义）", showticklabels=False,
                         showgrid=False, scaleanchor=None)
    fig.update_layout(dragmode="pan")
    return fig


def _level_name(level):
    return "等级未知" if level == theme.LEVEL_UNAVAILABLE else f"相对等级 {level}"


def _hover_text(p):
    prob = "unavailable" if p.probability is None else f"{p.probability:.4f}"
    return (f"pipe_id {p.pipe_id}<br>BH {p.bh}<br>p {prob}"
            f"<br>相对等级 {p.relative_risk_level}"
            f"<br>分量 {p.component_id if p.component_id is not None else 'unavailable'}")


def component_summary(pipes):
    """分量计数与冲突计数，用于页面标记。仅统计，不做推断。"""
    counts = {}
    for p in pipes:
        if p.component_id is None:
            continue
        counts[p.component_id] = counts.get(p.component_id, 0) + 1
    conflicts = sum(1 for p in pipes if p.coordinate_conflict)
    return {
        "n_components": len(counts),
        "largest": sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:5],
        "n_conflict": conflicts,
        "n_with_geometry": sum(1 for p in pipes if p.has_geometry),
    }