"""示意图测试（§4 拓扑与可视化数据契约）。

只验证呈现行为：坐标纪律、分量/冲突标记、纯拓扑布局的预算降级。
不验证任何业务量——等级、百分位、优先值都只是被读取的字段。
"""

from app import adapter, sketch, theme


def _pipes(n=8):
    return adapter.load_package_set(adapter.FIXTURE_DIR).pipes[:n]


def test_coordinate_layout_draws_segments_without_note():
    fig, note = sketch.build_figure(_pipes(), "light", layout="coordinate")
    assert note is None
    assert len(fig.data) >= 1
    assert "源坐标" in fig.layout.title.text


def test_topology_layout_has_no_coordinate_meaning():
    """纯拓扑布局不得把布局坐标当作源坐标呈现。"""
    fig, note = sketch.build_figure(_pipes(), "light", layout="topology")
    assert note is None
    assert "拓扑" in fig.layout.title.text
    assert "无坐标含义" in fig.layout.xaxis.title.text


def test_topology_over_budget_falls_back_and_says_so():
    """超出节点预算时退回坐标示意，并明确说明，不静默换口径。"""
    pipes = list(_pipes())
    # 构造超过预算的节点数：每条管段给独立端点
    big = []
    for i in range(sketch.TOPOLOGY_NODE_BUDGET + 10):
        p = pipes[0]
        big.append(type(p)(**{**p.__dict__,
                              "pipe_id": f"X{i}",
                              "x_start": float(i), "y_start": 0.0,
                              "x_end": float(i) + 0.5, "y_end": 1.0}))
    fig, note = sketch.build_figure(big, "light", layout="topology")
    assert note and "节点预算" in note
    assert "源坐标" in fig.layout.title.text


def test_both_modes_produce_figures():
    for mode in ("light", "dark"):
        fig, _ = sketch.build_figure(_pipes(), mode, layout="coordinate")
        assert fig.layout.paper_bgcolor == (theme.SURFACE_DARK if mode == "dark"
                                            else theme.SURFACE_LIGHT)


def test_conflict_marker_uses_shape_not_color_only():
    """坐标冲突用菱形标记（形状通道），不与等级色相冲突。"""
    pipes = adapter.load_package_set(adapter.FIXTURE_DIR).pipes
    conflict = [p for p in pipes if p.coordinate_conflict]
    assert conflict, "夹具应含坐标冲突样例"
    fig, _ = sketch.build_figure(pipes, "light", layout="coordinate")
    symbols = [t.marker.symbol for t in fig.data if t.mode == "markers"]
    assert "diamond" in symbols


def test_highlight_trace_added_for_single_pipe():
    p = _pipes(1)[0]
    fig, _ = sketch.build_figure([p], "light", layout="coordinate", highlight=p)
    names = [t.name for t in fig.data]
    assert "当前管段" in names


def test_unknown_level_uses_neutral_not_a_ramp_step():
    """等级未知用中性灰，不占用色相槽位。"""
    colors = theme.level_colors("light")
    assert colors["unavailable"] == theme.NEUTRAL
    assert colors["unavailable"] not in theme.LEVEL_RAMP["light"].values()


def test_level_ramp_is_ordered_light_to_dark():
    """相对等级是有序类别：浅→深，让顺序可从颜色读出。"""
    ramp = theme.LEVEL_RAMP["light"]
    assert list(ramp) == list(theme.LEVEL_ORDER)
    assert ramp["L1"] != ramp["L4"]


def test_component_summary_counts_only():
    summary = sketch.component_summary(_pipes())
    assert summary["n_components"] >= 1
    assert summary["n_with_geometry"] == 8
    assert summary["n_conflict"] == 1


def test_figure_omits_pipes_without_geometry():
    """没有几何记录的管段不进入图，也不凭空补坐标。"""
    p = _pipes(1)[0]
    stripped = type(p)(**{**p.__dict__, "has_geometry": False})
    fig, _ = sketch.build_figure([stripped], "light", layout="coordinate")
    assert len(fig.data) == 0