"""展示令牌：颜色、字体与 Plotly 版式。

颜色不是手挑的，按 dataviz 方法按职责取值：
- 相对等级是**有序**类别（L1<L2<L3<L4），因此用单色相 ordinal 阶梯，
  而不是分类色板，让读者从颜色本身看出顺序。
- 两个模式各自取值，并已用 `validate_palette.js --ordinal` 在各自底色上验证通过：
  light `#86b6ef,#5598e7,#2a78d6,#184f95`（浅端 2.06:1）；
  dark  `#184f95,#256abf,#3987e5,#86b6ef`（浅端 2.15:1，暗色下翻转锚点）。
- 等级未知用中性灰，不占用任何色相槽位。
- 坐标冲突是数据质量状态，用**形状**通道（菱形标记）而非颜色，避免与等级色相冲突。
"""

SURFACE_LIGHT = "#fcfcfb"
SURFACE_DARK = "#1a1a19"

TEXT = {
    "light": {"primary": "#0b0b0b", "secondary": "#52514e", "muted": "#898781"},
    "dark": {"primary": "#ffffff", "secondary": "#c3c2b7", "muted": "#898781"},
}

# 有序等级阶梯（L1 最浅 → L4 最深）
LEVEL_RAMP = {
    "light": {"L1": "#86b6ef", "L2": "#5598e7", "L3": "#2a78d6", "L4": "#184f95"},
    "dark": {"L1": "#184f95", "L2": "#256abf", "L3": "#3987e5", "L4": "#86b6ef"},
}
LEVEL_ORDER = ("L1", "L2", "L3", "L4")
LEVEL_UNAVAILABLE = "unavailable"
NEUTRAL = "#898781"          # 等级未知 / 无概率：中性，不占用色相

# 状态色（固定，永不主题化）：只用于数据质量提示，且始终配图标与文字
STATUS = {
    "good": "#0ca30c",
    "warning": "#fab219",
    "serious": "#ec835a",
    "critical": "#d03b3b",
}

FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'

QUALITY_FLAG_LABELS = {
    "coordinate_conflict": "节点坐标冲突（空间结果待核）",
    "prediction_invalid": "预测无效（p 缺失或非有限）",
    "explanation_pending": "解释未就绪",
    "semantics_unverified": "字段语义未核",
    "traffic_proxy": "道路类型作交通代理",
    "diameter_out_of_range": "管径超出训练范围，未外推",
    "press_missing": "运行压力缺失",
    "proxy_imputed:facility": "设施类别未知，代理代填",
    "not_pre_event": "运行快照，非事件前口径",
    "may_be_post_event": "可能含事件后信息",
    "zero_means_unrecorded": "0 表示未记录，非真实 0",
}


def mode_of(theme_type) -> str:
    """Streamlit 主题类型 → 本模块的模式名。无法判定时退回 light。"""
    return "dark" if theme_type == "dark" else "light"


def level_colors(mode: str) -> dict:
    colors = dict(LEVEL_RAMP.get(mode, LEVEL_RAMP["light"]))
    colors[LEVEL_UNAVAILABLE] = NEUTRAL
    return colors


def plotly_layout(mode: str, *, title: str = "", height: int = 620) -> dict:
    """统一的 Plotly 版式：细笔画、发丝网格、留白，坐标轴文字用中性墨色。"""
    ink = TEXT.get(mode, TEXT["light"])
    surface = SURFACE_DARK if mode == "dark" else SURFACE_LIGHT
    grid = "#2c2c2a" if mode == "dark" else "#e1e0d9"
    axis = "#383835" if mode == "dark" else "#c3c2b7"
    return {
        "title": {"text": title, "font": {"size": 15, "color": ink["primary"], "family": FONT}},
        "paper_bgcolor": surface,
        "plot_bgcolor": surface,
        "font": {"family": FONT, "color": ink["secondary"], "size": 12},
        "height": height,
        "margin": {"l": 56, "r": 20, "t": 48 if title else 16, "b": 52},
        "hovermode": "closest",
        "xaxis": {
            "title": {"text": "源坐标 X（单位未知，非经纬度）", "font": {"color": ink["muted"]}},
            "gridcolor": grid, "zeroline": False, "linecolor": axis,
            "tickfont": {"color": ink["muted"]}, "showline": True,
        },
        "yaxis": {
            "title": {"text": "源坐标 Y（单位未知，非经纬度）", "font": {"color": ink["muted"]}},
            "gridcolor": grid, "zeroline": False, "linecolor": axis,
            "tickfont": {"color": ink["muted"]}, "showline": True,
            "scaleanchor": "x", "scaleratio": 1,
        },
        "legend": {
            "orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0,
            "font": {"color": ink["secondary"]}, "bgcolor": "rgba(0,0,0,0)",
        },
    }