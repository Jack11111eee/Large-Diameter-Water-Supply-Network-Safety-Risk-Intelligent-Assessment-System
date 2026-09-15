"""只读数据加载与字段白名单（§3.1、§3.3、§9）。

训练代码仅能读取版本化白名单与独立标签；不对多工作表简单合并后使用全部数值列。
原始数据保持只读，本模块不写入任何 XLSX。
"""

import hashlib
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
ATTR_FILE = ROOT / "DemoPipes属性数据.xlsx"
EVENT_FILE = ROOT / "历史爆管记录_2024.xlsx"
WHITELIST = ROOT / "configs" / "features" / "whitelist.json"

ATTR_SHEET = "全部属性"
EVENT_SHEET = "2024年爆管记录"
SCORE_SHEET = "管段风险评分"

# 已核验的文件指纹（数据分析报告 附录 A）
EXPECTED_SHA256 = {
    "DemoPipes属性数据.xlsx":
        "c9ec7fe516686e5adb5fb64c4584424e1aa738611be943c93a8112fc53b1b18b",
    "历史爆管记录_2024.xlsx":
        "f7dbc6f766ba08ad58eba5973dfb03178009d22fcceb4e93e950645a8919aea7",
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _strip_code(col):
    """'管龄(年)\\n(PIPEAGE)' -> 'PIPEAGE'。"""
    m = re.search(r"\n\(([^)]+)\)$", col)
    return m.group(1) if m else col


def load_attributes():
    """只读加载属性表，列名归一为字段代码。"""
    p = pd.read_excel(ATTR_FILE, sheet_name=ATTR_SHEET)
    p.columns = [_strip_code(c) for c in p.columns]
    return p


def load_events():
    return pd.read_excel(EVENT_FILE, sheet_name=EVENT_SHEET)


def load_scorecard():
    """官方评分表。仅审计与展示使用，不进入训练矩阵（§3.2 禁止字段）。"""
    return pd.read_excel(EVENT_FILE, sheet_name=SCORE_SHEET)


def load_whitelist():
    return json.loads(WHITELIST.read_text(encoding="utf-8"))


def make_labels(pipes, events):
    """一管一行标签（§3.3 规则 1）。不按事件行切训练集。"""
    counts = pipes["ID"].map(events.groupby("管道ID").size()).fillna(0).astype(int)
    return counts, counts.gt(0).astype(int)


LAYER_SEPARATOR = "+"


def resolve_layer(spec):
    """解析特征层规格，支持 '+' 组合（§3.2）。

    按声明顺序拼接并去重；单一层名的行为与原来一致。
    """
    wl = load_whitelist()
    layers = wl["layers"]
    names = [s for s in str(spec).split(LAYER_SEPARATOR) if s]
    if not names:
        raise KeyError(f"空特征层规格 {spec!r}")
    unknown = [n for n in names if n not in layers]
    if unknown:
        raise KeyError(f"未知特征层 {unknown}；可用: {sorted(layers)}")

    cols = []
    for n in names:
        for c in layers[n]:
            if c not in cols:
                cols.append(c)

    overlap = set(cols) & set(wl["forbidden_fields"])
    if overlap:
        raise ValueError(f"白名单层 {spec} 含禁止字段: {sorted(overlap)}")
    return cols


def feature_matrix(pipes, layer):
    """按白名单层级导出训练矩阵。仅含该层字段，禁止字段不进入（§3.4）。

    支持 '+' 组合层；F3 拓扑列由节点 ID 派生后并入（§3.2）。
    """
    cols = resolve_layer(layer)

    missing = [c for c in cols if c not in pipes.columns]
    if missing:
        derivable = set(load_whitelist()["layers"]["F3_topology"])
        if not set(missing) <= derivable:
            raise KeyError(f"白名单层 {layer} 引用了不存在的列: {missing}")
        from src.data.topology import derive_topology_features

        derived = derive_topology_features(pipes)
        frame = pipes.assign(**{c: derived[c] for c in missing})
    else:
        frame = pipes

    return frame[cols].copy()


def with_derived_features(pipes):
    """补齐可由节点 ID 派生的列（F3），供训练路径使用（§3.2）。

    训练路径把原始属性表直接交给模型，模型自行选列；F3 列不在原始表中，
    必须在进入模型前派生。幂等：已存在则不重复派生，索引与顺序不变。
    """
    missing = [c for c in load_whitelist()["layers"]["F3_topology"]
               if c not in pipes.columns]
    if not missing:
        return pipes
    from src.data.topology import derive_topology_features

    derived = derive_topology_features(pipes)
    return pipes.assign(**{c: derived[c] for c in missing})


def numeric_fields():
    """白名单声明的数值字段（§3.2）。供预处理区分数值与类别。"""
    return frozenset(load_whitelist()["numeric_fields"])


def feature_group(field):
    """字段的解释分组（§7.1）。"""
    for name, fields in load_whitelist()["feature_groups"].items():
        if field in fields:
            return name
    return "未分组"


def assert_no_forbidden_columns(columns):
    """泄漏防护测试：训练矩阵不得含任何禁止字段及其别名（§3.4）。"""

    wl = load_whitelist()
    forbidden = set(wl["forbidden_fields"])
    hit = sorted(forbidden & set(columns))
    if hit:
        raise ValueError(f"训练矩阵含禁止字段: {hit}")


def source_refs(field_codes):
    """生成来源追踪记录（§3.1）。"""

    src = load_whitelist()["source_columns"]
    out = {}
    for code in field_codes:
        if code not in src:
            raise KeyError(f"字段 {code!r} 无来源追踪定义")
        out[code] = dict(src[code])
    return out