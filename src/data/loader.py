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


def feature_matrix(pipes, layer):
    """按白名单层级导出训练矩阵。仅含该层字段，禁止字段不进入（§3.4）。"""

    wl = load_whitelist()
    if layer not in wl["layers"]:
        raise KeyError(f"未知特征层 {layer!r}；可用: {sorted(wl['layers'])}")

    cols = wl["layers"][layer]
    forbidden = set(wl["forbidden_fields"])
    overlap = set(cols) & forbidden
    if overlap:
        raise ValueError(f"白名单层 {layer} 含禁止字段: {sorted(overlap)}")

    missing = [c for c in cols if c not in pipes.columns]
    if missing:
        raise KeyError(f"白名单层 {layer} 引用了不存在的列: {missing}")

    return pipes[cols].copy()


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