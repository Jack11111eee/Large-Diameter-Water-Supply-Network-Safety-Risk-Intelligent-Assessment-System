"""数据审计（§3.4、§6、里程碑 §5.2）。

复现数据分析报告 V2.0 的核验数值。审计为只读，不修改原始文件。
"""

import math
import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from src.data import loader


def audit_file_versions():
    """文件指纹（数据分析报告 附录 A）。"""
    out = {}
    for name, expected in loader.EXPECTED_SHA256.items():
        real = loader.sha256(loader.ROOT / name)
        out[name] = {"expected": expected, "actual": real, "match": real == expected}
    return out


def audit_keys_and_events(pipes, events, scorecard):
    """唯一键与事件关联（数据报告 §2.1）。"""
    sc = scorecard.set_index("管道ID").loc[pipes["ID"]].reset_index()
    counts = pipes["ID"].map(events.groupby("管道ID").size()).fillna(0).astype(int)
    return {
        "n_pipes": len(pipes),
        "id_unique": bool(pipes["ID"].is_unique),
        "bh_unique": bool(pipes["BH"].is_unique),
        "scorecard_id_unique": bool(scorecard["管道ID"].is_unique),
        "n_events": len(events),
        "events_all_linked": bool(events["管道ID"].isin(pipes["ID"]).all()),
        "counts": counts.value_counts().sort_index().to_dict(),
        "n_positive": int((counts > 0).sum()),
        "event_total": int(counts.sum()),
        "positive_rate": float(counts.gt(0).mean()),
        "accid_equals_count": bool(np.array_equal(pipes["ACCID"], counts)),
        "scorecard_count_equals": bool(np.array_equal(sc["2024年爆管次数"], counts)),
        "bh_consistent": bool(
            (events["管段编号"] == events["管道ID"].map(pipes.set_index("ID")["BH"])).all()
        ),
        "zero_share": float((counts == 0).mean()),
        "poisson_zero": math.exp(-counts.mean()),
    }


def audit_leakage(pipes, events, scorecard):
    """标签与评分泄漏（数据报告 §3.2、§4.1）。"""
    sc = scorecard.set_index("管道ID").loc[pipes["ID"]].reset_index()
    counts = pipes["ID"].map(events.groupby("管道ID").size()).fillna(0).astype(int)
    y = counts.gt(0).to_numpy()
    return {
        "accid_is_label": bool(np.array_equal(pipes["ACCID"], counts)),
        "ops_score_auc": _auc(sc["运维管理分(25)"], y),
        "total_score_auc": _auc(sc["综合风险评分"], y),
        "base_attr_score_auc": _auc(sc["基础属性分(30)"], y),
    }


def _auc(values, y):
    """描述性排序 AUC，重复值平均秩（数据报告 §4.1）。"""
    v = pd.Series(np.asarray(values))
    n = int(y.sum())
    r = v.rank()
    return float((r[y].sum() - n * (n + 1) / 2) / (n * (len(y) - n)))


def audit_single_field_auc(pipes, events):
    """单字段描述性排序 AUC（数据报告 §4.1）。"""
    counts = pipes["ID"].map(events.groupby("管道ID").size()).fillna(0).astype(int)
    y = counts.gt(0).to_numpy()
    out = {}
    for col in ["PIPEAGE", "INSPF", "REPCO2", "RENYR", "PRESS"]:
        out[col] = _auc(pipes[col], y)
    return out


def audit_age_bins(pipes, events):
    """管龄分箱（数据报告 §4.2）。"""
    counts = pipes["ID"].map(events.groupby("管道ID").size()).fillna(0).astype(int)
    y = counts.gt(0).astype(int)
    bins = pd.cut(pipes["PIPEAGE"], [0, 5, 15, 25, 35, 45, 55, 65, float("inf")],
                  right=False)
    g = pd.DataFrame({"y": y, "age_bin": bins}).groupby(
        "age_bin", observed=False)["y"].agg(["count", "sum", "mean"])
    return {str(k): {"count": int(r["count"]), "positive": int(r["sum"]),
                     "rate": (None if pd.isna(r["mean"]) else float(r["mean"]))}
            for k, r in g.iterrows()}


def audit_material(pipes, events):
    counts = pipes["ID"].map(events.groupby("管道ID").size()).fillna(0).astype(int)
    y = counts.gt(0).astype(int)
    g = pd.DataFrame({"y": y, "cz": pipes["CZ"]}).groupby("cz")["y"].agg(
        ["count", "sum", "mean"])
    return {str(k): {"count": int(r["count"]), "positive": int(r["sum"]),
                     "rate": float(r["mean"])} for k, r in g.iterrows()}


def audit_diameter(pipes, events):
    """管径统计（数据报告 §4.3）。"""
    counts = pipes["ID"].map(events.groupby("管道ID").size()).fillna(0).astype(int)
    y = counts.gt(0).to_numpy()
    dn300 = pipes["GJ"] == 300
    dn800 = pipes["GJ"] >= 800
    return {
        "gj_min": int(pipes["GJ"].min()),
        "gj_max": int(pipes["GJ"].max()),
        "dn300_count": int(dn300.sum()),
        "dn300_positive": int(y[dn300].sum()),
        "dn300_events": int(counts[dn300].sum()),
        "dn800_count": int(dn800.sum()),
        "dn800_positive": int(y[dn800].sum()),
        "dn800_rate": float(y[dn800].mean()),
    }


def audit_months(events):
    m = pd.to_datetime(events["爆管日期"]).dt.month.value_counts().sort_index()
    return {int(k): int(v) for k, v in m.items()}


def audit_missing(pipes):
    return {c: int(pipes[c].isna().sum())
            for c in ["JSNF", "SSQY", "QYBS", "SZDL"]}


def audit_constant_columns(pipes):
    """CL 与 QSDW 为单一值（数据报告 §5.3）。"""
    return {
        "CL": pipes["CL"].unique().tolist(),
        "QSDW": pipes["QSDW"].unique().tolist(),
        "constant": [c for c in ["CL", "QSDW"] if pipes[c].nunique() == 1],
    }


def audit_depth_corr(pipes):
    return pipes[["BURDEP", "QDMS", "ZDMS"]].corr().round(6).to_numpy().tolist()


def audit_flow_ratio(pipes):
    """FLOW 单位疑点（数据报告 §5.4）。"""
    q = np.pi * (pipes["GJ"] / 1000) ** 2 / 4 * pipes["VELOC"] * 3600
    ratio = pipes["FLOW"] / q
    return {"min": float(ratio.min()), "median": float(ratio.median()),
            "max": float(ratio.max())}


def audit_score_difference(pipes, scorecard):
    """分项和与总分差异（数据报告 §5.2）。"""
    sc = scorecard.set_index("管道ID").loc[pipes["ID"]].reset_index()
    parts = ["基础属性分(30)", "水力运行分(25)", "运维管理分(25)", "周边环境分(20)"]
    diff = (sc[parts].sum(axis=1) - sc["综合风险评分"]).abs()
    return {
        "n_gt_015": int((diff > 0.15).sum()),
        "max_diff": float(diff.max()),
        "risk_ge_75_attribute": int((pipes["RISK"] >= 75).sum()),
        "scorecard_levels": {str(k): int(v) for k, v in
                             sc["风险等级"].value_counts().items()},
    }


def audit_traffic_road_crosstab(pipes):
    """交通负荷与道路类型完全对应（数据报告 §5.2）。"""
    ct = pd.crosstab(pipes["TRAF"], pipes["RDTYPE"])
    return {"table": ct.to_dict(), "perfect_match": bool((ct > 0).sum().sum() == ct.shape[0])}


def audit_topology(pipes):
    """数字 ID 图统计（数据报告 §6）。"""
    out = {}
    for label, (start, end) in {
        "numeric_id": ("QSJD", "JSJDID"),
        "raw_string": ("SNODID", "ENODID"),
        "node_code": ("QSJDBH", "JSJDBH"),
    }.items():
        adj, degree = defaultdict(set), Counter()
        for u, v in zip(pipes[start], pipes[end]):
            adj[u].add(v)
            adj[v].add(u)
            degree[u] += 1
            degree[v] += 1
        seen, sizes = set(), []
        for u in adj:
            if u in seen:
                continue
            stack, size = [u], 0
            seen.add(u)
            while stack:
                x = stack.pop()
                size += 1
                for w in adj[x]:
                    if w not in seen:
                        seen.add(w)
                        stack.append(w)
            sizes.append(size)
        out[label] = {
            "nodes": len(adj), "edges": len(pipes), "components": len(sizes),
            "cycles": len(pipes) - len(adj) + len(sizes),
            "degree2": sum(d == 2 for d in degree.values()),
            "largest_component": max(sizes),
        }
    return out


def audit_coordinates(pipes):
    """端点坐标与冲突（数据报告 §6）。"""
    a = pipes["SNODID"].str.extract(r"^([\d.]+)-([\d.]+)$").astype(float).to_numpy()
    b = pipes["ENODID"].str.extract(r"^([\d.]+)-([\d.]+)$").astype(float).to_numpy()
    coords = defaultdict(list)
    for node, xy in zip(list(pipes["QSJD"]) + list(pipes["JSJDID"]), np.r_[a, b]):
        coords[node].append(xy)
    spans = []
    for values in coords.values():
        arr = np.array(values)
        spans.append(np.linalg.norm(arr[:, None, :] - arr[None, :, :], axis=2).max())
    allxy = np.r_[a, b]
    return {
        "x_min": float(allxy[:, 0].min()), "x_max": float(allxy[:, 0].max()),
        "y_min": float(allxy[:, 1].min()), "y_max": float(allxy[:, 1].max()),
        "conflict_nonzero": int(sum(d > 1e-9 for d in spans)),
        "conflict_gt_001": int(sum(d > 0.01 for d in spans)),
        "conflict_gt_1": int(sum(d > 1 for d in spans)),
        "max_span": float(max(spans)),
    }


def audit_inspection_dates(pipes):
    return {"min": str(pipes["LSTINSP"].min()), "max": str(pipes["LSTINSP"].max())}


def run_full_audit():
    """完整审计，返回全部核验数值。"""
    pipes = loader.load_attributes()
    events = loader.load_events()
    scorecard = loader.load_scorecard()
    return {
        "file_versions": audit_file_versions(),
        "keys_and_events": audit_keys_and_events(pipes, events, scorecard),
        "leakage": audit_leakage(pipes, events, scorecard),
        "single_field_auc": audit_single_field_auc(pipes, events),
        "age_bins": audit_age_bins(pipes, events),
        "material": audit_material(pipes, events),
        "diameter": audit_diameter(pipes, events),
        "months": audit_months(events),
        "missing": audit_missing(pipes),
        "constant_columns": audit_constant_columns(pipes),
        "depth_corr": audit_depth_corr(pipes),
        "flow_ratio": audit_flow_ratio(pipes),
        "score_difference": audit_score_difference(pipes, scorecard),
        "traffic_road": audit_traffic_road_crosstab(pipes),
        "topology": audit_topology(pipes),
        "coordinates": audit_coordinates(pipes),
        "inspection_dates": audit_inspection_dates(pipes),
    }


if __name__ == "__main__":
    import json
    print(json.dumps(run_full_audit(), ensure_ascii=False, indent=2, default=str))