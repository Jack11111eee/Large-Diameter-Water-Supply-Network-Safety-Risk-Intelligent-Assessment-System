"""发布清单与产物构建（§13.4、里程碑 §2.2）。

生产者：A。消费者：B、C。
各表按 pipe_id 验证，而非按行位置拼接（§13.6）。
普通发布包不含 y_true（§13.4）。
"""

import hashlib
import json
from pathlib import Path

import numpy as np

from src.data import loader

SCHEMA_VERSION = "1.0.0"

# 坐标冲突阈值：跨度超过 1 个源坐标单位才标记（数据报告 §6）。
# 小差异可能是精度或格式问题，较大差异需要标记。
COORD_CONFLICT_THRESHOLD = 1.0


def _fingerprint(payload):
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True,
                   separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def compute_components(pipes):
    """数字 ID 图的连通分量（§4）。每个 pipe_id 为独立边，无向多重图。"""
    parent = {}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for u, v in zip(pipes["QSJD"], pipes["JSJDID"]):
        parent.setdefault(u, u)
        parent.setdefault(v, v)
        ru, rv = find(u), find(v)
        if ru != rv:
            parent[ru] = rv

    roots, comp = {}, []
    for u in pipes["QSJD"]:
        r = find(u)
        if r not in roots:
            roots[r] = len(roots)
        comp.append(roots[r])
    return comp


def compute_coordinate_conflicts(pipes):
    """节点坐标冲突标记（§4、§6）。

    保留冲突列表，不自动跨节点吸附或合并。
    """
    from collections import defaultdict

    a = pipes["SNODID"].str.extract(r"^([\d.]+)-([\d.]+)$").astype(float).to_numpy()
    b = pipes["ENODID"].str.extract(r"^([\d.]+)-([\d.]+)$").astype(float).to_numpy()
    coords = defaultdict(list)
    for node, xy in zip(list(pipes["QSJD"]) + list(pipes["JSJDID"]), np.r_[a, b]):
        coords[node].append(xy)

    conflict_nodes = set()
    for node, values in coords.items():
        arr = np.array(values)
        if len(arr) < 2:
            continue
        span = np.linalg.norm(arr[:, None, :] - arr[None, :, :], axis=2).max()
        if span > COORD_CONFLICT_THRESHOLD:
            conflict_nodes.add(node)

    flags = [
        bool(u in conflict_nodes or v in conflict_nodes)
        for u, v in zip(pipes["QSJD"], pipes["JSJDID"])
    ]
    return flags, sorted(conflict_nodes)


# 决策所需的展示字段：R02 触发条件为 OPSST 原字段（§8.4），
# 运行状态快照需同时可见 PRESS 是否缺失（§8.5 T03）。
# 二者属 F2 展示字段，非禁止字段；加入展示包不改变训练白名单（§3.2）。
DECISION_DISPLAY_FIELDS = ("OPSST", "PRESS")


def build_standard_attributes(pipes, *, data_version, run_id):
    """标准展示属性与质量包。不含目标标签、ACCID 或默认历史事件（§13.4）。"""
    wl = loader.load_whitelist()
    fields = list(wl["layers"]["F1_base_environment"]) + list(DECISION_DISPLAY_FIELDS)
    refs = loader.source_refs([c for c in fields if c in wl["source_columns"]])
    conflict_flags, _ = compute_coordinate_conflicts(pipes)

    rows = []
    for i, (_, r) in enumerate(pipes.iterrows()):
        attrs = {}
        for c in fields:
            v = r[c]
            # JSON 不传 NaN（§13.3）
            if isinstance(v, float) and not np.isfinite(v):
                v = None
            elif hasattr(v, "item"):
                v = v.item()
            attrs[c] = v

        qf = []
        for c in fields:
            qf.extend(wl["source_columns"].get(c, {}).get("quality_flags", []))
        if conflict_flags[i]:
            qf.append("coordinate_conflict")

        rows.append({
            "schema_version": SCHEMA_VERSION,
            "data_version": data_version,
            "run_id": run_id,
            "prediction_mode": "oof_replay",
            "data_kind": "real_standard",
            "pipe_id": str(r["ID"]),
            "bh": str(r["BH"]),
            "attributes": attrs,
            "source_refs": refs,
            "quality_flags": sorted(set(qf)),
        })
    return rows


def build_geometry(pipes, *, data_version, run_id):
    """几何包。每条边自身起终点源坐标；坐标系未知必须声明（§4、§13.4）。"""
    a = pipes["SNODID"].str.extract(r"^([\d.]+)-([\d.]+)$").astype(float).to_numpy()
    b = pipes["ENODID"].str.extract(r"^([\d.]+)-([\d.]+)$").astype(float).to_numpy()
    comps = compute_components(pipes)
    conflict_flags, _ = compute_coordinate_conflicts(pipes)

    rows = []
    for i, (_, r) in enumerate(pipes.iterrows()):
        rows.append({
            "schema_version": SCHEMA_VERSION,
            "data_version": data_version,
            "run_id": run_id,
            "prediction_mode": "oof_replay",
            "data_kind": "real_standard",
            "pipe_id": str(r["ID"]),
            "x_start": float(a[i][0]), "y_start": float(a[i][1]),
            "x_end": float(b[i][0]), "y_end": float(b[i][1]),
            "component_id": int(comps[i]),
            "coordinate_conflict": bool(conflict_flags[i]),
            "crs_known": False,
        })
    return rows


def build_predictions(oof_df, *, data_version, run_id):
    """预测包。不含 y_true（§13.4）。"""
    rows = []
    for r in oof_df.itertuples():
        flags = []
        if not np.isfinite(r.p):
            flags.append("prediction_invalid")
        rows.append({
            "schema_version": SCHEMA_VERSION,
            "data_version": data_version,
            "run_id": run_id,
            "prediction_mode": "oof_replay",
            "data_kind": "real_standard",
            "pipe_id": str(r.pipe_id),
            "p": float(r.p),
            "model_id": str(r.model_id),
            "round_id": str(r.round_id),
            "fold": int(r.outer_fold),
            "calibrated": bool(r.calibrated),
            "quality_flags": flags,
        })
    return rows


def build_manifest(artifacts, *, data_version, run_id, model_id, seed):
    """发布清单：列出数据、模型、参考分布、解释、配置的版本（§13.6）。"""
    entries = {}
    for name, rows in artifacts.items():
        entries[name] = {
            "n_rows": len(rows),
            "fingerprint": _fingerprint(rows),
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "data_version": data_version,
        "run_id": run_id,
        "model_id": model_id,
        "seed": seed,
        "artifacts": entries,
        "prediction_mode": "oof_replay",
        "data_kind": "real_standard",
    }


def verify_by_pipe_id(*tables):
    """各表按 pipe_id 验证，而非按行位置拼接（§13.6）。

    tables: (名称, 记录列表) 的序列。
    返回共同 pipe_id 集合；任何表出现重复 ID 则拒绝。
    """
    sets = {}
    for name, rows in tables:
        ids = [r["pipe_id"] for r in rows]
        if len(ids) != len(set(ids)):
            raise ValueError(f"{name}: pipe_id 重复")
        sets[name] = set(ids)
    names = list(sets)
    common = set.intersection(*sets.values()) if sets else set()
    for n in names:
        if sets[n] != common:
            missing = len(sets[n] - common)
            raise ValueError(f"{n}: 与其他表 pipe_id 不一致，差异 {missing} 条")
    return common


def data_version_of(pipe_ids):
    """数据版本：基于管段 ID 集合与文件指纹的确定性标识。"""
    return _fingerprint({
        "pipe_ids": sorted(pipe_ids),
        "files": loader.EXPECTED_SHA256,
    })[:16]


def run_id_of(model_id, seed, data_version):
    return _fingerprint({"model": model_id, "seed": seed,
                         "data": data_version})[:16]


def save_json(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path