"""无标签拓扑特征派生（§3.2 F3、§6.2）。

只用节点 ID 结构，不读标签、事件或任何官方评分字段。
F3 在全网（含测试折边）上计算，属转导式增强：部署需披露「全网结构已知」的设定（§6.2）。
"""

from collections import defaultdict

import pandas as pd


def component_ids(pipes):
    """数字 ID 图的连通分量（§4）。每个 pipe_id 为独立边，无向多重图。

    原位于 src/integration/release.py；为修正依赖方向（src/data 不应依赖
    src/integration）而迁入，实现逐字保留。
    """
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


def derive_topology_features(pipes):
    """派生 F3 拓扑列（§3.2）。

    node_degree     起点节点在无向多重图中的度（平行边重复计数）
    node_adjacency  起点节点的不同邻居节点数（不含平行边）
    component_size  所在连通分量的管段数

    仅由 (QSJD, JSJDID) 决定，与标签无关，可重复复现。
    """
    degree = defaultdict(int)
    neighbours = defaultdict(set)
    for u, v in zip(pipes["QSJD"], pipes["JSJDID"]):
        degree[u] += 1
        degree[v] += 1
        neighbours[u].add(v)
        neighbours[v].add(u)

    comps = component_ids(pipes)
    size = defaultdict(int)
    for c in comps:
        size[c] += 1

    starts = list(pipes["QSJD"])
    return pd.DataFrame(
        {
            "node_degree": [degree[u] for u in starts],
            "node_adjacency": [len(neighbours[u]) for u in starts],
            "component_size": [size[c] for c in comps],
        },
        index=pipes.index,
    )