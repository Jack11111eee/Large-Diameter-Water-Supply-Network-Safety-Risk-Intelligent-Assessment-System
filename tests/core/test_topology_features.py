"""F3 无标签拓扑特征测试（§3.2 F3、§6.2）。

拓扑只由 (QSJD, JSJDID) 决定，不读标签、事件或官方评分字段。
"""

import pandas as pd
import pytest

from src.data import loader, topology


@pytest.fixture(scope="module")
def pipes():
    return loader.load_attributes()


def _toy():
    """三角分量 (A,B,C) + 带平行边的二节点分量 (D,E)。"""
    return pd.DataFrame(
        {
            "QSJD": ["A", "B", "C", "D", "D"],
            "JSJDID": ["B", "C", "A", "E", "E"],
        }
    )


# ---- 定义正确性 ----

def test_topology_values_on_toy_graph():
    """度按平行边重复计数，邻接数按不同邻居计数，二者刻意不同。"""
    t = topology.derive_topology_features(_toy())

    # 起点度：A/B/C 各 2；D 有两条平行边 -> 2
    assert list(t["node_degree"]) == [2, 2, 2, 2, 2]
    # 不同邻居数：A/B/C 各 2；D 只有 E 一个邻居（平行边不增加）
    assert list(t["node_adjacency"]) == [2, 2, 2, 1, 1]
    # 分量管段数：三角分量 3，平行边分量 2
    assert list(t["component_size"]) == [3, 3, 3, 2, 2]


def test_topology_columns_and_index():
    t = topology.derive_topology_features(_toy())
    assert list(t.columns) == ["node_degree", "node_adjacency", "component_size"]
    assert len(t) == 5


def test_topology_deterministic():
    """同一输入两次派生逐值一致（§3.2）。"""
    a = topology.derive_topology_features(_toy())
    b = topology.derive_topology_features(_toy())
    assert a.equals(b)


def test_topology_needs_only_node_ids():
    """只给节点 ID 即可派生——不读标签或任何其他字段（§3.2）。"""
    t = topology.derive_topology_features(_toy())
    assert len(t) == 5


def test_component_ids_matches_release_delegate(pipes):
    """src.data.topology 与原 release.compute_components 输出一致（§4）。"""
    from src.integration import release

    assert topology.component_ids(pipes) == release.compute_components(pipes)


def test_topology_derived_on_real_data(pipes):
    """真实 7,288 条可派生，且无缺失、全为整数。"""
    t = topology.derive_topology_features(pipes)
    assert len(t) == 7288
    assert not t.isna().any().any()
    for c in t.columns:
        assert (t[c] >= 1).all()
        assert t[c].dtype.kind in "iu"