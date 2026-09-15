"""分组划分与内层同边界测试（§6.2）。

随机 CV 是冻结默认路径，必须逐字节不变；分组协议只作补充实验。
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold

from src.data import loader
from src.evaluation import runner, split


@pytest.fixture(scope="module")
def pipes():
    return loader.load_attributes()


@pytest.fixture(scope="module")
def labels(pipes):
    counts, _ = loader.make_labels(pipes, loader.load_events())
    return counts


def _ids_y_groups(pipes, counts):
    ids = pipes["ID"].astype(str).to_numpy()
    y = counts.gt(0).astype(int).to_numpy()
    groups, meta = split.group_ids(pipes, "road")
    return ids, y, groups, meta


# ---- 随机默认路径必须不变（§6.1）----

def test_random_scheme_folds_unchanged(pipes, labels):
    """随机方案折分配与冻结实现逐折一致——分组特性不得扰动主协议。"""
    ids, y, _, _ = _ids_y_groups(pipes, labels)
    expected = {
        i: np.sort(t) for i, (_, t) in enumerate(
            StratifiedKFold(n_splits=5, shuffle=True, random_state=20260914)
            .split(np.asarray(ids), y))
    }
    got = split.make_outer_folds(ids, y)
    assert set(got) == set(expected)
    for k in expected:
        assert np.array_equal(got[k], expected[k])


def test_fingerprint_unchanged_for_random_scheme(pipes, labels):
    t = split.split_table(pipes, labels)
    assert split.table_fingerprint(t) == split.table_fingerprint(t, scheme="random")
    assert split.table_fingerprint(t, scheme="road") != split.table_fingerprint(t)


def test_random_scheme_table_has_no_group_column(pipes, labels):
    """随机划分表保持原四列，labels_and_splits 契约不受影响。"""
    t = split.split_table(pipes, labels)
    assert list(t.columns) == ["pipe_id", "y_true", "count_2024", "outer_fold"]


def test_stratified_kfold_never_reached_when_grouped(pipes, labels, monkeypatch):
    """分组路径不得回落到 StratifiedKFold（§6.2）。"""
    ids, y, groups, _ = _ids_y_groups(pipes, labels)

    def _boom(*a, **k):
        raise AssertionError("分组协议不得使用 StratifiedKFold")

    monkeypatch.setattr(split, "StratifiedKFold", _boom)
    folds = split.make_outer_folds(ids, y, groups=groups)
    assert len(folds) == 5
    inner, status = split.inner_splits(y, groups=groups)
    assert status in {"ok", "reduced_folds"}


# ---- 分组正确性 ----

def test_no_group_spans_outer_folds(pipes, labels):
    ids, y, groups, _ = _ids_y_groups(pipes, labels)
    folds = split.make_outer_folds(ids, y, groups=groups)
    for f, test_idx in folds.items():
        train_idx = np.setdiff1d(np.arange(len(groups)), test_idx)
        assert not (set(groups[train_idx]) & set(groups[test_idx]))


def test_inner_splits_share_outer_group_boundary(pipes, labels):
    """内层不得拆散同路管段（§6.2）——这是最易被违反的一条。"""
    ids, y, groups, _ = _ids_y_groups(pipes, labels)
    folds = split.make_outer_folds(ids, y, groups=groups)
    for test_idx in folds.values():
        train_idx = np.setdiff1d(np.arange(len(groups)), test_idx)
        g_tr = groups[train_idx]
        splits, status = split.inner_splits(y[train_idx], groups=g_tr)
        assert status == "ok"
        for tr, va in splits:
            assert not (set(g_tr[tr]) & set(g_tr[va]))


def test_missing_group_is_singleton(pipes, labels):
    """道路缺失的管段自成一组：不与他组合并，也不丢弃（§6.2）。"""
    _, _, groups, meta = _ids_y_groups(pipes, labels)
    assert meta["n_missing_rows"] == 1
    missing = [g for g in groups if str(g).startswith("__MISSING__")]
    assert len(missing) == 1
    assert list(groups).count(missing[0]) == 1


def test_group_ids_rejects_unknown_scheme(pipes):
    with pytest.raises(KeyError, match="未实现的分组方案"):
        split.group_ids(pipes, "nope")


def test_assert_groups_respected_detects_violation():
    groups = np.array(["a", "a", "b", "b"], dtype=object)
    with pytest.raises(ValueError, match="禁止拆散同组管段"):
        split.assert_groups_respected(groups, [(np.array([0, 2]), np.array([1, 3]))])


def test_grouped_fold_sizes_reported_per_fold(pipes, labels):
    """折规模可能不等，仍逐折报告，不以 pooled 替代（§6.2）。"""
    ids, y, groups, _ = _ids_y_groups(pipes, labels)
    t = split.split_table(pipes, labels, groups=groups)
    summary = split.fold_summary(t, pipes)
    assert set(summary) == {0, 1, 2, 3, 4}
    assert sum(v["n_test"] for v in summary.values()) == len(pipes)
    for v in summary.values():
        assert v["n_groups"] >= 1
        assert "pipeage_median" in v and "material_counts" in v
    assert sum(v["n_missing_group_rows"] for v in summary.values()) == 1


def test_grouped_diagnostics_carry_disclosure(pipes, labels):
    ids, y, groups, _ = _ids_y_groups(pipes, labels)
    folds = split.make_outer_folds(ids, y, groups=groups)
    diag = split.grouped_fold_diagnostics(groups, y, folds)
    assert "与随机 CV 接近不证明无泄漏" in diag["disclosure"]


# ---- 退化折：按预定义无效处理，不随机拆组补齐（§6.1）----

def _degenerate_setup():
    """每个外层训练集都含两类，但分组内层无法分出两类测试折。

    折 0 训练集 = 组 b∪c（各一类，分组内层只能二选一 -> 退化）；
    折 1 训练集 = 组 a（单组 -> 退化）。
    """
    frame = pd.DataFrame({
        "ID": [f"P{i}" for i in range(1, 11)],
        "PIPEAGE": [float(10 * i) for i in range(1, 11)],
    })
    counts = pd.Series([1, 0, 1, 1, 1, 1, 0, 0, 0, 0])
    groups = np.array(["a", "a", "b", "b", "b", "b", "c", "c", "c", "c"],
                      dtype=object)
    return frame, counts, groups, {0: np.array([0, 1]), 1: np.arange(2, 10)}


def test_degenerate_inner_folds_marked_invalid_not_randomly_split():
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    groups = np.array(["b", "b", "b", "b", "c", "c", "c", "c"], dtype=object)
    splits, status = split.inner_splits(y, groups=groups)
    assert status == "degenerate"
    assert splits == []


def test_degenerate_fold_falls_back_to_uncalibrated_control():
    """退化时退回未校准对照并标记，不伪造校准（§5.3、§6.1）。"""
    frame, counts, groups, folds = _degenerate_setup()
    df = runner.run_oof(frame, counts, model_name="B2_age_logreg",
                        folds=folds, groups=groups)
    assert not df["calibrated"].any()
    assert all("calibration_unavailable_degenerate_groups" in f
               for f in df["quality_flags"])


# ---- 折状态复用（§7.1）----

def test_fold_state_returns_fitted_models(pipes, labels):
    folds = split.make_outer_folds(
        pipes["ID"].astype(str).to_numpy(),
        labels.gt(0).astype(int).to_numpy())
    df, state = runner.run_oof(pipes, labels, model_name="B2_age_logreg",
                               folds=folds, return_fold_state=True)
    assert set(state) == set(folds)
    for f, s in state.items():
        assert len(s["models"]) == len(s["calibrators"]) == 3
        assert len(s["train_index"]) + len(s["test_index"]) == len(pipes)
    # 折状态不改变预测
    plain = runner.run_oof(pipes, labels, model_name="B2_age_logreg", folds=folds)
    assert np.allclose(df["p"], plain["p"])