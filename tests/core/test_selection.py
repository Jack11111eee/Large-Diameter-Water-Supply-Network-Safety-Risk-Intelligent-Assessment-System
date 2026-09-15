"""候选选择测试（§5.1、§5.2）。

M2 具名风险是「用外层成绩回选模型」——本文件同时守住该风险的不可达性。
"""

import numpy as np
import pandas as pd
import pytest

from src.data import loader
from src.evaluation import selection, split
from src.models import registry


@pytest.fixture(scope="module")
def data():
    pipes = loader.load_attributes()
    counts, _ = loader.make_labels(pipes, loader.load_events())
    return pipes, counts


# ---- 预注册候选 ----

def test_declared_candidates_match_config():
    """展开的候选与参数网格与预注册配置逐条一致（§5.1）。"""
    cfg = registry.load_candidates()
    cands = selection.candidates_from_config(["PIPEAGE", "CZ"])
    expected = [(s["name"], tuple(sorted(p.items())))
                for s in cfg["candidates"] for p in s["param_grid"]]
    got = [(c.name, tuple(sorted(c.params.items()))) for c in cands]
    assert sorted(got) == sorted(expected)
    assert len(cands) == len(expected)


def test_candidates_ordered_by_complexity_rank():
    """容差并列时按显式复杂度序优先简单模型，不依赖字典插入序。"""
    cands = selection.candidates_from_config(["PIPEAGE"])
    ranks = [c.complexity_rank for c in cands]
    assert ranks == sorted(ranks)
    families = {c.family for c in cands}
    assert families == {"linear", "tree"}


def test_registry_rejects_unknown_family():
    with pytest.raises(KeyError, match="未知候选族"):
        registry.factory_of({"name": "x", "family": "gnn"}, ["PIPEAGE"])


# ---- 选择只用内层信息 ----

class _Const:
    def __init__(self, p):
        self.p = p

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        return np.full(len(X), self.p)


class _Perfect:
    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        return X["sig"].to_numpy(dtype=float)


def _toy():
    X = pd.DataFrame({
        "PIPEAGE": [1.0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        "sig": [0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0],
    })
    return X, X["sig"].to_numpy()


def _cand(name, rank, factory):
    return selection.Candidate(name=name, family="fake", complexity_rank=rank,
                               params={"p": 0.0}, factory=factory)


def test_tolerance_prefers_simpler_candidate():
    """内层 AP 并列时选复杂度序更低者（§5.2）。"""
    X, y = _toy()
    cands = [_cand("simple", 0, lambda p: _Const(0.5)),
             _cand("complex", 1, lambda p: _Const(0.5))]
    chosen, log = selection.select_within_train(X, y, cands)
    assert chosen.name == "simple"
    assert log["tie_break"] == "prefer_lower_complexity_rank"
    assert log["tie_break_fired"] is False


def test_clearly_better_candidate_wins():
    """超出容差时选内层 AP 更高者——简单模型不是被硬保送。"""
    X, y = _toy()
    simple = _cand("simple", 0, lambda p: _Const(0.5))
    complex_ = _cand("complex", 1, lambda p: _Perfect())
    chosen, log = selection.select_within_train(X, y, [simple, complex_])
    assert chosen.name == "complex"
    assert log["per_candidate"][complex_.label]["mean_ap"] > \
        log["per_candidate"][simple.label]["mean_ap"]


def test_selection_log_has_no_outer_key():
    """选择日志不得含任何外层成绩键（§5.2 禁止用外层成绩回选）。"""
    X, y = _toy()
    cands = [_cand("simple", 0, lambda p: _Const(0.5))]
    _, log = selection.select_within_train(X, y, cands)
    assert not [k for k in log if "outer" in k.lower()]


def test_degenerate_inner_folds_pick_first_in_order():
    """内层退化时按预定义顺序取首个候选，不随机拆组（§6.1）。"""
    X, y = _toy()
    groups = np.array(["a"] * 12, dtype=object)  # 单组 -> 退化
    cands = [_cand("simple", 0, lambda p: _Const(0.5)),
             _cand("complex", 1, lambda p: _Perfect())]
    chosen, log = selection.select_within_train(X, y, cands, groups=groups)
    assert log["status"] == "degenerate"
    assert chosen.name == "simple"
    assert "退化" in log["note"]


def test_selection_log_records_criterion_and_tolerance():
    X, y = _toy()
    _, log = selection.select_within_train(
        X, y, [_cand("simple", 0, lambda p: _Const(0.5))], tolerance=0.01)
    assert log["criterion"] == "inner_ap"
    assert log["tolerance"] == 0.01
    assert log["n_inner_folds"] == 3


# ---- 真实数据上的逐折运行 ----

FAST_CONFIG = {
    "candidates_version": "test_fast",
    "selection": {"criterion": "inner_ap", "tolerance": 0.005},
    "candidates": [{
        "name": "logreg", "family": "linear", "complexity_rank": 0,
        "param_grid": [{"C": 1.0}],
    }],
}


@pytest.fixture(scope="module")
def run(data):
    pipes, counts = data
    cols = loader.resolve_layer("F1_base_environment")
    cands = selection.candidates_from_config(cols, config=FAST_CONFIG)
    folds = split.make_outer_folds(
        pipes["ID"].astype(str).to_numpy(),
        counts.gt(0).astype(int).to_numpy())
    return selection.select_and_run(pipes, counts, candidates=cands, folds=folds)


def test_select_and_run_covers_every_pipe_once(run):
    df, log = run
    assert len(df) == 7288
    assert df.pipe_id.is_unique
    assert set(log) == {0, 1, 2, 3, 4}


def test_oof_has_no_per_candidate_columns(run):
    """折外表每管段一行，不得按候选分列——否则等于泄露外层比较。"""
    df, _ = run
    assert "logreg" not in df.columns
    for c in df.columns:
        assert not c.startswith("candidate_")
    assert {"pipe_id", "p", "model_id", "raw_score"} <= set(df.columns)


def test_raw_score_is_log_odds_not_probability(run):
    """raw_score 与 p 尺度不同：树/线性候选都必须是原始分数（§7.1）。"""
    df, _ = run
    assert (df["p"] >= 0).all() and (df["p"] <= 1).all()
    assert (df["raw_score"].abs() > 1).any()