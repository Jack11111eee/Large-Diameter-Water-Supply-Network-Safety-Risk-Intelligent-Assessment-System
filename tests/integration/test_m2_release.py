"""M2 真实数据发布校验（里程碑 §6 验证项）。

M1 复核的教训：干净夹具的单测漏掉了两个真实数据缺陷。本文件在**真实
7,288 条**上跑分组补充协议（§6.2），断言产物自洽、契约通过、分组边界
被真正尊重。不可降级。
"""

import json
from collections import Counter

import pytest

from src.contracts import validate_package
from src.evaluation import split
from src.integration import pipeline

N_PIPES = 7288


@pytest.fixture(scope="module")
def grouped(tmp_path_factory):
    out = tmp_path_factory.mktemp("m2_grouped")
    result = pipeline.build_grouped(out)
    files = {p.name: json.loads(p.read_text(encoding="utf-8"))
             for p in out.iterdir() if p.suffix == ".json"}
    files["labels_and_splits"] = (out / "labels_and_splits.csv").read_text(
        encoding="utf-8")
    return out, result, files


# ---- 产物自洽 ----

def test_grouped_release_covers_all_pipes(grouped):
    _, _, f = grouped
    assert len(f["standard_attributes.json"]) == N_PIPES
    assert len(f["predictions.json"]) == N_PIPES
    assert len(f["geometry.json"]) == N_PIPES


def test_grouped_evaluation_is_contract_rows(grouped):
    """扁平成绩必须整体过契约校验（§13.4）。"""
    _, _, f = grouped
    rows = f["evaluation.json"]
    validate_package("evaluation", rows)
    assert {r["fold"] for r in rows if r["aggregation"] == "per_fold"} == \
        {0, 1, 2, 3, 4}


def test_grouped_has_no_failed_fold(grouped):
    _, _, f = grouped
    macro = [r for r in f["evaluation.json"] if r["aggregation"] == "macro_mean"]
    assert macro
    for r in macro:
        assert r["valid_folds"] == r["total_folds"] == 5, r["metric"]


def test_all_artifacts_share_one_run_and_data_version(grouped):
    """跨产物 run_id / data_version 必须一致（§13.3）。"""
    _, result, f = grouped
    for name in ("standard_attributes.json", "geometry.json",
                 "predictions.json", "reference_bundle.json",
                 "evaluation.json"):
        rows = f[name]
        assert {r["run_id"] for r in rows} == {result["run_id"]}, name
        assert {r["data_version"] for r in rows} == {result["data_version"]}, name
    assert f["manifest.json"]["run_id"] == result["run_id"]


def test_grouped_round_id_carries_scheme(grouped):
    """分组发布的轮次 ID 带方案后缀，不与随机发布撞 ID（§6.2）。"""
    _, _, f = grouped
    assert {r["round_id"] for r in f["predictions.json"]} == {"seed20260914-road"}
    assert split.round_id_of(20260914) not in \
        {r["round_id"] for r in f["predictions.json"]}


def test_grouped_run_id_differs_from_random(grouped):
    _, result, _ = grouped
    random_run = pipeline.release.run_id_of(pipeline.MODEL_ID, pipeline.SEED,
                                            result["data_version"])
    assert result["run_id"] != random_run


# ---- 分组边界被真正尊重 ----

def test_no_road_spans_two_folds(grouped):
    """同一道路不得跨外层折——分组协议的核心承诺（§6.2）。"""
    _, _, f = grouped
    lines = f["labels_and_splits"].splitlines()
    header = lines[0].split(",")
    gi, fi = header.index("group_id"), header.index("outer_fold")
    road_folds = {}
    for line in lines[1:]:
        parts = line.split(",")
        road_folds.setdefault(parts[gi], set()).add(parts[fi])
    crossing = {r: fs for r, fs in road_folds.items() if len(fs) > 1}
    assert not crossing, f"道路跨折：{list(crossing)[:3]}"


def test_grouping_metadata_is_disclosed(grouped):
    """缺失分组值的处理必须披露，不得静默合并或丢弃（§6.2）。"""
    _, _, f = grouped
    grouping = f["manifest.json"]["split"]["grouping"]
    assert grouping["scheme"] == "road"
    assert grouping["column"] == "SZDL"
    assert grouping["n_missing_rows"] == 1
    assert grouping["n_groups"] >= 100


def test_szdl_never_enters_the_training_matrix(grouped):
    """SZDL 只作分组，永不进训练矩阵（§3.2 隔离字段）。"""
    from src.data import loader

    for layer in ("F1_base_environment", "F2_operational_snapshot"):
        assert "SZDL" not in loader.resolve_layer(layer)


# ---- 发布清单即审计数据源 ----

def test_manifest_carries_audit_sections(grouped):
    _, result, f = grouped
    m = f["manifest.json"]
    assert set(m["configs"]) == {"evaluation_protocol", "candidates", "whitelist"}
    assert m["split"]["scheme"] == "road"
    assert m["split"]["seed"] == split.SEED
    assert m["split"]["n_outer"] == split.N_OUTER
    assert m["split"]["table_fingerprint"]
    assert len(m["split"]["folds"]) == 5
    assert set(m["data_files"]) >= {"DemoPipes属性数据.xlsx"}
    assert m["training_fingerprint"] == result["manifest"]["training_fingerprint"]
    assert len(m["training_fingerprint"]) == 16


def test_manifest_lists_evaluation_artifact(grouped):
    _, _, f = grouped
    assert "evaluation" in f["manifest.json"]["artifacts"]


def test_training_fingerprint_matches_full_fit(grouped):
    """折外与全量拟合在同一数据上必须得到同一训练指纹（§6.5）。"""
    from src.data import loader
    from src.integration import fullfit

    pipes = loader.load_attributes()
    counts, _ = loader.make_labels(pipes, loader.load_events())
    _, _, f = grouped
    assert f["manifest.json"]["training_fingerprint"] == \
        fullfit.training_fingerprint(pipes, counts)


# ---- 概率链路在真实数据上仍然有效 ----

def test_top100_precision_beats_base_rate(grouped):
    """真实数据 sanity：Top-100 精度高于基率，否则概率链路失效（§6.4）。"""
    _, _, f = grouped
    y = {}
    for line in f["labels_and_splits"].splitlines()[1:]:
        parts = line.split(",")
        y[parts[0]] = int(parts[1])
    preds = sorted(f["predictions.json"], key=lambda r: -r["p"])[:100]
    hits = sum(y[r["pipe_id"]] for r in preds)
    base = sum(y.values()) / len(y)
    assert hits / 100 > base
    assert hits >= 5


def test_probabilities_are_valid(grouped):
    _, _, f = grouped
    for r in f["predictions.json"]:
        assert 0.0 <= r["p"] <= 1.0


def test_no_forbidden_ceiling_claim_in_release(grouped):
    """不得以 AUC 上限解释差异——该说法已被撤回（数据报告 §8）。"""
    _, _, f = grouped
    blob = json.dumps(f["manifest.json"], ensure_ascii=False)
    for token in ("天花板", "0.80 上限", "AUC 上限"):
        assert token not in blob


def test_grade_distribution_covers_every_pipe(grouped):
    """分级合计必须等于管段数，不得丢行（§13.6）。"""
    _, _, f = grouped
    dist = Counter(r["prediction_mode"] for r in f["predictions.json"])
    assert dist == {"oof_replay": N_PIPES}