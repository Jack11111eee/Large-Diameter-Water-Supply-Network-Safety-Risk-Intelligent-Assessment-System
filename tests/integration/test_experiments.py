"""实验产物包测试（§5.2、§5.3、§10 W3）。

实验包描述候选选择流程，不是发布成绩。两条纪律：选择日志不含任何外层
成绩；实验包与发布包版本口径可核对（同 data_version、同划分指纹）。
"""

import json

import pytest

from src.data import loader
from src.integration import experiments, pipeline, release

# 依赖真实 7,288 条数据或完整发布构建，耗时较长（见 pytest.ini）
pytestmark = pytest.mark.slow

N = 600


@pytest.fixture(scope="module")
def small():
    pipes = loader.load_attributes().iloc[:N].reset_index(drop=True)
    counts, _ = loader.make_labels(pipes, loader.load_events())
    return pipes, counts


@pytest.fixture(scope="module")
def built(tmp_path_factory, small):
    pipes, counts = small
    monkeypatch_target = (pipes, counts)
    import unittest.mock as mock

    out = tmp_path_factory.mktemp("experiments")
    with mock.patch.object(loader, "load_attributes", lambda: monkeypatch_target[0]), \
            mock.patch.object(loader, "load_events", loader.load_events), \
            mock.patch.object(loader, "make_labels",
                              lambda p, e: (monkeypatch_target[1],
                                            monkeypatch_target[1].gt(0))):
        result = experiments.build_experiments(out)
    payloads = {p.name: json.loads(p.read_text(encoding="utf-8"))
                for p in out.iterdir()}
    return out, result, payloads


def test_all_four_files_are_written(built):
    out, _, payloads = built
    assert sorted(payloads) == ["ablation.json", "calibration_report.json",
                                "manifest.json", "selection_log.json"]


def test_experiment_manifest_points_back_at_the_release(built):
    _, result, payloads = built
    manifest = payloads["manifest.json"]
    assert manifest["data_version"] == result["data_version"]
    assert manifest["structure"] == experiments.EXPERIMENT_STRUCTURE
    assert manifest["split"]["table_fingerprint"]
    assert "不是发布成绩" in manifest["note"]


def test_experiment_run_id_differs_from_the_release_run_id(built):
    """实验包与发布包是两条路径，不得撞同一 run_id（§13.3）。"""
    _, result, payloads = built
    release_run = pipeline.release_run_id(
        result["data_version"], "random",
        payloads["manifest.json"]["split"]["table_fingerprint"])
    assert payloads["manifest.json"]["run_id"] != release_run


def test_selection_log_has_no_outer_scores(built):
    """「用外层成绩回选模型」必须结构上不可达（M2 具名风险，§5.2）。"""
    _, _, payloads = built
    blob = json.dumps(payloads["selection_log.json"], ensure_ascii=False)
    for forbidden in ("outer", "oof", "test_fold", "roc_auc", "pooled"):
        assert forbidden not in blob, f"选择日志出现了外层信息 {forbidden!r}"


def test_selection_log_records_the_criterion_and_order(built):
    _, _, payloads = built
    log = payloads["selection_log.json"]
    assert log["criterion"] == "inner_ap"
    assert log["tolerance"] == 0.005
    assert len(log["folds"]) == 5
    for rec in log["folds"].values():
        assert rec["chosen"] in rec["per_candidate"]
        assert rec["candidate_order"]


def test_calibration_report_is_per_fold_and_forbids_brier_only_claim(built):
    _, _, payloads = built
    report = payloads["calibration_report.json"]
    assert report["method"] == "sigmoid"
    assert report["control"] == "uncalibrated"
    assert len(report["per_fold"]) == 5
    for rec in report["per_fold"].values():
        assert "不构成校准更好的证据" in rec["conclusion"]
        for bins in (rec["bins_calibrated"], rec["bins_uncalibrated"]):
            assert bins
            for b in bins:
                assert "n" in b and "n_positive" in b


def test_ablation_covers_structures_and_carries_disclosures(built):
    _, _, payloads = built
    abl = payloads["ablation.json"]
    assert set(abl["structures"]) == set(experiments.ablation.STRUCTURES)
    for rec in abl["structures"].values():
        assert rec["paired_delta_vs_M1"]["metric"] == abl["metric"]
    assert set(abl["disclosures"]) == {"F2", "F3", "common"}


def test_ablation_never_invokes_a_retracted_ceiling(built):
    """不得以 AUC 上限解释差异——该说法已被撤回（数据报告 §8）。

    披露串本身会**否定**这个说法（「不以任何 AUC 上限解释差异」），
    故只在结构与读数上扫描，不把披露串当违规。
    """
    _, _, payloads = built
    abl = payloads["ablation.json"]
    readings = json.dumps(abl["structures"], ensure_ascii=False)
    for token in ("天花板", "0.80", "上限"):
        assert token not in readings
    assert "天花板" not in json.dumps(abl["disclosures"], ensure_ascii=False)
    # 披露必须明确否定该说法
    assert "不以任何 AUC 上限解释差异" in abl["disclosures"]["common"]


def test_ablation_deltas_are_per_fold_first(built):
    """增益必须是逐折配对差值，而不是两个宏均值之差（§10 W3）。"""
    _, _, payloads = built
    for rec in payloads["ablation.json"]["structures"].values():
        delta = rec["paired_delta_vs_M1"]
        assert set(delta["per_fold_delta"]) == set(rec["per_fold"])
        if delta["n_folds"]:
            mean = sum(delta["per_fold_delta"].values()) / delta["n_folds"]
            assert mean == pytest.approx(delta["mean_delta"])


def test_experiment_output_is_deterministic(built, tmp_path, small):
    import unittest.mock as mock

    pipes, counts = small
    out2 = tmp_path / "again"
    with mock.patch.object(loader, "load_attributes", lambda: pipes), \
            mock.patch.object(loader, "make_labels",
                              lambda p, e: (counts, counts.gt(0))):
        experiments.build_experiments(out2)
    _, _, payloads = built
    assert json.loads((out2 / "manifest.json").read_text(encoding="utf-8")) \
        == payloads["manifest.json"]
