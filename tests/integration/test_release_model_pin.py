"""发布模型标识的钉扎测试（§5.1、§5.2、§13.3）。

发布包用哪个模型是评审口径的一部分，不能悄悄换。这里把发布侧模型标识
写成显式声明：改动发布模型，本文件必须跟着改——**测试失败是设计意图**，
不是回归。

M2 把发布模型从 B2_age_logreg（仅管龄，单特征）换成了 F1 层上的候选
选择流程（逐外层折在内层 AP 上选候选，容差 0.005 内优先简单模型）。
B2 保留为文档化的基线，不再是发布模型。改发布模型时三件事一起做：

  1. 重建 outputs/ 四个目录
  2. 在 里程碑与实施计划.md 记成发布模型变更，附新旧对照成绩
  3. 更新本文件的 DECLARED_* 常量
"""

import json

import pytest

from src.integration import pipeline
from src.models import registry

# 依赖真实 7,288 条数据的完整发布构建，耗时较长（见 pytest.ini）
pytestmark = pytest.mark.slow

# 声明的发布模型。改这两个值 = 一次发布模型变更（见模块 docstring）。
DECLARED_FLOW_ID = "F1_candidate_flow"
DECLARED_STRUCTURE = "F1_base_environment"


@pytest.fixture(scope="module")
def released(tmp_path_factory):
    out = tmp_path_factory.mktemp("pin_release")
    pipeline.build_all(out)
    return {p.name: json.loads(p.read_text(encoding="utf-8"))
            for p in out.iterdir() if p.suffix == ".json"}


# ---- 声明本身 ----

def test_declared_release_model_is_the_frozen_one():
    """改了 pipeline 常量却没改本文件，必须失败——这是钉子的锚点。"""
    assert pipeline.RELEASE_FLOW_ID == DECLARED_FLOW_ID
    assert pipeline.RELEASE_STRUCTURE == DECLARED_STRUCTURE


# ---- 发布包真的用了声明的模型 ----

def test_manifest_carries_the_declared_flow(released):
    assert released["manifest.json"]["model_id"] == DECLARED_FLOW_ID


def test_release_is_not_the_m1_baseline(released):
    """B2_age_logreg 已不是发布模型。

    只比对名字不够——名字对了但特征还是单列，等于没换。故一并断言解释
    带多个特征贡献：单特征基线只可能有一个。
    """
    assert released["manifest.json"]["model_id"] != "B2_age_logreg"
    contribs = [len(r["contributions"]) for r in released["explanation.json"]
                if r["status"] == "ok"]
    assert contribs and min(contribs) > 1


def test_every_prediction_uses_a_registered_candidate(released):
    """逐行 model_id 必须是预注册候选名，不得发布没登记过的模型。"""
    known = {c["name"] for c in registry.load_candidates()["candidates"]}
    used = {r["model_id"] for r in released["predictions.json"]}
    assert used and used <= known, f"发布了未注册的候选：{used - known}"


# ---- 清单与产物说的是同一件事 ----

def test_manifest_records_the_candidate_chosen_per_fold(released):
    sel = released["manifest.json"]["selection"]
    assert sel["flow"] == DECLARED_FLOW_ID
    assert sel["structure"] == DECLARED_STRUCTURE
    assert sel["criterion"] == "inner_ap"
    assert sorted(sel["chosen_per_fold"], key=int) == ["0", "1", "2", "3", "4"]
    assert set(sel["chosen_per_fold"].values()) == set(sel["distinct_chosen"])


def test_per_fold_choice_matches_the_predictions(released):
    """清单里逐折选中的候选，必须与预测表逐折的 model_id 对得上。"""
    sel = released["manifest.json"]["selection"]
    by_fold = {}
    for r in released["predictions.json"]:
        by_fold.setdefault(str(r["fold"]), set()).add(r["model_id"])
    for fold, chosen in sel["chosen_per_fold"].items():
        assert by_fold[fold] == {chosen.split("[")[0]}, fold


def test_explanation_model_id_matches_the_prediction(released):
    """解释逐管段的 model_id 必须等于预测的——适配器据此拒绝顶替（§7.1）。"""
    pred = {r["pipe_id"]: r["model_id"] for r in released["predictions.json"]}
    expl = {r["pipe_id"]: r["model_id"] for r in released["explanation.json"]}
    assert expl
    assert {pid: pred[pid] for pid in expl} == expl


def test_run_id_is_recomputable_from_the_declared_spec(released):
    """run_id 必须能由声明的结构/划分/候选集复算出来（§13.3）。"""
    manifest = released["manifest.json"]
    assert manifest["run_id"] == pipeline.release_run_id(
        manifest["data_version"], manifest["split"]["scheme"],
        manifest["split"]["table_fingerprint"])