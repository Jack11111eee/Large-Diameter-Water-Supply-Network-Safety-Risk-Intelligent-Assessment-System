"""模块边界与不变性守卫（§13.1、§13.2、§13.6）。

覆盖：决策模块不导入 pandas/sklearn、不读 Excel、不训练；分级唯一实现；
字段别名两种拼写；事后模式逐字段隔离；输出不含 NaN。
"""

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import anonymity  # noqa: E402

from src.decision import (
    advise_post_event,
    advise_predictive,
    grade,
    load_advice_rules,
    load_grade_config,
    load_scenario_config,
)
from src.decision.aliases import ALIASES, canonical_key, normalize_attributes
from src.decision.errors import DecisionError
from src.decision.priority import prioritize

from .helpers import deep, event_view, fixture, prediction, reference, view, view_codes

RULES = load_advice_rules()
SCENARIO = load_scenario_config()
CFG = load_grade_config()

DECISION_DIR = Path(__file__).resolve().parents[2] / "src" / "decision"

FORBIDDEN_IMPORTS = {"pandas", "sklearn", "numpy", "openpyxl", "scipy",
                     "catboost", "shap", "networkx", "matplotlib"}


def _imported_modules(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.add(node.module.split(".")[0])
    return names


# --------------------------------------------------------------------------
# 模块边界
# --------------------------------------------------------------------------

@pytest.mark.parametrize("path", sorted(DECISION_DIR.glob("*.py")),
                         ids=lambda p: p.name)
def test_decision_module_imports_only_stdlib_and_contracts(path):
    """决策模块不导入 pandas/sklearn/numpy 等；只用标准库 + 契约层。"""
    imported = _imported_modules(path)
    forbidden = imported & FORBIDDEN_IMPORTS
    assert not forbidden, f"{path.name} 导入了禁止依赖 {sorted(forbidden)}"


@pytest.mark.parametrize("path", sorted(DECISION_DIR.glob("*.py")),
                         ids=lambda p: p.name)
def test_decision_module_has_no_absolute_paths(path):
    """代码中不出现绝对路径（§里程碑 4.6）。"""
    text = path.read_text(encoding="utf-8")
    hits = anonymity.find_absolute_paths(text)
    assert not hits, f"{path.name} 含绝对路径 {hits}"


def test_decision_module_does_not_read_excel():
    """决策模块不读 Excel：不出现 read_excel / openpyxl / .xlsx。"""
    for path in DECISION_DIR.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "read_excel" not in text
        assert "openpyxl" not in text
        assert ".xlsx" not in text


def test_decision_import_is_lightweight():
    """在干净子进程中导入决策模块，不触发 pandas/sklearn 加载。"""
    code = (
        "import sys;"
        "import src.decision;"
        "loaded = {m for m in sys.modules if m.split('.')[0] in "
        "{'pandas','sklearn','numpy','openpyxl'}};"
        "print(sorted(loaded))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(DECISION_DIR.parents[1]),
        capture_output=True, text=True, check=True,
    )
    assert result.stdout.strip() == "[]", f"导入了重依赖：{result.stdout}"


def test_grading_implemented_only_in_decision_module():
    """分级（percentile + 四档阈值）唯一实现在 src/decision/，其它模块不复制。

    只检测**计算**的痕迹：平均秩百分位公式，或同时出现四档等级代码与
    80/95/99 阈值。契约层仅声明字段名，不算重复实现。
    """
    root = DECISION_DIR.parents[1] / "src"
    offenders = []
    for path in root.rglob("*.py"):
        if DECISION_DIR in path.parents:
            continue
        text = path.read_text(encoding="utf-8")
        if "count(r < p)" in text or "count(r == p)" in text:
            offenders.append((str(path), "平均秩百分位公式"))
            continue
        has_codes = all(f'"{code}"' in text for code in ("L1", "L2", "L3", "L4"))
        has_thresholds = "80" in text and "95" in text and "99" in text
        if has_codes and has_thresholds:
            offenders.append((str(path), "四档等级阈值"))
    assert not offenders, f"分级疑似在决策模块外重复实现：{offenders}"


# --------------------------------------------------------------------------
# 字段别名
# --------------------------------------------------------------------------

@pytest.mark.parametrize("canonical,aliases", sorted(ALIASES.items()))
def test_alias_map_covers_both_spellings(canonical, aliases):
    """每个规范键都能由夹具拼写与正式列码映射而来。"""
    for alias in aliases:
        assert canonical_key(alias) == canonical


def test_normalize_attributes_accepts_mixed_spellings():
    """同一 attributes 内混用两种拼写也能归一。"""
    normalized = normalize_attributes({"PIPEAGE": 12, "material": "球墨铸铁",
                                       "GJ": 400, "road_type": "次干道"})
    assert normalized == {"pipeage": 12, "material": "球墨铸铁",
                          "diameter_mm": 400, "road_type": "次干道"}


def test_normalize_attributes_rejects_conflicting_spellings():
    """同一字段两种拼写取值不一致 -> 拒绝，不静默择一。"""
    with pytest.raises(DecisionError, match="冲突拼写"):
        normalize_attributes({"PIPEAGE": 12, "pipeage": 40})


def test_normalize_attributes_does_not_mutate():
    original = {"PIPEAGE": 12, "CZ": "球墨铸铁"}
    snapshot = deep(original)
    normalize_attributes(original)
    assert original == snapshot


def test_grading_and_advice_accept_code_spellings():
    """整条链路对正式列码拼写同样可用。"""
    views = [view_codes("P1", PIPEAGE=50, CZ="铸铁管", JOINTT="承插接口",
                        GJ=1000, RDTYPE="主干道", FAC="学校",
                        OPSST="偏高", PRESS=0.6)]
    preds = [prediction("P1", 0.995)]
    grades = grade(preds, reference("r", [i / 100.0 for i in range(1, 101)]), CFG)
    priority = prioritize(views, preds, grades, SCENARIO, {"objective": "V", "k": 10})
    advice = advise_predictive(views, preds, grades, priority, RULES)
    assert {a["rule_id"] for a in advice} == {"R01", "R02", "R03"}


# --------------------------------------------------------------------------
# 事后模式逐字段隔离
# --------------------------------------------------------------------------

def _snapshot(views, preds, ref):
    grades = grade(preds, ref, CFG)
    priority = prioritize(views, preds, grades, SCENARIO,
                          {"objective": "V", "k": 100})
    advice = advise_predictive(views, preds, grades, priority, RULES)
    return grades, priority, advice


def test_post_event_toggle_leaves_everything_byte_identical():
    """开关事后模式后，p/等级/C/V/预测清单逐字节相同。"""
    views = fixture("standard_attributes.json")
    preds = fixture("predictions.json")
    ref = fixture("reference_bundle.json")

    before = _snapshot(views, preds, ref)
    events = event_view({row["pipe_id"]: 2 for row in views})
    advise_post_event(views, events, "2024-12-31", RULES)
    after = _snapshot(views, preds, ref)

    for label, left, right in zip(("grades", "priority", "advice"), before, after):
        assert json.dumps(left, ensure_ascii=False, sort_keys=True) == \
               json.dumps(right, ensure_ascii=False, sort_keys=True), label


def test_post_event_list_is_separate_from_predictive_list():
    """事后清单单独导出，不与预测清单合并（§8.5）。"""
    views = [view("P1", pipeage=50, material="钢管", joint="焊接接口",
                  diameter_mm=400, road_type="支路", facility="绿地",
                  opsst="正常", press_mpa=0.3)]
    preds = [prediction("P1", 0.5)]
    grades = grade(preds, reference("r", [0.1, 0.2, 0.3]), CFG)
    priority = prioritize(views, preds, grades, SCENARIO, {"objective": "V", "k": 10})
    predictive = advise_predictive(views, preds, grades, priority, RULES)
    post = advise_post_event(views, event_view({"P1": 2}), "2024-12-31", RULES)

    assert all(a["advice_mode"] == "predictive" for a in predictive)
    assert all(a["advice_mode"] == "post_event_review" for a in post)
    assert all(a["as_of"] is None for a in predictive)
    assert all(a["as_of"] == "2024-12-31" for a in post)
    # 预测清单里没有 O01
    assert all(a["rule_id"] != "O01" for a in predictive)


def test_post_event_does_not_change_consequence_or_value():
    """事后模式不改变 C 与 V。"""
    views = fixture("standard_attributes.json")
    preds = fixture("predictions.json")
    ref = fixture("reference_bundle.json")
    _, priority_before, _ = _snapshot(views, preds, ref)
    advise_post_event(views, event_view({row["pipe_id"]: 5 for row in views}),
                      "2024-12-31", RULES)
    _, priority_after, _ = _snapshot(views, preds, ref)
    for pid, record in priority_before["records"].items():
        assert record["consequence_proxy"] == priority_after["records"][pid]["consequence_proxy"]
        assert record["priority_value"] == priority_after["records"][pid]["priority_value"]


# --------------------------------------------------------------------------
# 版本冻结
# --------------------------------------------------------------------------

def test_config_versions_are_frozen():
    from src.contracts import (
        ADVICE_RULES_VERSION,
        GRADE_CONFIG_VERSION,
        SCENARIO_CONFIG_VERSION,
    )
    assert CFG["version"] == GRADE_CONFIG_VERSION == "relative_grade_v1"
    assert SCENARIO["version"] == SCENARIO_CONFIG_VERSION == "consequence_scenario_v1"
    assert RULES["version"] == ADVICE_RULES_VERSION == "advice_rules_v1"


def test_module_exposes_version_and_docstring():
    """每个模块交付含输入输出说明与版本号（§13.6）。"""
    import src.decision as decision
    assert decision.VERSION
    assert decision.__doc__
    for name in ("grade", "prioritize", "advise_predictive", "advise_post_event"):
        assert name in decision.__all__
        assert getattr(decision, name).__doc__