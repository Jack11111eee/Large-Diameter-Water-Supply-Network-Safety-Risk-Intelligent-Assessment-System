"""界面边界守卫：`app/` 不得包含任何业务计算（§13.1、§13.6、里程碑 §4.1）。

这是单人实施下最关键的一条纪律：百分位、后果代理 C、优先值 V、建议规则只在
`src/decision/` 唯一实现，界面只展示。本测试用 AST 扫描 `app/` 源码守住它。
"""

import ast
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import anonymity  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
APP = ROOT / "app"

# 业务计算关键词：出现在**计算**里即视为在界面里重算业务逻辑
BUSINESS_NAMES = (
    "percentile", "consequence", "priority", "advise", "advice_rule",
    "grade_config", "c_min", "backpack",
)
# 允许的例外：
# - `select_top_k` 只对已给出的展示列排序截取，不计算任何新量；
# - `_default_order` 是排序键，取负只为倒序，不产生新的业务量。
ALLOWED_SUFFIXES = ("select_top_k", "_default_order")

FORBIDDEN_IMPORTS = ("src.models", "src.data")


def _app_sources():
    return sorted(p for p in APP.rglob("*.py") if "__pycache__" not in p.parts)


def _tree(path):
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _calls_streamlit(node):
    """该函数体是否调用 Streamlit：是则为纯展示函数，不承担计算职责。"""
    for child in ast.walk(node):
        if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute):
            value = child.func.value
            if isinstance(value, ast.Name) and value.id == "st":
                return True
    return False


def _business_names_in(node):
    """函数体内被直接引用、且名字含业务关键词的标识符。"""
    found = set()
    for child in ast.walk(node):
        name = None
        if isinstance(child, ast.Name):
            name = child.id
        elif isinstance(child, ast.Attribute):
            name = child.attr
        if name and any(token in name.lower() for token in BUSINESS_NAMES):
            found.add(name)
    return found


def test_app_has_sources():
    assert _app_sources(), "app/ 下没有找到任何源码"


@pytest.mark.parametrize("path", _app_sources(), ids=lambda p: str(p.relative_to(ROOT)))
def test_app_does_not_import_models_or_data(path):
    """界面不导入 src.models / src.data（§13.3、§13.6）。"""
    imported = set()
    for node in ast.walk(_tree(path)):
        if isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    for module in imported:
        for forbidden in FORBIDDEN_IMPORTS:
            assert not module.startswith(forbidden), (
                f"{path.relative_to(ROOT)} 导入了 {module}，界面不得依赖训练或原始数据模块")


@pytest.mark.parametrize("path", _app_sources(), ids=lambda p: str(p.relative_to(ROOT)))
def test_app_defines_no_business_functions(path):
    """界面不得定义百分位 / 后果 / 优先值 / 建议规则的**计算**函数。

    纯展示函数（函数体调用 Streamlit）豁免命名检查——展示后果字段不等于计算后果；
    真正的计算由下面的算术检查兜住。
    """
    offenders = []
    for node in ast.walk(_tree(path)):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name.endswith(ALLOWED_SUFFIXES) or _calls_streamlit(node):
            continue
        if any(token in node.name.lower() for token in BUSINESS_NAMES):
            offenders.append(node.name)
    assert not offenders, (
        f"{path.relative_to(ROOT)} 定义了疑似业务计算的函数 {offenders}；"
        "分级/后果/优先值/建议规则只能在 src/decision/ 实现")


@pytest.mark.parametrize("path", _app_sources(), ids=lambda p: str(p.relative_to(ROOT)))
def test_app_does_no_arithmetic_on_business_quantities(path):
    """界面不得对百分位 / 后果 / 优先值做算术（第二套计算的真正形态）。

    `is` / `is not` 只是存在性判断，不是计算，因此不计入。
    """
    offenders = []
    for node in ast.walk(_tree(path)):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name.endswith(ALLOWED_SUFFIXES):
            continue
        for child in ast.walk(node):
            if isinstance(child, ast.Compare):
                # 纯同一性比较（x is None / x is not None）不算算术
                if all(isinstance(op, (ast.Is, ast.IsNot)) for op in child.ops):
                    continue
            elif not isinstance(child, (ast.BinOp, ast.AugAssign)):
                continue
            referenced = set()
            for operand in ast.walk(child):
                if isinstance(operand, ast.Name):
                    referenced.add(operand.id)
                elif isinstance(operand, ast.Attribute):
                    referenced.add(operand.attr)
            hit = {n for n in referenced
                   if any(token in n.lower() for token in BUSINESS_NAMES)}
            if hit:
                offenders.append(f"{node.name}:{sorted(hit)}")
    assert not offenders, (
        f"{path.relative_to(ROOT)} 对业务量做了算术运算 {offenders}；"
        "界面只能展示 src/decision/ 给出的结果")


@pytest.mark.parametrize("path", _app_sources(), ids=lambda p: str(p.relative_to(ROOT)))
def test_app_does_not_define_rule_or_grade_tables(path):
    """界面不得自带规则表或分级阈值表（第二套实现的常见形态）。"""
    offenders = []
    for node in ast.walk(_tree(path)):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if isinstance(target, ast.Name):
                lowered = target.id.lower()
                if any(token in lowered for token in ("rule_table", "grade_table",
                                                      "level_thresholds",
                                                      "consequence_weights")):
                    offenders.append(target.id)
    assert not offenders, (
        f"{path.relative_to(ROOT)} 疑似自带业务配置表 {offenders}；"
        "配置归 configs/decision/，实现归 src/decision/")


def test_app_has_no_absolute_paths():
    """代码中不得出现绝对路径（可移植性）。"""
    for path in _app_sources():
        text = path.read_text(encoding="utf-8")
        hits = anonymity.find_absolute_paths(text)
        assert not hits, f"{path.relative_to(ROOT)} 含绝对路径 {hits}"


def test_app_has_no_school_name():
    """任何文件不得出现校名（赛题硬性扣分项）。

    禁用词表在被 gitignore 排除的 .anonymity_tokens，不把校名写进公开仓库。
    """
    tokens = anonymity.banned_names()
    if not tokens:
        pytest.skip("无本地禁用词表 .anonymity_tokens")
    for path in _app_sources():
        text = path.read_text(encoding="utf-8")
        hits = anonymity.find_banned(text, tokens)
        assert not hits, f"{path.relative_to(ROOT)} 含禁用名称 {hits}"


def test_streamlit_imports_are_confined():
    """Streamlit 调用只出现在界面层，适配器保持可独立测试（§13.6）。"""
    adapter_src = (APP / "adapter.py").read_text(encoding="utf-8")
    assert "import streamlit" not in adapter_src
    assert "streamlit as st" not in adapter_src


def test_views_expose_render_entrypoints():
    """每个页面各有一个 render(package, mode) 入口。"""
    for name in ("overview", "detail", "resources", "post_event"):
        tree = _tree(APP / "views" / f"{name}.py")
        funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
        assert "render" in funcs, f"views/{name}.py 缺少 render 入口"
        assert [a.arg for a in funcs["render"].args.args] == ["package", "mode"]