"""事后运维复核测试（§8.5、§9 第 4 项）。

两条纪律：事后模式必须显式给 as_of（不隐式取系统当前日）；
事后建议是独立清单，不改变预测侧任何数值。
"""

import ast
import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from app import adapter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import anonymity  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
MAIN = ROOT / "app" / "main.py"
PAGE = ROOT / "app" / "views" / "post_event.py"
PAGE_NAME = "事后运维复核"
TIMEOUT = 300

AS_OFS = ("2023-12-31", "2024-03-31", "2024-06-30", "2024-12-31")


@pytest.fixture(scope="module")
def package():
    return adapter.load_package_set()


def _open_page(page=PAGE_NAME):
    at = AppTest.from_file(str(MAIN), default_timeout=TIMEOUT)
    at.run()
    at.sidebar.radio[0].set_value(page).run()
    return at


def _rendered_total(at):
    """页面自述的事后建议条数。"""
    import re

    for caption in at.caption:
        match = re.search(r"advise_post_event\(\)` 返回，共 (\d+) 条", caption.value)
        if match:
            return int(match.group(1))
    raise AssertionError("页面没有给出事后建议条数说明")


def package_rows(as_of):
    return adapter.load_package_set().decision.post_event(as_of)


# ---- 适配器入口 ----

def test_default_as_of_is_an_explicit_constant():
    """默认截止日是常量，不是系统当前日（§8.5）。"""
    import datetime

    assert adapter.DEFAULT_AS_OF == "2024-12-31"
    assert adapter.DEFAULT_AS_OF != datetime.date.today().isoformat()


def test_post_event_entry_is_exposed(package):
    assert package.decision.post_event_fn is not None
    assert package.decision.post_event(adapter.DEFAULT_AS_OF)


def test_missing_event_view_is_honest():
    """没有事件视图时回调仍存在，但决策模块自己拒绝——界面据此显示空态。"""
    from src.decision import DecisionError

    views = [{"pipe_id": "P1", "attributes": {}, "quality_flags": []}]
    from src.decision import advise_post_event, load_advice_rules

    with pytest.raises(DecisionError, match="event_view"):
        advise_post_event(views, None, adapter.DEFAULT_AS_OF, load_advice_rules())


# ---- as_of 真正生效 ----

def test_as_of_is_honoured(package):
    """截止日越早，可见的历史事件越少；O01 触发数单调不减（§8.5）。"""
    counts = []
    for as_of in AS_OFS:
        rows = package.decision.post_event(as_of)
        assert {r["advice_mode"] for r in rows} == {"post_event_review"}
        assert {r["as_of"] for r in rows} == {as_of}
        counts.append(sum(1 for r in rows if r["rule_id"] == "O01"))
    assert counts == sorted(counts)
    # 全部事件都在 2024 年，故 2023 年底之前一条都不可见
    assert counts[0] == 0
    assert counts[-1] > 0


def test_late_events_are_invisible_before_the_cutoff(package):
    """截止日之后的事件不得计入：触发值须与事件视图逐管段对得上（§8.5）。"""
    as_of = "2024-06-30"
    expected = {r["pipe_id"]: sum(1 for d in r["events"] if d <= as_of)
                for r in package.event_view}
    rows = package.decision.post_event(as_of)
    o01 = {r["pipe_id"]: r["trigger_values"]["pre_as_of_event_count"]
           for r in rows if r["rule_id"] == "O01"}
    assert o01, "该截止日下应有一条以上历史事件复核"
    for pipe_id, count in o01.items():
        assert count == expected[pipe_id], pipe_id
        assert count >= 1
    # 触发集合恰为「截止日之前有事件」的管段，不多不少
    assert set(o01) == {pid for pid, n in expected.items() if n >= 1}


# ---- 与预测侧隔离 ----

def test_post_event_does_not_touch_the_prediction_side(package):
    """事后模式不改变 p / 等级 / C / V，也不改变预测清单（§8.5）。"""
    before = [(p.pipe_id, p.probability, p.relative_risk_level,
               p.consequence_proxy, p.priority_value) for p in package.pipes]
    package.decision.post_event(adapter.DEFAULT_AS_OF)
    after = [(p.pipe_id, p.probability, p.relative_risk_level,
              p.consequence_proxy, p.priority_value) for p in package.pipes]
    assert before == after


def test_page_has_no_rule_ids_of_its_own():
    """界面不得自带规则实现：规则 id 只出现在 configs/decision/ 与 src/decision/（§13.1）。"""
    src = PAGE.read_text(encoding="utf-8")
    for rule_id in ("O01", "R01", "R02", "R03", "D01", "D02", "R00"):
        assert rule_id not in src, f"views/post_event.py 出现了规则 id {rule_id}"


def test_page_does_not_read_raw_events():
    """界面不读原始爆管记录：只消费 event_view 产物（§13.3）。"""
    src = PAGE.read_text(encoding="utf-8")
    for reader in ("read_excel", "read_csv", "import pandas", "open("):
        assert reader not in src, f"views/post_event.py 出现了原始数据读取 {reader!r}"


# ---- 页面 ----

def test_page_defaults_to_off():
    """默认关闭：不开启时不调用事后入口，显示诚实空态。"""
    at = _open_page()
    assert not at.exception, [str(e.value) for e in at.exception]
    assert at.toggle[0].value is False
    assert any("默认关闭" in m.value for m in at.info)


def test_page_renders_when_enabled():
    at = _open_page()
    at.toggle[0].set_value(True).run()
    assert not at.exception, [str(e.value) for e in at.exception]
    assert at.text_input[0].value == adapter.DEFAULT_AS_OF
    assert not any("默认关闭" in m.value for m in at.info)


def test_page_uses_the_given_as_of():
    """改截止日会真正改变结果，界面不吞掉这个输入。"""
    at = _open_page()
    at.toggle[0].set_value(True).run()
    late = _rendered_total(at)
    at.text_input[0].set_value("2024-03-31").run()
    assert not at.exception, [str(e.value) for e in at.exception]
    early = _rendered_total(at)
    assert early < late
    # 页面显示的条数与决策模块自己算出的条数一致，界面不增删
    assert late == len(package_rows(adapter.DEFAULT_AS_OF))
    assert early == len(package_rows("2024-03-31"))


def test_page_refuses_an_illegal_as_of():
    """非法截止日由决策模块拒绝，界面如实显示原因，不自行换值（§8.5）。"""
    at = _open_page()
    at.toggle[0].set_value(True).run()
    at.text_input[0].set_value("not-a-date").run()
    assert not at.exception, [str(e.value) for e in at.exception]
    assert any("拒绝" in e.value for e in at.error)


def test_page_renders_on_fixture_fallback(monkeypatch):
    """夹具包没有事件视图：页面显示诚实空态，不崩、不填数。"""
    monkeypatch.setenv(adapter.ENV_PACKAGE_DIR, "/nonexistent/package/dir")
    at = _open_page()
    at.toggle[0].set_value(True).run()
    assert not at.exception, [str(e.value) for e in at.exception]
    assert any("没有事件视图" in m.value for m in at.info)


def test_page_has_no_absolute_paths_and_no_school_name():
    text = PAGE.read_text(encoding="utf-8")
    assert not anonymity.find_absolute_paths(text)
    tokens = anonymity.banned_names()
    if tokens:
        assert not anonymity.find_banned(text, tokens)