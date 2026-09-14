"""决策模块（Track D，M1）。

VERSION: 0.1.0（模块版本）
配置版本：relative_grade_v1 / consequence_scenario_v1 / advice_rules_v1

职责（§13.2）：相对等级、后果代理、概率及综合 Top-K、建议规则、事后建议隔离。
**唯一实现**百分位分级、C、V 与建议规则（§13.1）；界面只展示，不重算。

输入输出
--------
四个冻结函数（§13.4）：

    grade(predictions, reference_bundle, grade_config) -> grades
    prioritize(pipe_views, predictions, grades, scenario, selection_request)
        -> priority_result
    advise_predictive(pipe_views, predictions, grades, priority_result, rules)
        -> predictive_advice
    advise_post_event(pipe_views, event_view, as_of, rules) -> post_event_advice

输入为契约形状的普通 dict/list（`src/contracts` 的产物 schema），字段拼写
`pipeage`/`material`/`joint`/`diameter_mm`/`road_type`/`facility`/`opsst`/`press_mpa`
与正式列码 `PIPEAGE`/`CZ`/`JOINTT`/`GJ`/`RDTYPE`/`FAC`/`OPSST`/`PRESS` 均可（见 `aliases`）。

约束
----
不导入 pandas/sklearn，不读 Excel，不训练，不修改传入参数；纯 Python + 标准库。
JSON 不传 NaN/Infinity，不可用值用 `null` + `quality_flags`（§13.3）。

可运行样例
----------
    from src.decision import (grade, prioritize, advise_predictive,
                              load_grade_config, load_scenario_config,
                              load_advice_rules)

    grades = grade(predictions, reference_bundle, load_grade_config())
    priority = prioritize(views, predictions, grades, load_scenario_config(),
                          {"objective": "V", "k": 100})
    advice = advise_predictive(views, predictions, grades, priority,
                               load_advice_rules())

失败处理
--------
版本/模式/唯一键不匹配、必需字段缺失、数值非有限或越界、事后模式未给合法
`as_of` → 抛 `DecisionError` 拒绝接入并说明原因，禁止按行号对齐或静默丢行（§13.3）。
"""

from .config import (
    load_advice_rules,
    load_grade_config,
    load_scenario_config,
)
from .consequence import compute_consequence, compute_consequences
from .errors import DecisionError
from .grading import assign_level, grade, percentile_of
from .priority import prioritize
from .advice import advise_post_event, advise_predictive

VERSION = "0.1.0"

__all__ = [
    "VERSION",
    "grade",
    "prioritize",
    "advise_predictive",
    "advise_post_event",
    "load_grade_config",
    "load_scenario_config",
    "load_advice_rules",
    "compute_consequence",
    "compute_consequences",
    "percentile_of",
    "assign_level",
    "DecisionError",
]