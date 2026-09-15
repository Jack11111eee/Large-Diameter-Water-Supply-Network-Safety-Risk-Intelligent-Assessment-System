"""契约层：schema、版本、错误表示。

依据 项目方案与技术路线.md §13.3 与 §13.4。
所有产物的生产者/消费者见 §13.4 契约表。
"""

from .versions import (
    SCHEMA_VERSION,
    GRADE_CONFIG_VERSION,
    SCENARIO_CONFIG_VERSION,
    ADVICE_RULES_VERSION,
    MODEL_CONFIG_VERSION,
    EVALUATION_CONFIG_VERSION,
    DataKind,
    PredictionMode,
    Scale,
    ExplainStatus,
)
from .schema import (
    PRODUCTS,
    ContractError,
    validate_package,
    assert_no_nonfinite,
    reject,
)

__all__ = [
    "SCHEMA_VERSION",
    "GRADE_CONFIG_VERSION",
    "SCENARIO_CONFIG_VERSION",
    "ADVICE_RULES_VERSION",
    "MODEL_CONFIG_VERSION",
    "EVALUATION_CONFIG_VERSION",
    "DataKind",
    "PredictionMode",
    "Scale",
    "ExplainStatus",
    "PRODUCTS",
    "ContractError",
    "validate_package",
    "assert_no_nonfinite",
    "reject",
]