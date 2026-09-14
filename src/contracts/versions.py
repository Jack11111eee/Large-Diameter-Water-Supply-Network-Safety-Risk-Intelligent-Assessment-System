"""版本常量与枚举。

版本不兼容时必须拒绝接入（§13.3）。历史结果保留旧版本，不覆盖（§5.3.1）。
"""

from enum import Enum

# 契约 schema 版本。字段结构变更时递增。
SCHEMA_VERSION = "1.0.0"

# 相对风险等级配置版本（§5.3.1）。
GRADE_CONFIG_VERSION = "relative_grade_v1"

# 后果代理情景版本（§8.1）。
SCENARIO_CONFIG_VERSION = "consequence_scenario_v1"

# 建议规则版本（§8.4）。
ADVICE_RULES_VERSION = "advice_rules_v1"


class DataKind(str, Enum):
    """产物来源。样例与正式产物必须分开（§13.3）。"""

    REAL_STANDARD = "real_standard"
    SYNTHETIC_FIXTURE = "synthetic_fixture"


class PredictionMode(str, Enum):
    """三种产物模式必须分开（§2.3）。"""

    OOF_REPLAY = "oof_replay"        # 折外回放，可用于评测
    FULL_FIT = "full_fit"            # 全量拟合，不能用原训练标签证明泛化
    SCENARIO = "scenario"            # 假设情景，不能宣称真实收益


class Scale(str, Enum):
    """解释尺度。树模型 SHAP 解释 log-odds 时不得写成概率百分点（§7.1）。"""

    LOG_ODDS = "log_odds"
    PROBABILITY = "probability"


class ExplainStatus(str, Enum):
    """解释可用性。未就绪时必须显式 unavailable，不得顶替（§13.5）。"""

    OK = "ok"
    UNAVAILABLE = "unavailable"