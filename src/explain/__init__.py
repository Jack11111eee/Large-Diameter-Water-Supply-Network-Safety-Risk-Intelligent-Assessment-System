"""模型解释（§7.1）。

树模型用 SHAP，线性模型用编码后特征的 SHAP；尺度固定为原始输出（log-odds）。
校准概率与原模型 SHAP 分别展示，不得混用，也不得把归一化贡献称为事故原因占比。
"""

from src.explain.artifacts import (
    ATTRIBUTION_NAME,
    DEFAULT_TOP_K,
    NOT_FINAL_PROBABILITY_FLAG,
    TRUNCATED_FLAG,
    build_explanation_rows,
    verify_rows_additivity,
)
from src.explain.attribution import (
    background_sample,
    check_additivity,
    explain_model,
    explainable,
    raw_scores,
)

__all__ = [
    "ATTRIBUTION_NAME",
    "DEFAULT_TOP_K",
    "NOT_FINAL_PROBABILITY_FLAG",
    "TRUNCATED_FLAG",
    "background_sample",
    "build_explanation_rows",
    "check_additivity",
    "explain_model",
    "explainable",
    "raw_scores",
    "verify_rows_additivity",
]