"""决策配置加载（`configs/decision/`）。

只用相对包路径定位，不写绝对路径；只读 JSON，不读 Excel。
"""

import json
from pathlib import Path

from .errors import DecisionError

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "decision"


def _load(name):
    path = CONFIG_DIR / name
    if not path.is_file():
        raise DecisionError(f"缺少决策配置文件 {name}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DecisionError(f"决策配置文件 {name} 不是合法 JSON：{exc}") from exc


def load_grade_config():
    """等级配置（`relative_grade_v1`）。"""
    return _load("grade_config.json")


def load_scenario_config():
    """后果代理情景配置（`consequence_scenario_v1`）。"""
    return _load("scenario_config.json")


def load_advice_rules():
    """建议规则配置（`advice_rules_v1`）。"""
    return _load("advice_rules.json")