"""评测协议配置（§6.1、§6.2）。

`configs/evaluation/protocol.json` 是评测协议的单一可读描述，供发布清单与
实验审计页展示。数值必须与 `src/evaluation/split.py` 的冻结常量一致——
由测试钉住，不靠人工同步。
"""

import json
from pathlib import Path

from src.models import registry

ROOT = Path(__file__).resolve().parent.parent.parent
PROTOCOL_FILE = ROOT / "configs" / "evaluation" / "protocol.json"


def load_protocol(path=None):
    """读取评测协议配置。"""
    p = Path(path) if path else PROTOCOL_FILE
    return json.loads(p.read_text(encoding="utf-8"))


def config_versions():
    """发布清单的 `configs` 段：本次运行所用配置的版本（§13.6）。

    只列 A 侧产物实际依赖的配置；决策侧配置由 B 在其产物中自述。
    """
    from src.data import loader

    return {
        "evaluation_protocol": load_protocol()["version"],
        "candidates": registry.load_candidates()["candidates_version"],
        "whitelist": loader.load_whitelist()["whitelist_version"],
    }