"""候选模型注册表（§5.1、§5.2）。

候选与参数范围在 configs/models/candidates.json 预注册。
注册表只把声明转成工厂，不做任何选择——选择在 evaluation.selection。
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
CANDIDATES_FILE = ROOT / "configs" / "models" / "candidates.json"


def load_candidates(path=None):
    """读取预注册候选配置。"""
    p = Path(path) if path else CANDIDATES_FILE
    return json.loads(p.read_text(encoding="utf-8"))


def factory_of(candidate, layer_columns, *, seed=None):
    """把一条候选声明转成 `params -> 模型实例` 的工厂。"""
    family = candidate["family"]
    if family == "linear":
        from src.models.baselines import LayerLogReg

        return lambda params: LayerLogReg(layer_columns, **params)
    if family == "tree":
        from src.models.tree import HGBTClassifier

        return lambda params: HGBTClassifier(
            layer_columns, random_state=seed, **params)
    raise KeyError(f"未知候选族 {family!r}")