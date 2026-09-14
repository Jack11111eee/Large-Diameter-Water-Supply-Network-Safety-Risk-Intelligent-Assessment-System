"""匿名检查：禁用名称与绝对路径（§里程碑 4.6）。

赛题要求任何交付物不得出现校名；仓库为公开仓库，因此**禁用名称本身**
也不能以字面量写进受版本控制的文件。禁用词表放在被 `.gitignore` 排除的
`.anonymity_tokens`（每行一个），本地与提交前检查读取它。

文件缺失时（如全新克隆）跳过具名词检查，但绝对路径检查始终执行。
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOKENS_FILE = ROOT / ".anonymity_tokens"

# 绝对路径标记：与具体机构无关，可直接内联。
# 覆盖字符串字面量（带引号）与裸路径两种写法；C:\ 单独列出以捕获 Windows 路径。
ABSOLUTE_PATH_TOKENS = ('"/Users/', "'/Users/", '"/home/', "'/home/",
                        '"/tmp/', "/Users/", "/home/", "C:\\")


def banned_names():
    """从本地禁用词表读取校名等禁用名称。缺失则返回空列表。"""
    if not TOKENS_FILE.exists():
        return []
    return [line.strip() for line in
            TOKENS_FILE.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")]


def find_banned(text, tokens):
    """返回 text 中命中的禁用词。"""
    return [t for t in tokens if t in text]


def find_absolute_paths(text):
    return [t for t in ABSOLUTE_PATH_TOKENS if t in text]