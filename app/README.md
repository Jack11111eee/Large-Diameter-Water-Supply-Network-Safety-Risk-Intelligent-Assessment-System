# app/ — Streamlit 界面（M1 范围）

大口径供水管网安全风险智能评估与决策 —— 界面模块（§9、§13.2）。

## 启动

```bash
python3 -m streamlit run app/main.py
```

浏览器打开终端提示的本地地址（默认 http://localhost:8501）。默认本地离线运行，
不联网、不叠加真实底图。

指定数据包目录（可选）：

```bash
APP_PACKAGE_DIR=/path/to/package python3 -m streamlit run app/main.py
```

## 数据来源与回退

| 情况 | 行为 |
|---|---|
| `outputs/release/` 有完整产物 | 装载正式产物，徽章显示 `real_standard` |
| `outputs/release/` **不存在** | 回退到 `tests/fixtures/`，徽章显示 `synthetic_fixture` 并提示"不是正式结果" |
| `outputs/release/` 存在但**违反契约** | **拒绝接入并显示原因**，不回退（避免真实问题被夹具掩盖） |

回退只在"产物不存在"时发生。版本不匹配、重复 `pipe_id`、缺失字段、表间 ID 集合不一致
一律拒绝接入并说明原因（§13.3）。

## 页面

1. **风险总览与清单** — 概率、相对等级、固定百分位、本地坐标示意（源坐标直线 / 纯拓扑布局切换）、
   分量与坐标冲突标记、风险清单。
2. **管段详情** — 按 `pipe_id` / `BH` 检索；源字段与质量标记、模型口径、解释（或 `unavailable`）、
   后果代理、优先值、规则触发依据与建议动作。
3. **资源清单** — 预算 K、目标切换（p / C / V）、独立按 p 视图（标注未进入当前目标清单的管段）、
   L3/L4 视图、参数版本、导出。
4. **事后运维复核** — 默认**关闭**；开启后以显式 `as_of`（不取系统当前日）调用决策模块的
   `advise_post_event()`。事后建议是独立清单，不改变 p、等级、C、V 或预测清单（§8.5）。
5. **数据审计** — 只读发布清单：数据文件指纹、配置版本、划分与分组诊断、产物行数与指纹、
   质量标记分布。不碰 pandas，不读原始 XLSX（§13.6）。
6. **实验审计** — 发布成绩按聚合口径分块（逐折为主；`pooled_oof_replay` 单独成块并标注
   **不替代逐折主成绩**）；另有可选的实验包（候选选择日志、校准对照、结构消融）。
   实验包与发布包**不同源时拒绝并列展示**（§6.4.1、§13.3）。

页级徽章：预测模式（`oof_replay` / `full_fit` / `scenario`）与数据来源
（`real_standard` / `synthetic_fixture`）。

## 模块边界（§13.1）

**`app/` 不含任何业务计算。** 百分位、后果代理 `C`、优先值 `V`、建议规则只在
`src/decision/` 唯一实现；界面只展示其结果。`tests/app/test_app_boundaries.py`
用 AST 扫描守住这条边界（禁止定义计算函数、禁止对业务量做算术、禁止导入
`src.models` / `src.data`、禁止自带规则表）。

| 文件 | 职责 |
|---|---|
| `adapter.py` | 唯一的装载、契约校验与按 `pipe_id` 连接入口；**不导入 Streamlit**，可独立测试 |
| `theme.py` | 颜色与 Plotly 版式令牌 |
| `sketch.py` | 本地坐标 / 拓扑示意图，只做呈现 |
| `components.py` | 共用展示组件（徽章、空 / 错误状态、属性表、解释面板） |
| `views/` | 页面模块，每个导出一个 `render(package, mode)`；审计两页只展示已给出的读数 |
| `main.py` | 入口：装载 → 徽章 → 分发到页面 |

> 页面模块放在 `views/` 而非 `pages/`：Streamlit 会把 `app/pages/` 自动识别为
> 多页导航，与 `main.py` 的侧边栏分发冲突。

## 决策模块未接入时的行为

界面**不写第二套实现**。适配器按以下顺序取决策结果：

1. `src/decision/` 可导入且 `grade()` 返回可识别结果 → 直接使用（`decision_module`）；
2. 否则展示本次装载目录的 `decision.json`，缺失时回退到冻结的
   `tests/fixtures/decision.json`（`frozen_package`），并在页面明确标注；
3. 两者都没有 → 相关字段显示 `unavailable`（`absent`）。

`grade()` 抛错或返回形状不可识别时同样退回冻结包，绝不本地重算。

## 坐标纪律（§4）

源坐标单位未知、CRS 未知。示意图只画直线，**不叠加真实底图，不输出米制半径、
真实管长、供水范围或最近阀门距离**；页面固定标注"源坐标，CRS 未知，非真实经纬度"。
坐标冲突边保留原始端点、以菱形标记，不跨节点自动吸附或合并。

## 测试

```bash
python3 -m pytest tests/app/ -q
```

不需要两个原始 XLSX，也不需要 `src/models/`。用 Streamlit 自带的 `AppTest` 驱动页面，
无需浏览器。

离线运行的最小目录集（§13.6，已实测）：

```text
app/  src/（无 src/models/、无 src/data/）  tests/  configs/  pytest.ini
```

`configs/decision/` 归决策模块所有；界面经模块自带的 `load_grade_config()` /
`load_scenario_config()` 读取，模块未提供加载器时才回退读仓库内的同名文件。