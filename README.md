# 运行说明

大口径供水管网安全风险智能评估与决策 —— 当前进度 M1（最小闭环打通）

赛题：JBGS-2026-11，湖州市水务集团有限公司。

---

## 1. 环境

```bash
python3 --version          # 3.13.2（实测）
python3 -m pip install pandas numpy scikit-learn pytest openpyxl streamlit plotly networkx
```

实测版本：pandas 2.3.3、numpy 2.3.5、scikit-learn 1.8.0、pytest 9.1.1、streamlit 1.63.0、plotly 7.0.0、networkx 3.6.1。

M0 路径只依赖前四项。决策与界面模块依赖 stdlib（决策）与 streamlit/plotly/networkx（界面）。
CatBoost / SHAP 不在 M1 路径上（留 M2）。

## 2. 跑测试

```bash
python3 -m pytest tests/             # 411 项，约 2 分钟
python3 -m pytest tests/core/        # 契约、审计、指标、模型
python3 -m pytest tests/decision/    # 分级、后果、Top-K、建议规则
python3 -m pytest tests/app/         # 界面适配器与页面
python3 -m pytest tests/integration/ # M0 出口判定 + M1 闭环
```

`tests/integration/test_m1_closure.py` 是 M1 门禁：真实属性 → 折外概率 →
逐折成绩 → 概率 Top-K → 页面，并断言页面与导出不重算业务逻辑。

### 匿名检查（§里程碑 4.6）

赛题禁止交付物出现校名。仓库为公开仓库，因此**禁用名称本身也不写进受版本控制的文件**：
词表放在被 `.gitignore` 排除的 `.anonymity_tokens`（每行一个），由 `tests/anonymity.py`
读取，`test_app_has_no_school_name` 等守卫用它扫描。

该文件不存在时（如全新克隆）具名词检查会 `skip`，绝对路径检查始终执行。
本地维护者应保留该文件，提交前跑一次 `python3 -m pytest tests/ -q` 确认匿名检查未 skip。

## 2.1 启动界面

```bash
python3 -m streamlit run app/main.py
```

详见 `app/README.md`。

## 3. 构建发布包

```bash
python3 -m src.integration.pipeline            # 输出到 outputs/release/
python3 -m src.integration.pipeline --out /tmp/m0
```

产物：`standard_attributes.json`、`geometry.json`、`predictions.json`、
`reference_bundle.json`、`manifest.json`、`evaluation.json`、`labels_and_splits.csv`。

`outputs/` 按保密要求管理，不纳入版本控制。

## 4. 目录与模块边界（§13.2）

| 目录 | 模块 | 说明 |
|---|---|---|
| `src/contracts/` | 核心 | schema、版本、错误表示 |
| `src/data/` | 核心 | 只读加载、白名单、折内预处理 |
| `src/audit/` | 核心 | 数据审计 |
| `src/models/` | 核心 | B0/B1/B2 基线、校准 |
| `src/evaluation/` | 核心 | 划分、指标、折外预测、参考分布包 |
| `src/integration/` | 核心 | 发布清单、M0 入口 |
| `src/decision/` | 决策 | 分级、后果代理、Top-K、建议 |
| `src/explain/report_gen/` | 决策 | 业务建议生成 |
| `app/` | 界面 | 界面 |
| `configs/contracts/`、`features/`、`models/`、`evaluation/` | 核心 | 公共配置 |
| `configs/decision/` | 决策 | 等级、后果、建议规则配置 |
| `tests/core/`、`tests/integration/` | 核心 | |
| `tests/decision/` | 决策 | |
| `tests/app/` | 界面 | |

## 5. 决策与界面的离线入口（§13.3）

**决策与界面模块不需要两个原始 XLSX，也不需要导入 `src/models/`。**

夹具目录：`tests/fixtures/`

| 文件 | 用途 |
|---|---|
| `standard_attributes.json` | 8 类边界管段属性 + 质量标记 |
| `geometry.json` | 边起终点坐标、分量、冲突标记、`crs_known=false` |
| `predictions.json` | 概率（含 `p=null` 的无效预测） |
| `explanation.json` | 解释（含 `status=unavailable`） |
| `decision.json` | 决策输出样例（供界面对样例开发） |
| `reference_bundle.json` | 参考分布样例 |
| `expected_rules.json` | 预期规则触发参考（不是正式结果） |

夹具统一标 `data_kind=synthetic_fixture`。**正式报告禁止引用样例指标**（§13.3）。

契约层 `src/contracts/` 不依赖 pandas/sklearn，可轻量导入：

```python
from src.contracts import validate_package, PRODUCTS, ContractError
```

## 6. 公共约定（§13.3）

- `pipe_id` 为规范字符串；`BH` 仅作展示，不各自重新生成 ID。
- JSON 不传 `NaN`/`Infinity`；不可用值用 `null` + `quality_flags`。
- 事后访问用显式 `as_of`，不隐式取系统当前日。
- 数据包含 `schema_version`、`data_version`、`run_id`、`prediction_mode`、`data_kind`。
- 版本/模式/唯一键不匹配时**拒绝接入并说明原因**，禁止按行号对齐或静默丢行。

## 7. 决策模块接口（§13.4，签名已冻结）

```python
grade(predictions, reference_bundle, grade_config)               -> grades
prioritize(pipe_views, predictions, grades, scenario,
           selection_request)                                     -> priority_result
advise_predictive(pipe_views, predictions, grades,
                  priority_result, rules)                         -> predictive_advice
advise_post_event(pipe_views, event_view, as_of, rules)           -> post_event_advice
```

这些函数不读 Excel、不调用训练、不写原始数据、不修改传入表。
界面只通过固定适配器调用，**不在页面复制 percentile / C / V / 建议规则**。

## 8. 已完成 / 未完成

**M0 已完成**：契约、夹具、审计、白名单、固定划分、年龄基线折外预测、四类聚合评测、参考分布包、发布清单、M0 入口。

**M1 已完成**：决策模块（分级 / 后果代理 / 数量预算 Top-K / 建议规则 T01–T10 / 事后隔离）、
界面四页（总览清单 / 管段详情 / 资源清单 / 事后运维复核，含空错误态与导出）、M1 闭环集成测试。
展示包补充 `OPSST`/`PRESS`（决策 R02 依赖，属 F2 展示字段，不进训练白名单）。

**未完成（留 M2/M3）**：CatBoost 候选、F2/F3 消融、SHAP 归因、空间/道路敏感性协议、
界面的事后运维页与审计页、可选成本背包、全量拟合模型。

## 9. 已修正的文档与实现问题

| 位置 | 问题 | 处理 |
|---|---|---|
| 方案 §6.4.1 | 反例折 B 第二项写 `(1,0.8)`，会使折 B 的 AUC 无定义且拼接 AUC=1.0，与原文自述的 0.75/5-6 矛盾 | 按唯一自洽解释 `(0,0.8)` 实现，测试注明 |
| 方案 §3.4 | 泄漏测试要求"固定已保存的划分"，原实现每次重算折 | `run_oof` 增加 `folds` 参数，测试传入固定划分 |
| 方案 §8.4 | 规则表无 O01 行，但执行顺序含 O01、T07/T08 要求 O01 | 按 §8.5 散文的触发条件实现 O01 |
| 实现 | 参考分布包 `data_version` 误用 `round_id`（`seed20260914`），与其余产物版本不一致，违反 §13.3 | 改为传入真实 `data_version`；加回归测试 |
| 实现 | D01 把**全表**标记 `semantics_unverified` 当作逐管段冲突，真实数据上命中 100% 管段 | 只收逐管段冲突；D01 命中降为 29 条（坐标冲突数），加回归测试 |