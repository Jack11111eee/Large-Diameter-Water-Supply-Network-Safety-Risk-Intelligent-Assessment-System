# M0 运行说明

大口径供水管网安全风险智能评估与决策 —— M0 里程碑（契约冻结 + 基线链路）

依据：`里程碑与分工实施.md` §2.2、§13.6。

---

## 1. 环境

```bash
python3 --version          # 3.13.2（实测）
python3 -m pip install pandas numpy scikit-learn pytest openpyxl
```

实测版本：pandas 2.3.3、numpy 2.3.5、scikit-learn 1.8.0、pytest 9.1.1。

M0 只依赖上述四项。CatBoost / Streamlit / SHAP / NetworkX 不在 M0 路径上（留 M2）。

## 2. 跑测试

```bash
python3 -m pytest tests/            # 115 项，约 30 秒
python3 -m pytest tests/core/       # 契约、审计、指标、模型
python3 -m pytest tests/integration/ # M0 出口判定
```

## 3. 构建发布包

```bash
python3 -m src.integration.pipeline            # 输出到 outputs/release/
python3 -m src.integration.pipeline --out /tmp/m0
```

产物：`standard_attributes.json`、`geometry.json`、`predictions.json`、
`reference_bundle.json`、`manifest.json`、`evaluation.json`、`labels_and_splits.csv`。

`outputs/` 按保密要求管理，不纳入版本控制。

## 4. 目录与所有权（§13.2）

| 目录 | 所有者 | 说明 |
|---|---|---|
| `src/contracts/` | A | schema、版本、错误表示 |
| `src/data/` | A | 只读加载、白名单、折内预处理 |
| `src/audit/` | A | 数据审计 |
| `src/models/` | A | B0/B1/B2 基线、校准 |
| `src/evaluation/` | A | 划分、指标、折外预测、参考分布包 |
| `src/integration/` | A | 发布清单、M0 入口 |
| `src/decision/` | **B** | 分级、后果代理、Top-K、建议（M0 期间由 B 建） |
| `src/explain/report_gen/` | **B** | 业务建议生成 |
| `app/` | **C** | 界面 |
| `configs/contracts/`、`features/`、`models/`、`evaluation/` | A | 公共配置 |
| `configs/decision/` | **B** | 等级、后果、建议规则配置 |
| `tests/core/`、`tests/integration/` | A | |
| `tests/decision/` | **B** | |
| `tests/app/` | **C** | |

## 5. B 与 C 的离线入口（§13.3）

**B、C 不需要两个原始 XLSX，也不需要导入 `src/models/`。**

夹具目录：`tests/fixtures/`

| 文件 | 用途 |
|---|---|
| `standard_attributes.json` | 8 类边界管段属性 + 质量标记 |
| `geometry.json` | 边起终点坐标、分量、冲突标记、`crs_known=false` |
| `predictions.json` | 概率（含 `p=null` 的无效预测） |
| `explanation.json` | 解释（含 `status=unavailable`） |
| `decision.json` | 决策输出样例（供 C 对样例开发） |
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

## 7. B 的接口（§13.4，签名已冻结）

```python
grade(predictions, reference_bundle, grade_config)               -> grades
prioritize(pipe_views, predictions, grades, scenario,
           selection_request)                                     -> priority_result
advise_predictive(pipe_views, predictions, grades,
                  priority_result, rules)                         -> predictive_advice
advise_post_event(pipe_views, event_view, as_of, rules)           -> post_event_advice
```

这些函数不读 Excel、不调用训练、不写原始数据、不修改传入表。
C 只通过固定适配器调用，**不在页面复制 percentile / C / V / 建议规则**。

## 8. M0 已完成 / 未完成

**已完成**：契约、夹具、审计、白名单、固定划分、年龄基线折外预测、四类聚合评测、参考分布包、发布清单、M0 入口。

**未完成（留 M2/M3）**：CatBoost 候选、F2/F3 消融、SHAP 归因、空间/道路敏感性协议、界面、决策模块实现。

## 9. 已修正的文档问题

| 位置 | 问题 | 处理 |
|---|---|---|
| 方案 §6.4.1 | 反例折 B 第二项写 `(1,0.8)`，会使折 B 的 AUC 无定义且拼接 AUC=1.0，与原文自述的 0.75/5-6 矛盾 | 按唯一自洽解释 `(0,0.8)` 实现，测试注明 |
| 方案 §3.4 | 泄漏测试要求"固定已保存的划分"，原实现每次重算折 | `run_oof` 增加 `folds` 参数，测试传入固定划分 |