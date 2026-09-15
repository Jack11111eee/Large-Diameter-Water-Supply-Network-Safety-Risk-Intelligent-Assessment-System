"""页面 5：数据审计（§9、§13.3、§13.6）。

只展示发布清单与包级元数据：数据文件指纹、配置版本、划分与分组、产物
行数与指纹、质量标记统计。**不碰 pandas，不读原始 XLSX**，也不重算任何
业务量——清单本身就是审计数据源。
"""

from collections import Counter

import streamlit as st

from .. import components, theme

UNKNOWN = components.UNKNOWN


def render(package, mode):
    st.subheader("数据审计")
    st.caption("展示本次发布所用的数据文件、配置版本、划分与分组、产物指纹与质量标记；"
               "只读发布清单，不读原始表格、不重算任何业务量（§13.6）。")

    report = package.report
    _load_report(report)
    _data_files(package.manifest)
    _configs(package.manifest)
    _split(package.manifest)
    _artifacts(package.manifest)
    _quality_flags(package)


def _load_report(report):
    st.markdown("#### 装载报告")
    # 值一律转字符串：同一列混合类型会让 Arrow 序列化回退并告警
    rows = [
        {"项": "装载目录", "值": str(report.directory)},
        {"项": "来源", "值": components.SOURCE_LABELS.get(report.source, report.source)},
        {"项": "预测模式", "值": str(report.prediction_mode)},
        {"项": "schema_version", "值": str(report.schema_version)},
        {"项": "data_version", "值": str(report.data_version)},
        {"项": "run_id", "值": str(report.run_id)},
        {"项": "model_id", "值": str(report.model_id or UNKNOWN)},
        {"项": "缺失产物", "值": "、".join(report.missing_products) or "无"},
    ]
    st.dataframe(rows, width="stretch", hide_index=True)
    if report.fallback_reason:
        st.info(f"已回退到夹具目录：{report.fallback_reason}", icon="ℹ️")
    for note in report.warnings:
        st.warning(note, icon="⚠️")


def _data_files(manifest):
    st.markdown("#### 数据文件指纹")
    files = (manifest or {}).get("data_files")
    if not files:
        components.empty_state("发布清单没有 data_files 段。",
                               detail="清单由 pipeline 生成；缺失时界面不自行计算文件摘要。",
                               icon="📭")
        return
    st.dataframe([{"文件": name, "SHA256": str(digest)}
                  for name, digest in sorted(files.items())],
                 width="stretch", hide_index=True)
    st.caption("数据版本由管段 ID 集合与这些文件指纹共同确定（§13.3）。")


def _configs(manifest):
    st.markdown("#### 配置版本")
    configs = (manifest or {}).get("configs")
    if not configs:
        components.empty_state("发布清单没有 configs 段。", icon="📭")
        return
    st.dataframe([{"配置": name, "版本": str(version)}
                  for name, version in sorted(configs.items())],
                 width="stretch", hide_index=True)


def _split(manifest):
    st.markdown("#### 划分与分组")
    meta = (manifest or {}).get("split")
    if not meta:
        components.empty_state("发布清单没有 split 段。", icon="📭")
        return

    st.dataframe([
        {"项": "划分方案 scheme", "值": str(meta.get("scheme"))},
        {"项": "预注册种子 seed", "值": str(meta.get("seed"))},
        {"项": "外层折数 n_outer", "值": str(meta.get("n_outer"))},
        {"项": "内层折数 n_inner", "值": str(meta.get("n_inner"))},
        {"项": "划分表指纹", "值": str(meta.get("table_fingerprint"))},
    ], width="stretch", hide_index=True)

    folds = meta.get("folds") or {}
    if folds:
        st.caption("逐折规模与正例数（§6.4.1 展示项）：")
        rows = []
        for fold, rec in sorted(folds.items(), key=lambda kv: int(kv[0])):
            row = {"折": fold, "测试管段": rec.get("n_test"),
                   "正例": rec.get("n_positive"),
                   "基率": components.fmt(rec.get("base_rate"), 4)}
            if "n_groups" in rec:
                row["分组数"] = rec["n_groups"]
                row["缺分组行"] = rec["n_missing_group_rows"]
            rows.append(row)
        st.dataframe(rows, width="stretch", hide_index=True)

    grouping = meta.get("grouping")
    if grouping:
        st.markdown("##### 分组诊断")
        st.dataframe([
            {"项": "分组方案", "值": str(grouping.get("scheme"))},
            {"项": "分组列", "值": str(grouping.get("column"))},
            {"项": "分组数", "值": str(grouping.get("n_groups"))},
            {"项": "缺失分组值的管段数", "值": str(grouping.get("n_missing_rows"))},
        ], width="stretch", hide_index=True)
        st.caption("缺失分组值作单例组：既不与其他未分组管段合并，也不丢弃（§6.2）。"
                   "分组列只作分组，永不进训练矩阵（§3.2）。")

    diagnostics = meta.get("grouped_diagnostics")
    if diagnostics:
        st.info(diagnostics.get("disclosure", ""), icon="ℹ️")


def _artifacts(manifest):
    st.markdown("#### 产物行数与指纹")
    artifacts = (manifest or {}).get("artifacts")
    if not artifacts:
        components.empty_state("发布清单没有 artifacts 段。", icon="📭")
        return
    st.dataframe([
        {"产物": name,
         "行数": entry.get("n_rows", UNKNOWN),
         "指纹": (entry.get("fingerprint") or "")[:16] + "…"}
        for name, entry in sorted(artifacts.items())
    ], width="stretch", hide_index=True)
    st.caption("指纹为发布时对整包内容计算的摘要；跨运行应逐字节一致。")


def _quality_flags(package):
    st.markdown("#### 质量标记分布")
    if not package.pipes:
        components.empty_state("清单为空，无法统计质量标记。", icon="📭")
        return
    counter = Counter()
    for pipe in package.pipes:
        if not pipe.quality_flags:
            counter["（无标记）"] += 1
            continue
        for flag in pipe.quality_flags:
            counter[flag] += 1
    st.dataframe([
        {"质量标记": theme.QUALITY_FLAG_LABELS.get(flag, flag),
         "标记代码": flag, "管段数": n}
        for flag, n in counter.most_common()
    ], width="stretch", hide_index=True)
    st.caption(f"统计口径：按管段计数，共 {len(package.pipes):,} 条；"
               "一根管段可带多个标记，故合计可大于管段数。"
               "标记由发布包给出，界面不重算。")