# JFK airline operational risk project

## Overview | 项目概述
This repository contains a reproducible operational-risk workflow for JFK arrival delays, cancellations, and diversions using BTS delay-cause data from 2020 to 2025.

本仓库保存了一个可复现的 JFK 到达航班运营风险分析流程，基于 2020 到 2025 年的 BTS delay-cause 数据，对延误、取消、改降、状态依赖、年度聚合风险、尾部风险和敏感度分析进行建模。

The project is organized around one quantitative chain:
- risk identification from BTS delay causes
- frequency modeling with `Poisson` and `Negative Binomial`
- severity modeling with `Lognormal` and `Weibull`
- disrupted-state definition and two-state Markov dependence
- annual aggregate-risk simulation in delay-equivalent minutes
- EVT tail extension
- standalone sensitivity analysis

项目围绕一条完整的定量分析主线展开：
- 基于 BTS 延误原因的风险识别
- `Poisson` 与 `Negative Binomial` 的频率建模比较
- `Lognormal` 与 `Weibull` 的严重度建模比较
- 扰动状态划分与两状态 Markov 依赖
- 以 `delay-equivalent minutes` 为口径的年度聚合风险模拟
- EVT 尾部扩展
- 独立敏感度分析

## Research framing | 研究定位
The project does not attempt to predict individual flight delays. Instead, it builds a course-aligned operational-risk framework at the airline-month and airport-month levels.

项目并不试图预测单个航班是否延误，而是以 airline-month 和 airport-month 为主要粒度，构建一套贴合课程方法的 operational-risk 框架。

Two data layers are used:
- `airline-month`: frequency and severity modeling
- `airport-month`: state classification and Markov dependence

项目使用两层数据结构：
- `airline-month`：用于频率和严重度建模
- `airport-month`：用于状态划分和 Markov 依赖分析

The operational loss proxy is defined in minutes:

\[
\text{Delay-equivalent minutes}
=
\text{arrival delay minutes}
+ 180 \times \text{cancelled arrivals}
+ 240 \times \text{diverted arrivals}
\]

项目采用分钟口径的运营损失代理：

\[
\text{Delay-equivalent minutes}
=
\text{arrival delay minutes}
+ 180 \times \text{cancelled arrivals}
+ 240 \times \text{diverted arrivals}
\]

## Repository structure | 仓库结构
```text
.
|-- data
|   |-- raw
|   `-- processed
|-- reports
|   |-- charts
|   |-- latex
|   |   |-- build
|   |   |-- sections
|   |   `-- jfk_operational_risk_report.tex
|   |-- modeling_chapter
|   |   |-- en
|   |   `-- zh
|   `-- summary
|-- scripts
|   |-- build_operational_risk_assets.py
|   `-- build_multiyear_operational_risk_project.py
|-- .gitignore
|-- environment.yml
`-- README.md
```

## Main scripts | 主要脚本
- `scripts/build_operational_risk_assets.py`
  Builds the cleaned descriptive datasets, summary tables, and descriptive charts.
- `scripts/build_multiyear_operational_risk_project.py`
  Builds the multi-year modeling inputs, model-comparison tables, state panel, transition matrix, annual aggregate-risk outputs, EVT extension, sensitivity analysis, and the final quantitative figures.

- `scripts/build_operational_risk_assets.py`
  生成描述性清洗数据、汇总表和基础图表。
- `scripts/build_multiyear_operational_risk_project.py`
  生成多年份建模输入、模型比较表、状态面板、转移矩阵、年度聚合风险结果、EVT 扩展、敏感度分析以及主要定量图表。

## Key outputs | 关键输出

### Processed datasets | 处理后数据
- `data/processed/jfk_airline_month_modeling_input.csv`
- `data/processed/jfk_airport_month_state_panel.csv`
- `data/processed/jfk_frequency_model_summary.csv`
- `data/processed/jfk_severity_model_summary.csv`
- `data/processed/jfk_state_definition_comparison.csv`
- `data/processed/jfk_markov_transition_matrix.csv`
- `data/processed/jfk_aggregate_risk_scenario_metrics.csv`
- `data/processed/jfk_evt_tail_fit_summary.csv`
- `data/processed/jfk_evt_tail_threshold_sensitivity.csv`
- `data/processed/jfk_sensitivity_analysis_metrics.csv`

### Figures | 图表
Core figures are stored in `reports/charts/`, including:
- monthly trend and cause breakdown
- risk heat map
- disrupted-state diagnostics
- Markov transition matrix
- frequency and severity AIC comparison
- aggregate annual risk distributions
- EVT tail-fit and threshold sensitivity
- sensitivity tornado, heatmap, and profile plots

核心图表统一保存在 `reports/charts/`，包括：
- 月度趋势与原因分解
- 风险热力图
- 扰动状态诊断图
- Markov 转移矩阵
- 频率与严重度的 AIC 对比图
- 年度 aggregate risk 分布图
- EVT 尾部拟合与阈值敏感度图
- 敏感度龙卷图、热力图和 profile 图

### Reports | 报告
- Main report entry:
  `reports/latex/jfk_operational_risk_report.tex`
- Main compiled report:
  `reports/latex/build/jfk_operational_risk_report.pdf`
- Supplementary bilingual modeling chapter:
  `reports/modeling_chapter/en/quantitative_modeling_report_en.pdf`
  `reports/modeling_chapter/zh/quantitative_modeling_report_zh.pdf`

- 正式主报告入口：
  `reports/latex/jfk_operational_risk_report.tex`
- 正式主报告 PDF：
  `reports/latex/build/jfk_operational_risk_report.pdf`
- 补充的中英双语建模章节：
  `reports/modeling_chapter/en/quantitative_modeling_report_en.pdf`
  `reports/modeling_chapter/zh/quantitative_modeling_report_zh.pdf`

## Modeling content | 建模内容
The core modeling layer covers:
- candidate-model comparison using AIC
- count modeling for delays, cancellations, diversions, and reconstructed disruption blocks
- minutes-based severity modeling
- composite disrupted-state definition
- two-state Markov transition analysis
- annual scenario-based aggregate simulation
- EVT tail interpretation
- single-factor and multi-factor sensitivity analysis

核心建模层包括：
- 使用 AIC 的候选模型比较
- 对延误、取消、改降以及重构风险分块的计数建模
- 基于分钟口径的严重度建模
- 综合 disrupted-state 划分规则
- 两状态 Markov 转移分析
- 基于年度情景的 aggregate simulation
- EVT 尾部解释
- 单因子和多因子敏感度分析

## Reproducibility | 复现方式

### 1. Create the environment | 创建环境
```bash
conda env create -f environment.yml
conda activate operational_risk
```

### 2. Rebuild the descriptive assets | 重建描述性输出
```bash
python scripts/build_operational_risk_assets.py
```

### 3. Rebuild the multi-year modeling outputs | 重建多年份建模输出
```bash
python scripts/build_multiyear_operational_risk_project.py
```

### 4. Compile the main report | 编译主报告
Use:
```text
reports/latex/jfk_operational_risk_report.tex
```

使用：
```text
reports/latex/jfk_operational_risk_report.tex
```

## Notes | 说明
- Cache folders, Python bytecode, and LaTeX temporary files are not part of the deliverable and are excluded from the cleaned repository.
- The repository keeps generated processed datasets and figures because they are part of the reproducible project output.
- The bilingual modeling chapter is retained as a supplementary write-up focused on the mathematical modeling component.

- 缓存目录、Python 字节码和 LaTeX 临时编译文件不属于最终交付内容，已从整理后的仓库中排除。
- 仓库保留生成后的处理数据和图表，因为它们属于项目可复现输出的一部分。
- 中英双语建模章节作为补充材料保留，重点服务于项目中的数学建模部分。
