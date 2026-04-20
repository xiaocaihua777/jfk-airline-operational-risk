# JFK Airline Delay Operational Risk Project

## Overview | 项目概述
This repository contains a reproducible operational-risk workflow for JFK airline delays, cancellations, and diversions.

本仓库保存了一个可复现的 JFK 航班延误运营风险分析流程，覆盖延误、取消、备降与状态依赖分析。

The project is organized around one full analytical chain:
- risk identification from BTS delay causes
- frequency modeling with `Poisson` vs `Negative Binomial`
- severity modeling with `Lognormal` vs `Weibull`
- state definition and two-state Markov dependence
- aggregate annual risk under operating scenarios

项目围绕一条完整的分析链条展开：
- 基于 BTS 延误原因的风险识别
- `Poisson` 与 `Negative Binomial` 的频率建模比较
- `Lognormal` 与 `Weibull` 的严重度建模比较
- 状态划分与两状态 Markov 依赖分析
- 不同运营情景下的年度 aggregate risk

## Final Deliverables | 最终交付
- `reports/dashboard/jfk_modeling_process_showcase_zh.html`
  Chinese process-oriented showcase report for explanation and discussion.
- `reports/summary/jfk_modeling_process_showcase_pdf_export_guide.md`
  Chinese guide for exporting the showcase HTML to PDF from the browser.

- `reports/dashboard/jfk_modeling_process_showcase_zh.html`
  中文建模过程展示版 HTML，用于过程说明、组内讨论和展示。
- `reports/summary/jfk_modeling_process_showcase_pdf_export_guide.md`
  中文 PDF 导出说明，指导如何通过浏览器将 HTML 打印为 PDF。

## Repository Structure | 仓库结构
```text
.
|-- data
|   |-- raw
|   `-- processed
|-- reports
|   |-- charts
|   |-- dashboard
|   |   `-- jfk_modeling_process_showcase_zh.html
|   `-- summary
|       `-- jfk_modeling_process_showcase_pdf_export_guide.md
|-- scripts
|   |-- build_operational_risk_assets.py
|   |-- build_multiyear_operational_risk_project.py
|   |-- build_chinese_modeling_process_report.py
|   `-- build_chinese_modeling_process_showcase.py
|-- .gitignore
|-- environment.yml
`-- README.md
```

## Script Roles | 脚本说明
- `scripts/build_operational_risk_assets.py`
  Builds descriptive cleaned datasets and reusable descriptive chart assets.
- `scripts/build_multiyear_operational_risk_project.py`
  Builds the multi-year modeling datasets, model summaries, transition matrix, aggregate-risk metrics, and the main charts.
- `scripts/build_chinese_modeling_process_report.py`
  Provides shared helpers and report-building utilities used by the showcase generator.
- `scripts/build_chinese_modeling_process_showcase.py`
  Generates the final Chinese showcase HTML and the Chinese PDF export guide.

- `scripts/build_operational_risk_assets.py`
  生成描述性清洗数据和可复用的基础图表资产。
- `scripts/build_multiyear_operational_risk_project.py`
  生成多年份建模数据、模型汇总、转移矩阵、aggregate risk 指标和主图表。
- `scripts/build_chinese_modeling_process_report.py`
  提供中文报告生成所需的共享函数和报告构建工具。
- `scripts/build_chinese_modeling_process_showcase.py`
  生成最终中文展示版 HTML 和中文 PDF 导出说明。

## Main Charts | 主要图表
The final project currently uses these chart outputs:
- `chart_1_multiyear_monthly_trend`
- `chart_2_delay_cause_breakdown`
- `chart_3_risk_heatmap`
- `chart_4_disruption_state_impact`
- `chart_5_monthly_disrupted_share`
- `chart_6_aggregate_risk_scenarios`
- `chart_7_markov_transition_matrix`
- `chart_8_frequency_aic_comparison`
- `chart_9_severity_aic_comparison`
- `chart_10_state_rule_comparison`

当前项目主要使用这些图表输出：
- `chart_1_multiyear_monthly_trend`
- `chart_2_delay_cause_breakdown`
- `chart_3_risk_heatmap`
- `chart_4_disruption_state_impact`
- `chart_5_monthly_disrupted_share`
- `chart_6_aggregate_risk_scenarios`
- `chart_7_markov_transition_matrix`
- `chart_8_frequency_aic_comparison`
- `chart_9_severity_aic_comparison`
- `chart_10_state_rule_comparison`

## Main Processed Datasets | 主要处理后数据
- `jfk_airline_month_modeling_input.csv`
  Airline-month modeling input for frequency and severity analysis.
- `jfk_airport_month_state_panel.csv`
  Airport-month panel used for state definition and Markov analysis.
- `jfk_frequency_model_summary.csv`
  Frequency-model comparison results.
- `jfk_severity_model_summary.csv`
  Severity-model comparison results.
- `jfk_state_definition_comparison.csv`
  Candidate state-rule comparison.
- `jfk_markov_transition_matrix.csv`
  Two-state transition matrix.
- `jfk_aggregate_risk_scenario_metrics.csv`
  Aggregate-risk scenario metrics used in the final project outputs.

- `jfk_airline_month_modeling_input.csv`
  航司-月份层面的建模输入数据，用于频率和严重度分析。
- `jfk_airport_month_state_panel.csv`
  机场-月份层面的状态面板数据，用于状态划分与 Markov 分析。
- `jfk_frequency_model_summary.csv`
  频率模型对比结果。
- `jfk_severity_model_summary.csv`
  严重度模型对比结果。
- `jfk_state_definition_comparison.csv`
  候选状态规则比较结果。
- `jfk_markov_transition_matrix.csv`
  两状态转移矩阵。
- `jfk_aggregate_risk_scenario_metrics.csv`
  最终项目使用的 aggregate risk 情景指标。

Other files in `data/processed` are supportive cleaned datasets and summary tables retained for reproducibility.

`data/processed` 中其余文件属于辅助清洗数据和汇总表，为了保证流程可复现而保留。

## Quick Start | 快速开始
### 1. Create the environment | 创建环境
```bash
conda env create -f environment.yml
conda activate operational_risk
```

### 2. Rebuild the required assets | 重建主要输出
```bash
python scripts/build_operational_risk_assets.py
python scripts/build_multiyear_operational_risk_project.py
python scripts/build_chinese_modeling_process_showcase.py
```

### 3. Open the final report | 打开最终报告
```text
reports/dashboard/jfk_modeling_process_showcase_zh.html
```

## How To View The HTML | 如何查看 HTML
GitHub can display the source code of the HTML file, but it does not behave like a local browser opening the full report with relative assets.

在 GitHub 页面里，你可以直接看到 HTML 源文件，但 GitHub 不会像本地浏览器那样完整加载该报告依赖的相对路径资源。

Practical options:
- Download the repository or the HTML file and open `reports/dashboard/jfk_modeling_process_showcase_zh.html` locally in a browser.
- Use the `Raw` view only for source inspection, not for the final visual presentation.
- If you want an online preview, the better solution is to enable GitHub Pages and publish the `reports/` content or a dedicated `docs/` folder.

实际可用的方式：
- 下载整个仓库，或下载 HTML 文件后，在本地浏览器中打开 `reports/dashboard/jfk_modeling_process_showcase_zh.html`。
- `Raw` 视图只适合查看源码，不适合看最终展示效果。
- 如果想在线打开并分享展示，建议启用 GitHub Pages，把 `reports/` 或单独的 `docs/` 目录发布出去。

## Notes | 说明
- Cache folders, temporary preview files, and Python bytecode are not part of the deliverable.
- The repository is organized around the retained Chinese showcase output and the reproducible data pipeline.
- If you later want to regenerate alternative presentation formats, do it from the retained scripts rather than from deleted generated outputs.

- 缓存目录、临时预览文件和 Python 字节码不属于最终交付内容。
- 当前仓库围绕保留的中文展示版输出和可复现的数据建模流程组织。
- 如果后续还要生成其他展示形式，建议从保留脚本重新生成，而不是恢复已删除的产物。
