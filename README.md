# JFK Airline Delay Operational Risk Project

## Overview
This repository contains a reproducible operational-risk workflow for JFK airline delays, cancellations, and diversions.

The project is organized around one full analytical chain:
- risk identification from BTS delay causes
- frequency modeling with `Poisson` vs `Negative Binomial`
- severity modeling with `Lognormal` vs `Weibull`
- state definition and two-state Markov dependence
- aggregate annual risk under operating scenarios

## Final Deliverables
- `reports/dashboard/jfk_modeling_process_showcase_zh.html`
  Chinese process-oriented showcase report for internal explanation and discussion.
- `reports/summary/jfk_modeling_process_showcase_pdf_export_guide.md`
  Chinese guide for exporting the showcase HTML to PDF from the browser.

## Repository Structure
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

## Script Roles
- `scripts/build_operational_risk_assets.py`
  Builds descriptive cleaned datasets and reusable descriptive chart assets.
- `scripts/build_multiyear_operational_risk_project.py`
  Builds the multi-year modeling datasets, model summaries, transition matrix, aggregate-risk metrics, and the main charts.
- `scripts/build_chinese_modeling_process_report.py`
  Provides shared helpers and report-building utilities used by the showcase generator.
- `scripts/build_chinese_modeling_process_showcase.py`
  Generates the final Chinese showcase HTML and the Chinese PDF export guide.

## Main Charts
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

## Main Processed Datasets
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

Other files in `data/processed` are supportive cleaned datasets and summary tables retained for reproducibility.

## Quick Start
### 1. Create the environment
```bash
conda env create -f environment.yml
conda activate operational_risk
```

### 2. Rebuild the required assets
```bash
python scripts/build_operational_risk_assets.py
python scripts/build_multiyear_operational_risk_project.py
python scripts/build_chinese_modeling_process_showcase.py
```

### 3. Open the final report
```text
reports/dashboard/jfk_modeling_process_showcase_zh.html
```

## Notes
- Cache folders, temporary preview files, and Python bytecode are not part of the deliverable.
- The repository is organized around the retained Chinese showcase output and the reproducible data pipeline.
- If you later want to regenerate alternative presentation formats, do it from the retained scripts rather than from deleted generated outputs.
