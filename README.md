# JFK Airline Delay Operational Risk Project

This repository contains the data preparation, descriptive analysis, charts, and offline dashboard for an operational risk management project focused on airline delays, cancellations, and diversions at JFK Airport using BTS On-Time Performance data.

## Project structure

```text
.
├── data
│   ├── raw
│   │   ├── Airline_Delay_Cause.csv
│   │   └── Download_Column_Definitions.xlsx
│   └── processed
│       ├── jfk_airline_delay_readable_full.csv
│       ├── jfk_airline_delay_core_cleaned.csv
│       ├── jfk_column_dictionary.csv
│       ├── jfk_monthly_risk_summary.csv
│       ├── jfk_delay_cause_summary.csv
│       └── jfk_airline_risk_profile.csv
├── reports
│   ├── charts
│   ├── dashboard
│   └── summary
├── scripts
│   └── build_operational_risk_assets.py
├── environment.yml
└── README.md
```

## Environment

Recommended conda environment name: `operational_risk`

Create it from the repo file:

```bash
conda env create -f environment.yml
```

If you already created the environment locally:

```bash
source /home/simon033/miniconda3/bin/activate operational_risk
```

## Rebuild the outputs

```bash
/home/simon033/miniconda3/envs/operational_risk/bin/python scripts/build_operational_risk_assets.py
```

## Main deliverables

- `data/raw/`
  The original BTS dataset and the official column-definition file.
- `data/processed/jfk_airline_delay_readable_full.csv`
  Readable full dataset with airline codes replaced by full names and clearer field names.
- `data/processed/jfk_airline_delay_core_cleaned.csv`
  Cleaned core dataset for later frequency and severity modeling.
- `data/processed/jfk_column_dictionary.csv`
  Data dictionary with original column names, readable names, and plain-language explanations.
- `reports/charts/`
  High-resolution PNG charts plus SVG vector versions for PowerPoint or print.
- `reports/dashboard/jfk_operational_risk_dashboard.html`
  Standalone offline interactive dashboard.
- `reports/summary/jfk_descriptive_analysis_summary.md`
  Short narrative summary of cleaning results and key descriptive findings.

## Cleaning logic

- Drop rows where `total_arrival_flights` is missing or zero.
- Fill missing delay-minute fields with `0` when the airline operated flights that month.
- Drop rows where `delayed_arrivals_15_plus` exceeds `total_arrival_flights`.

For this JFK sample, no rows were removed by those rules, but the pipeline keeps the checks in place for reproducibility and later modeling work.
