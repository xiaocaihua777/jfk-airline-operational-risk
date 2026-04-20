from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(BASE_DIR / ".cache" / "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
REPORTS_DIR = BASE_DIR / "reports"
CHART_DIR = REPORTS_DIR / "charts"
SUMMARY_DIR = REPORTS_DIR / "summary"

READABLE_FULL = PROCESSED_DIR / "jfk_airline_delay_readable_full.csv"
CORE_CLEANED = PROCESSED_DIR / "jfk_airline_delay_core_cleaned.csv"
MONTHLY_SUMMARY = PROCESSED_DIR / "jfk_monthly_risk_summary.csv"
SEASONAL_SUMMARY = PROCESSED_DIR / "jfk_seasonal_risk_summary.csv"
CAUSE_SUMMARY = PROCESSED_DIR / "jfk_delay_cause_summary.csv"
AIRLINE_PROFILE = PROCESSED_DIR / "jfk_airline_risk_profile.csv"
AIRPORT_MONTH_PANEL = PROCESSED_DIR / "jfk_airport_month_state_panel.csv"
STATE_RULE_COMPARISON = PROCESSED_DIR / "jfk_state_definition_comparison.csv"
STATE_TRANSITIONS = PROCESSED_DIR / "jfk_markov_transition_matrix.csv"
MODELING_INPUT = PROCESSED_DIR / "jfk_airline_month_modeling_input.csv"
FREQUENCY_SUMMARY = PROCESSED_DIR / "jfk_frequency_model_summary.csv"
SEVERITY_SUMMARY = PROCESSED_DIR / "jfk_severity_model_summary.csv"
AGGREGATE_SIMULATIONS = PROCESSED_DIR / "jfk_aggregate_risk_simulations.csv"
SCENARIO_METRICS = PROCESSED_DIR / "jfk_aggregate_risk_scenario_metrics.csv"

STAT_SUMMARY = SUMMARY_DIR / "jfk_statistical_modeling_summary.md"
RISK_IDENTIFICATION_SUMMARY = SUMMARY_DIR / "jfk_risk_identification_summary.md"
METHODOLOGY_SUMMARY = SUMMARY_DIR / "jfk_methodology_notes.md"

RAW_GLOB = "Airline_Delay_Cause*.csv"
CHART_STYLE = {
    "primary": "#0F4C5C",
    "secondary": "#E36414",
    "accent": "#2A9D8F",
    "red": "#D1495B",
    "sand": "#F6F2E8",
    "grid": "#D9D2C3",
    "ink": "#3D405B",
}

AIRLINE_NAME_FIXES = {
    "American Airlines Network": "American Airlines",
    "Alaska Airlines Network": "Alaska Airlines",
    "Delta Air Lines Network": "Delta Air Lines",
    "Endeavor Air Inc.": "Endeavor Air",
    "Hawaiian Airlines Network": "Hawaiian Airlines",
    "SkyWest Airlines Inc.": "SkyWest Airlines",
}

READABLE_COLUMN_MAP = {
    "year": "year",
    "month": "month",
    "carrier_name": "airline_name",
    "airport": "airport_code",
    "airport_name": "airport_name",
    "arr_flights": "total_arrival_flights",
    "arr_del15": "delayed_arrivals_15_plus",
    "carrier_ct": "estimated_delayed_arrivals_due_to_airline",
    "weather_ct": "estimated_delayed_arrivals_due_to_weather",
    "nas_ct": "estimated_delayed_arrivals_due_to_nas",
    "security_ct": "estimated_delayed_arrivals_due_to_security",
    "late_aircraft_ct": "estimated_delayed_arrivals_due_to_late_aircraft",
    "arr_cancelled": "cancelled_arrivals",
    "arr_diverted": "diverted_arrivals",
    "arr_delay": "total_arrival_delay_minutes",
    "carrier_delay": "airline_delay_minutes",
    "weather_delay": "weather_delay_minutes",
    "nas_delay": "nas_delay_minutes",
    "security_delay": "security_delay_minutes",
    "late_aircraft_delay": "late_aircraft_delay_minutes",
}

READABLE_COLUMN_ORDER = [
    "year",
    "month",
    "airline_name",
    "airport_code",
    "airport_name",
    "total_arrival_flights",
    "delayed_arrivals_15_plus",
    "estimated_delayed_arrivals_due_to_airline",
    "estimated_delayed_arrivals_due_to_weather",
    "estimated_delayed_arrivals_due_to_nas",
    "estimated_delayed_arrivals_due_to_security",
    "estimated_delayed_arrivals_due_to_late_aircraft",
    "cancelled_arrivals",
    "diverted_arrivals",
    "total_arrival_delay_minutes",
    "airline_delay_minutes",
    "weather_delay_minutes",
    "nas_delay_minutes",
    "security_delay_minutes",
    "late_aircraft_delay_minutes",
]

NUMERIC_COLUMNS = [
    "year",
    "month",
    "arr_flights",
    "arr_del15",
    "carrier_ct",
    "weather_ct",
    "nas_ct",
    "security_ct",
    "late_aircraft_ct",
    "arr_cancelled",
    "arr_diverted",
    "arr_delay",
    "carrier_delay",
    "weather_delay",
    "nas_delay",
    "security_delay",
    "late_aircraft_delay",
]

CAUSE_LABELS = {
    "airline_delay_minutes": "Airline",
    "weather_delay_minutes": "Weather",
    "nas_delay_minutes": "NAS",
    "security_delay_minutes": "Security",
    "late_aircraft_delay_minutes": "Late Aircraft",
}

CANCELLATION_PENALTY_MINUTES = 180.0
DIVERSION_PENALTY_MINUTES = 240.0
SIMULATION_RUNS = 500
RANDOM_SEED = 7337


def ensure_output_directories() -> None:
    for directory in (PROCESSED_DIR, CHART_DIR, SUMMARY_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def apply_style() -> None:
    sns.set_theme(
        style="whitegrid",
        rc={
            "axes.facecolor": CHART_STYLE["sand"],
            "figure.facecolor": CHART_STYLE["sand"],
            "axes.edgecolor": CHART_STYLE["grid"],
            "grid.color": CHART_STYLE["grid"],
            "axes.labelcolor": CHART_STYLE["ink"],
            "text.color": CHART_STYLE["ink"],
            "xtick.color": CHART_STYLE["ink"],
            "ytick.color": CHART_STYLE["ink"],
        },
    )


def save_chart(fig: plt.Figure, stem: str) -> None:
    fig.savefig(CHART_DIR / f"{stem}.png", dpi=320, bbox_inches="tight")
    fig.savefig(CHART_DIR / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def month_name(month: int) -> str:
    labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    return labels[int(month) - 1]


def load_multiyear_raw() -> tuple[pd.DataFrame, list[str]]:
    raw_files = sorted(path for path in RAW_DIR.glob(RAW_GLOB) if path.name.lower().endswith(".csv"))
    if not raw_files:
        raise FileNotFoundError(f"No raw files matching {RAW_GLOB} were found in {RAW_DIR}.")

    frames: list[pd.DataFrame] = []
    used_files: list[str] = []
    for path in raw_files:
        df = pd.read_csv(path)
        expected = {"year", "month", "carrier", "carrier_name", "airport", "airport_name", "arr_flights"}
        if not expected.issubset(df.columns):
            continue
        frames.append(df)
        used_files.append(path.name)

    if not frames:
        raise ValueError("Raw files were found, but none matched the expected BTS schema.")

    raw_df = pd.concat(frames, ignore_index=True).drop_duplicates().reset_index(drop=True)
    raw_df["carrier_name"] = raw_df["carrier_name"].replace(AIRLINE_NAME_FIXES)
    raw_df["airport_name"] = raw_df["airport_name"].replace(
        {"New York, NY: John F. Kennedy International": "John F. Kennedy International Airport (JFK)"}
    )
    for column in NUMERIC_COLUMNS:
        raw_df[column] = pd.to_numeric(raw_df[column], errors="coerce")
    raw_df = raw_df.loc[raw_df["airport"].eq("JFK")].copy()
    raw_df["year"] = raw_df["year"].astype("Int64")
    raw_df["month"] = raw_df["month"].astype("Int64")
    raw_df = raw_df.sort_values(["year", "month", "carrier_name"]).reset_index(drop=True)
    return raw_df, used_files


def build_readable_and_core(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    readable = raw_df.drop(columns=["carrier"]).rename(columns=READABLE_COLUMN_MAP)
    readable = readable[READABLE_COLUMN_ORDER].copy()

    int_like = [
        "year",
        "month",
        "total_arrival_flights",
        "delayed_arrivals_15_plus",
        "cancelled_arrivals",
        "diverted_arrivals",
        "total_arrival_delay_minutes",
        "airline_delay_minutes",
        "weather_delay_minutes",
        "nas_delay_minutes",
        "security_delay_minutes",
        "late_aircraft_delay_minutes",
    ]
    readable[int_like] = readable[int_like].fillna(0).astype(int)
    readable.to_csv(READABLE_FULL, index=False)

    cleaned = readable.copy()
    initial_rows = len(cleaned)
    no_op_mask = cleaned["total_arrival_flights"].isna() | (cleaned["total_arrival_flights"] <= 0)
    removed_no_op = int(no_op_mask.sum())
    cleaned = cleaned.loc[~no_op_mask].copy()

    fill_zero_columns = [
        "delayed_arrivals_15_plus",
        "cancelled_arrivals",
        "diverted_arrivals",
        "total_arrival_delay_minutes",
        "airline_delay_minutes",
        "weather_delay_minutes",
        "nas_delay_minutes",
        "security_delay_minutes",
        "late_aircraft_delay_minutes",
        "estimated_delayed_arrivals_due_to_airline",
        "estimated_delayed_arrivals_due_to_weather",
        "estimated_delayed_arrivals_due_to_nas",
        "estimated_delayed_arrivals_due_to_security",
        "estimated_delayed_arrivals_due_to_late_aircraft",
    ]
    cleaned[fill_zero_columns] = cleaned[fill_zero_columns].fillna(0)
    logic_mask = cleaned["delayed_arrivals_15_plus"] > cleaned["total_arrival_flights"]
    removed_logic = int(logic_mask.sum())
    cleaned = cleaned.loc[~logic_mask].copy()
    cleaned = cleaned.sort_values(["year", "month", "airline_name"]).reset_index(drop=True)
    cleaned.to_csv(CORE_CLEANED, index=False)

    summary = {
        "initial_rows": initial_rows,
        "removed_for_no_operations": removed_no_op,
        "removed_for_logic_issue": removed_logic,
        "final_rows": int(len(cleaned)),
    }
    return readable, cleaned, summary


def build_summaries(cleaned: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    monthly = (
        cleaned.groupby(["year", "month"], as_index=False)
        .agg(
            airlines_in_sample=("airline_name", "nunique"),
            total_arrival_flights=("total_arrival_flights", "sum"),
            delayed_arrivals_15_plus=("delayed_arrivals_15_plus", "sum"),
            cancelled_arrivals=("cancelled_arrivals", "sum"),
            diverted_arrivals=("diverted_arrivals", "sum"),
            total_arrival_delay_minutes=("total_arrival_delay_minutes", "sum"),
            airline_delay_minutes=("airline_delay_minutes", "sum"),
            weather_delay_minutes=("weather_delay_minutes", "sum"),
            nas_delay_minutes=("nas_delay_minutes", "sum"),
            security_delay_minutes=("security_delay_minutes", "sum"),
            late_aircraft_delay_minutes=("late_aircraft_delay_minutes", "sum"),
        )
        .sort_values(["year", "month"])
        .reset_index(drop=True)
    )
    monthly["period"] = pd.to_datetime(dict(year=monthly["year"], month=monthly["month"], day=1))
    monthly["period_label"] = monthly["period"].dt.strftime("%Y-%m")
    monthly["delay_rate"] = monthly["delayed_arrivals_15_plus"] / monthly["total_arrival_flights"]
    monthly["cancellation_rate"] = monthly["cancelled_arrivals"] / monthly["total_arrival_flights"]
    monthly["diversion_rate"] = monthly["diverted_arrivals"] / monthly["total_arrival_flights"]
    monthly["avg_delay_minutes_per_delayed_flight"] = np.where(
        monthly["delayed_arrivals_15_plus"] > 0,
        monthly["total_arrival_delay_minutes"] / monthly["delayed_arrivals_15_plus"],
        0.0,
    )
    monthly.to_csv(MONTHLY_SUMMARY, index=False)

    seasonal = (
        monthly.groupby("month", as_index=False)
        .agg(
            years_observed=("year", "nunique"),
            mean_delay_rate=("delay_rate", "mean"),
            median_delay_rate=("delay_rate", "median"),
            mean_cancellation_rate=("cancellation_rate", "mean"),
            mean_impact_minutes=("total_arrival_delay_minutes", "mean"),
        )
        .sort_values("month")
        .reset_index(drop=True)
    )
    seasonal["month_label"] = seasonal["month"].map(month_name)
    seasonal.to_csv(SEASONAL_SUMMARY, index=False)

    cause = pd.DataFrame(
        {
            "cause": [CAUSE_LABELS[column] for column in CAUSE_LABELS],
            "delay_minutes": [monthly[column].sum() for column in CAUSE_LABELS],
        }
    )
    cause["share_of_total_delay_minutes"] = cause["delay_minutes"] / cause["delay_minutes"].sum()
    cause = cause.sort_values("delay_minutes", ascending=False).reset_index(drop=True)
    cause.to_csv(CAUSE_SUMMARY, index=False)

    airline_profile = (
        cleaned.groupby("airline_name", as_index=False)
        .agg(
            total_arrival_flights=("total_arrival_flights", "sum"),
            delayed_arrivals_15_plus=("delayed_arrivals_15_plus", "sum"),
            cancelled_arrivals=("cancelled_arrivals", "sum"),
            diverted_arrivals=("diverted_arrivals", "sum"),
            total_arrival_delay_minutes=("total_arrival_delay_minutes", "sum"),
            airline_delay_minutes=("airline_delay_minutes", "sum"),
            weather_delay_minutes=("weather_delay_minutes", "sum"),
            nas_delay_minutes=("nas_delay_minutes", "sum"),
            security_delay_minutes=("security_delay_minutes", "sum"),
            late_aircraft_delay_minutes=("late_aircraft_delay_minutes", "sum"),
            active_years=("year", "nunique"),
        )
        .sort_values("airline_name")
        .reset_index(drop=True)
    )
    airline_profile["delay_rate"] = airline_profile["delayed_arrivals_15_plus"] / airline_profile["total_arrival_flights"]
    airline_profile["cancellation_rate"] = airline_profile["cancelled_arrivals"] / airline_profile["total_arrival_flights"]
    airline_profile["avg_delay_minutes_per_delayed_flight"] = np.where(
        airline_profile["delayed_arrivals_15_plus"] > 0,
        airline_profile["total_arrival_delay_minutes"] / airline_profile["delayed_arrivals_15_plus"],
        0.0,
    )
    airline_profile.to_csv(AIRLINE_PROFILE, index=False)

    return monthly, seasonal, cause, airline_profile


def build_airport_state_panel(monthly: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, str, bool]:
    airport_month = monthly.copy()
    airport_month["internal_ops_delay_minutes"] = airport_month["airline_delay_minutes"] + airport_month["late_aircraft_delay_minutes"]
    airport_month["system_ops_delay_minutes"] = airport_month["nas_delay_minutes"]
    airport_month["external_ops_delay_minutes"] = airport_month["weather_delay_minutes"] + airport_month["security_delay_minutes"]
    airport_month["disruption_impact_minutes"] = (
        airport_month["total_arrival_delay_minutes"]
        + airport_month["cancelled_arrivals"] * CANCELLATION_PENALTY_MINUTES
        + airport_month["diverted_arrivals"] * DIVERSION_PENALTY_MINUTES
    )

    impact_threshold = airport_month["disruption_impact_minutes"].quantile(0.75)
    airport_month["state_impact_q75"] = np.where(
        airport_month["disruption_impact_minutes"] >= impact_threshold, "disrupted", "normal"
    )

    delay_std = airport_month["delay_rate"].std(ddof=0)
    cancel_std = airport_month["cancellation_rate"].std(ddof=0)
    impact_std = airport_month["disruption_impact_minutes"].std(ddof=0)
    delay_rate_z = ((airport_month["delay_rate"] - airport_month["delay_rate"].mean()) / delay_std) if delay_std > 0 else 0
    cancel_rate_z = ((airport_month["cancellation_rate"] - airport_month["cancellation_rate"].mean()) / cancel_std) if cancel_std > 0 else 0
    impact_z = ((airport_month["disruption_impact_minutes"] - airport_month["disruption_impact_minutes"].mean()) / impact_std) if impact_std > 0 else 0
    composite = pd.Series(delay_rate_z, index=airport_month.index) + pd.Series(cancel_rate_z, index=airport_month.index) + pd.Series(impact_z, index=airport_month.index)
    composite_threshold = composite.quantile(0.75)
    airport_month["state_composite"] = np.where(composite >= composite_threshold, "disrupted", "normal")

    comparison_rows = []
    for rule, column in {
        "impact_q75": "state_impact_q75",
        "delay_cancel_impact_composite": "state_composite",
    }.items():
        disrupted_share = float((airport_month[column] == "disrupted").mean())
        yearly = (
            airport_month.assign(rule_state=airport_month[column])
            .groupby("year")
            .apply(lambda grp: float((grp["rule_state"] == "disrupted").mean()), include_groups=False)
        )
        share_std = float(yearly.std(ddof=0)) if len(yearly) > 1 else 0.0
        score = abs(disrupted_share - 0.25) + share_std
        comparison_rows.append(
            {
                "rule_name": rule,
                "disrupted_share": disrupted_share,
                "yearly_disrupted_share_std": share_std,
                "disrupted_months": int((airport_month[column] == "disrupted").sum()),
                "selection_score": score,
            }
        )

    comparison = pd.DataFrame(comparison_rows).sort_values("selection_score").reset_index(drop=True)
    comparison["selected"] = False
    selected_rule = "impact_q75" if len(airport_month) < 24 else str(comparison.iloc[0]["rule_name"])
    comparison.loc[comparison["rule_name"] == selected_rule, "selected"] = True
    comparison.to_csv(STATE_RULE_COMPARISON, index=False)

    selected_column = "state_impact_q75" if selected_rule == "impact_q75" else "state_composite"
    airport_month["operational_state"] = airport_month[selected_column]
    stable_transition = len(airport_month) >= 24 and airport_month["operational_state"].nunique() == 2
    if stable_transition:
        lag = airport_month["operational_state"].shift(1)
        starts = lag.iloc[1:]
        stable_transition = int((starts == "normal").sum()) >= 4 and int((starts == "disrupted").sum()) >= 4

    airport_month.to_csv(AIRPORT_MONTH_PANEL, index=False)
    return airport_month, comparison, selected_rule, stable_transition


def estimate_transition_matrix(airport_month: pd.DataFrame, stable_transition: bool) -> pd.DataFrame:
    if not stable_transition:
        matrix = pd.DataFrame(
            [
                {
                    "from_state": "not_estimated",
                    "to_state": "not_estimated",
                    "transition_probability": np.nan,
                    "note": "Transition matrix not estimated because the month-level sample is too short or the state split is unstable.",
                }
            ]
        )
        matrix.to_csv(STATE_TRANSITIONS, index=False)
        return matrix

    panel = airport_month.sort_values(["year", "month"]).reset_index(drop=True)
    transitions = [(panel.loc[idx, "operational_state"], panel.loc[idx + 1, "operational_state"]) for idx in range(len(panel) - 1)]
    rows = []
    for start in ("normal", "disrupted"):
        total = sum(1 for from_state, _ in transitions if from_state == start)
        for end in ("normal", "disrupted"):
            count = sum(1 for from_state, to_state in transitions if from_state == start and to_state == end)
            rows.append(
                {
                    "from_state": start,
                    "to_state": end,
                    "transition_probability": count / total if total else np.nan,
                    "transition_count": count,
                }
            )
    matrix = pd.DataFrame(rows)
    matrix.to_csv(STATE_TRANSITIONS, index=False)
    return matrix


def build_modeling_input(cleaned: pd.DataFrame, airport_month: pd.DataFrame) -> pd.DataFrame:
    model_df = cleaned.copy()
    model_df["internal_ops_count"] = model_df["estimated_delayed_arrivals_due_to_airline"] + model_df["estimated_delayed_arrivals_due_to_late_aircraft"]
    model_df["system_ops_count"] = model_df["estimated_delayed_arrivals_due_to_nas"]
    model_df["external_ops_count"] = model_df["estimated_delayed_arrivals_due_to_weather"] + model_df["estimated_delayed_arrivals_due_to_security"]
    model_df["internal_ops_count_rounded"] = model_df["internal_ops_count"].round().astype(int)
    model_df["system_ops_count_rounded"] = model_df["system_ops_count"].round().astype(int)
    model_df["external_ops_count_rounded"] = model_df["external_ops_count"].round().astype(int)

    model_df["internal_ops_delay_minutes"] = model_df["airline_delay_minutes"] + model_df["late_aircraft_delay_minutes"]
    model_df["system_ops_delay_minutes"] = model_df["nas_delay_minutes"]
    model_df["external_ops_delay_minutes"] = model_df["weather_delay_minutes"] + model_df["security_delay_minutes"]

    model_df["delay_rate"] = model_df["delayed_arrivals_15_plus"] / model_df["total_arrival_flights"]
    model_df["cancellation_rate"] = model_df["cancelled_arrivals"] / model_df["total_arrival_flights"]
    model_df["diversion_rate"] = model_df["diverted_arrivals"] / model_df["total_arrival_flights"]
    model_df["avg_delay_minutes_per_delayed_flight"] = np.where(
        model_df["delayed_arrivals_15_plus"] > 0,
        model_df["total_arrival_delay_minutes"] / model_df["delayed_arrivals_15_plus"],
        0.0,
    )
    model_df["avg_internal_minutes_per_event"] = np.where(
        model_df["internal_ops_count"] > 0,
        model_df["internal_ops_delay_minutes"] / model_df["internal_ops_count"],
        0.0,
    )
    model_df["avg_system_minutes_per_event"] = np.where(
        model_df["system_ops_count"] > 0,
        model_df["system_ops_delay_minutes"] / model_df["system_ops_count"],
        0.0,
    )
    model_df["avg_external_minutes_per_event"] = np.where(
        model_df["external_ops_count"] > 0,
        model_df["external_ops_delay_minutes"] / model_df["external_ops_count"],
        0.0,
    )
    model_df["disruption_impact_minutes"] = (
        model_df["total_arrival_delay_minutes"]
        + model_df["cancelled_arrivals"] * CANCELLATION_PENALTY_MINUTES
        + model_df["diverted_arrivals"] * DIVERSION_PENALTY_MINUTES
    )

    model_df = model_df.merge(
        airport_month[["year", "month", "period_label", "operational_state"]],
        on=["year", "month"],
        how="left",
    )
    model_df = model_df.sort_values(["year", "month", "airline_name"]).reset_index(drop=True)
    model_df.to_csv(MODELING_INPUT, index=False)
    return model_df


def fit_poisson(data: np.ndarray) -> tuple[dict[str, float], float, float]:
    lam = float(np.mean(data))
    ll = float(np.sum(stats.poisson.logpmf(data, mu=lam)))
    return {"lambda": lam}, ll, 2 - 2 * ll


def fit_negative_binomial(data: np.ndarray) -> tuple[dict[str, float], float, float]:
    mean = float(np.mean(data))
    variance = float(np.var(data, ddof=1)) if len(data) > 1 else mean
    if variance <= mean:
        dispersion = 1e6
    else:
        dispersion = (mean * mean) / (variance - mean)
    p = dispersion / (dispersion + mean)
    ll = float(np.sum(stats.nbinom.logpmf(data, dispersion, p)))
    return {"dispersion": dispersion, "p": p}, ll, 4 - 2 * ll


def map_variable_to_series(variable: str) -> str:
    return {
        "total_delay_count": "delayed_arrivals_15_plus",
        "cancellation_count": "cancelled_arrivals",
        "diversion_count": "diverted_arrivals",
        "internal_ops_count": "internal_ops_count_rounded",
        "system_ops_count": "system_ops_count_rounded",
        "external_ops_count": "external_ops_count_rounded",
    }[variable]


def build_frequency_models(model_df: pd.DataFrame) -> pd.DataFrame:
    targets = {
        "total_delay_count": model_df["delayed_arrivals_15_plus"].to_numpy(dtype=int),
        "cancellation_count": model_df["cancelled_arrivals"].to_numpy(dtype=int),
        "diversion_count": model_df["diverted_arrivals"].to_numpy(dtype=int),
        "internal_ops_count": model_df["internal_ops_count_rounded"].to_numpy(dtype=int),
        "system_ops_count": model_df["system_ops_count_rounded"].to_numpy(dtype=int),
        "external_ops_count": model_df["external_ops_count_rounded"].to_numpy(dtype=int),
    }

    rows: list[dict[str, float | str | bool]] = []
    for variable, values in targets.items():
        poisson_params, poisson_ll, poisson_aic = fit_poisson(values)
        nb_params, nb_ll, nb_aic = fit_negative_binomial(values)
        series_name = map_variable_to_series(variable)
        normal_mean = float(model_df.loc[model_df["operational_state"] == "normal", series_name].mean())
        disrupted_mean = float(model_df.loc[model_df["operational_state"] == "disrupted", series_name].mean())
        rows.extend(
            [
                {
                    "variable": variable,
                    "distribution": "poisson",
                    "param_1": poisson_params["lambda"],
                    "param_2": np.nan,
                    "log_likelihood": poisson_ll,
                    "aic": poisson_aic,
                    "overall_mean": float(np.mean(values)),
                    "overall_variance": float(np.var(values, ddof=1)),
                    "normal_state_mean": normal_mean,
                    "disrupted_state_mean": disrupted_mean,
                },
                {
                    "variable": variable,
                    "distribution": "negative_binomial",
                    "param_1": nb_params["dispersion"],
                    "param_2": nb_params["p"],
                    "log_likelihood": nb_ll,
                    "aic": nb_aic,
                    "overall_mean": float(np.mean(values)),
                    "overall_variance": float(np.var(values, ddof=1)),
                    "normal_state_mean": normal_mean,
                    "disrupted_state_mean": disrupted_mean,
                },
            ]
        )
    summary = pd.DataFrame(rows)
    summary["selected"] = summary.groupby("variable")["aic"].transform("min") == summary["aic"]
    summary.to_csv(FREQUENCY_SUMMARY, index=False)
    return summary


def fit_lognorm(data: np.ndarray) -> tuple[tuple[float, float, float], float]:
    params = stats.lognorm.fit(data, floc=0)
    ll = float(np.sum(stats.lognorm.logpdf(data, *params)))
    return params, ll


def fit_weibull(data: np.ndarray) -> tuple[tuple[float, float, float], float]:
    params = stats.weibull_min.fit(data, floc=0)
    ll = float(np.sum(stats.weibull_min.logpdf(data, *params)))
    return params, ll


def build_severity_models(model_df: pd.DataFrame) -> pd.DataFrame:
    targets = {
        "total_avg_delay_minutes": "avg_delay_minutes_per_delayed_flight",
        "internal_avg_delay_minutes": "avg_internal_minutes_per_event",
        "system_avg_delay_minutes": "avg_system_minutes_per_event",
        "external_avg_delay_minutes": "avg_external_minutes_per_event",
    }

    rows: list[dict[str, float | str | bool]] = []
    for variable, series_name in targets.items():
        values = model_df.loc[model_df[series_name] > 0, series_name].to_numpy()
        lognorm_params, lognorm_ll = fit_lognorm(values)
        weibull_params, weibull_ll = fit_weibull(values)
        normal_mean = float(model_df.loc[(model_df["operational_state"] == "normal") & (model_df[series_name] > 0), series_name].mean())
        disrupted_mean = float(model_df.loc[(model_df["operational_state"] == "disrupted") & (model_df[series_name] > 0), series_name].mean())
        rows.extend(
            [
                {
                    "variable": variable,
                    "distribution": "lognorm",
                    "param_1": float(lognorm_params[0]),
                    "param_2": float(lognorm_params[2]),
                    "log_likelihood": lognorm_ll,
                    "aic": 4 - 2 * lognorm_ll,
                    "sample_size": int(len(values)),
                    "normal_state_mean": normal_mean,
                    "disrupted_state_mean": disrupted_mean,
                },
                {
                    "variable": variable,
                    "distribution": "weibull_min",
                    "param_1": float(weibull_params[0]),
                    "param_2": float(weibull_params[2]),
                    "log_likelihood": weibull_ll,
                    "aic": 4 - 2 * weibull_ll,
                    "sample_size": int(len(values)),
                    "normal_state_mean": normal_mean,
                    "disrupted_state_mean": disrupted_mean,
                },
            ]
        )

    summary = pd.DataFrame(rows)
    summary["selected"] = summary.groupby("variable")["aic"].transform("min") == summary["aic"]
    summary.to_csv(SEVERITY_SUMMARY, index=False)
    return summary


def draw_count(model_name: str, params: pd.Series, multiplier: float, rng: np.random.Generator) -> int:
    if model_name == "poisson":
        lam = max(float(params["param_1"]) * multiplier, 1e-9)
        return int(rng.poisson(lam))

    dispersion = max(float(params["param_1"]), 1e-9)
    base_p = min(max(float(params["param_2"]), 1e-9), 1 - 1e-9)
    base_mean = dispersion * (1 - base_p) / base_p
    scaled_mean = max(base_mean * multiplier, 1e-9)
    if dispersion >= 1e5:
        return int(rng.poisson(scaled_mean))
    p = dispersion / (dispersion + scaled_mean)
    return int(rng.negative_binomial(dispersion, p))


def draw_severity(model_name: str, params: pd.Series, multiplier: float, rng: np.random.Generator) -> float:
    if model_name == "lognorm":
        sigma = float(params["param_1"])
        scale = max(float(params["param_2"]), 1e-9)
        value = rng.lognormal(mean=float(np.log(scale)), sigma=sigma)
    else:
        shape = max(float(params["param_1"]), 1e-9)
        scale = max(float(params["param_2"]), 1e-9)
        value = rng.weibull(shape) * scale
    return float(value) * multiplier


def simulate_states(
    airport_month: pd.DataFrame,
    transition_matrix: pd.DataFrame,
    stable_transition: bool,
    months_to_simulate: int,
    rng: np.random.Generator,
) -> list[str]:
    observed_states = airport_month["operational_state"].tolist()
    if not stable_transition:
        disrupted_share = float((airport_month["operational_state"] == "disrupted").mean())
        return ["disrupted" if rng.random() < disrupted_share else "normal" for _ in range(months_to_simulate)]

    transition_lookup = {
        (row["from_state"], row["to_state"]): float(row["transition_probability"])
        for _, row in transition_matrix.iterrows()
        if pd.notna(row["transition_probability"])
    }
    current_state = observed_states[0]
    sequence = [current_state]
    for _ in range(months_to_simulate - 1):
        p_disrupted = transition_lookup[(current_state, "disrupted")]
        current_state = "disrupted" if rng.random() < p_disrupted else "normal"
        sequence.append(current_state)
    return sequence


def simulate_aggregate_risk(
    model_df: pd.DataFrame,
    airport_month: pd.DataFrame,
    transition_matrix: pd.DataFrame,
    stable_transition: bool,
    frequency_df: pd.DataFrame,
    severity_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_frequency = frequency_df.loc[frequency_df["selected"]].set_index("variable")
    selected_severity = severity_df.loc[severity_df["selected"]].set_index("variable")
    month_sizes = airport_month["airlines_in_sample"].to_numpy(dtype=int)
    rng = np.random.default_rng(RANDOM_SEED)

    scenarios = {
        "base": {
            "normal": {"internal": 1.00, "system": 1.00, "external": 1.00, "cancel": 1.00, "divert": 1.00, "sev_internal": 1.00, "sev_system": 1.00, "sev_external": 1.00},
            "disrupted": {"internal": 1.10, "system": 1.20, "external": 1.25, "cancel": 1.15, "divert": 1.10, "sev_internal": 1.05, "sev_system": 1.08, "sev_external": 1.10},
        },
        "disruption_stress": {
            "normal": {"internal": 1.05, "system": 1.10, "external": 1.10, "cancel": 1.05, "divert": 1.05, "sev_internal": 1.03, "sev_system": 1.05, "sev_external": 1.05},
            "disrupted": {"internal": 1.25, "system": 1.35, "external": 1.45, "cancel": 1.30, "divert": 1.20, "sev_internal": 1.10, "sev_system": 1.15, "sev_external": 1.20},
        },
        "weather_shock": {
            "normal": {"internal": 1.00, "system": 1.05, "external": 1.25, "cancel": 1.10, "divert": 1.08, "sev_internal": 1.00, "sev_system": 1.03, "sev_external": 1.12},
            "disrupted": {"internal": 1.05, "system": 1.15, "external": 1.65, "cancel": 1.40, "divert": 1.30, "sev_internal": 1.02, "sev_system": 1.08, "sev_external": 1.25},
        },
        "holiday_peak": {
            "normal": {"internal": 1.12, "system": 1.12, "external": 1.08, "cancel": 1.08, "divert": 1.05, "sev_internal": 1.05, "sev_system": 1.05, "sev_external": 1.03},
            "disrupted": {"internal": 1.22, "system": 1.28, "external": 1.18, "cancel": 1.22, "divert": 1.15, "sev_internal": 1.10, "sev_system": 1.12, "sev_external": 1.08},
        },
    }

    simulation_rows: list[dict[str, float | str | int]] = []
    metrics_rows: list[dict[str, float | str]] = []
    for scenario_name, config in scenarios.items():
        annual_totals = []
        for sim_id in range(1, SIMULATION_RUNS + 1):
            state_sequence = simulate_states(airport_month, transition_matrix, stable_transition, 12, rng)
            annual_total = 0.0
            for state in state_sequence:
                month_airlines = int(rng.choice(month_sizes))
                state_cfg = config[state]
                monthly_total = 0.0
                for _ in range(month_airlines):
                    internal_count = draw_count(selected_frequency.loc["internal_ops_count", "distribution"], selected_frequency.loc["internal_ops_count"], state_cfg["internal"], rng)
                    system_count = draw_count(selected_frequency.loc["system_ops_count", "distribution"], selected_frequency.loc["system_ops_count"], state_cfg["system"], rng)
                    external_count = draw_count(selected_frequency.loc["external_ops_count", "distribution"], selected_frequency.loc["external_ops_count"], state_cfg["external"], rng)
                    cancel_count = draw_count(selected_frequency.loc["cancellation_count", "distribution"], selected_frequency.loc["cancellation_count"], state_cfg["cancel"], rng)
                    divert_count = draw_count(selected_frequency.loc["diversion_count", "distribution"], selected_frequency.loc["diversion_count"], state_cfg["divert"], rng)

                    if internal_count > 0:
                        monthly_total += internal_count * draw_severity(selected_severity.loc["internal_avg_delay_minutes", "distribution"], selected_severity.loc["internal_avg_delay_minutes"], state_cfg["sev_internal"], rng)
                    if system_count > 0:
                        monthly_total += system_count * draw_severity(selected_severity.loc["system_avg_delay_minutes", "distribution"], selected_severity.loc["system_avg_delay_minutes"], state_cfg["sev_system"], rng)
                    if external_count > 0:
                        monthly_total += external_count * draw_severity(selected_severity.loc["external_avg_delay_minutes", "distribution"], selected_severity.loc["external_avg_delay_minutes"], state_cfg["sev_external"], rng)

                    monthly_total += cancel_count * CANCELLATION_PENALTY_MINUTES
                    monthly_total += divert_count * DIVERSION_PENALTY_MINUTES
                annual_total += monthly_total

            annual_totals.append(annual_total)
            simulation_rows.append(
                {
                    "scenario": scenario_name,
                    "simulation_id": sim_id,
                    "aggregate_annual_impact_minutes": round(annual_total, 2),
                }
            )

        annual_series = pd.Series(annual_totals)
        var95 = float(annual_series.quantile(0.95))
        var99 = float(annual_series.quantile(0.99))
        metrics_rows.append(
            {
                "scenario": scenario_name,
                "expected_impact_minutes": round(float(annual_series.mean()), 2),
                "var_95_minutes": round(var95, 2),
                "tvar_95_minutes": round(float(annual_series.loc[annual_series >= var95].mean()), 2),
                "var_99_minutes": round(var99, 2),
                "tvar_99_minutes": round(float(annual_series.loc[annual_series >= var99].mean()), 2),
            }
        )

    simulations = pd.DataFrame(simulation_rows)
    metrics = pd.DataFrame(metrics_rows)
    simulations.to_csv(AGGREGATE_SIMULATIONS, index=False)
    metrics.to_csv(SCENARIO_METRICS, index=False)
    return simulations, metrics


def build_risk_identification_assets(
    monthly: pd.DataFrame,
    airport_month: pd.DataFrame,
    cause: pd.DataFrame,
) -> pd.DataFrame:
    frequency_proxy = {
        "Internal airline-operational disruption": float((airport_month["airline_delay_minutes"] + airport_month["late_aircraft_delay_minutes"]).mean()),
        "System/NAS disruption": float(airport_month["nas_delay_minutes"].mean()),
        "External shock disruption": float((airport_month["weather_delay_minutes"] + airport_month["security_delay_minutes"]).mean()),
    }
    severity_proxy = {
        "Internal airline-operational disruption": float(airport_month["internal_ops_delay_minutes"].mean()),
        "System/NAS disruption": float(airport_month["system_ops_delay_minutes"].mean()),
        "External shock disruption": float(airport_month["external_ops_delay_minutes"].mean()),
    }

    def bucket(values: dict[str, float]) -> dict[str, str]:
        pct = pd.Series(values).rank(pct=True)
        out: dict[str, str] = {}
        for key, value in pct.items():
            if value <= 1 / 3:
                out[key] = "Low"
            elif value <= 2 / 3:
                out[key] = "Medium"
            else:
                out[key] = "High"
        return out

    freq_bucket = bucket(frequency_proxy)
    sev_bucket = bucket(severity_proxy)
    heatmap = pd.DataFrame(
        [
            {
                "risk_block": risk,
                "frequency_proxy_minutes": round(frequency_proxy[risk], 2),
                "severity_proxy_minutes": round(severity_proxy[risk], 2),
                "frequency_bucket": freq_bucket[risk],
                "severity_bucket": sev_bucket[risk],
            }
            for risk in frequency_proxy
        ]
    )
    heatmap.to_csv(PROCESSED_DIR / "jfk_risk_heatmap.csv", index=False)

    peak_month = airport_month.loc[airport_month["disruption_impact_minutes"].idxmax()]
    top_cause = cause.iloc[0]
    md = f"""# JFK Risk Identification Summary

## SWOT

- Strengths: JFK has a diversified airline mix and repeatable monthly operations, so disruption patterns can be compared consistently across carriers and time.
- Weaknesses: Internal and late-aircraft delays create knock-on effects, and cancellation surges can rapidly inflate delay-equivalent minutes.
- Opportunities: Better turnaround buffers, crew and maintenance coordination, and stronger congestion planning can target the largest modeled risk blocks.
- Threats: Weather and NAS shocks create tail-risk months where aggregate operational impact rises sharply.

## Risk Heat Map Summary

{heatmap.to_markdown(index=False)}

## 5 Whys For The Peak Disruption Month

- Peak airport-month by delay-equivalent minutes: {peak_month['period_label']} with {peak_month['disruption_impact_minutes']:,.0f} minutes.
- Dominant delay-severity driver in the overall sample: {top_cause['cause']} ({top_cause['share_of_total_delay_minutes']:.1%} of total delay minutes).
- Why 1: Too many arrivals accumulated delay-equivalent minutes in the same month.
- Why 2: Late aircraft and airline-driven disruptions propagated through the schedule.
- Why 3: Aircraft rotation and recovery buffers were not strong enough to absorb upstream delays.
- Why 4: Once the system entered a disrupted state, cancellations and long delays amplified the monthly impact.
- Why 5: Operational resilience depended on coordination quality, recovery playbooks, and contingency capacity rather than on a single airline-level fix.

## Management Focus

- Internal disruption: strengthen turnaround buffers, crew coordination, and maintenance recovery planning.
- System/NAS disruption: improve congestion mitigation and airport-wide coordination during high-pressure months.
- External shock disruption: maintain weather contingency protocols and faster recovery actions for cancellation-heavy periods.
"""
    RISK_IDENTIFICATION_SUMMARY.write_text(md, encoding="utf-8")
    return heatmap


def build_methodology_summary(
    used_files: list[str],
    cleaned_summary: dict[str, int],
    selected_rule: str,
    stable_transition: bool,
    airport_month: pd.DataFrame,
) -> None:
    years = sorted(int(year) for year in airport_month["year"].unique())
    md = f"""# JFK Methodology Notes

## Data scope

- Raw BTS files detected: {", ".join(used_files)}
- Years present in the current workspace sample: {years}
- Rows before cleaning: {cleaned_summary['initial_rows']}
- Rows removed for no operations: {cleaned_summary['removed_for_no_operations']}
- Rows removed for logic inconsistencies: {cleaned_summary['removed_for_logic_issue']}
- Final airline-month rows: {cleaned_summary['final_rows']}

## Loss proxy

- All main results use operational impact in minutes rather than monetary loss.
- Delay-equivalent minutes are defined as:
  `total arrival delay minutes + cancelled arrivals * {int(CANCELLATION_PENALTY_MINUTES)} + diverted arrivals * {int(DIVERSION_PENALTY_MINUTES)}`
- This keeps the analysis close to the observed BTS data and avoids unsupported financial-cost assumptions.

## Frequency models

- Candidate count models: Poisson and Negative Binomial.
- Target variables: delayed arrivals, cancellations, diversions, and internal/system/external disruption counts.

## Severity models

- Candidate severity models: Lognormal and Weibull.
- Target variables: average delay minutes per delayed flight and cause-specific average minutes per event.

## Dependence framework

- Airport-month states are classified into `normal` and `disrupted`.
- State rule selected: `{selected_rule}`
- Transition matrix estimated: {'Yes' if stable_transition else 'No'}
- If transition estimation is unstable, the project keeps a simplified disruption-state framework instead of forcing a full Markov chain claim.

## Aggregate risk

- Scenarios simulated: base, disruption_stress, weather_shock, holiday_peak.
- Main reported metrics: expected impact, VaR95, TVaR95, VaR99, TVaR99.
"""
    METHODOLOGY_SUMMARY.write_text(md, encoding="utf-8")


def build_summary_markdown(
    used_files: list[str],
    cleaned_summary: dict[str, int],
    monthly: pd.DataFrame,
    cause: pd.DataFrame,
    airport_month: pd.DataFrame,
    comparison: pd.DataFrame,
    transition_matrix: pd.DataFrame,
    stable_transition: bool,
    frequency_df: pd.DataFrame,
    severity_df: pd.DataFrame,
    scenario_metrics: pd.DataFrame,
) -> None:
    selected_frequency = frequency_df.loc[frequency_df["selected"], ["variable", "distribution", "aic"]]
    selected_severity = severity_df.loc[severity_df["selected"], ["variable", "distribution", "aic"]]
    peak_month = airport_month.loc[airport_month["disruption_impact_minutes"].idxmax()]
    top_cause = cause.iloc[0]
    selected_rule = comparison.loc[comparison["selected"], "rule_name"].iloc[0]

    lines = [
        "# JFK Statistical Modeling Summary",
        "",
        "## Current data coverage",
        f"- Raw files used: {', '.join(used_files)}",
        f"- Years present in current sample: {', '.join(str(int(year)) for year in sorted(monthly['year'].unique()))}",
        f"- Airline-month rows after cleaning: {cleaned_summary['final_rows']}",
        f"- Airport-month observations available: {len(airport_month)}",
        "",
        "## Risk identification highlights",
        f"- Peak airport-month by delay-equivalent minutes: {peak_month['period_label']} with {peak_month['disruption_impact_minutes']:,.0f} minutes.",
        f"- Largest delay-severity driver across the sample: {top_cause['cause']} at {top_cause['share_of_total_delay_minutes']:.1%} of total delay minutes.",
        f"- Selected state-identification rule: {selected_rule}",
        "",
        "## Selected frequency models",
    ]
    for row in selected_frequency.itertuples(index=False):
        lines.append(f"- {row.variable}: {row.distribution} (AIC {row.aic:.2f})")

    lines.extend(["", "## Selected severity models"])
    for row in selected_severity.itertuples(index=False):
        lines.append(f"- {row.variable}: {row.distribution} (AIC {row.aic:.2f})")

    lines.extend(["", "## Dependence status"])
    if stable_transition:
        lines.append("- Two-state monthly transition matrix estimated successfully.")
        for row in transition_matrix.itertuples(index=False):
            lines.append(f"- {row.from_state} -> {row.to_state}: {row.transition_probability:.3f} (count {int(row.transition_count)})")
    else:
        lines.append("- Transition matrix not estimated because the current month-level sample is still too short or unstable.")
        lines.append("- The project therefore retains a simplified normal/disrupted state framework for now.")

    lines.extend(["", "## Aggregate risk scenarios"])
    for row in scenario_metrics.itertuples(index=False):
        lines.append(
            f"- {row.scenario}: expected {row.expected_impact_minutes:,.0f} minutes, "
            f"VaR95 {row.var_95_minutes:,.0f}, TVaR95 {row.tvar_95_minutes:,.0f}, "
            f"VaR99 {row.var_99_minutes:,.0f}, TVaR99 {row.tvar_99_minutes:,.0f}"
        )

    lines.extend(
        [
            "",
            "## Recommendation mapping",
            "- Internal disruption results support turnaround buffers, crew coordination, and maintenance recovery planning.",
            "- System/NAS results support congestion mitigation and airport-wide coordination for high-pressure months.",
            "- External shock results support stronger weather contingency and recovery protocols during cancellation-heavy periods.",
        ]
    )
    STAT_SUMMARY.write_text("\n".join(lines), encoding="utf-8")


def build_charts(
    monthly: pd.DataFrame,
    seasonal: pd.DataFrame,
    heatmap: pd.DataFrame,
    model_df: pd.DataFrame,
    airport_month: pd.DataFrame,
    simulations: pd.DataFrame,
    transition_matrix: pd.DataFrame,
    stable_transition: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(monthly["period"], monthly["delayed_arrivals_15_plus"], color=CHART_STYLE["primary"], linewidth=2.5, label="Delayed arrivals")
    ax.plot(monthly["period"], monthly["cancelled_arrivals"], color=CHART_STYLE["secondary"], linewidth=2.5, label="Cancelled arrivals")
    ax.set_title("JFK Multi-Year Delay and Cancellation Trend")
    ax.set_xlabel("Month")
    ax.set_ylabel("Flights")
    ax.legend(frameon=False)
    fig.tight_layout()
    save_chart(fig, "chart_1_multiyear_monthly_trend")

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=seasonal, x="month_label", y="mean_delay_rate", color=CHART_STYLE["accent"], ax=ax)
    ax.set_title("Average Delay Rate by Month of Year")
    ax.set_xlabel("Month")
    ax.set_ylabel("Average delay rate")
    fig.tight_layout()
    bucket_to_num = {"Low": 1, "Medium": 2, "High": 3}
    fig, ax = plt.subplots(figsize=(8, 4.8))
    sns.heatmap(
        pd.DataFrame(
            {
                "Frequency": heatmap.set_index("risk_block")["frequency_bucket"].map(bucket_to_num),
                "Severity": heatmap.set_index("risk_block")["severity_bucket"].map(bucket_to_num),
            }
        ),
        annot=pd.DataFrame(
            {
                "Frequency": heatmap.set_index("risk_block")["frequency_bucket"],
                "Severity": heatmap.set_index("risk_block")["severity_bucket"],
            }
        ),
        fmt="",
        cmap="YlOrRd",
        cbar=False,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Operational Risk Heat Map")
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.tight_layout()
    save_chart(fig, "chart_3_risk_heatmap")

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(
        data=model_df,
        x="operational_state",
        y="disruption_impact_minutes",
        hue="operational_state",
        palette={"normal": CHART_STYLE["primary"], "disrupted": CHART_STYLE["secondary"]},
        legend=False,
        ax=ax,
    )
    ax.set_title("Delay-Equivalent Minutes by Airport State")
    ax.set_xlabel("Airport-month state")
    ax.set_ylabel("Airline-month impact minutes")
    fig.tight_layout()
    save_chart(fig, "chart_4_disruption_state_impact")

    fig, ax = plt.subplots(figsize=(14, 5))
    temp = airport_month.copy()
    temp["state_flag"] = np.where(temp["operational_state"] == "disrupted", 1, 0)
    sns.barplot(data=temp, x="period_label", y="state_flag", color=CHART_STYLE["accent"], ax=ax)
    ax.set_title("Airport-Month Disruption State Timeline")
    ax.set_xlabel("Period")
    ax.set_ylabel("Disrupted = 1")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    save_chart(fig, "chart_5_monthly_disrupted_share")

    fig, ax = plt.subplots(figsize=(10, 6))
    for scenario_name, group in simulations.groupby("scenario"):
        sns.kdeplot(group["aggregate_annual_impact_minutes"], ax=ax, label=scenario_name, linewidth=2)
    ax.set_title("Annual Aggregate Operational Impact by Scenario")
    ax.set_xlabel("Annual delay-equivalent minutes")
    ax.set_ylabel("Density")
    ax.legend(frameon=False)
    fig.tight_layout()
    save_chart(fig, "chart_6_aggregate_risk_scenarios")

    if stable_transition:
        pivot = transition_matrix.pivot(index="from_state", columns="to_state", values="transition_probability")
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="Blues", linewidths=0.5, ax=ax)
        ax.set_title("Monthly Transition Matrix")
        ax.set_xlabel("To state")
        ax.set_ylabel("From state")
        fig.tight_layout()
        save_chart(fig, "chart_7_markov_transition_matrix")


def main() -> None:
    ensure_output_directories()
    apply_style()

    raw_df, used_files = load_multiyear_raw()
    _, cleaned, cleaning_summary = build_readable_and_core(raw_df)
    monthly, seasonal, cause, _ = build_summaries(cleaned)
    airport_month, comparison, selected_rule, stable_transition = build_airport_state_panel(monthly)
    transition_matrix = estimate_transition_matrix(airport_month, stable_transition)
    model_df = build_modeling_input(cleaned, airport_month)
    frequency_df = build_frequency_models(model_df)
    severity_df = build_severity_models(model_df)
    simulations, scenario_metrics = simulate_aggregate_risk(
        model_df,
        airport_month,
        transition_matrix,
        stable_transition,
        frequency_df,
        severity_df,
    )
    heatmap = build_risk_identification_assets(monthly, airport_month, cause)
    build_charts(monthly, seasonal, heatmap, model_df, airport_month, simulations, transition_matrix, stable_transition)

    print("Multi-year operational-risk outputs written to:")
    print(f"  Readable full dataset: {READABLE_FULL}")
    print(f"  Core cleaned dataset: {CORE_CLEANED}")
    print(f"  Airport-month state panel: {AIRPORT_MONTH_PANEL}")
    print(f"  Frequency summary: {FREQUENCY_SUMMARY}")
    print(f"  Severity summary: {SEVERITY_SUMMARY}")
    print(f"  Aggregate scenario metrics: {SCENARIO_METRICS}")
    print(f"  Required multi-year charts: {CHART_DIR}")


if __name__ == "__main__":
    main()
