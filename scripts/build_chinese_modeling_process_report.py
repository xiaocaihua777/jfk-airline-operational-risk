from __future__ import annotations

import html
import os
from pathlib import Path
from textwrap import dedent

BASE_DIR = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(BASE_DIR / ".cache" / "matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


PROCESSED_DIR = BASE_DIR / "data" / "processed"
REPORTS_DIR = BASE_DIR / "reports"
CHART_DIR = REPORTS_DIR / "charts"
DASHBOARD_DIR = REPORTS_DIR / "dashboard"
SUMMARY_DIR = REPORTS_DIR / "summary"

FREQUENCY_SUMMARY = PROCESSED_DIR / "jfk_frequency_model_summary.csv"
SEVERITY_SUMMARY = PROCESSED_DIR / "jfk_severity_model_summary.csv"
STATE_RULE_COMPARISON = PROCESSED_DIR / "jfk_state_definition_comparison.csv"
STATE_TRANSITIONS = PROCESSED_DIR / "jfk_markov_transition_matrix.csv"
SCENARIO_METRICS = PROCESSED_DIR / "jfk_aggregate_risk_scenario_metrics.csv"
MODELING_INPUT = PROCESSED_DIR / "jfk_airline_month_modeling_input.csv"
AIRPORT_MONTH_PANEL = PROCESSED_DIR / "jfk_airport_month_state_panel.csv"
RISK_HEATMAP = PROCESSED_DIR / "jfk_risk_heatmap.csv"
CAUSE_SUMMARY = PROCESSED_DIR / "jfk_delay_cause_summary.csv"

OUTPUT_HTML = DASHBOARD_DIR / "jfk_modeling_process_report_zh.html"
OUTPUT_GUIDE = SUMMARY_DIR / "jfk_modeling_process_report_pdf_export_guide.md"

FREQUENCY_AIC_CHART = "chart_8_frequency_aic_comparison"
SEVERITY_AIC_CHART = "chart_9_severity_aic_comparison"
STATE_RULE_CHART = "chart_10_state_rule_comparison"
FREQUENCY_AIC_CHART_EN = "chart_8_frequency_aic_comparison_en"
SEVERITY_AIC_CHART_EN = "chart_9_severity_aic_comparison_en"
STATE_RULE_CHART_EN = "chart_10_state_rule_comparison_en"

CHART_STYLE = {
    "primary": "#0F4C5C",
    "secondary": "#C66A2B",
    "accent": "#2A9D8F",
    "ink": "#20323B",
    "muted": "#61727B",
    "sand": "#F5F1E8",
    "line": "#D5CEC0",
    "danger": "#B24A48",
    "gold": "#C6A15B",
}

FREQUENCY_VARIABLE_LABELS = {
    "total_delay_count": "总延误班次",
    "cancellation_count": "取消班次",
    "diversion_count": "备降班次",
    "internal_ops_count": "内部运营扰动次数",
    "system_ops_count": "系统 / NAS 扰动次数",
    "external_ops_count": "外部冲击扰动次数",
}

FREQUENCY_VARIABLE_LABELS_EN = {
    "total_delay_count": "Total delayed arrivals",
    "cancellation_count": "Cancelled arrivals",
    "diversion_count": "Diverted arrivals",
    "internal_ops_count": "Internal disruption counts",
    "system_ops_count": "System / NAS disruption counts",
    "external_ops_count": "External shock disruption counts",
}

SEVERITY_VARIABLE_LABELS = {
    "total_avg_delay_minutes": "总体平均延误分钟",
    "internal_avg_delay_minutes": "内部扰动平均分钟",
    "system_avg_delay_minutes": "系统 / NAS 平均分钟",
    "external_avg_delay_minutes": "外部冲击平均分钟",
}

SEVERITY_VARIABLE_LABELS_EN = {
    "total_avg_delay_minutes": "Overall average delay minutes",
    "internal_avg_delay_minutes": "Internal average delay minutes",
    "system_avg_delay_minutes": "System / NAS average delay minutes",
    "external_avg_delay_minutes": "External shock average delay minutes",
}

DISTRIBUTION_LABELS = {
    "poisson": "Poisson",
    "negative_binomial": "Negative Binomial",
    "lognorm": "Lognormal",
    "weibull_min": "Weibull",
}

STATE_RULE_LABELS = {
    "impact_q75": "impact_q75",
    "delay_cancel_impact_composite": "delay_cancel_impact_composite",
}

SCENARIO_LABELS = {
    "base": "基准情景",
    "disruption_stress": "扰动加压情景",
    "weather_shock": "天气冲击情景",
    "holiday_peak": "节假日高峰情景",
}

COURSE_MAPPING = [
    ("Lecture 1", "风险分类、SWOT、heat map、5 Whys"),
    ("Lecture 3", "severity distributions、VaR / TVaR"),
    ("Lecture 4", "Poisson vs Negative Binomial"),
    ("Lecture 5", "aggregate impact"),
    ("Lecture 7", "state-based dependence、transition matrix"),
]

RISK_METHOD_CARDS = [
    (
        "Internal / internal airline-operational disruption",
        "由 <b>Airline</b> 与 <b>Late Aircraft</b> 组合而成，强调班表恢复、机组衔接、维修与前序航班传导。",
    ),
    (
        "System / system disruption",
        "对应 <b>NAS</b> 相关扰动，强调机场容量、空域拥堵与系统级协调压力。",
    ),
    (
        "External / external shock disruption",
        "由 <b>Weather</b> 与 <b>Security</b> 组合，强调外部冲击和恢复韧性，而不是单一航司内部效率。",
    ),
]

DISTRIBUTION_EXPLANATIONS = [
    (
        "Poisson",
        "适合先作为计数型事件的最基础参照。它假设平均数和方差接近，因此可以用来检验数据是否接近“均匀随机到达”。",
    ),
    (
        "Negative Binomial",
        "当延误、取消或备降在某些月份明显聚集时，方差通常会高于平均数。它比 Poisson 更能描述高波动和事件扎堆的情形。",
    ),
    (
        "Lognormal",
        "适合描述多数事件不算极端、但少量事件会拉出长尾的分钟损失。对右偏分布的平均严重度尤其常见。",
    ),
    (
        "Weibull",
        "适合在分钟损失随强度阶段性变化时提供更灵活的形状。它常被拿来与 Lognormal 比较，以确认哪种尾部和主体形状更贴近样本。",
    ),
]


def ensure_output_directories() -> None:
    for directory in (CHART_DIR, DASHBOARD_DIR, SUMMARY_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def apply_style() -> None:
    sns.set_theme(
        style="whitegrid",
        rc={
            "axes.facecolor": "#FBF8F1",
            "figure.facecolor": "#FBF8F1",
            "axes.edgecolor": CHART_STYLE["line"],
            "grid.color": CHART_STYLE["line"],
            "axes.labelcolor": CHART_STYLE["ink"],
            "text.color": CHART_STYLE["ink"],
            "xtick.color": CHART_STYLE["ink"],
            "ytick.color": CHART_STYLE["ink"],
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"],
            "axes.unicode_minus": False,
        },
    )


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required input file: {path}")
    return pd.read_csv(path)


def to_bool(value: object) -> bool:
    return str(value).strip().lower() == "true"


def format_int(value: float | int) -> str:
    return f"{int(round(float(value))):,}"


def format_float(value: float | int, digits: int = 2) -> str:
    return f"{float(value):,.{digits}f}"


def format_pct(value: float | int, digits: int = 1) -> str:
    return f"{float(value) * 100:.{digits}f}%"


def save_chart(fig: plt.Figure, stem: str) -> None:
    fig.savefig(CHART_DIR / f"{stem}.png", dpi=320, bbox_inches="tight")
    fig.savefig(CHART_DIR / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def build_frequency_table(frequency_df: pd.DataFrame) -> pd.DataFrame:
    freq = frequency_df.copy()
    freq["selected"] = freq["selected"].map(to_bool)
    freq["aic"] = freq["aic"].astype(float)
    freq["overall_mean"] = freq["overall_mean"].astype(float)
    freq["overall_variance"] = freq["overall_variance"].astype(float)
    freq["normal_state_mean"] = freq["normal_state_mean"].astype(float)
    freq["disrupted_state_mean"] = freq["disrupted_state_mean"].astype(float)
    rows: list[dict[str, object]] = []
    for variable, group in freq.groupby("variable", sort=False):
        ordered = group.sort_values("aic").reset_index(drop=True)
        best = ordered.iloc[0]
        alt = ordered.iloc[1]
        rows.append(
            {
                "variable": variable,
                "label": FREQUENCY_VARIABLE_LABELS[variable],
                "selected_model": DISTRIBUTION_LABELS[str(best["distribution"])],
                "selected_model_raw": str(best["distribution"]),
                "selected_aic": float(best["aic"]),
                "comparison_model": DISTRIBUTION_LABELS[str(alt["distribution"])],
                "comparison_model_raw": str(alt["distribution"]),
                "comparison_aic": float(alt["aic"]),
                "delta_aic": float(alt["aic"]) - float(best["aic"]),
                "overall_mean": float(best["overall_mean"]),
                "overall_variance": float(best["overall_variance"]),
                "normal_state_mean": float(best["normal_state_mean"]),
                "disrupted_state_mean": float(best["disrupted_state_mean"]),
            }
        )
    return pd.DataFrame(rows)


def build_severity_table(severity_df: pd.DataFrame) -> pd.DataFrame:
    sev = severity_df.copy()
    sev["selected"] = sev["selected"].map(to_bool)
    sev["aic"] = sev["aic"].astype(float)
    sev["sample_size"] = sev["sample_size"].astype(int)
    sev["normal_state_mean"] = sev["normal_state_mean"].astype(float)
    sev["disrupted_state_mean"] = sev["disrupted_state_mean"].astype(float)
    rows: list[dict[str, object]] = []
    for variable, group in sev.groupby("variable", sort=False):
        ordered = group.sort_values("aic").reset_index(drop=True)
        best = ordered.iloc[0]
        alt = ordered.iloc[1]
        rows.append(
            {
                "variable": variable,
                "label": SEVERITY_VARIABLE_LABELS[variable],
                "selected_model": DISTRIBUTION_LABELS[str(best["distribution"])],
                "selected_model_raw": str(best["distribution"]),
                "selected_aic": float(best["aic"]),
                "comparison_model": DISTRIBUTION_LABELS[str(alt["distribution"])],
                "comparison_model_raw": str(alt["distribution"]),
                "comparison_aic": float(alt["aic"]),
                "delta_aic": float(alt["aic"]) - float(best["aic"]),
                "sample_size": int(best["sample_size"]),
                "normal_state_mean": float(best["normal_state_mean"]),
                "disrupted_state_mean": float(best["disrupted_state_mean"]),
            }
        )
    return pd.DataFrame(rows)


def build_state_rule_table(state_rule_df: pd.DataFrame) -> pd.DataFrame:
    rules = state_rule_df.copy()
    rules["selected"] = rules["selected"].map(to_bool)
    for column in ("disrupted_share", "yearly_disrupted_share_std", "selection_score"):
        rules[column] = rules[column].astype(float)
    rules["disrupted_months"] = rules["disrupted_months"].astype(int)
    rules["label"] = rules["rule_name"].map(STATE_RULE_LABELS)
    return rules.sort_values("selection_score").reset_index(drop=True)


def build_transition_table(transition_df: pd.DataFrame) -> pd.DataFrame:
    transition = transition_df.copy()
    transition["transition_probability"] = transition["transition_probability"].astype(float)
    transition["transition_count"] = transition["transition_count"].astype(int)
    return transition


def build_scenario_table(metrics_df: pd.DataFrame) -> pd.DataFrame:
    scenarios = metrics_df.copy()
    for column in (
        "expected_impact_minutes",
        "var_95_minutes",
        "tvar_95_minutes",
        "var_99_minutes",
        "tvar_99_minutes",
    ):
        scenarios[column] = scenarios[column].astype(float)
    scenarios["label"] = scenarios["scenario"].map(SCENARIO_LABELS)
    return scenarios


def build_frequency_aic_chart(freq_table: pd.DataFrame) -> None:
    chart_df = freq_table.sort_values("delta_aic", ascending=True).copy()
    fig, ax = plt.subplots(figsize=(11, 6.2))
    y_pos = np.arange(len(chart_df))
    ax.barh(y_pos, chart_df["delta_aic"], color=CHART_STYLE["primary"], alpha=0.88)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(chart_df["label"])
    ax.set_xlabel("较差模型相对最优模型的 ΔAIC")
    ax.set_title("Frequency 候选模型 AIC 差值对比（ΔAIC 越大，最优模型优势越明显）")
    max_gap = float(chart_df["delta_aic"].max())
    for idx, row in enumerate(chart_df.itertuples(index=False)):
        annotation = (
            f"{row.selected_model} 最优 | AIC={row.selected_aic:,.1f}\n"
            f"对比模型 {row.comparison_model} | ΔAIC={row.delta_aic:,.1f}"
        )
        ax.text(float(row.delta_aic) + max_gap * 0.02, idx, annotation, va="center", fontsize=9)
    fig.tight_layout()
    save_chart(fig, FREQUENCY_AIC_CHART)


def build_severity_aic_chart(sev_table: pd.DataFrame) -> None:
    chart_df = sev_table.sort_values("delta_aic", ascending=True).copy()
    fig, ax = plt.subplots(figsize=(10.2, 5.8))
    y_pos = np.arange(len(chart_df))
    ax.barh(y_pos, chart_df["delta_aic"], color=CHART_STYLE["accent"], alpha=0.9)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(chart_df["label"])
    ax.set_xlabel("较差模型相对最优模型的 ΔAIC")
    ax.set_title("Severity 候选模型 AIC 差值对比")
    max_gap = float(chart_df["delta_aic"].max())
    for idx, row in enumerate(chart_df.itertuples(index=False)):
        annotation = (
            f"{row.selected_model} 最优 | AIC={row.selected_aic:,.1f}\n"
            f"对比模型 {row.comparison_model} | ΔAIC={row.delta_aic:,.1f}"
        )
        ax.text(float(row.delta_aic) + max_gap * 0.03, idx, annotation, va="center", fontsize=9)
    fig.tight_layout()
    save_chart(fig, SEVERITY_AIC_CHART)


def build_state_rule_chart(state_rules: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 5.8))
    selected = state_rules.loc[state_rules["selected"]].iloc[0]
    for row in state_rules.itertuples(index=False):
        color = CHART_STYLE["secondary"] if row.selected else CHART_STYLE["muted"]
        size = 150 if row.selected else 110
        ax.scatter(row.disrupted_share, row.yearly_disrupted_share_std, s=size, color=color, zorder=3)
        ax.text(
            float(row.disrupted_share) + 0.008,
            float(row.yearly_disrupted_share_std) + 0.002,
            f"{row.label}\nshare={row.disrupted_share:.2f}, std={row.yearly_disrupted_share_std:.3f}",
            fontsize=9,
        )
    ax.axvline(0.25, color=CHART_STYLE["gold"], linestyle="--", linewidth=1.2, label="目标 disrupted share = 25%")
    ax.set_xlim(0.20, 0.30)
    ax.set_ylim(0.0, max(state_rules["yearly_disrupted_share_std"]) * 1.25)
    ax.set_xlabel("Disrupted share")
    ax.set_ylabel("Yearly disrupted-share std")
    ax.set_title(f"状态规则比较：{selected['label']} 更接近 25% 且跨年更稳定")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    save_chart(fig, STATE_RULE_CHART)


def card(title: str, body_html: str, tone: str = "") -> str:
    class_attr = f"card {tone}".strip()
    return (
        f'<article class="{class_attr}">'
        f'<div class="card-title">{html.escape(title)}</div>'
        f'<div class="card-body">{body_html}</div>'
        f"</article>"
    )


def metric(label: str, value: str, note: str) -> str:
    return (
        '<div class="metric-card">'
        f'<div class="metric-label">{html.escape(label)}</div>'
        f'<div class="metric-value">{value}</div>'
        f'<div class="metric-note">{note}</div>'
        "</div>"
    )


def image_card(title: str, image_src: str, caption: str, image_class: str = "") -> str:
    class_attr = f"image-frame {image_class}".strip()
    return (
        '<article class="card image-card">'
        f'<div class="card-title">{html.escape(title)}</div>'
        f'<div class="{class_attr}"><img src="{image_src}" alt="{html.escape(title)}"></div>'
        f'<div class="image-caption">{caption}</div>'
        "</article>"
    )


def render_compact_table(headers: list[str], rows: list[list[str]], table_class: str = "") -> str:
    class_attr = f"data-table {table_class}".strip()
    header_html = "".join(f"<th>{html.escape(header)}</th>" for header in headers)
    body_html = ""
    for row in rows:
        body_html += "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>"
    return f'<table class="{class_attr}"><thead><tr>{header_html}</tr></thead><tbody>{body_html}</tbody></table>'


def page(chapter: str, title: str, summary: str, content_html: str, page_index: int, total_pages: int) -> str:
    progress = int(round(((page_index + 1) / total_pages) * 100))
    return dedent(
        f"""
        <section class="report-page">
          <div class="hd">
            <div class="chapter-tag">{html.escape(chapter)}</div>
            <div class="page-title">{html.escape(title)}</div>
          </div>
          <div class="ct">
            {content_html}
          </div>
          <div class="sm">
            <span class="summary-pill">本页摘要</span>
            <span>{summary}</span>
          </div>
          <div class="ft">
            <span>JFK 中文建模过程版报告</span>
            <div class="footer-progress">
              <div class="footer-progress-bar"><span style="width:{progress}%;"></span></div>
              <span>{page_index + 1:02d}/{total_pages:02d}</span>
            </div>
          </div>
        </section>
        """
    ).strip()


def build_pdf_guide() -> str:
    return dedent(
        f"""
        # 中文建模过程版报告 PDF 导出说明

        ## 目标文件
        - HTML：`{OUTPUT_HTML.relative_to(BASE_DIR)}`
        - 建议导出 PDF 名称：`jfk_modeling_process_report_zh.pdf`

        ## 推荐浏览器
        - Microsoft Edge
        - Google Chrome

        ## 导出步骤
        1. 用 Edge 或 Chrome 打开 `reports/dashboard/jfk_modeling_process_report_zh.html`。
        2. 等待页面内图表和图片全部加载完成，再打开打印面板。
        3. 使用 `Ctrl+P`。
        4. 目标打印机选择“另存为 PDF”。
        5. 布局选择“横向”。
        6. 勾选“背景图形”。
        7. 缩放保持默认或“适合页面”，不要手动压缩到 70% 以下。
        8. 边距选择“默认”或“最小”，不要启用页眉页脚。
        9. 预览确认每页单独分页后导出。

        ## 导出时应检查的点
        - 每页都是独立页面，没有两页内容挤在同一张纸上。
        - 图表标题、表格列名、公式框没有被裁切。
        - 背景色、浅色卡片和分隔线都被保留下来。
        - 中文没有乱码。

        ## 这轮实现的边界
        - 当前流程默认走“打印友好 HTML + 浏览器导出 PDF”。
        - 本轮没有增加自动化 PDF 生成依赖。
        - 如果后续需要自动导出，可再单独增加 headless browser 或 reportlab 路线。
        """
    ).strip() + "\n"


def run_validations(
    freq_table: pd.DataFrame,
    sev_table: pd.DataFrame,
    state_rules: pd.DataFrame,
    transitions: pd.DataFrame,
    scenarios: pd.DataFrame,
    html_output: str,
) -> None:
    page_count = html_output.count('class="report-page"')
    if page_count != 14:
        raise ValueError(f"Expected 14 report pages, found {page_count}.")
    if len(freq_table) != 6:
        raise ValueError(f"Expected 6 frequency result rows, found {len(freq_table)}.")
    if len(sev_table) != 4:
        raise ValueError(f"Expected 4 severity result rows, found {len(sev_table)}.")
    if len(state_rules) != 2:
        raise ValueError(f"Expected 2 state-rule rows, found {len(state_rules)}.")
    if len(transitions) != 4:
        raise ValueError(f"Expected 4 transition rows, found {len(transitions)}.")
    if len(scenarios) != 4:
        raise ValueError(f"Expected 4 scenario rows, found {len(scenarios)}.")
    if not OUTPUT_HTML.exists():
        raise FileNotFoundError(f"Expected HTML output not found: {OUTPUT_HTML}")
    if not OUTPUT_GUIDE.exists():
        raise FileNotFoundError(f"Expected guide output not found: {OUTPUT_GUIDE}")
    for stem in (FREQUENCY_AIC_CHART, SEVERITY_AIC_CHART, STATE_RULE_CHART):
        for suffix in ("png", "svg"):
            path = CHART_DIR / f"{stem}.{suffix}"
            if not path.exists():
                raise FileNotFoundError(f"Expected chart output not found: {path}")


def build_report_html(
    freq_table: pd.DataFrame,
    sev_table: pd.DataFrame,
    state_rules: pd.DataFrame,
    transitions: pd.DataFrame,
    scenarios: pd.DataFrame,
    model_df: pd.DataFrame,
    airport_panel: pd.DataFrame,
    risk_heatmap: pd.DataFrame,
    cause_summary: pd.DataFrame,
) -> str:
    total_pages = 14
    years = sorted(int(year) for year in airport_panel["year"].unique())
    airline_months = len(model_df)
    airport_months = len(airport_panel)
    disrupted_months = int((airport_panel["operational_state"] == "disrupted").sum())
    total_flights = float(airport_panel["total_arrival_flights"].sum())
    mean_delay_rate = float(airport_panel["delay_rate"].mean())
    mean_cancellation_rate = float(airport_panel["cancellation_rate"].mean())
    total_disruption_impact = float(airport_panel["disruption_impact_minutes"].sum())
    selected_rule = state_rules.loc[state_rules["selected"]].iloc[0]
    peak_month = airport_panel.loc[airport_panel["disruption_impact_minutes"].idxmax()]
    peak_delay_rate_month = airport_panel.loc[airport_panel["delay_rate"].idxmax()]
    top_cause = cause_summary.sort_values("share_of_total_delay_minutes", ascending=False).iloc[0]
    total_delay_minutes = float(cause_summary["delay_minutes"].astype(float).sum())
    cause_share_map = {
        str(row["cause"]): float(row["share_of_total_delay_minutes"])
        for row in cause_summary.to_dict(orient="records")
    }
    internal_share = cause_share_map.get("Airline", 0.0) + cause_share_map.get("Late Aircraft", 0.0)
    system_share = cause_share_map.get("NAS", 0.0)
    external_share = cause_share_map.get("Weather", 0.0) + cause_share_map.get("Security", 0.0)
    freq_all_nb = int((freq_table["selected_model_raw"] == "negative_binomial").sum())
    sev_lognorm = int((sev_table["selected_model_raw"] == "lognorm").sum())
    sev_weibull = int((sev_table["selected_model_raw"] == "weibull_min").sum())
    freq_biggest_gap = freq_table.sort_values("delta_aic", ascending=False).iloc[0]
    sev_biggest_gap = sev_table.sort_values("delta_aic", ascending=False).iloc[0]
    p_nd = float(
        transitions.loc[
            (transitions["from_state"] == "normal") & (transitions["to_state"] == "disrupted"),
            "transition_probability",
        ].iloc[0]
    )
    p_dd = float(
        transitions.loc[
            (transitions["from_state"] == "disrupted") & (transitions["to_state"] == "disrupted"),
            "transition_probability",
        ].iloc[0]
    )
    p_dn = float(
        transitions.loc[
            (transitions["from_state"] == "disrupted") & (transitions["to_state"] == "normal"),
            "transition_probability",
        ].iloc[0]
    )
    transition_ratio = p_dd / p_nd if p_nd else float("nan")
    scenario_peak = scenarios.sort_values("var_95_minutes", ascending=False).iloc[0]
    base_expected = float(scenarios.loc[scenarios["scenario"] == "base", "expected_impact_minutes"].iloc[0])
    base_var95 = float(scenarios.loc[scenarios["scenario"] == "base", "var_95_minutes"].iloc[0])

    risk_rows = []
    for row in risk_heatmap.itertuples(index=False):
        risk_rows.append(
            [
                html.escape(str(row.risk_block)),
                html.escape(str(row.frequency_bucket)),
                html.escape(str(row.severity_bucket)),
                format_int(row.frequency_proxy_minutes),
            ]
        )

    freq_rows = []
    for row in freq_table.itertuples(index=False):
        freq_rows.append(
            [
                html.escape(row.label),
                format_float(row.selected_aic, 1),
                format_float(row.comparison_aic, 1),
                html.escape(row.selected_model),
                format_float(row.delta_aic, 1),
            ]
        )

    sev_rows = []
    for row in sev_table.itertuples(index=False):
        sev_rows.append(
            [
                html.escape(row.label),
                format_float(row.selected_aic, 1),
                format_float(row.comparison_aic, 1),
                html.escape(row.selected_model),
                format_float(row.delta_aic, 1),
            ]
        )

    state_rows = []
    for row in state_rules.itertuples(index=False):
        state_rows.append(
            [
                f'<span class="table-strong">{html.escape(row.label)}</span>',
                format_pct(row.disrupted_share, 1),
                format_float(row.yearly_disrupted_share_std, 3),
                format_float(row.selection_score, 3),
                "是" if row.selected else "否",
            ]
        )

    transition_rows = []
    for row in transitions.itertuples(index=False):
        transition_rows.append(
            [
                f"{row.from_state} → {row.to_state}",
                format_float(row.transition_probability, 3),
                format_int(row.transition_count),
            ]
        )

    scenario_rows = []
    for row in scenarios.itertuples(index=False):
        scenario_rows.append(
            [
                html.escape(row.label),
                format_int(row.expected_impact_minutes),
                format_int(row.var_95_minutes),
                format_int(row.tvar_95_minutes),
            ]
        )

    lecture_rows = []
    for lecture, topic in COURSE_MAPPING:
        lecture_rows.append([html.escape(lecture), html.escape(topic)])

    risk_mapping_rows = [
        ["Airline + Late Aircraft", "internal", f"{format_pct(internal_share, 1)} of total delay minutes"],
        ["NAS", "system", f"{format_pct(system_share, 1)} of total delay minutes"],
        ["Weather + Security", "external", f"{format_pct(external_share, 1)} of total delay minutes"],
    ]

    freq_detail_rows = []
    for row in freq_table.itertuples(index=False):
        freq_detail_rows.append(
            [
                html.escape(row.label),
                format_float(row.overall_mean, 1),
                format_float(row.overall_variance, 1),
                format_float(row.disrupted_state_mean - row.normal_state_mean, 1),
            ]
        )

    sev_detail_rows = []
    for row in sev_table.itertuples(index=False):
        sev_detail_rows.append(
            [
                html.escape(row.label),
                format_int(row.sample_size),
                format_float(row.normal_state_mean, 1),
                format_float(row.disrupted_state_mean, 1),
            ]
        )

    objective_rows = [
        [
            "计数数据方差远高于均值",
            "6 个 frequency 变量全部选中 Negative Binomial",
            "扰动事件明显不是均匀、独立、低波动到达",
        ],
        [
            "internal 分钟占比约 71.3%",
            "风险分类后 internal 落在高频高严重度区",
            "内部运行链条确实是主要风险源，不是人为硬分",
        ],
        [
            f"disrupted 月份 {format_int(disrupted_months)} / {format_int(airport_months)}",
            f"P(D→D)={format_float(p_dd,3)} 高于 P(N→D)={format_float(p_nd,3)}",
            "状态依赖与持续性具有经验性支持",
        ],
        [
            "节假日高峰与高压情景更危险",
            f"{scenario_peak['label']} 的 VaR95 最高",
            "情景结果与运营直觉一致，不是反常输出",
        ],
    ]

    operator_rows = [
        ["航司内部运行", "加强 turnaround buffer、机组与维修恢复", "internal 风险占主导，且 delay / late aircraft 是主要来源"],
        ["机场 / NAS 协同", "在高压月份提前协调容量与时隙", "system 风险在 severity 上有明显上移"],
        ["天气与节假日应对", "在 holiday / weather 情景前布置 contingency", "aggregate risk 在高峰与天气冲击下显著抬升"],
        ["月度风险监控", "监控 delay rate、cancellation rate、impact 指标", "这些指标直接进入 composite state rule"],
    ]

    p00_content = dedent(
        f"""
        <div class="cover-page">
          <div class="cover-hero">
            <div class="cover-eyebrow">JFK Operational Risk Project · 中文过程解释版</div>
            <h1>从模型为什么这样选，到结果应该怎样解释</h1>
            <p>
              这份报告不只展示最终结论，而是把 <b>候选分布</b>、<b>AIC 选择逻辑</b>、
              <b>状态划分规则</b> 与 <b>两状态 Markov 依赖</b> 串成一条完整叙事链，
              让组员能够直接理解“为什么是这些模型、为什么这些结果可信、为什么解释要克制”。
            </p>
          </div>
          <div class="cover-metrics">
            {metric("样本年份", f"{years[0]}–{years[-1]}", "连续多年机场月度样本")}
            {metric("airline-month", format_int(airline_months), "频率与严重度建模主粒度")}
            {metric("airport-month", format_int(airport_months), "状态划分与转移矩阵粒度")}
            {metric("当前主线", "过程 + 结果", "面向组内解释与讨论")}
          </div>
          <div class="cover-band">
            <div class="band-card">
              <div class="band-label">报告对象</div>
              <div class="band-value">内部讨论 / 课程对齐 / 后续 submission 前置整理</div>
            </div>
            <div class="band-card">
              <div class="band-label">证据口径</div>
              <div class="band-value">AIC 相对优选，状态依赖为经验性证据，不夸大为强统计证明</div>
            </div>
          </div>
        </div>
        """
    ).strip()

    p01_content = dedent(
        f"""
        <div class="layout layout-i">
          <div class="metrics-row">
            {metric("项目目标", "解释建模过程", "从选择逻辑到结果解读全部中文化")}
            {metric("数据主粒度", "airline-month", "用于 frequency / severity")}
            {metric("状态粒度", "airport-month", "用于 state / transition")}
            {metric("状态月份", format_int(disrupted_months), "selected rule 下的 disrupted 月份")}
          </div>
          <div class="two-col">
            <div class="stack">
              {card("项目目标与损失口径", "<p>本项目关注的不是单次航班预测，而是 <b>JFK 机场多年度运营扰动风险</b>。我们真正要回答的是：扰动多久会发生一次、发生后通常有多严重、状态会不会持续，以及这些信息汇总后 annual impact 大约会落在哪个范围。</p><p>所有主结果统一用 <b>delay-equivalent minutes</b> 表达。这样做的好处是：第一，分钟损失直接来自 BTS 数据；第二，取消和备降可以通过固定 penalty 纳入；第三，避免引入难以 defend 的 monetary assumption。</p><div class='formula-box'>delay-equivalent minutes = total arrival delay minutes + 180 × cancelled arrivals + 240 × diverted arrivals</div>", "primary")}
              {card("数据结构为什么分两层", "<p><b>airline-month</b> 是分布拟合层。它保留不同航司、不同月份之间的差异，因此适合估计 count 的 frequency 以及平均分钟损失的 severity。</p><p><b>airport-month</b> 是状态解释层。它把机场整体压力汇总成一个月度状态，用来回答“这个月是正常还是扰动”“扰动后下个月是否更可能继续扰动”。</p><p>这两层结构分开后，报告逻辑会更清楚：先做个体层的分布选择，再做系统层的状态依赖，而不是把所有问题塞进同一张表里。</p>")}
            </div>
            <div class="stack">
              {image_card("多年度运营波动总览", "../charts/chart_1_multiyear_monthly_trend.png", "复用现有多年度趋势图，用来说明研究对象已经从单年描述扩展到多年过程建模。", "image-short")}
              {card("当前数据范围的关键信息", f"<ul><li>年份覆盖：<b>{years[0]}–{years[-1]}</b>，这使状态讨论不再局限于单年偶然波动。</li><li>频率/严重度样本：<b>{format_int(airline_months)}</b> 个 airline-month，足以比较候选分布。</li><li>状态样本：<b>{format_int(airport_months)}</b> 个 airport-month，其中 <b>{format_int(disrupted_months)}</b> 个被识别为 disrupted。</li><li>峰值月份：<b>{peak_month['period_label']}</b>，impact 为 <b>{format_int(peak_month['disruption_impact_minutes'])}</b> 分钟。</li></ul><p>换句话说，这份报告研究的是一个 <b>多年、月度、机场运营风险</b> 问题，而不是单个航班调度问题。</p>")}
            </div>
          </div>
        </div>
        """
    ).strip()

    risk_cards_html = "".join(card(title, body, "accent") for title, body in RISK_METHOD_CARDS)
    p02_content = dedent(
        f"""
        <div class="layout layout-g">
          <div class="two-col">
            <div class="stack">
              {card("原始 BTS cause taxonomy 是什么", "<p>原始字段来自 <b>Airline</b>、<b>Weather</b>、<b>NAS</b>、<b>Security</b>、<b>Late Aircraft</b>。这套 taxonomy 的优点是数据来源清楚、与官方表格一致，但它的缺点是更像“记录原因的数据库口径”，而不是“管理者理解风险的课程口径”。</p><p>例如 <b>Late Aircraft</b> 从数据角度是独立 cause，但从风险识别角度，它通常仍然属于内部运行链条失稳后的传导结果。</p>", "primary")}
              {card("为什么要重构成 internal / system / external", "<p>课程中的风险识别强调 <b>责任边界</b>、<b>控制手段</b> 和 <b>管理动作</b>。因此我们把 cause taxonomy 重新整理为 internal、system、external 三类，让后续建议能直接对应“航司内部改善”“机场 / NAS 协同”“外部冲击应对”三种管理语言。</p><p>这一步不是在做统计显著性检验，而是在做 <b>风险识别框架重写</b>：把原始字段翻译成更适合课堂与管理讨论的 operational risk 结构。</p>")}
              <div class="triple-stack">
                {risk_cards_html}
              </div>
            </div>
            <div class="stack">
              {image_card("课程化后的风险热力图", "../charts/chart_3_risk_heatmap.png", "这部分属于风险识别与解释层，不是显著性检验层。", "image-medium")}
              {card("BTS taxonomy 到课程语言的映射", render_compact_table(["原始 causes", "课程风险块", "数据占比提示"], risk_mapping_rows, "tight-table"))}
              {card("SWOT / heat map / 5 Whys 在项目中的作用", f"<ul><li><b>SWOT</b>：把长期运营弱点与外部威胁说清楚，回答“为什么这个项目值得做”。</li><li><b>Risk heat map</b>：把 internal / system / external 的相对强弱视觉化，回答“哪个风险块最值得优先管理”。</li><li><b>5 Whys</b>：围绕峰值月份 <b>{peak_month['period_label']}</b> 解释延误冲击如何从原因层一路放大到结果层。</li><li>整体上，这一层回答“<b>风险是什么、为什么值得管</b>”，而不是“统计上显著吗”。</li></ul><p>当前结果里，internal 类风险对应的分钟损失占比约 <b>{format_pct(internal_share, 1)}</b>，远高于 system 和 external，这也是后续管理建议偏向内部恢复能力的原因。</p>")}
            </div>
          </div>
        </div>
        """
    ).strip()

    distribution_cards_html = "".join(card(title, f"<p>{body}</p>", "accent") for title, body in DISTRIBUTION_EXPLANATIONS)
    p03_content = dedent(
        f"""
        <div class="layout layout-j">
          {card("为什么是这四个候选分布", "<p>Frequency 只比较 <b>Poisson</b> 和 <b>Negative Binomial</b>，因为这一层真正的问题是：count 数据到底像不像“均匀、独立、波动有限”的标准计数过程，还是已经出现明显的聚集和 over-dispersion。</p><p>Severity 只比较 <b>Lognormal</b> 和 <b>Weibull</b>，因为这一层真正的问题是：分钟损失是更像一种典型右偏长尾分布，还是更像形状弹性更强的另一类右偏分布。</p><p>这不是把所有理论分布都堆进来，而是挑选最能支撑课程叙事、又与数据形态匹配的最小比较集。候选集过大，反而会让解释失焦。</p>", "primary")}
          <div class="two-col bottom-split">
            <div class="quad-grid">
              {distribution_cards_html}
            </div>
            <div class="stack">
              {card("Frequency 侧要回答的其实是两个问题", "<ul><li>事件是不是近似均匀、独立、波动有限？</li><li>如果不是，波动是不是明显大于均值，也就是出现 clustering 与 over-dispersion？</li></ul><p>因此 Poisson 与 Negative Binomial 的比较，本质上是在比较“简单计数世界”与“更高波动世界”。如果数据的方差远大于均值，Poisson 往往会过度简化现实。</p>")}
              {card("Severity 侧关心的是分钟损失形状", "<ul><li>大多数月份的平均分钟损失在哪个区间聚集？</li><li>极端月份会不会把尾部拉长？</li><li>不同 risk block 的尾部是不是同一种形状？</li></ul><p>因此 Lognormal 与 Weibull 的比较，本质上是在比较右偏分钟损失的两种常见解释。它不是在问“哪个分布永远正确”，而是在问“对当前变量，哪个工作模型更贴近样本”。</p>")}
            </div>
          </div>
        </div>
        """
    ).strip()

    p04_content = dedent(
        f"""
        <div class="layout layout-g">
          <div class="two-col">
            <div class="stack">
              {card("AIC 是什么", "<p><b>AIC = 2k - 2\\ln(L)</b>。其中 <b>k</b> 是模型参数个数，<b>L</b> 是似然。它的核心任务不是给出显著性结论，而是用同一把尺子平衡 <b>拟合质量</b> 与 <b>模型复杂度</b>。</p><p>如果只看拟合优度，复杂模型往往更容易占优；如果只看简单性，又会忽略数据形状。AIC 的意义就在于把这两个目标压缩进同一个相对比较指标里。</p><div class='formula-box'>AIC 越低越好，但只能在同一问题、同一候选集内部做相对比较。</div>", "primary")}
              {card("项目里的具体使用流程", "<div class='step-flow'><div class='step-box'><span>1</span><b>选候选模型</b><small>Frequency: Poisson vs NB<br>Severity: Lognormal vs Weibull</small></div><div class='step-box'><span>2</span><b>拟合参数</b><small>对每个目标变量分别估计分布参数</small></div><div class='step-box'><span>3</span><b>计算 AIC</b><small>把拟合优度和复杂度折成一个分数</small></div><div class='step-box'><span>4</span><b>选择较低者</b><small>保留 AIC 更低的模型，并记录 ΔAIC</small></div></div><p>换句话说，项目里的选择流程并不是“看上去像哪个就选哪个”，而是先限定候选集，再用同一准则比较所有候选模型。</p>")}
            </div>
            <div class="stack">
              {card("怎样读 ΔAIC", "<ul><li><b>ΔAIC 接近 0</b>：两个候选模型差不多，解释应更谨慎。</li><li><b>ΔAIC 明显大于 10</b>：通常可以认为较低 AIC 的模型有很强优势。</li><li><b>ΔAIC 非常大</b>：往往说明较差模型在这个变量上明显不合适。</li></ul><p>所以我们不仅看“谁最低”，也看“低了多少”。这决定了最后的解释口吻是保守还是明确。</p>", "warning")}
              {card("为什么这里要单独讲 AIC", f"<p>如果只展示最终模型名称，读者会看到“为什么总是 Negative Binomial、为什么 severity 有时是 Weibull 有时是 Lognormal”，但不知道比较标准是什么。AIC 页面把这个中间环节单独拉出来，后面 Frequency 与 Severity 两页的结果才有解释基础。</p><p>例如在 Frequency 里，<b>{freq_biggest_gap['label']}</b> 的 ΔAIC 达到 <b>{format_float(freq_biggest_gap['delta_aic'], 1)}</b>，说明 Poisson 与 Negative Binomial 的差异并不是边缘性的；而在 Severity 里，不同变量的 ΔAIC 大小不同，解释力度也应该随之不同。</p><p><b>AIC 不是显著性检验</b>，但它仍然是一个足够严谨的相对模型选择标准。</p>")}
            </div>
          </div>
        </div>
        """
    ).strip()

    p05_content = dedent(
        f"""
        <div class="layout layout-g">
          <div class="two-col">
            <div class="stack">
              {card("Frequency 结果表", render_compact_table(["变量", "最优 AIC", "对比 AIC", "最优模型", "ΔAIC"], freq_rows, "tight-table"), "primary")}
              {card("结果应该怎么解释", f"<ul><li><b>{freq_all_nb}/6</b> 个 count 变量全部选择 <b>Negative Binomial</b>，说明数据波动显著高于简单 Poisson 世界。</li><li><b>{freq_biggest_gap['label']}</b> 的 ΔAIC 最大，意味着该变量的过度离散最明显。</li><li>以总体延误班次为例，normal 状态均值约 <b>{format_float(freq_table.loc[freq_table['variable'] == 'total_delay_count', 'normal_state_mean'].iloc[0], 1)}</b>，disrupted 状态升到 <b>{format_float(freq_table.loc[freq_table['variable'] == 'total_delay_count', 'disrupted_state_mean'].iloc[0], 1)}</b>，状态差异不只是概念上的，而是量化可见的。</li><li>因此这里的核心结论不是“NB 更高级”，而是 <b>JFK 的扰动计数过程明显不稳定、会聚集</b>。</li></ul>")}
            </div>
            <div class="stack">
              {image_card("Frequency AIC 对比图", f"../charts/{FREQUENCY_AIC_CHART}.png", "每个变量都展示较差模型相对最优模型的 ΔAIC，便于判断优势强弱。", "image-medium")}
              {card("方差为何支持 Negative Binomial", render_compact_table(["变量", "均值", "方差", "扰动态增量"], freq_detail_rows, "tight-table"))}
              {card("课程语义下的含义", "<p>Lecture 4 讲的是 loss number model。这里的结论并不是“Poisson 完全错误”，而是说在多年度 JFK 样本里，<b>事件到达不够均匀、存在聚集与高波动</b>，因此用 Negative Binomial 作为工作模型更稳妥。</p><p>如果报告里只写“最终选了 NB”，读者可能会觉得这是形式上的模型切换；但把均值、方差和 disrupted 态增量放在一起看，就能更直观地理解为什么 Poisson 不够用。</p>")}
            </div>
          </div>
        </div>
        """
    ).strip()

    p06_content = dedent(
        f"""
        <div class="layout layout-g">
          <div class="two-col">
            <div class="stack">
              {card("Severity 结果表", render_compact_table(["变量", "最优 AIC", "对比 AIC", "最优模型", "ΔAIC"], sev_rows, "tight-table"), "primary")}
              {card("结果应该怎么解释", f"<ul><li><b>总体平均延误分钟</b> 选择 <b>Weibull</b>，而分风险块的平均分钟更偏向 <b>Lognormal</b>。</li><li>也就是说，总体分钟损失的形状与分块后的分钟损失形状并不完全一致；把所有 risk block 混在一起后，整体形状会发生变化。</li><li>ΔAIC 最大的是 <b>{sev_biggest_gap['label']}</b>，达到 <b>{format_float(sev_biggest_gap['delta_aic'], 1)}</b>，这说明该变量的候选模型差异最清晰。</li><li>因此这里的结论不是“某个分布普适”，而是 <b>不同 severity 变量需要各自建模</b>。</li></ul>")}
            </div>
            <div class="stack">
              {image_card("Severity AIC 对比图", f"../charts/{SEVERITY_AIC_CHART}.png", "图中保留 AIC 差值，用来支撑为什么不同变量会落在不同分布上。", "image-medium")}
              {card("Severity 的样本与状态差异", render_compact_table(["变量", "样本量", "normal 平均", "disrupted 平均"], sev_detail_rows, "tight-table"))}
              {card("Lecture 3 对应关系", f"<p>Severity 这一层对应的是损失强度的分布假设，后面再与 aggregate risk 对接到 <b>VaR / TVaR</b>。当前结果里，<b>Lognormal {sev_lognorm} 个</b>、<b>Weibull {sev_weibull} 个</b>，说明分钟损失并不是所有 risk block 都共享同一种尾部形状。</p><p>同时，disrupted 状态下的平均分钟往往高于 normal 状态，这也解释了为什么 severity 与 dependence 后面需要连起来看。</p>")}
            </div>
          </div>
        </div>
        """
    ).strip()

    p07_content = dedent(
        f"""
        <div class="layout layout-g">
          <div class="two-col">
            <div class="stack">
              {card("为什么不是一开始就正式做 Markov", "<p>如果只有单年样本，状态切换次数太少，很难稳定估计转移概率。单年数据更适合做“高风险月份识别”或“状态框架草图”，但不足以支撑正式的 transition matrix 解释。</p><p>多年度样本把机场月度状态扩展到 <b>72</b> 个观测点后，才有资格升级到两状态框架；但即便如此，结论仍应表述为 <b>经验性状态依赖证据</b>，而不是形式化强证明。</p>", "primary")}
              {card("状态规则比较表", render_compact_table(["规则", "disrupted share", "跨年 std", "score", "selected"], state_rows, "tight-table"))}
              {card("为什么最终选 composite rule", f"<ul><li><b>impact_q75</b> 和 <b>delay_cancel_impact_composite</b> 的 disrupted share 都是 <b>25%</b>。</li><li>但 composite rule 的 yearly std 为 <b>{format_float(selected_rule['yearly_disrupted_share_std'], 3)}</b>，低于 impact_q75 的波动。</li><li>因此最终选择它，不是因为它“更复杂”，而是因为它在跨年分布上更稳定。</li><li>报告里要强调：我们追求的是 <b>稳定且可解释</b> 的状态划分，而不是追求规则形式越复杂越好。</li></ul>", "warning")}
            </div>
            <div class="stack">
              {image_card("状态规则稳定性比较", f"../charts/{STATE_RULE_CHART}.png", "横轴越接近 25%、纵轴越低，说明跨年 disrupted 占比更稳定。", "image-medium")}
              {card("规则含义的解释方式", "<p><b>impact_q75</b> 只看整体冲击分钟是否进入高分位，因此直观，但信息来源较单一。</p><div class='formula-box'>impact_q75: disruption_impact_minutes ≥ Q75</div><p><b>delay_cancel_impact_composite</b> 同时纳入延误率、取消率与冲击分钟的标准化信息，因此更像一个综合运营压力指标。</p><div class='formula-box'>composite = z(delay rate) + z(cancellation rate) + z(impact minutes)<br>若 composite ≥ Q75，则判为 disrupted</div><p>这一步属于“状态划分规则设计”，仍然不是显著性检验。</p>")}
            </div>
          </div>
        </div>
        """
    ).strip()

    p08_content = dedent(
        f"""
        <div class="layout layout-g">
          <div class="two-col">
            <div class="stack">
              {image_card("两状态 Markov 转移矩阵", "../charts/chart_7_markov_transition_matrix.png", "复用已有转移矩阵图，并在本页给出可解释口径。", "image-medium")}
              {card("转移概率表", render_compact_table(["转移", "概率", "计数"], transition_rows, "tight-table"))}
            </div>
            <div class="stack">
              {card("P(N→D) 与 P(D→D) 的差异意味着什么", f"<ul><li><b>P(N→D) = {format_float(p_nd, 3)}</b>：正常月进入 disrupted 的概率不高，但不是可忽略。</li><li><b>P(D→D) = {format_float(p_dd, 3)}</b>：一旦进入 disrupted，继续停留在 disrupted 的概率约是 <b>{format_float(transition_ratio, 2)}</b> 倍于正常月直接跌入 disrupted 的概率。</li><li>这说明状态依赖不是“有没有”，而是“存在一定持续性”。</li><li>换成管理语言，就是：<b>系统一旦进入坏状态，恢复并不是自动发生的</b>。</li></ul>", "primary")}
              {card("为什么这不是夸大的强证明", f"<ul><li>样本仍然只是 <b>{format_int(airport_months)}</b> 个 airport-month。</li><li>转移概率应解读为 <b>近似估计</b>，而不是结构参数的最终定论。</li><li>同时 <b>P(D→N) = {format_float(p_dn, 3)}</b> 仍高于 <b>P(D→D)</b>，说明 disrupted 并非吸收态，系统仍有恢复能力。</li><li>因此最稳妥的表述是：<b>存在经验性状态依赖与一定持续性</b>，而不是“已经证明强 Markov 机制”。</li></ul>", "warning")}
              {card("推荐的课堂口径", "<p>本项目可以说：在多年度样本下，机场运营状态呈现出 <b>有说服力的经验性状态依赖</b>。这比单纯展示一个 heatmap 或分位规则更进一步，但仍保持学术上的克制。</p><p>如果老师追问为什么可以用两状态框架，最好的回答是：<b>因为我们先做了稳定的状态定义，再在足够长的月度样本上观察转移，而不是先假定 Markov、再硬把数据塞进去。</b></p>")}
            </div>
          </div>
        </div>
        """
    ).strip()

    p09_content = dedent(
        f"""
        <div class="layout layout-g">
          <div class="two-col">
            <div class="stack">
              {image_card("Aggregate risk 情景结果", "../charts/chart_6_aggregate_risk_scenarios.png", "复用已有情景曲线，用于连接 severity / dependence 到 annual impact。", "image-medium")}
              {card("Aggregate risk 指标表", render_compact_table(["情景", "Expected", "VaR95", "TVaR95"], scenario_rows, "tight-table"), "primary")}
              {card("为什么要做四个情景", f"<p>Aggregate risk 不是只给一个平均值，而是要展示在不同运行压力下 annual impact 会如何移动。基准情景提供正常参考，天气与节假日情景代表外部冲击与季节性压力，disruption stress 代表系统处于更脆弱状态时的上移风险。</p><p>相对基准情景，风险最高的 <b>{scenario_peak['label']}</b> 在 Expected 上高出约 <b>{format_int(scenario_peak['expected_impact_minutes'] - base_expected)}</b> 分钟，在 VaR95 上高出约 <b>{format_int(scenario_peak['var_95_minutes'] - base_var95)}</b> 分钟。</p>")}
            </div>
            <div class="stack">
              {card("课程对应关系", render_compact_table(["Lecture", "本项目对应内容"], lecture_rows, "tight-table"))}
              {card("当前结论", f"<ul><li>Frequency 层显示：JFK 运营扰动更像 <b>高波动、会聚集</b> 的计数过程，而不是简单均匀到达。</li><li>Severity 层显示：分钟损失形状在总体与分风险块之间并不完全一致，因此不能用一个单一分布解释所有强度问题。</li><li>Dependence 层显示：disrupted 状态存在持续性，但应保持经验性表述。</li><li>Aggregate risk 层显示：<b>{scenario_peak['label']}</b> 的 <b>VaR95 = {format_int(scenario_peak['var_95_minutes'])}</b> 分钟，是当前四种情景中风险最高的一档。</li></ul><p>把这四层连起来看，项目已经从 descriptive analysis 升级成一个可解释的 operational risk framework。</p>")}
              {card("最后的诚实说明", f"<p>分布选择是 <b>AIC 相对优选</b>，风险分类是 <b>理论映射</b>，Markov 是 <b>经验性状态依赖证据</b>。这三类证据的强度不同，正是本报告要刻意讲清楚的地方。</p><p>样本里的最大 delay cause 仍然是 <b>{top_cause['cause']}</b>，占总延误分钟约 <b>{format_pct(top_cause['share_of_total_delay_minutes'], 1)}</b>，因此管理建议依旧应优先落在内部恢复与系统协调上，而不是把重点完全放在天气这类外部冲击上。</p>", "warning")}
            </div>
          </div>
        </div>
        """
    ).strip()

    p10_content = dedent(
        f"""
        <div class="layout layout-i">
          <div class="metrics-row">
            {metric("选题理由", "真实运营问题", "不是抽象金融损失，而是机场每天都会面对的问题")}
            {metric("数据来源", "BTS 官方数据", "公开、连续、字段清楚，适合做可复现项目")}
            {metric("现实规模", format_int(total_flights), "当前样本覆盖的总到港架次")}
            {metric("项目价值", "可解释 + 可落地", "既能展示方法，也能给运营建议")}
          </div>
          <div class="two-col">
            <div class="stack">
              {card("为什么选择飞机延误做风险分析", f"<p>飞机延误是一个非常适合做运营风险分析的题目，因为它同时具备三种特征：<b>真实存在的运行压力</b>、<b>公开可复现的数据</b>、<b>可以被量化的损失</b>。</p><p>在当前样本里，我们覆盖了 <b>{years[0]}–{years[-1]}</b> 年、总计 <b>{format_int(total_flights)}</b> 架次到港航班。延误、取消、备降不仅影响旅客体验，更会通过班表恢复、机组衔接、维修与机场协同持续放大，因此它天然就是一个 operational risk 问题。</p>", "primary")}
              {card("为什么这个题目适合展示项目丰富程度", "<p>很多题目只能停留在描述统计或定性讨论，但飞机延误数据同时支持 <b>风险识别、频率建模、严重度建模、状态依赖和 aggregate risk</b>。</p><p>这意味着项目可以形成一条完整方法链，而不是只做一张图或一张表。你想展示“项目够不够丰富”，这个题目本身就允许你把方法、事实、解释和管理启示放进同一个 HTML 里。</p>")}
            </div>
            <div class="stack">
              {card("为什么不是直接做 monetary risk", "<p>如果强行把延误全部换算成货币损失，模型会立刻依赖大量难以 defend 的外部假设，例如每分钟延误的统一成本、不同航司的成本结构和取消 / 备降的间接成本。</p><p>相比之下，<b>delay-equivalent minutes</b> 更接近原始数据，也更适合作为课程项目中的 operational impact proxy。</p>")}
              {card("这个项目最终想回答什么", "<ul><li>为什么选择机场延误做风险分析是合理的？</li><li>为什么 risk taxonomy 要重构？</li><li>为什么比较这四个分布、为什么用 AIC 选？</li><li>模型结果是否符合客观事实？</li><li>为什么要引入状态与 Markov，最后又能给运营商什么帮助？</li></ul><p>这就是整份 HTML 的主线。</p>")}
            </div>
          </div>
        </div>
        """
    ).strip()

    p11_content = dedent(
        f"""
        <div class="layout layout-g">
          <div class="two-col">
            <div class="stack">
              {image_card("风险识别结果总览", "../charts/chart_3_risk_heatmap.png", "热力图让 risk identification 从概念进入相对强弱判断。", "image-medium")}
              {card("风险识别结果摘录", render_compact_table(["风险块", "频率桶", "严重度桶", "频率代理分钟"], risk_rows, "tight-table"), "primary")}
            </div>
            <div class="stack">
              {card("风险分类后的结果说明了什么", f"<p>internal 风险分钟占比约 <b>{format_pct(internal_share,1)}</b>，system 约 <b>{format_pct(system_share,1)}</b>，external 约 <b>{format_pct(external_share,1)}</b>。这说明分类后的排序并不是反直觉的：项目识别出来的主要风险，确实就是原始数据里最重要的分钟损失来源。</p><p>因此 risk classification 不是为了讲故事而重命名，而是把原始 BTS 原因翻译成更适合风险管理和课程表达的结构。</p>", "primary")}
              {card("为什么这一页很重要", "<p>如果只讲后面的分布和 AIC，整份项目会显得像一份统计作业；把 risk identification 单独拉出来，才能说明项目起点是现实运营问题，而不是先有模型再找数据。</p><p>这也是体现项目完整度的重要一环：先识别风险，再解释风险，最后才对风险建模。</p>")}
            </div>
          </div>
        </div>
        """
    ).strip()

    p12_content = dedent(
        f"""
        <div class="layout layout-g">
          <div class="two-col">
            <div class="stack">
              {card("模型结果是否符合客观事实", "<p>这一步不是额外点缀，而是整个项目可信度的核心。如果一个模型虽然数学上能拟合数据，但给出的世界与真实运营逻辑完全相反，那么它就不该被当成最终解释。</p><p>因此在展示结果时，必须把模型输出和客观事实逐项对照，而不是只报出“最优模型名字”。</p>", "primary")}
              {card("事实与模型之间的对应", render_compact_table(["观察到的事实", "模型结果", "解释"], objective_rows, "tight-table"))}
            </div>
            <div class="stack">
              {card("为什么这一步能体现项目丰富程度", "<p>很多项目做到这里就结束了，只停留在“模型跑出来了”。但把结果与现实运营事实对照，说明这份工作不只是会用工具，而是会判断这些结果是不是在说真话。</p><p>这会让老师和组员更容易接受：模型不是硬套上的，而是在真实数据和真实运行经验之间相互印证出来的。</p>")}
              {card("一个最直观的总结", f"<p>Negative Binomial 对应“方差远大于均值”的事实，internal 风险主导对应原始 delay cause 的分钟占比结构，holiday peak 情景风险最高对应高峰运行更脆弱的现实直觉，而 P(D→D) 高于 P(N→D) 则对应系统进入坏状态后恢复更慢的运营经验。</p><p>这些一致性共同提高了模型解释的可信度。</p>", "warning")}
            </div>
          </div>
        </div>
        """
    ).strip()

    p13_content = dedent(
        f"""
        <div class="layout layout-g">
          <div class="two-col">
            <div class="stack">
              {card("这个风险分析最终能给运营商什么帮助", "<p>这份项目的价值不在于给出一个抽象分数，而在于把运营商能采取的动作和量化证据绑定起来。也就是说，它应该帮助运营方回答：资源应该优先投入哪里、哪些月份更该提前准备、哪些指标最值得持续监控。</p>", "primary")}
              {card("运营建议矩阵", render_compact_table(["对象 / 场景", "可执行帮助", "对应证据"], operator_rows, "tight-table"))}
            </div>
            <div class="stack">
              {card("为什么这份项目已经足够完整", "<ul><li>它从现实运营问题出发，而不是从一个现成模型出发。</li><li>它先做风险识别，再做 frequency、severity、dependence 和 aggregate risk。</li><li>它解释了为什么选题、为什么分类、为什么选分布、为什么用 AIC、为什么做 Markov。</li><li>它还专门检查了模型结果是否符合客观事实。</li></ul><p>这几层叠在一起，已经构成一份完整的 operational risk 项目，而不是单纯的技术展示页。</p>")}
              {card("最终呈现口径", "<p>你在展示这份 HTML 时，可以把它定义为：<b>一个从真实机场运行问题出发，经过课程化风险识别、统计建模、状态依赖分析，再回到管理启示的完整 operational risk project</b>。</p><p>这样讲，既能让组员看懂过程，也能让老师看到项目的丰富程度和完整度。</p>", "warning")}
            </div>
          </div>
        </div>
        """
    ).strip()

    pages = [
        page("P00 封面", "中文建模过程版报告", "这是一份面向组员理解和讨论的解释型报告，而不是只展示结论的结果页。", p00_content, 0, total_pages),
        page("P01 选题动机", "为什么选择飞机延误做风险分析", "先把选题的现实价值、数据可得性和课程契合度讲清楚。", p10_content, 1, total_pages),
        page("P02 项目目标与数据结构", "项目目标与数据结构", "先把研究对象、损失口径和两层数据结构说清楚，后面的模型页才有基础。", p01_content, 2, total_pages),
        page("P03 风险分类方法", "为什么这么进行风险分类", "这一页解决“风险是什么、为什么这样分”，它属于识别层，不属于显著性检验层。", p02_content, 3, total_pages),
        page("P04 风险识别结果", "风险识别结果与现实含义", "热力图和分钟占比一起说明分类并不是人为硬分。", p11_content, 4, total_pages),
        page("P05 候选分布介绍", "候选分布介绍与比较理由", "这一页回答“为什么只比较这几种分布”，防止后续结果看起来像黑箱。", p03_content, 5, total_pages),
        page("P06 分布选择方法", "抉择分布类型时使用了什么方法", "AIC 提供的是相对比较标准，不是显著性检验。", p04_content, 6, total_pages),
        page("P07 Frequency 结果", "Frequency 结果与解释", "所有 count 变量都更支持 Negative Binomial，说明事件并不接近均匀随机到达。", p05_content, 7, total_pages),
        page("P08 Severity 结果", "Severity 结果与解释", "Severity 的最优分布并不完全统一，这恰好说明分钟损失形状具有层次差异。", p06_content, 8, total_pages),
        page("P09 客观事实一致性", "这些选择是否符合客观事实", "这一页专门把模型结果与现实运营事实做对照。", p12_content, 9, total_pages),
        page("P10 状态划分规则", "为什么进行状态划分与 Markov 分析", "先解释为什么需要状态框架，再解释为什么不能轻率做 Markov。", p07_content, 10, total_pages),
        page("P11 状态依赖解释", "Markov 分析结果与解释", "P(N→D) 与 P(D→D) 的差异支持经验性状态依赖，但不应夸大为强证明。", p08_content, 11, total_pages),
        page("P12 Aggregate risk", "最终 aggregate risk 结果是什么", "把频率、严重度和状态依赖收束到 annual impact 的情景结果上。", p09_content, 12, total_pages),
        page("P13 运营帮助与项目完整度", "这个项目最终能给运营商什么帮助", "最后用管理动作和项目完整度收束整份 HTML。", p13_content, 13, total_pages),
    ]

    return dedent(
        """
        <!doctype html>
        <html lang="zh-CN">
        <head>
          <meta charset="utf-8">
          <meta name="viewport" content="width=device-width, initial-scale=1">
          <title>JFK 中文建模过程版报告</title>
          <style>
            :root {
              --bg: #efe9de;
              --page: rgba(255, 252, 247, 0.94);
              --card: rgba(255, 255, 255, 0.88);
              --ink: #20323B;
              --muted: #61727B;
              --line: #D5CEC0;
              --primary: #0F4C5C;
              --accent: #2A9D8F;
              --shadow: 0 20px 44px rgba(32, 50, 59, 0.12);
              --font-display: "Georgia", "Times New Roman", serif;
              --font-body: "PingFang SC", "Microsoft YaHei", "Noto Sans SC", sans-serif;
              --font-mono: "Consolas", "SFMono-Regular", monospace;
            }
            *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
            body { background: var(--bg); font-family: var(--font-body); color: var(--ink); padding: 26px; position: relative; }
            body::before { content: ""; position: fixed; inset: 0; pointer-events: none; z-index: 0; background: radial-gradient(circle at 12% 12%, rgba(15, 76, 92, 0.12), transparent 24%), radial-gradient(circle at 88% 8%, rgba(198, 106, 43, 0.12), transparent 22%), linear-gradient(180deg, rgba(255,255,255,0.36), transparent 18%); }
            body::after { content: ""; position: fixed; inset: 0; pointer-events: none; z-index: 0; background-image: linear-gradient(rgba(255,255,255,0.06) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.06) 1px, transparent 1px); background-size: 28px 28px; opacity: 0.22; }
            .report-stack { position: relative; z-index: 1; width: fit-content; margin: 0 auto; }
            .report-page { width: 1017px; height: 720px; min-width: 1017px; max-width: 1017px; min-height: 720px; max-height: 720px; overflow: hidden; background: var(--page); border: 1px solid rgba(32, 50, 59, 0.08); border-radius: 30px; box-shadow: var(--shadow); margin: 0 auto 28px; }
            .hd { height: 72px; padding: 0 25px; display: flex; align-items: center; justify-content: space-between; border-bottom: 1px solid rgba(32, 50, 59, 0.08); }
            .ct { height: 580px; padding: 0 25px; overflow: hidden; }
            .sm { height: 48px; padding: 0 25px; display: flex; align-items: center; gap: 10px; border-top: 1px solid rgba(32, 50, 59, 0.06); }
            .ft { height: 20px; padding: 0 25px; display: flex; align-items: center; justify-content: space-between; color: var(--muted); font-size: 10px; }
            .chapter-tag { display: inline-flex; align-items: center; padding: 6px 10px; border-radius: 999px; background: rgba(15, 76, 92, 0.08); color: var(--primary); font-size: 10px; letter-spacing: 0.08em; text-transform: uppercase; font-weight: 700; }
            .page-title { font-size: 14px; font-weight: 700; color: var(--ink); }
            .summary-pill { display: inline-flex; align-items: center; padding: 4px 8px; border-radius: 999px; background: rgba(42, 157, 143, 0.08); color: var(--accent); font-size: 10px; font-weight: 700; flex-shrink: 0; }
            .footer-progress { display: flex; align-items: center; gap: 6px; }
            .footer-progress-bar { width: 90px; height: 3px; background: rgba(15, 76, 92, 0.10); border-radius: 999px; overflow: hidden; }
            .footer-progress-bar span { display: block; height: 100%; background: var(--primary); }
            .layout { height: calc(580px - 8px); padding-top: 8px; display: grid; gap: 10px; }
            .layout-i { grid-template-rows: 92px 470px; }
            .layout-g { grid-template-rows: 572px; }
            .layout-j { grid-template-rows: 156px 406px; }
            .metrics-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; height: 92px; }
            .two-col { display: grid; grid-template-columns: 1.02fr 0.98fr; gap: 10px; height: 100%; }
            .bottom-split { grid-template-columns: 1fr 0.95fr; }
            .stack { display: grid; gap: 10px; height: 100%; grid-auto-rows: minmax(0, 1fr); }
            .triple-stack { display: grid; gap: 10px; grid-template-rows: repeat(3, 1fr); min-height: 0; }
            .quad-grid { display: grid; gap: 10px; grid-template-columns: repeat(2, 1fr); grid-template-rows: repeat(2, 1fr); height: 100%; }
            .card { background: var(--card); border: 1px solid rgba(32, 50, 59, 0.08); border-radius: 20px; padding: 12px 14px; overflow: hidden; display: flex; flex-direction: column; gap: 8px; min-height: 0; }
            .card.primary { background: linear-gradient(180deg, rgba(15, 76, 92, 0.11), rgba(255,255,255,0.95)); border-color: rgba(15, 76, 92, 0.15); }
            .card.accent { background: linear-gradient(180deg, rgba(42, 157, 143, 0.08), rgba(255,255,255,0.95)); }
            .card.warning { background: linear-gradient(180deg, rgba(198, 161, 91, 0.12), rgba(255,255,255,0.95)); }
            .card-title { font-size: 13px; font-weight: 700; color: var(--ink); line-height: 1.35; }
            .card-body { font-size: 11px; line-height: 1.48; color: var(--ink); overflow: hidden; }
            .card-body p + p, .card-body ul + p, .card-body p + ul, .card-body ul + ul { margin-top: 6px; }
            .card-body ul { padding-left: 16px; }
            .card-body li + li { margin-top: 4px; }
            .metric-card { background: rgba(255,255,255,0.92); border: 1px solid rgba(32, 50, 59, 0.07); border-radius: 18px; padding: 12px 13px; display: flex; flex-direction: column; justify-content: center; gap: 5px; min-height: 0; }
            .metric-label { font-size: 10px; color: var(--muted); letter-spacing: 0.06em; text-transform: uppercase; font-weight: 700; }
            .metric-value { font-family: var(--font-display); font-size: 26px; line-height: 1; color: var(--primary); font-weight: 700; }
            .metric-note { font-size: 10px; color: var(--muted); line-height: 1.35; }
            .formula-box { margin-top: 6px; padding: 8px 10px; border-radius: 14px; border: 1px solid rgba(15, 76, 92, 0.14); background: rgba(255,255,255,0.74); font-family: var(--font-mono); font-size: 10px; line-height: 1.5; }
            .step-flow { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin-top: 4px; }
            .step-box { min-height: 104px; border-radius: 16px; border: 1px solid rgba(32, 50, 59, 0.08); background: rgba(255,255,255,0.88); padding: 10px; display: flex; flex-direction: column; gap: 5px; }
            .step-box span { width: 22px; height: 22px; border-radius: 999px; display: inline-flex; align-items: center; justify-content: center; background: var(--primary); color: white; font-size: 11px; font-weight: 700; }
            .step-box b { font-size: 11px; }
            .step-box small { font-size: 10px; line-height: 1.4; color: var(--muted); }
            .data-table { width: 100%; border-collapse: collapse; font-size: 10px; }
            .data-table th, .data-table td { border-bottom: 1px solid rgba(32, 50, 59, 0.08); padding: 6px 5px; text-align: left; vertical-align: top; }
            .data-table th { color: var(--muted); font-size: 9px; text-transform: uppercase; letter-spacing: 0.05em; }
            .tight-table td, .tight-table th { padding-top: 5px; padding-bottom: 5px; }
            .table-strong { color: var(--primary); font-weight: 700; }
            .image-card { gap: 7px; }
            .image-frame { width: 100%; border-radius: 16px; border: 1px solid rgba(32, 50, 59, 0.08); overflow: hidden; background: rgba(255,255,255,0.94); display: flex; align-items: center; justify-content: center; }
            .image-frame img { width: 100%; height: 100%; object-fit: cover; display: block; }
            .image-medium { height: 210px; }
            .image-short { height: 188px; }
            .image-caption { color: var(--muted); font-size: 10px; line-height: 1.4; }
            .cover-page { height: calc(580px - 8px); padding-top: 6px; display: grid; grid-template-rows: 336px 120px 96px; gap: 10px; }
            .cover-hero { border-radius: 28px; padding: 26px 28px; background: radial-gradient(circle at 82% 18%, rgba(255,255,255,0.10), transparent 18%), linear-gradient(135deg, rgba(15, 76, 92, 0.98), rgba(42, 157, 143, 0.92)); color: #F5FBFC; position: relative; overflow: hidden; }
            .cover-hero::after { content: ""; position: absolute; width: 260px; height: 260px; right: -90px; top: -70px; border-radius: 50%; background: rgba(255,255,255,0.08); }
            .cover-eyebrow { display: inline-flex; align-items: center; padding: 6px 10px; border-radius: 999px; background: rgba(255,255,255,0.12); font-size: 10px; letter-spacing: 0.08em; text-transform: uppercase; font-weight: 700; }
            .cover-hero h1 { margin-top: 16px; font-size: 38px; line-height: 1.08; max-width: 12ch; font-family: var(--font-display); }
            .cover-hero p { margin-top: 14px; max-width: 60ch; font-size: 13px; line-height: 1.62; color: rgba(245, 251, 252, 0.92); }
            .cover-metrics { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; }
            .cover-band { display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; }
            .band-card { border-radius: 18px; border: 1px solid rgba(32, 50, 59, 0.08); background: rgba(255,255,255,0.88); padding: 12px 14px; display: flex; flex-direction: column; gap: 5px; }
            .band-label { font-size: 10px; letter-spacing: 0.05em; text-transform: uppercase; color: var(--muted); font-weight: 700; }
            .band-value { font-size: 12px; line-height: 1.45; color: var(--ink); }
            @media print {
              @page { size: A4 landscape; margin: 8mm; }
              body { padding: 0; background: white; -webkit-print-color-adjust: exact; print-color-adjust: exact; }
              body::before, body::after { display: none; }
              .report-page { margin: 0 0 6mm 0; box-shadow: none; border-radius: 0; border: 1px solid rgba(32, 50, 59, 0.10); break-after: page; page-break-after: always; }
              .report-page:last-child { break-after: auto; page-break-after: auto; }
            }
          </style>
        </head>
        <body>
          <main class="report-stack">
        """
    ).strip() + "".join(pages) + "</main></body></html>\n"


def build_frequency_aic_chart(freq_table: pd.DataFrame, english: bool = False, stem: str | None = None) -> None:
    chart_df = freq_table.sort_values("delta_aic", ascending=True).copy()
    chart_df["chart_label"] = chart_df["variable"].map(FREQUENCY_VARIABLE_LABELS_EN if english else FREQUENCY_VARIABLE_LABELS)
    chart_df["chart_label"] = chart_df["chart_label"].fillna(chart_df["label"])
    xlabel = (
        "Delta AIC of the weaker model relative to the preferred model"
        if english
        else "较差模型相对最优模型的 ΔAIC"
    )
    title = (
        "Frequency Candidate Model Comparison by Delta AIC"
        if english
        else "Frequency 候选模型 AIC 差值对比"
    )
    annotation_template = (
        "{selected} preferred | AIC={aic:,.1f}\nAgainst {comparison} | Delta AIC={delta:,.1f}"
        if english
        else "{selected} 最优 | AIC={aic:,.1f}\n对比模型 {comparison} | ΔAIC={delta:,.1f}"
    )
    chart_stem = stem or (FREQUENCY_AIC_CHART_EN if english else FREQUENCY_AIC_CHART)

    fig, ax = plt.subplots(figsize=(11, 6.2))
    y_pos = np.arange(len(chart_df))
    ax.barh(y_pos, chart_df["delta_aic"], color=CHART_STYLE["primary"], alpha=0.88)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(chart_df["chart_label"])
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    max_gap = float(chart_df["delta_aic"].max())
    for idx, row in enumerate(chart_df.itertuples(index=False)):
        annotation = annotation_template.format(
            selected=row.selected_model,
            aic=row.selected_aic,
            comparison=row.comparison_model,
            delta=row.delta_aic,
        )
        ax.text(float(row.delta_aic) + max_gap * 0.02, idx, annotation, va="center", fontsize=9)
    fig.tight_layout()
    save_chart(fig, chart_stem)


def build_severity_aic_chart(sev_table: pd.DataFrame, english: bool = False, stem: str | None = None) -> None:
    chart_df = sev_table.sort_values("delta_aic", ascending=True).copy()
    chart_df["chart_label"] = chart_df["variable"].map(SEVERITY_VARIABLE_LABELS_EN if english else SEVERITY_VARIABLE_LABELS)
    chart_df["chart_label"] = chart_df["chart_label"].fillna(chart_df["label"])
    xlabel = (
        "Delta AIC of the weaker model relative to the preferred model"
        if english
        else "较差模型相对最优模型的 ΔAIC"
    )
    title = (
        "Severity Candidate Model Comparison by Delta AIC"
        if english
        else "Severity 候选模型 AIC 差值对比"
    )
    annotation_template = (
        "{selected} preferred | AIC={aic:,.1f}\nAgainst {comparison} | Delta AIC={delta:,.1f}"
        if english
        else "{selected} 最优 | AIC={aic:,.1f}\n对比模型 {comparison} | ΔAIC={delta:,.1f}"
    )
    chart_stem = stem or (SEVERITY_AIC_CHART_EN if english else SEVERITY_AIC_CHART)

    fig, ax = plt.subplots(figsize=(10.2, 5.8))
    y_pos = np.arange(len(chart_df))
    ax.barh(y_pos, chart_df["delta_aic"], color=CHART_STYLE["accent"], alpha=0.9)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(chart_df["chart_label"])
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    max_gap = float(chart_df["delta_aic"].max())
    for idx, row in enumerate(chart_df.itertuples(index=False)):
        annotation = annotation_template.format(
            selected=row.selected_model,
            aic=row.selected_aic,
            comparison=row.comparison_model,
            delta=row.delta_aic,
        )
        ax.text(float(row.delta_aic) + max_gap * 0.03, idx, annotation, va="center", fontsize=9)
    fig.tight_layout()
    save_chart(fig, chart_stem)


def build_state_rule_chart(state_rules: pd.DataFrame, english: bool = False, stem: str | None = None) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 5.8))
    selected = state_rules.loc[state_rules["selected"]].iloc[0]
    for row in state_rules.itertuples(index=False):
        color = CHART_STYLE["secondary"] if row.selected else CHART_STYLE["muted"]
        size = 150 if row.selected else 110
        ax.scatter(row.disrupted_share, row.yearly_disrupted_share_std, s=size, color=color, zorder=3)
        ax.text(
            float(row.disrupted_share) + 0.008,
            float(row.yearly_disrupted_share_std) + 0.002,
            f"{row.label}\nshare={row.disrupted_share:.2f}, std={row.yearly_disrupted_share_std:.3f}",
            fontsize=9,
        )
    if english:
        ax.axvline(0.25, color=CHART_STYLE["gold"], linestyle="--", linewidth=1.2, label="Target disrupted share = 25%")
        ax.set_title(f"State Rule Comparison: {selected['label']} is closer to 25% and more stable")
        chart_stem = stem or STATE_RULE_CHART_EN
    else:
        ax.axvline(0.25, color=CHART_STYLE["gold"], linestyle="--", linewidth=1.2, label="目标 disrupted share = 25%")
        ax.set_title(f"状态规则比较：{selected['label']} 更接近 25% 且跨年更稳定")
        chart_stem = stem or STATE_RULE_CHART
    ax.set_xlim(0.20, 0.30)
    ax.set_ylim(0.0, max(state_rules["yearly_disrupted_share_std"]) * 1.25)
    ax.set_xlabel("Disrupted share")
    ax.set_ylabel("Yearly disrupted-share std")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    save_chart(fig, chart_stem)


def main() -> None:
    ensure_output_directories()
    apply_style()

    frequency_df = load_csv(FREQUENCY_SUMMARY)
    severity_df = load_csv(SEVERITY_SUMMARY)
    state_rule_df = load_csv(STATE_RULE_COMPARISON)
    transition_df = load_csv(STATE_TRANSITIONS)
    metrics_df = load_csv(SCENARIO_METRICS)
    model_df = load_csv(MODELING_INPUT)
    airport_panel = load_csv(AIRPORT_MONTH_PANEL)
    risk_heatmap = load_csv(RISK_HEATMAP)
    cause_summary = load_csv(CAUSE_SUMMARY)

    freq_table = build_frequency_table(frequency_df)
    sev_table = build_severity_table(severity_df)
    state_rules = build_state_rule_table(state_rule_df)
    transitions = build_transition_table(transition_df)
    scenarios = build_scenario_table(metrics_df)

    build_frequency_aic_chart(freq_table)
    build_severity_aic_chart(sev_table)
    build_state_rule_chart(state_rules)

    html_output = build_report_html(
        freq_table,
        sev_table,
        state_rules,
        transitions,
        scenarios,
        model_df,
        airport_panel,
        risk_heatmap,
        cause_summary,
    )
    OUTPUT_HTML.write_text(html_output, encoding="utf-8")
    OUTPUT_GUIDE.write_text(build_pdf_guide(), encoding="utf-8")
    run_validations(freq_table, sev_table, state_rules, transitions, scenarios, html_output)

    print("Chinese modeling-process report outputs written to:")
    print(f"  HTML report: {OUTPUT_HTML}")
    print(f"  PDF guide: {OUTPUT_GUIDE}")
    print(f"  Frequency AIC chart: {CHART_DIR / (FREQUENCY_AIC_CHART + '.png')}")
    print(f"  Severity AIC chart: {CHART_DIR / (SEVERITY_AIC_CHART + '.png')}")
    print(f"  State rule chart: {CHART_DIR / (STATE_RULE_CHART + '.png')}")


if __name__ == "__main__":
    main()
