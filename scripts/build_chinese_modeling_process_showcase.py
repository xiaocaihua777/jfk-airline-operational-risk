from __future__ import annotations

import html
import re
import shutil
import subprocess
from textwrap import dedent, fill

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch

import build_chinese_modeling_process_report as base


SHOWCASE_HTML = base.DASHBOARD_DIR / "jfk_modeling_process_showcase_zh.html"
SHOWCASE_HTML_EN = base.DASHBOARD_DIR / "jfk_modeling_process_showcase_en.html"
SHOWCASE_GUIDE = base.SUMMARY_DIR / "jfk_modeling_process_showcase_pdf_export_guide.md"
SHOWCASE_GUIDE_EN = base.SUMMARY_DIR / "jfk_modeling_process_showcase_pdf_export_guide_en.md"
SHOWCASE_PDF_EN = base.SUMMARY_DIR / "jfk_modeling_process_showcase_en.pdf"
TMP_PDF_DIR = base.BASE_DIR / "tmp" / "pdfs"
TMP_EN_HTML = TMP_PDF_DIR / "jfk_modeling_process_showcase_en_tmp.html"
ENGLISH_PDF_PAGES = 9


def stat_block(label: str, value: str, note: str, tone: str = "") -> str:
    class_attr = f"stat-block {tone}".strip()
    return (
        f'<div class="{class_attr}">'
        f'<div class="stat-label">{html.escape(label)}</div>'
        f'<div class="stat-value">{value}</div>'
        f'<div class="stat-note">{note}</div>'
        "</div>"
    )


def panel(title: str, body_html: str, tone: str = "") -> str:
    class_attr = f"panel {tone}".strip()
    return (
        f'<section class="{class_attr}">'
        f'<div class="panel-title">{html.escape(title)}</div>'
        f'<div class="panel-body">{body_html}</div>'
        "</section>"
    )


def visual(title: str, image_src: str, caption: str, frame_class: str = "") -> str:
    class_attr = f"visual-frame {frame_class}".strip()
    return (
        '<section class="visual-block">'
        f'<div class="panel-title">{html.escape(title)}</div>'
        f'<div class="{class_attr}"><img src="{image_src}" alt="{html.escape(title)}"></div>'
        f'<div class="visual-caption">{caption}</div>'
        "</section>"
    )


def showcase_page(
    chapter: str,
    title: str,
    summary: str,
    content_html: str,
    page_index: int,
    total_pages: int,
    tone: str = "",
) -> str:
    progress = int(round(((page_index + 1) / total_pages) * 100))
    class_attr = f"report-page {tone}".strip()
    return dedent(
        f"""
        <section class="{class_attr}" style="--enter-delay:{page_index * 70}ms;">
          <div class="hd">
            <div class="chapter-tag">{html.escape(chapter)}</div>
            <div class="page-title">{html.escape(title)}</div>
          </div>
          <div class="ct">
            {content_html}
          </div>
        </section>
        """
    ).strip()


def build_showcase_guide() -> str:
    return dedent(
        f"""
        # 中文项目展示版 HTML 导出说明

        ## 目标文件
        - HTML：`{SHOWCASE_HTML.relative_to(base.BASE_DIR)}`
        - 建议导出 PDF 名称：`jfk_modeling_process_showcase_zh.pdf`

        ## 推荐浏览器
        - Microsoft Edge
        - Google Chrome

        ## 导出步骤
        1. 打开 `reports/dashboard/jfk_modeling_process_showcase_zh.html`。
        2. 等待图表与背景全部加载完成。
        3. 使用 `Ctrl+P` 进入打印面板。
        4. 打印机选择“另存为 PDF”。
        5. 布局选择“横向”。
        6. 勾选“背景图形”。
        7. 缩放使用默认或“适合页面”。
        8. 页边距选择“默认”或“最小”，关闭页眉页脚。
        9. 预览确认每页单独分页后导出。

        ## 导出时应重点检查
        - 深色封面和分节页的背景是否被保留。
        - 图表、公式和关键数字是否完整显示。
        - 标题与正文之间没有重叠。
        - 中文无乱码，页脚页码连续。

        ## 本版特点
        - 这一版更偏项目展示与答辩，不是说明书式报告。
        - 视觉权重更集中，重点结论会被前置放大。
        - 仍保留打印友好样式，可直接导出 PDF。
        """
    ).strip() + "\n"


def build_showcase_guide_en() -> str:
    return dedent(
        f"""
        # English Showcase HTML Export Guide

        ## Target files
        - HTML: `{SHOWCASE_HTML_EN.relative_to(base.BASE_DIR)}`
        - Suggested PDF name: `jfk_modeling_process_showcase_en.pdf`

        ## Recommended browsers
        - Microsoft Edge
        - Google Chrome

        ## Export steps
        1. Open `reports/dashboard/jfk_modeling_process_showcase_en.html`.
        2. Wait until all charts and backgrounds are fully loaded.
        3. Press `Ctrl+P`.
        4. Choose “Save as PDF”.
        5. Set layout to `Landscape`.
        6. Enable `Background graphics`.
        7. Keep scaling at default or `Fit to page`.
        8. Use `Default` or `Minimum` margins and disable headers/footers.
        9. Confirm each page is separated correctly in print preview, then export.

        ## What to check before export
        - Dark cover / section backgrounds are preserved.
        - Charts, formulas, and key figures are fully visible.
        - Titles and body text do not overlap.
        - Page order is correct and no text is clipped.

        ## Notes
        - This file is the English counterpart of the final showcase report.
        - It is designed for browser-based print-to-PDF export.
        """
    ).strip() + "\n"


def run_validations(html_output: str) -> None:
    page_count = html_output.count('class="report-page')
    if page_count != 12:
        raise ValueError(f"Expected 12 showcase pages, found {page_count}.")
    if not SHOWCASE_HTML.exists():
        raise FileNotFoundError(f"Expected showcase HTML output not found: {SHOWCASE_HTML}")
    if not SHOWCASE_GUIDE.exists():
        raise FileNotFoundError(f"Expected showcase guide output not found: {SHOWCASE_GUIDE}")
    if not SHOWCASE_HTML_EN.exists():
        raise FileNotFoundError(f"Expected English showcase HTML output not found: {SHOWCASE_HTML_EN}")
    if not SHOWCASE_GUIDE_EN.exists():
        raise FileNotFoundError(f"Expected English showcase guide output not found: {SHOWCASE_GUIDE_EN}")


def build_showcase_html(
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
    total_pages = 12

    years = sorted(int(year) for year in airport_panel["year"].unique())
    airline_months = len(model_df)
    airport_months = len(airport_panel)
    disrupted_months = int((airport_panel["operational_state"] == "disrupted").sum())
    total_flights = float(airport_panel["total_arrival_flights"].sum())
    mean_delay_rate = float(airport_panel["delay_rate"].mean())
    mean_cancellation_rate = float(airport_panel["cancellation_rate"].mean())
    total_disruption_impact = float(airport_panel["disruption_impact_minutes"].sum())
    peak_month = airport_panel.loc[airport_panel["disruption_impact_minutes"].idxmax()]
    peak_delay_rate_month = airport_panel.loc[airport_panel["delay_rate"].idxmax()]

    cause_share_map = {
        str(row["cause"]): float(row["share_of_total_delay_minutes"])
        for row in cause_summary.to_dict(orient="records")
    }
    internal_share = cause_share_map.get("Airline", 0.0) + cause_share_map.get("Late Aircraft", 0.0)
    system_share = cause_share_map.get("NAS", 0.0)
    external_share = cause_share_map.get("Weather", 0.0) + cause_share_map.get("Security", 0.0)
    top_cause = cause_summary.sort_values("share_of_total_delay_minutes", ascending=False).iloc[0]

    freq_biggest_gap = freq_table.sort_values("delta_aic", ascending=False).iloc[0]
    sev_biggest_gap = sev_table.sort_values("delta_aic", ascending=False).iloc[0]
    freq_all_nb = int((freq_table["selected_model_raw"] == "negative_binomial").sum())
    sev_lognorm = int((sev_table["selected_model_raw"] == "lognorm").sum())
    sev_weibull = int((sev_table["selected_model_raw"] == "weibull_min").sum())

    selected_rule = state_rules.loc[state_rules["selected"]].iloc[0]
    alt_rule = state_rules.loc[~state_rules["selected"]].iloc[0]
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
    scenario_rank = scenarios.sort_values("var_95_minutes", ascending=False).copy()
    base_expected = float(scenarios.loc[scenarios["scenario"] == "base", "expected_impact_minutes"].iloc[0])
    base_var95 = float(scenarios.loc[scenarios["scenario"] == "base", "var_95_minutes"].iloc[0])

    freq_rows = []
    for row in freq_table.itertuples(index=False):
        freq_rows.append(
            [
                html.escape(row.label),
                row.selected_model,
                base.format_float(row.delta_aic, 1),
                base.format_float(row.overall_mean, 1),
                base.format_float(row.overall_variance, 1),
            ]
        )

    sev_rows = []
    for row in sev_table.itertuples(index=False):
        sev_rows.append(
            [
                html.escape(row.label),
                row.selected_model,
                base.format_float(row.delta_aic, 1),
                base.format_int(row.sample_size),
                base.format_float(row.disrupted_state_mean - row.normal_state_mean, 1),
            ]
        )

    mapping_rows = [
        ["Airline + Late Aircraft", "internal", f"{base.format_pct(internal_share, 1)}"],
        ["NAS", "system", f"{base.format_pct(system_share, 1)}"],
        ["Weather + Security", "external", f"{base.format_pct(external_share, 1)}"],
    ]

    scenario_rows = []
    for row in scenario_rank.itertuples(index=False):
        scenario_rows.append(
            [
                html.escape(row.label),
                base.format_int(row.expected_impact_minutes),
                base.format_int(row.var_95_minutes),
                base.format_int(row.tvar_95_minutes),
            ]
        )

    reality_rows = [
        [
            "计数型扰动不是均匀到达",
            f"6/{len(freq_table)} 个 frequency 变量全部选中 Negative Binomial",
            "说明方差显著高于均值，Poisson 过于理想化。",
        ],
        [
            "主要风险确实来自内部运行链条",
            f"internal 分钟占比 {base.format_pct(internal_share, 1)}",
            "与原始 cause 结构一致，不是为了好讲而重命名。",
        ],
        [
            "坏状态有持续性",
            f"P(D→D)={base.format_float(p_dd, 3)} > P(N→D)={base.format_float(p_nd, 3)}",
            "说明系统进入扰动态后，恢复并不是自动完成。",
        ],
        [
            "高峰情景更危险",
            f"{scenario_peak['label']} 的 VaR95 最高",
            "与节假日和高压运行更脆弱的现实经验一致。",
        ],
    ]

    operator_rows = [
        ["航司内部恢复", "加大 turnaround buffer，优先补强机组与维修衔接", "internal 风险占主导，且 top cause 是 Airline。"],
        ["机场 / NAS 协同", "在高压月份提前做容量、时隙与地面资源协调", "system 风险在 severity 上有明显抬升。"],
        ["季节与天气应对", "对 holiday / weather 情景配置 contingency plan", "aggregate risk 在这两类情景下显著上移。"],
        ["月度预警机制", "持续监控 delay rate、cancellation rate 与 impact composite", "它们直接决定状态识别与坏状态延续判断。"],
    ]

    p00 = dedent(
        f"""
        <div class="poster">
          <div class="poster-copy">
            <div class="eyebrow">JFK Operational Risk Showcase</div>
            <h1>为什么飞机延误<br>值得被做成一整套风险项目</h1>
            <p>
              这不是一份“把结果排版得更好看”的 HTML。
              这是一份从 <b>选题动机</b>、<b>风险识别</b>、<b>分布选择</b>、
              <b>状态依赖</b> 到 <b>运营启示</b> 的完整项目展示版。
            </p>
            <div class="poster-line"></div>
            <div class="poster-caption">
              如果你想让读者感受到“这个项目不只是会跑模型，而是真的设计过方法链”，这就是应该呈现出来的样子。
            </div>
          </div>
          <div class="poster-stats">
            {stat_block("样本年份", f"{years[0]}–{years[-1]}", "多年度机场月度序列", "light")}
            {stat_block("覆盖架次", base.format_int(total_flights), "总到港航班", "light")}
            {stat_block("研究粒度", f"{base.format_int(airline_months)} / {base.format_int(airport_months)}", "airline-month / airport-month", "light")}
          </div>
          <div class="poster-roadmap">
            <span>选题价值</span>
            <span>风险识别</span>
            <span>分布抉择</span>
            <span>状态依赖</span>
            <span>aggregate risk</span>
            <span>运营决策</span>
          </div>
        </div>
        """
    ).strip()

    p01 = dedent(
        f"""
        <div class="split split-hero">
          <div class="stack spacious">
            <div class="section-kicker">Why This Problem</div>
            <div class="big-claim">延误是最像 operational risk 的机场问题之一。</div>
            <div class="claim-text">
              它同时具备真实运营压力、公开可复现数据，以及可以量化的损失代理。
              这意味着它既能讲现实问题，也能撑起完整方法链，而不只是做一页描述统计。
              更重要的是，延误、取消、备降之间天然存在传导关系，所以这个题目非常适合把 frequency、severity、dependence 和 aggregate risk 串成一个完整故事。
            </div>
            <div class="inline-proof">
              <span>平均延误率 {base.format_pct(mean_delay_rate, 1)}</span>
              <span>平均取消率 {base.format_pct(mean_cancellation_rate, 2)}</span>
              <span>总冲击 {base.format_int(total_disruption_impact)} 分钟</span>
            </div>
            {panel("为什么不用 monetary risk 起手", "<p>因为货币化会立刻引入大量难以 defend 的外部假设。相比之下，<b>delay-equivalent minutes</b> 更贴近原始数据，也更适合作为课程项目中的 operational impact proxy。</p><p>这一步其实也让整份项目更诚实：我们承认当前数据最直接支持的是运营冲击而不是财务损失，因此先把 operational risk 讲清楚，再谈可能的成本外推，会比一开始就把所有东西硬换成钱更稳。</p><div class='formula-box'>delay-equivalent minutes = delay minutes + 180 × cancelled arrivals + 240 × diverted arrivals</div>")}
          </div>
          <div class="stack">
            {visual("多年度运营波动", "../charts/chart_1_multiyear_monthly_trend.png", "研究对象不是单个航班，而是持续多年的机场运营压力。", "visual-tall")}
            <div class="band-grid">
              {stat_block("峰值冲击月份", str(peak_month["period_label"]), f"{base.format_int(peak_month['disruption_impact_minutes'])} 分钟", "accent")}
              {stat_block("最高延误率月份", str(peak_delay_rate_month["period_label"]), base.format_pct(peak_delay_rate_month["delay_rate"], 1), "accent")}
            </div>
          </div>
        </div>
        """
    ).strip()

    p02 = dedent(
        f"""
        <div class="framework-page">
          <div class="section-kicker">Project Architecture</div>
          <div class="big-claim narrow">这份项目的价值，不在某一个模型，而在整条方法链。</div>
          <div class="framework-board">
            <div class="flow-step">
              <div class="flow-no">01</div>
              <h3>Risk Identification</h3>
              <p>先从 BTS cause taxonomy 出发，把原始原因重写成 internal / system / external。</p>
            </div>
            <div class="flow-step">
              <div class="flow-no">02</div>
              <h3>Frequency</h3>
              <p>判断扰动到达过程更像均匀计数，还是已经出现 clustering 与 over-dispersion。</p>
            </div>
            <div class="flow-step">
              <div class="flow-no">03</div>
              <h3>Severity</h3>
              <p>判断分钟损失的主体形状和尾部结构是不是可以用同一种分布解释。</p>
            </div>
            <div class="flow-step">
              <div class="flow-no">04</div>
              <h3>State Dependence</h3>
              <p>把机场月度状态从“高风险月份识别”升级到“坏状态会不会延续”。</p>
            </div>
            <div class="flow-step">
              <div class="flow-no">05</div>
              <h3>Aggregate Risk</h3>
              <p>把频率、严重度和依赖结构收束成 annual impact 的情景结果。</p>
            </div>
            <div class="flow-step">
              <div class="flow-no">06</div>
              <h3>Operator Actions</h3>
              <p>最后把统计输出翻译成运营商可以提前做什么、重点管什么。</p>
            </div>
          </div>
          <div class="framework-footer">
            <div>这就是它区别于“普通延误分析”的地方：不是只有结论，而是每一步都能解释为什么这样做。</div>
            <div class="capsule">from real operation problem to management action</div>
          </div>
        </div>
        """
    ).strip()

    p03 = dedent(
        f"""
        <div class="split">
          <div class="stack">
            <div class="section-kicker">Risk Language</div>
            <div class="big-claim narrow">为什么风险分类不能直接照搬 BTS taxonomy。</div>
            {panel("分类逻辑", "<p>原始字段是数据库语言，不是管理语言。课程里的风险识别更看重责任边界、控制手段和管理动作，所以要把 cause 重新组织成 <b>internal / system / external</b>。</p><p>例如 <b>Late Aircraft</b> 在原始表里是单独原因，但从运营链条看，它往往不是独立世界，而是前序航班延误、机组恢复、维修衔接失稳后的传导结果，因此更适合放回 internal disruption 的语义里。</p><p>这一步不是显著性检验，而是风险识别框架重写。</p>", "primary")}
            {panel("从原始原因到课程语言", base.render_compact_table(["原始 causes", "课程风险块", "分钟占比"], mapping_rows, "tight-table"))}
            <div class="band-grid thirds">
              {stat_block("Internal", base.format_pct(internal_share, 1), "Airline + Late Aircraft", "accent")}
              {stat_block("System", base.format_pct(system_share, 1), "NAS", "muted")}
              {stat_block("External", base.format_pct(external_share, 1), "Weather + Security", "muted")}
            </div>
          </div>
          <div class="stack">
            {visual("风险识别热力图", "../charts/chart_3_risk_heatmap.png", "risk identification 是项目起点，不是建模后的附属说明。", "visual-medium")}
            {visual("原始原因结构", "../charts/chart_2_delay_cause_breakdown.png", f"最大延误原因仍是 {top_cause['cause']}，占总延误分钟 {base.format_pct(top_cause['share_of_total_delay_minutes'], 1)}。", "visual-medium")}
          </div>
        </div>
        """
    ).strip()

    p04 = dedent(
        f"""
        <div class="split">
          <div class="stack">
            <div class="section-kicker">Distribution Choice</div>
            <div class="big-claim narrow">我们不是在堆分布，而是在回答两个不同的问题。</div>
            <div class="choice-lanes">
              <div class="choice-lane">
                <div class="lane-head">Frequency</div>
                <div class="lane-pair">
                  <div><b>Poisson</b><span>均值与方差接近时的基础计数世界。</span></div>
                  <div><b>Negative Binomial</b><span>更适合事件聚集、方差大于均值的高波动世界。</span></div>
                </div>
              </div>
              <div class="choice-lane">
                <div class="lane-head">Severity</div>
                <div class="lane-pair">
                  <div><b>Lognormal</b><span>右偏长尾，适合大多数月份不极端、少数月份拉长尾部。</span></div>
                  <div><b>Weibull</b><span>形状更灵活，用来检验分钟损失是否属于另一类右偏结构。</span></div>
                </div>
              </div>
            </div>
          </div>
          <div class="stack">
            {panel("AIC 是怎么用的", "<p><b>AIC = 2k - 2\\ln(L)</b>。它不是 p-value，不负责回答“显著不显著”，而是负责在同一候选集里平衡拟合质量与模型复杂度。</p><p>所以 AIC 的逻辑不是“证明这个分布是真的”，而是“在当前候选集里，这个分布作为工作模型更合适”。这也是为什么报告里要反复强调 relative fit，而不是 absolute truth。</p><div class='formula-box'>候选模型拟合 → 参数估计 → 计算 AIC → 选择 AIC 更低者</div>", "primary")}
            {panel("怎样读结果", f"<p>我们不仅看谁最低，也看低了多少。比如 Frequency 里 <b>{freq_biggest_gap['label']}</b> 的 ΔAIC 达到 <b>{base.format_float(freq_biggest_gap['delta_aic'], 1)}</b>，说明优势不是边缘性的；Severity 里最清晰的是 <b>{sev_biggest_gap['label']}</b>，ΔAIC 为 <b>{base.format_float(sev_biggest_gap['delta_aic'], 1)}</b>。</p><p>换句话说，AIC 让我们在“候选分布为什么这样定”和“最后为什么这样选”之间搭起了一座桥。没有这一页，后面的结果页很容易被读成黑箱。</p>")}
            <div class="step-ribbon">
              <span>候选集要小而合理</span>
              <span>比较标准要统一</span>
              <span>解释口吻要跟着 ΔAIC 强弱走</span>
            </div>
          </div>
        </div>
        """
    ).strip()

    p05 = dedent(
        f"""
        <div class="split split-bias">
          <div class="stack spacious">
            <div class="section-kicker">Frequency Finding</div>
            <div class="metric-hero">
              <div class="hero-number">{freq_all_nb}/6</div>
              <div class="hero-copy">
                <h2>所有 frequency 变量都选中了 Negative Binomial。</h2>
                <p>这不是模型偏好，而是数据在反复告诉我们：JFK 的扰动到达并不接近均匀、独立、低波动的 Poisson 世界。</p>
              </div>
            </div>
            {panel("最重要的读法", "<p>当方差显著高于均值，且 disrupted 状态下的计数整体抬升时，Poisson 会把真实世界压扁。Negative Binomial 更能容纳事件聚集与高波动。</p><p>这背后的运营含义也很直接：延误不是均匀散落在每个月里，而是会在某些压力月份集中爆发。对管理者来说，这意味着风险不是“平均分摊”的，而是存在高压窗口和恢复压力。</p>", "primary")}
            {panel("结果摘录", base.render_compact_table(["变量", "最优模型", "ΔAIC", "均值", "方差"], freq_rows, "tight-table"))}
          </div>
          <div class="stack">
            {visual("Frequency AIC 对比", f"../charts/{base.FREQUENCY_AIC_CHART}.png", "ΔAIC 越大，说明较优模型的优势越明确。", "visual-tall")}
          </div>
        </div>
        """
    ).strip()

    p06 = dedent(
        f"""
        <div class="split split-bias">
          <div class="stack spacious">
            <div class="section-kicker">Severity Finding</div>
            <div class="dual-hero">
              <div class="dual-box primary">
                <div class="dual-value">{sev_lognorm}</div>
                <div class="dual-label">Lognormal</div>
              </div>
              <div class="dual-box accent">
                <div class="dual-value">{sev_weibull}</div>
                <div class="dual-label">Weibull</div>
              </div>
            </div>
            <div class="claim-text">
              Severity 没有“一刀切”的统一答案。总体分钟损失和分风险块分钟损失的形状并不完全一样，
              这反而说明项目不是在把所有变量强塞进同一解释框架。
              对答辩来说，这一点反而很重要，因为它说明你们没有为了追求整齐而牺牲解释力。
            </div>
            {panel("结果摘录", base.render_compact_table(["变量", "最优模型", "ΔAIC", "样本量", "扰动态增量"], sev_rows, "tight-table"), "primary")}
          </div>
          <div class="stack">
            {visual("Severity AIC 对比", f"../charts/{base.SEVERITY_AIC_CHART}.png", "分钟损失的工作模型随变量而变，说明形状层次确实不同。", "visual-tall")}
            {panel("为什么这是好事", f"<p>最清晰的变量是 <b>{sev_biggest_gap['label']}</b>，ΔAIC 为 <b>{base.format_float(sev_biggest_gap['delta_aic'], 1)}</b>。这不是让结论更混乱，而是让解释更贴近变量本身。</p><p>如果所有 severity 变量都机械地落在同一种分布上，反而要怀疑是不是候选集太窄、或解释过度简化。当前结果更像真实世界：不同风险块的分钟损失尾部并不完全一样。</p>")}
          </div>
        </div>
        """
    ).strip()

    proof_items = "".join(
        f'<div class="proof-item"><div class="proof-title">{row[0]}</div><div class="proof-highlight">{row[1]}</div><div class="proof-text">{row[2]}</div></div>'
        for row in reality_rows
    )
    p07 = dedent(
        f"""
        <div class="reality-page">
          <div class="section-kicker">Reality Check</div>
          <div class="big-claim narrow">真正让项目站得住的，不是跑出模型，而是模型说的话和客观事实对得上。</div>
          <div class="proof-grid">
            {proof_items}
          </div>
          <div class="split compact">
            <div class="stack">
              {visual("状态影响差异", "../charts/chart_4_disruption_state_impact.png", "normal / disrupted 的影响差异是可以在图上直接看到的。", "visual-medium")}
            </div>
            <div class="stack">
              {panel("一句话总结", "<p>Negative Binomial 对应“方差远大于均值”的现实，internal 风险主导对应原始延误原因结构，holiday peak 风险最高符合高峰运行更脆弱的直觉，而 P(D→D) 高于 P(N→D) 则对应系统进入坏状态后恢复更慢的经验。</p><p>这一页的作用，其实是告诉听众：这份项目不是“统计结果看起来漂亮”，而是“统计结果和现实运行逻辑相互印证”。</p>", "warning")}
            </div>
          </div>
        </div>
        """
    ).strip()

    p08 = dedent(
        f"""
        <div class="split">
          <div class="stack">
            <div class="section-kicker">State Design</div>
            <div class="big-claim narrow">我们想做 Markov，不是为了多加一个方法，而是想回答“坏月份会不会延续”这个前面几层回答不了的问题。</div>
            {panel("为什么我们会想做 Markov", "<p>Frequency 和 severity 能告诉我们扰动出现得多不多、严重到什么程度，但它们默认看到的是“单个月份的分布形状”。</p><p>对机场运营来说，这还不够。管理者更关心的是：如果这个月已经进入高压状态，下个月会不会更容易继续坏下去？如果答案是会，那么运营风险就不只是单点冲击，而是带有持续性和恢复压力的过程。</p><p>所以 Markov 在这里的任务，是把问题从“高风险月份识别”升级为“状态会不会延续”。</p>", "primary")}
            {panel("为什么要先把状态定义清楚", f"<p>只有先把每个月定义成 <b>normal</b> 或 <b>disrupted</b>，后面才谈得上转移。否则所谓的 Markov 只是术语，没有稳定的状态基础。</p><p><b>{selected_rule['label']}</b> 与 <b>{alt_rule['label']}</b> 的 disrupted share 都是 <b>{base.format_pct(selected_rule['disrupted_share'], 1)}</b>，但前者跨年波动更低：<b>{base.format_float(selected_rule['yearly_disrupted_share_std'], 3)}</b> vs <b>{base.format_float(alt_rule['yearly_disrupted_share_std'], 3)}</b>。这意味着 composite rule 更适合拿来做状态依赖分析，因为它更稳定，也更像综合运营压力。</p><div class='formula-box'>composite = z(delay rate) + z(cancellation rate) + z(impact minutes)</div>", "warning")}
          </div>
          <div class="stack">
            {visual("状态规则比较", f"../charts/{base.STATE_RULE_CHART}.png", "先证明状态定义是稳定的，后面的转移分析才有意义。", "visual-medium")}
            {panel("什么样的结果才说明做 Markov 有意义", f"<p>如果转移结果显示 <b>P(D→D)</b> 和 <b>P(N→D)</b> 差不多，那么 Markov 只会告诉我们“坏状态没有明显惯性”，那它的附加价值就有限。</p><p>相反，如果进入 disrupted 之后继续留在 disrupted 的概率明显更高，就说明状态依赖确实存在，做 Markov 就不是多余的，而是在揭示前面分布模型看不到的时间结构。</p><p>也正因为如此，多年度样本扩展到 <b>{base.format_int(airport_months)}</b> 个 airport-month 后，Markov 才真正值得做。</p>")}
          </div>
        </div>
        """
    ).strip()

    p09 = dedent(
        f"""
        <div class="split split-hero">
          <div class="stack spacious">
            <div class="section-kicker">Markov Insight</div>
            <div class="ratio-panel">
              <div class="ratio-value">{base.format_float(transition_ratio, 2)}x</div>
              <div class="ratio-copy">结果表明，进入 disrupted 之后继续停留在 disrupted 的概率，约是正常月直接跌入 disrupted 的 <b>{base.format_float(transition_ratio, 2)}</b> 倍。</div>
            </div>
            {panel("结果为什么说明做 Markov 是有意义的", f"<p><b>P(N→D) = {base.format_float(p_nd, 3)}</b>，而 <b>P(D→D) = {base.format_float(p_dd, 3)}</b>。这说明坏状态不是随机独立地散落在时间线上，而是存在明显的延续倾向。</p><p>也就是说，Markov 在这里确实补充了新信息：如果只看前面的 frequency 和 severity，你知道风险高不高；但做了 Markov 之后，你还能知道系统一旦变坏，会不会更难立刻恢复。</p><p>因此，这一步不是重复已有结论，而是在证明“状态依赖”本身值得被纳入项目框架。</p>", "primary")}
            {panel("最终结果应该怎样解释", f"<p>坏状态不是吸收态，因为 <b>P(D→N) = {base.format_float(p_dn, 3)}</b> 仍然高于 <b>P(D→D)</b>，说明系统仍有恢复能力；但 <b>P(D→D)</b> 明显高于 <b>P(N→D)</b>，又说明恢复并不是自动完成的。</p><p>所以最稳妥的课堂口径是：<b>JFK 的月度运营状态存在有说服力的经验性状态依赖</b>。这不是过度夸大的强统计证明，但已经足以支持“坏月份会带来后续恢复压力”的管理解释。</p>")}
            <div class="step-ribbon">
              <span>先有动机：想知道坏状态会不会延续</span>
              <span>再有结论：结果表明确实存在延续倾向</span>
            </div>
          </div>
          <div class="stack">
            {visual("两状态转移矩阵", "../charts/chart_7_markov_transition_matrix.png", "Markov 的价值不只是给出矩阵，而是说明坏状态比随机跌入更容易延续。", "visual-tall")}
          </div>
        </div>
        """
    ).strip()

    p10 = dedent(
        f"""
        <div class="split">
          <div class="stack">
            <div class="section-kicker">Aggregate Risk</div>
            <div class="big-claim narrow">项目最终不是停在模型比较，而是回到 annual impact 会怎样移动。</div>
            {visual("情景风险曲线", "../charts/chart_6_aggregate_risk_scenarios.png", "用四个情景把 expected impact 和 tail risk 拉到同一张图里。", "visual-medium")}
            {panel("为什么要做四个情景", f"<p>基准情景是参考线，天气冲击与节假日高峰代表外部压力，disruption stress 代表系统处于更脆弱状态时的上移风险。相对基准，风险最高的 <b>{scenario_peak['label']}</b> 在 Expected 上高出 <b>{base.format_int(scenario_peak['expected_impact_minutes'] - base_expected)}</b> 分钟，在 VaR95 上高出 <b>{base.format_int(scenario_peak['var_95_minutes'] - base_var95)}</b> 分钟。</p><p>这一页真正想说明的是：项目最后没有停留在“哪个分布更好”，而是回到管理者会问的那句话上，<b>如果运行环境变差，年度风险会被抬高到什么程度</b>。</p>", "primary")}
          </div>
          <div class="stack">
            {panel("情景排序", base.render_compact_table(["情景", "Expected", "VaR95", "TVaR95"], scenario_rows, "tight-table"))}
            <div class="band-grid">
              {stat_block("最高 VaR95", base.format_int(scenario_peak["var_95_minutes"]), scenario_peak["label"], "accent")}
              {stat_block("基准 Expected", base.format_int(base_expected), "用于比较增量", "muted")}
            </div>
          </div>
        </div>
        """
    ).strip()

    action_items = "".join(
        f'<div class="action-item"><div class="action-head">{row[0]}</div><div class="action-main">{row[1]}</div><div class="action-note">{row[2]}</div></div>'
        for row in operator_rows
    )
    p11 = dedent(
        f"""
        <div class="finale">
          <div class="section-kicker">Operator Value</div>
          <div class="big-claim">这份风险分析，最终应该帮助运营商更早决定资源放在哪里。</div>
          <div class="action-grid">
            {action_items}
          </div>
          <div class="closing-band">
            <div>
              <div class="closing-title">为什么这版更像一个完整项目</div>
              <p>它不是从模型出发，而是从真实运营问题出发；它不是只给结果，而是把为什么这样分类、为什么只比这四个分布、为什么用 AIC、为什么可以做 Markov 全部串成了一条可解释的方法链。</p>
              <p>对答辩来说，这一点尤其重要，因为听众看到的不是若干孤立分析，而是一套从识别、建模到解释、再到行动建议的完整项目结构。</p>
            </div>
            <div>
              <div class="closing-title">最终展示口径</div>
              <p>你可以把它定义成：<b>一个从机场真实运行问题出发，经过风险识别、统计建模、状态依赖分析，再回到运营决策的完整 operational risk project。</b></p>
              <p>这样收束时，老师和组员看到的就不只是“你们做了很多页”，而是“你们确实完成了一条有设计感的方法链”。</p>
            </div>
          </div>
        </div>
        """
    ).strip()

    pages = [
        showcase_page("P00 Cover", "为什么飞机延误值得做成一整套风险项目", "用一个强开场先建立项目价值，而不是先掉进方法细节。", p00, 0, total_pages, "dark"),
        showcase_page("P01 Motivation", "为什么选择飞机延误作为风险分析对象", "让读者先相信这个问题值得研究，再进入方法。", p01, 1, total_pages),
        showcase_page("P02 Architecture", "这份项目到底比普通延误分析多做了什么", "把整条方法链一口气亮出来，让项目丰富度立刻可见。", p02, 2, total_pages, "ink"),
        showcase_page("P03 Risk Language", "为什么风险分类不能直接照搬原始 taxonomy", "先把风险语言改对，后面的模型才有管理意义。", p03, 3, total_pages),
        showcase_page("P04 Model Choice", "为什么只比较这四个分布，以及怎么选", "候选集要小而合理，比较标准要统一。", p04, 4, total_pages),
        showcase_page("P05 Frequency", "Frequency 结果告诉了我们什么", "把最强的计数层结论前置放大。", p05, 5, total_pages),
        showcase_page("P06 Severity", "Severity 结果为什么不能只用一个分布解释", "让读者看到变量之间的形状差异，而不是只记模型名字。", p06, 6, total_pages),
        showcase_page("P07 Reality Check", "模型结果是否真的符合客观事实", "这页负责提升整份项目的可信度。", p07, 7, total_pages),
        showcase_page("P08 Why Markov", "为什么我们想做 Markov 分析", "先说明做 Markov 想回答什么问题，以及什么时候这件事才值得做。", p08, 8, total_pages),
        showcase_page("P09 Markov Result", "Markov 结果为什么说明这一步有意义", "先证明状态依赖确实存在，再解释转移矩阵和管理含义。", p09, 9, total_pages, "ink"),
        showcase_page("P10 Aggregate Risk", "最终 annual impact 在不同情景下会怎样移动", "把频率、严重度和依赖结构收束成管理者能读的结果。", p10, 10, total_pages),
        showcase_page("P11 Operator Value", "这个项目最终能给运营商什么帮助", "最后一页不是总结，而是把统计输出翻译成行动。", p11, 11, total_pages, "dark"),
    ]

    return dedent(
        """
        <!doctype html>
        <html lang="zh-CN">
        <head>
          <meta charset="utf-8">
          <meta name="viewport" content="width=device-width, initial-scale=1">
          <title>JFK Operational Risk Showcase · 中文展示版</title>
          <style>
            :root {
              --bg: #d7ddd9;
              --page: #f7f5ef;
              --ink: #11212b;
              --muted: #60707a;
              --line: rgba(17, 33, 43, 0.14);
              --primary: #11212b;
              --accent: #d56f33;
              --shadow: 0 28px 50px rgba(10, 20, 30, 0.14);
              --font-body: "Microsoft YaHei", "PingFang SC", "Noto Sans SC", sans-serif;
              --font-display: "Georgia", "Times New Roman", serif;
              --font-mono: "Consolas", monospace;
            }
            *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
            body {
              background:
                radial-gradient(circle at 18% 14%, rgba(213, 111, 51, 0.10), transparent 22%),
                radial-gradient(circle at 82% 8%, rgba(17, 33, 43, 0.10), transparent 20%),
                linear-gradient(180deg, #eef1ec 0%, #d7ddd9 100%);
              color: var(--ink);
              font-family: var(--font-body);
              padding: 26px;
              min-height: 100vh;
            }
            body::before {
              content: "";
              position: fixed;
              inset: 0;
              pointer-events: none;
              background-image:
                linear-gradient(rgba(255,255,255,0.08) 1px, transparent 1px),
                linear-gradient(90deg, rgba(255,255,255,0.08) 1px, transparent 1px);
              background-size: 36px 36px;
              opacity: 0.28;
            }
            .report-stack { position: relative; z-index: 1; width: fit-content; margin: 0 auto; }
            .report-page {
              width: 1017px;
              height: 720px;
              min-width: 1017px;
              max-width: 1017px;
              min-height: 720px;
              max-height: 720px;
              overflow: hidden;
              background: var(--page);
              border-radius: 34px;
              box-shadow: var(--shadow);
              margin: 0 auto 28px;
              border: 1px solid rgba(17, 33, 43, 0.08);
              position: relative;
            }
            .report-page::after {
              content: "";
              position: absolute;
              inset: 0;
              pointer-events: none;
              background:
                linear-gradient(135deg, rgba(255,255,255,0.00), rgba(255,255,255,0.16)),
                linear-gradient(90deg, transparent 0%, rgba(255,255,255,0.06) 48%, transparent 100%);
              opacity: 0.65;
            }
            .report-page.dark { background: linear-gradient(145deg, #10202b, #18303a 58%, #223540); color: #eef5f4; }
            .report-page.ink { background: linear-gradient(180deg, #edf1ee, #f7f5ef); }
            .hd, .ct { position: relative; z-index: 1; }
            .hd {
              height: 72px;
              padding: 0 30px;
              display: flex;
              align-items: center;
              justify-content: space-between;
              border-bottom: 1px solid rgba(17, 33, 43, 0.10);
            }
            .dark .hd, .dark .sm { border-color: rgba(255,255,255,0.10); }
            .ct { height: 648px; padding: 12px 30px 14px; overflow: hidden; }
            .dark .page-title, .dark .chapter-tag { color: #e6efee; }
            .chapter-tag {
              display: inline-flex;
              align-items: center;
              padding: 6px 10px;
              border-radius: 999px;
              background: rgba(17, 33, 43, 0.07);
              color: var(--primary);
              font-size: 10px;
              letter-spacing: 0.12em;
              text-transform: uppercase;
              font-weight: 700;
            }
            .dark .chapter-tag { background: rgba(255,255,255,0.10); }
            .page-title { font-size: 14px; font-weight: 700; color: var(--primary); }
            .section-kicker {
              font-size: 10px;
              letter-spacing: 0.16em;
              text-transform: uppercase;
              font-weight: 700;
              color: var(--accent);
              margin-bottom: 10px;
            }
            .dark .section-kicker { color: #f0b996; }
            .big-claim {
              font-size: 36px;
              line-height: 1.08;
              letter-spacing: -0.02em;
              font-weight: 700;
              max-width: 15ch;
              color: var(--primary);
            }
            .big-claim.narrow { max-width: 17ch; }
            .dark .big-claim { color: #f4fbfa; }
            .claim-text {
              margin-top: 12px;
              max-width: 62ch;
              font-size: 13px;
              line-height: 1.7;
              color: var(--muted);
            }
            .dark .claim-text { color: rgba(238, 245, 244, 0.80); }

            .split { height: 568px; display: grid; grid-template-columns: 1.04fr 0.96fr; gap: 18px; }
            .split.compact { height: 220px; margin-top: 16px; }
            .split-hero { grid-template-columns: 1.02fr 0.98fr; }
            .split-bias { grid-template-columns: 1.08fr 0.92fr; }
            .stack { display: flex; flex-direction: column; gap: 14px; min-width: 0; }
            .stack.spacious { gap: 16px; }
            .finale { height: 568px; display: flex; flex-direction: column; }

            .poster { height: 568px; display: grid; grid-template-columns: 1.16fr 0.84fr; grid-template-rows: 1fr 86px; gap: 16px 18px; }
            .poster-copy { padding: 18px 0 0; display: flex; flex-direction: column; justify-content: center; }
            .eyebrow { font-size: 11px; letter-spacing: 0.22em; text-transform: uppercase; color: rgba(238, 245, 244, 0.76); font-weight: 700; }
            .poster-copy h1 { margin-top: 18px; font-size: 54px; line-height: 0.98; letter-spacing: -0.04em; max-width: 11ch; }
            .poster-copy p { margin-top: 18px; max-width: 58ch; font-size: 14px; line-height: 1.72; color: rgba(238, 245, 244, 0.82); }
            .poster-line { margin-top: 18px; width: 88px; height: 4px; border-radius: 999px; background: linear-gradient(90deg, var(--accent), rgba(255,255,255,0.18)); }
            .poster-caption { margin-top: 16px; max-width: 58ch; font-size: 12px; line-height: 1.7; color: rgba(238, 245, 244, 0.72); }
            .poster-stats { display: flex; flex-direction: column; justify-content: center; gap: 14px; }
            .poster-roadmap { grid-column: 1 / -1; display: grid; grid-template-columns: repeat(6, 1fr); gap: 10px; align-self: end; }
            .poster-roadmap span { display: flex; align-items: center; justify-content: center; min-height: 72px; border-radius: 999px; background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.10); font-size: 11px; letter-spacing: 0.05em; }

            .stat-block { min-height: 96px; padding: 14px 16px; background: rgba(255,255,255,0.74); border: 1px solid rgba(17, 33, 43, 0.08); border-radius: 22px; display: flex; flex-direction: column; justify-content: center; gap: 6px; }
            .stat-block.light { background: rgba(255,255,255,0.10); border-color: rgba(255,255,255,0.12); }
            .stat-block.accent { background: rgba(213, 111, 51, 0.10); border-color: rgba(213, 111, 51, 0.18); }
            .stat-block.muted { background: rgba(17, 33, 43, 0.05); }
            .stat-label { font-size: 10px; letter-spacing: 0.12em; text-transform: uppercase; font-weight: 700; color: var(--muted); }
            .light .stat-label, .dark .stat-label { color: rgba(238, 245, 244, 0.72); }
            .stat-value { font-size: 28px; line-height: 1; font-weight: 700; color: var(--primary); font-family: var(--font-display); }
            .dark .stat-value, .light .stat-value { color: #f4fbfa; }
            .stat-note { font-size: 11px; line-height: 1.45; color: var(--muted); }
            .dark .stat-note, .light .stat-note { color: rgba(238, 245, 244, 0.78); }

            .panel, .visual-block { padding: 16px 18px; border-radius: 24px; background: rgba(255,255,255,0.64); border: 1px solid rgba(17, 33, 43, 0.08); backdrop-filter: blur(6px); }
            .panel.primary { background: rgba(17, 33, 43, 0.92); color: #eef5f4; border-color: rgba(17, 33, 43, 0.94); }
            .panel.warning { background: rgba(213, 111, 51, 0.10); border-color: rgba(213, 111, 51, 0.22); }
            .dark .panel:not(.primary) { background: rgba(255,255,255,0.08); border-color: rgba(255,255,255,0.10); color: #eef5f4; }
            .panel-title { font-size: 11px; letter-spacing: 0.10em; text-transform: uppercase; font-weight: 700; color: var(--muted); margin-bottom: 12px; }
            .primary .panel-title { color: rgba(238, 245, 244, 0.72); }
            .dark .panel-title, .dark .visual-caption { color: rgba(238, 245, 244, 0.72); }
            .panel-body { font-size: 12px; line-height: 1.68; color: inherit; }
            .panel-body p + p, .panel-body ul + p, .panel-body p + ul { margin-top: 8px; }
            .panel-body ul { padding-left: 18px; }
            .panel-body li + li { margin-top: 5px; }
            .formula-box { margin-top: 8px; padding: 10px 12px; border-radius: 16px; background: rgba(255,255,255,0.12); border: 1px solid rgba(255,255,255,0.14); font-family: var(--font-mono); font-size: 10px; line-height: 1.55; }
            .report-page:not(.dark) .formula-box { background: rgba(17, 33, 43, 0.04); border-color: rgba(17, 33, 43, 0.10); }

            .visual-frame {
              width: 100%;
              border-radius: 18px;
              overflow: hidden;
              border: 1px solid rgba(17, 33, 43, 0.08);
              background: linear-gradient(180deg, rgba(255,255,255,0.98), rgba(246, 244, 239, 0.98));
              display: flex;
              align-items: center;
              justify-content: center;
              padding: 12px;
            }
            .visual-frame img {
              display: block;
              width: 100%;
              height: 100%;
              object-fit: contain;
              object-position: center center;
            }
            .visual-caption { margin-top: 10px; font-size: 10px; line-height: 1.55; color: var(--muted); }
            .visual-medium { height: 206px; }
            .visual-tall { height: 368px; }

            .inline-proof, .step-ribbon { display: flex; flex-wrap: wrap; gap: 10px; }
            .inline-proof span, .step-ribbon span, .capsule { display: inline-flex; align-items: center; justify-content: center; min-height: 38px; padding: 0 12px; border-radius: 999px; background: rgba(17, 33, 43, 0.06); border: 1px solid rgba(17, 33, 43, 0.08); font-size: 11px; line-height: 1.4; }
            .dark .inline-proof span, .dark .step-ribbon span { background: rgba(255,255,255,0.08); border-color: rgba(255,255,255,0.10); }
            .band-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px; }
            .band-grid.thirds { grid-template-columns: repeat(3, 1fr); }

            .framework-page { height: 568px; display: flex; flex-direction: column; justify-content: space-between; }
            .framework-board { margin-top: 18px; display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; }
            .flow-step { min-height: 146px; padding: 18px; border-radius: 24px; background: rgba(255,255,255,0.76); border: 1px solid rgba(17, 33, 43, 0.08); }
            .flow-no { font-size: 10px; letter-spacing: 0.18em; text-transform: uppercase; color: var(--accent); font-weight: 700; }
            .flow-step h3 { margin-top: 10px; font-size: 18px; line-height: 1.15; color: var(--primary); }
            .flow-step p { margin-top: 10px; font-size: 12px; line-height: 1.65; color: var(--muted); }
            .framework-footer { display: flex; align-items: center; justify-content: space-between; gap: 16px; padding-top: 14px; border-top: 1px solid rgba(17, 33, 43, 0.10); font-size: 12px; line-height: 1.6; color: var(--muted); }

            .choice-lanes { display: flex; flex-direction: column; gap: 14px; }
            .choice-lane { padding: 16px 18px; border-radius: 24px; background: rgba(255,255,255,0.74); border: 1px solid rgba(17, 33, 43, 0.08); }
            .lane-head { font-size: 11px; letter-spacing: 0.14em; text-transform: uppercase; font-weight: 700; color: var(--accent); margin-bottom: 12px; }
            .lane-pair { display: grid; grid-template-columns: repeat(2, 1fr); gap: 14px; }
            .lane-pair div { padding-top: 4px; border-top: 1px solid rgba(17, 33, 43, 0.10); font-size: 12px; line-height: 1.6; }
            .lane-pair b { display: block; margin-bottom: 6px; font-size: 15px; color: var(--primary); }
            .lane-pair span { color: var(--muted); }

            .metric-hero, .ratio-panel, .dual-hero, .closing-band, .action-grid, .proof-grid { display: grid; gap: 14px; }
            .metric-hero { grid-template-columns: 180px 1fr; align-items: center; }
            .hero-number, .ratio-value, .dual-value { font-size: 88px; line-height: 0.9; letter-spacing: -0.05em; font-family: var(--font-display); color: var(--primary); }
            .hero-copy h2 { font-size: 30px; line-height: 1.05; max-width: 12ch; color: var(--primary); }
            .hero-copy p { margin-top: 10px; font-size: 13px; line-height: 1.68; color: var(--muted); }
            .dual-hero { grid-template-columns: repeat(2, 1fr); }
            .dual-box { min-height: 180px; border-radius: 26px; padding: 20px; border: 1px solid rgba(17, 33, 43, 0.08); display: flex; flex-direction: column; justify-content: center; gap: 10px; }
            .dual-box.primary { background: rgba(17, 33, 43, 0.92); color: #eef5f4; }
            .dual-box.primary .dual-value { color: #eef5f4; }
            .dual-box.accent { background: rgba(213, 111, 51, 0.10); }
            .dual-label { font-size: 12px; letter-spacing: 0.14em; text-transform: uppercase; font-weight: 700; }

            .reality-page { height: 568px; display: flex; flex-direction: column; }
            .proof-grid { grid-template-columns: repeat(4, 1fr); margin-top: 18px; }
            .proof-item { min-height: 184px; padding: 16px; border-radius: 24px; background: rgba(255,255,255,0.72); border: 1px solid rgba(17, 33, 43, 0.08); display: flex; flex-direction: column; gap: 10px; }
            .proof-title { font-size: 11px; letter-spacing: 0.08em; text-transform: uppercase; font-weight: 700; color: var(--muted); }
            .proof-highlight { font-size: 20px; line-height: 1.2; color: var(--primary); font-weight: 700; }
            .proof-text { font-size: 12px; line-height: 1.62; color: var(--muted); }

            .ratio-panel { grid-template-columns: 170px 1fr; align-items: center; min-height: 170px; padding: 18px 20px; border-radius: 28px; background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.10); }
            .ratio-value { color: #eef5f4; }
            .ratio-copy { font-size: 15px; line-height: 1.62; color: rgba(238, 245, 244, 0.84); }

            .action-grid { margin-top: 20px; grid-template-columns: repeat(4, 1fr); }
            .action-item { min-height: 206px; padding: 18px; border-radius: 24px; background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.12); display: flex; flex-direction: column; gap: 10px; }
            .action-head { font-size: 11px; letter-spacing: 0.12em; text-transform: uppercase; color: #f0b996; font-weight: 700; }
            .action-main { font-size: 18px; line-height: 1.35; font-weight: 700; color: #f4fbfa; }
            .action-note { font-size: 12px; line-height: 1.65; color: rgba(238, 245, 244, 0.76); }
            .closing-band { margin-top: 18px; grid-template-columns: repeat(2, 1fr); }
            .closing-band > div { padding: 18px; border-radius: 24px; background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.10); }
            .closing-title { font-size: 11px; letter-spacing: 0.12em; text-transform: uppercase; color: #f0b996; font-weight: 700; margin-bottom: 10px; }
            .closing-band p { font-size: 12px; line-height: 1.7; color: rgba(238, 245, 244, 0.82); }

            .data-table { width: 100%; border-collapse: collapse; font-size: 10.5px; line-height: 1.45; }
            .data-table thead th { text-align: left; font-size: 10px; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); padding-bottom: 8px; border-bottom: 1px solid rgba(17, 33, 43, 0.10); }
            .data-table tbody td { padding: 7px 0; border-bottom: 1px solid rgba(17, 33, 43, 0.06); vertical-align: top; }
            .primary .data-table thead th, .primary .data-table tbody td { border-color: rgba(255,255,255,0.12); color: inherit; }
            .primary .data-table thead th { color: rgba(238, 245, 244, 0.70); }

            @media (prefers-reduced-motion: no-preference) {
              .report-page { opacity: 0; transform: translateY(16px); animation: page-enter 0.8s cubic-bezier(.2,.8,.2,1) forwards; animation-delay: var(--enter-delay); }
              .hero-number, .ratio-value, .dual-value, .stat-value { animation: settle 0.9s cubic-bezier(.18,.9,.2,1) both; animation-delay: calc(var(--enter-delay) + 120ms); }
            }
            @keyframes page-enter {
              from { opacity: 0; transform: translateY(16px); }
              to { opacity: 1; transform: translateY(0); }
            }
            @keyframes settle {
              from { opacity: 0; transform: translateY(10px) scale(0.98); }
              to { opacity: 1; transform: translateY(0) scale(1); }
            }

            @media print {
              @page { size: A4 landscape; margin: 8mm; }
              body { padding: 0; background: white; -webkit-print-color-adjust: exact; print-color-adjust: exact; }
              body::before { display: none; }
              .report-page { margin: 0 0 6mm 0; box-shadow: none; border-radius: 0; break-after: page; page-break-after: always; }
              .report-page:last-child { break-after: auto; page-break-after: auto; }
            }
          </style>
        </head>
        <body>
          <main class="report-stack">
        """
    ).strip() + "".join(pages) + "</main></body></html>\n"


def build_showcase_html_en(html_output_zh: str) -> str:
    html_output_en = html_output_zh.replace('lang="zh-CN"', 'lang="en"')
    html_output_en = html_output_en.replace(
        "<title>JFK Operational Risk Showcase · 中文展示版</title>",
        "<title>JFK Operational Risk Showcase · English Version</title>",
    )

    replacements = [
        ("中文展示版", "English Version"),
        ("为什么飞机延误值得做成一整套风险项目", "Why Flight Delays Deserve a Full Operational Risk Project"),
        ("为什么选择飞机延误作为风险分析对象", "Why Flight Delays Were Chosen as the Risk Analysis Problem"),
        ("这份项目到底比普通延误分析多做了什么", "What This Project Adds Beyond a Standard Delay Analysis"),
        ("为什么风险分类不能直接照搬原始 taxonomy", "Why the Risk Taxonomy Cannot Simply Copy the Raw BTS Categories"),
        ("为什么只比较这四个分布，以及怎么选", "Why These Four Distributions Were Compared and How the Choice Was Made"),
        ("Frequency 结果告诉了我们什么", "What the Frequency Results Tell Us"),
        ("Severity 结果为什么不能只用一个分布解释", "Why the Severity Results Cannot Be Explained by One Distribution Alone"),
        ("模型结果是否真的符合客观事实", "Do the Model Results Match Operational Reality"),
        ("为什么我们想做 Markov 分析", "Why We Wanted to Run a Markov Analysis"),
        ("Markov 结果为什么说明这一步有意义", "Why the Markov Results Show This Step Was Worth Doing"),
        ("最终 annual impact 在不同情景下会怎样移动", "How Annual Impact Shifts Across Scenarios"),
        ("这个项目最终能给运营商什么帮助", "What This Project Ultimately Helps Operators Do"),
        ("用一个强开场先建立项目价值，而不是先掉进方法细节。", "Open with project value first, not with method detail."),
        ("让读者先相信这个问题值得研究，再进入方法。", "Make the audience believe the problem matters before moving into method."),
        ("把整条方法链一口气亮出来，让项目丰富度立刻可见。", "Show the full method chain at once so the project depth is immediately visible."),
        ("先把风险语言改对，后面的模型才有管理意义。", "Get the risk language right first so the later models have management meaning."),
        ("候选集要小而合理，比较标准要统一。", "Keep the candidate set small and defensible, and keep the comparison rule consistent."),
        ("把最强的计数层结论前置放大。", "Bring the strongest count-model finding forward and make it explicit."),
        ("让读者看到变量之间的形状差异，而不是只记模型名字。", "Let the audience see shape differences across variables, not just model names."),
        ("这页负责提升整份项目的可信度。", "This page is where the project earns credibility."),
        ("先说明做 Markov 想回答什么问题，以及什么时候这件事才值得做。", "Explain first what Markov is meant to answer and when it is worth using."),
        ("先证明状态依赖确实存在，再解释转移矩阵和管理含义。", "Show that state dependence exists first, then interpret the transition matrix and its management meaning."),
        ("把频率、严重度和依赖结构收束成管理者能读的结果。", "Bring frequency, severity, and dependence together into a manager-readable result."),
        ("最后一页不是总结，而是把统计输出翻译成行动。", "The final page is not a recap; it translates statistical output into action."),
        ("这不是一份“把结果排版得更好看”的 HTML。", "This is not just an HTML file that makes the results look nicer."),
        ("这是一份从 <b>选题动机</b>、<b>风险识别</b>、<b>分布选择</b>、       <b>状态依赖</b> 到 <b>运营启示</b> 的完整项目展示版。", "It is a full project showcase that runs from <b>problem motivation</b>, <b>risk identification</b>, and <b>distribution choice</b> to <b>state dependence</b> and <b>operator implications</b>."),
        ("如果你想让读者感受到“这个项目不只是会跑模型，而是真的设计过方法链”，这就是应该呈现出来的样子。", "If you want the audience to feel that this project is not just model execution but a deliberately designed analytical framework, this is the form it should take."),
        ("样本年份", "Sample Window"),
        ("覆盖架次", "Flights Covered"),
        ("研究粒度", "Modeling Granularity"),
        ("多年度机场月度序列", "Multi-year airport-month series"),
        ("总到港航班", "Total arriving flights"),
        ("airline-month / airport-month", "airline-month / airport-month"),
        ("选题价值", "Problem Value"),
        ("风险识别", "Risk Identification"),
        ("分布抉择", "Distribution Choice"),
        ("状态依赖", "State Dependence"),
        ("运营决策", "Operator Action"),
        ("延误是最像 operational risk 的机场问题之一。", "Flight delay is one of the clearest operational risk problems in airport operations."),
        ("它同时具备真实运营压力、公开可复现数据，以及可以量化的损失代理。       这意味着它既能讲现实问题，也能撑起完整方法链，而不只是做一页描述统计。       更重要的是，延误、取消、备降之间天然存在传导关系，所以这个题目非常适合把 frequency、severity、dependence 和 aggregate risk 串成一个完整故事。", "It combines real operating pressure, public reproducible data, and a measurable loss proxy. That means it can support both a real-world story and a full analytical pipeline rather than a single descriptive slide. More importantly, delays, cancellations, and diversions naturally propagate into one another, so the topic is well suited to connect frequency, severity, dependence, and aggregate risk into one coherent narrative."),
        ("平均延误率", "Average delay rate"),
        ("平均取消率", "Average cancellation rate"),
        ("总冲击", "Total impact"),
        ("为什么不用 monetary risk 起手", "Why We Do Not Start with Monetary Risk"),
        ("因为货币化会立刻引入大量难以 defend 的外部假设。相比之下，<b>delay-equivalent minutes</b> 更贴近原始数据，也更适合作为课程项目中的 operational impact proxy。", "Because full monetization would immediately introduce many external assumptions that are hard to defend. By contrast, <b>delay-equivalent minutes</b> stay much closer to the raw data and work better as an operational impact proxy for this project."),
        ("这一步其实也让整份项目更诚实：我们承认当前数据最直接支持的是运营冲击而不是财务损失，因此先把 operational risk 讲清楚，再谈可能的成本外推，会比一开始就把所有东西硬换成钱更稳。", "This also makes the project more honest: the current data directly support operational impact much more clearly than financial loss, so it is more defensible to explain operational risk first and only then discuss possible cost extrapolation."),
        ("多年度运营波动", "Multi-Year Operating Variation"),
        ("研究对象不是单个航班，而是持续多年的机场运营压力。", "The object of study is not a single flight but a multi-year pattern of airport operating pressure."),
        ("峰值冲击月份", "Peak impact month"),
        ("最高延误率月份", "Highest delay-rate month"),
        ("这份项目的价值，不在某一个模型，而在整条方法链。", "The value of this project does not sit in one model; it sits in the full analytical chain."),
        ("这就是它区别于“普通延误分析”的地方：不是只有结论，而是每一步都能解释为什么这样做。", "This is what separates it from a standard delay analysis: it does not only report findings, it explains why each step was designed that way."),
        ("为什么风险分类不能直接照搬 BTS taxonomy。", "Why the risk classification cannot simply copy the BTS taxonomy."),
        ("分类逻辑", "Classification Logic"),
        ("从原始原因到课程语言", "From Raw Causes to Course Risk Language"),
        ("原始 causes", "Raw causes"),
        ("课程风险块", "Course risk block"),
        ("分钟占比", "Share of delay minutes"),
        ("风险识别热力图", "Risk Identification Heatmap"),
        ("risk identification 是项目起点，不是建模后的附属说明。", "Risk identification is the starting point of the project, not a post-model add-on."),
        ("原始原因结构", "Raw Cause Structure"),
        ("为什么只比较这四个分布，以及怎么选", "Why These Four Distributions Were Compared and How the Choice Was Made"),
        ("我们不是在堆分布，而是在回答两个不同的问题。", "We are not stacking distributions; we are answering two different modeling questions."),
        ("AIC 是怎么用的", "How AIC Is Used"),
        ("怎样读结果", "How to Read the Comparison"),
        ("候选集要小而合理", "Keep the candidate set small and defensible"),
        ("比较标准要统一", "Use one consistent comparison rule"),
        ("解释口吻要跟着 ΔAIC 强弱走", "Scale the wording to the strength of ΔAIC"),
        ("所有 frequency 变量都选中了 Negative Binomial。", "All frequency variables selected Negative Binomial."),
        ("这不是模型偏好，而是数据在反复告诉我们：JFK 的扰动到达并不接近均匀、独立、低波动的 Poisson 世界。", "This is not a stylistic preference for one model. It is the data repeatedly showing that disruptions at JFK do not behave like a uniform, independent, low-variance Poisson world."),
        ("最重要的读法", "How to Read This Finding"),
        ("结果摘录", "Result Snapshot"),
        ("Severity 没有“一刀切”的统一答案。总体分钟损失和分风险块分钟损失的形状并不完全一样，       这反而说明项目不是在把所有变量强塞进同一解释框架。       对答辩来说，这一点反而很重要，因为它说明你们没有为了追求整齐而牺牲解释力。", "Severity does not have a one-size-fits-all answer. The shape of total delay minutes is not identical to the shape of delay minutes within each risk block. That is actually a strength: it shows the project did not force all variables into one artificially tidy explanation. For presentation purposes, that matters because it shows interpretive accuracy was not sacrificed for visual neatness."),
        ("为什么这是好事", "Why This Is a Strength"),
        ("真正让项目站得住的，不是跑出模型，而是模型说的话和客观事实对得上。", "What makes the project credible is not merely fitting models, but showing that the models say things that match operational reality."),
        ("一句话总结", "One-Sentence Takeaway"),
        ("为什么我们想做 Markov 分析", "Why We Wanted a Markov Analysis"),
        ("我们想做 Markov，不是为了多加一个方法，而是想回答“坏月份会不会延续”这个前面几层回答不了的问题。", "We did not want Markov just to add another method. We wanted it because the earlier layers cannot answer one key question: do bad months tend to persist?"),
        ("为什么我们会想做 Markov", "Why We Wanted Markov"),
        ("为什么要先把状态定义清楚", "Why the State Definition Had to Be Settled First"),
        ("状态规则比较", "State Rule Comparison"),
        ("先证明状态定义是稳定的，后面的转移分析才有意义。", "The state definition has to be shown to be stable before the transition analysis is meaningful."),
        ("什么样的结果才说明做 Markov 有意义", "What Kind of Result Makes Markov Worthwhile"),
        ("Markov 结果为什么说明这一步有意义", "Why the Markov Results Show This Step Was Worth Doing"),
        ("结果表明，进入 disrupted 之后继续停留在 disrupted 的概率，约是正常月直接跌入 disrupted 的 <b>2.54</b> 倍。", "The results show that once the system enters a disrupted state, the probability of remaining disrupted is about <b>2.54</b> times the probability that a normal month directly falls into disruption."),
        ("结果为什么说明做 Markov 是有意义的", "Why the Results Show Markov Was Useful"),
        ("最终结果应该怎样解释", "How the Final Result Should Be Interpreted"),
        ("先有动机：想知道坏状态会不会延续", "First the motivation: do bad states persist"),
        ("再有结论：结果表明确实存在延续倾向", "Then the conclusion: the results show that persistence is real"),
        ("两状态转移矩阵", "Two-State Transition Matrix"),
        ("Markov 的价值不只是给出矩阵，而是说明坏状态比随机跌入更容易延续。", "The value of Markov is not just producing a matrix; it is showing that bad states are more likely to persist than to arise at random."),
        ("项目最终不是停在模型比较，而是回到 annual impact 会怎样移动。", "The project does not stop at model comparison; it returns to how annual impact shifts."),
        ("情景风险曲线", "Scenario Risk Curves"),
        ("用四个情景把 expected impact 和 tail risk 拉到同一张图里。", "Four scenarios place expected impact and tail risk onto one comparable view."),
        ("为什么要做四个情景", "Why Four Scenarios Were Used"),
        ("情景排序", "Scenario Ranking"),
        ("最高 VaR95", "Highest VaR95"),
        ("基准 Expected", "Base Expected"),
        ("用于比较增量", "Used as the baseline for increments"),
        ("这份风险分析，最终应该帮助运营商更早决定资源放在哪里。", "This risk analysis should ultimately help operators decide earlier where resources need to go."),
        ("为什么这版更像一个完整项目", "Why This Version Feels Like a Complete Project"),
        ("最终展示口径", "Recommended Final Framing"),
        ("对答辩来说，这一点尤其重要，因为听众看到的不是若干孤立分析，而是一套从识别、建模到解释、再到行动建议的完整项目结构。", "For a final presentation, this matters because the audience sees not isolated analyses, but a complete project structure moving from identification to modeling, interpretation, and action."),
        ("这样收束时，老师和组员看到的就不只是“你们做了很多页”，而是“你们确实完成了一条有设计感的方法链”。", "With this framing, the audience does not just feel that many slides were made; they see that a deliberately designed analytical chain was completed."),
        ("值得被做成一整套风险项目", "deserves a full operational risk project"),
        ("选题动机", "problem motivation"),
        ("分布选择", "distribution choice"),
        ("运营启示", "operator implications"),
        ("真实运营压力", "real operating pressure"),
        ("公开可复现数据", "public reproducible data"),
        ("可以量化的损失代理", "a measurable loss proxy"),
        ("延误、取消、备降之间天然存在传导关系，所以这个题目非常适合把 frequency、severity、dependence 和 aggregate risk 串成一个完整故事。", "Delays, cancellations, and diversions naturally propagate into one another, which makes this topic especially suitable for linking frequency, severity, dependence, and aggregate risk into one coherent story."),
        ("总冲击 14,856,758 分钟", "Total impact 14,856,758 minutes"),
        ("583,999 分钟", "583,999 minutes"),
        ("先从 BTS cause taxonomy 出发，把原始原因重写成 internal / system / external。", "Start from the BTS cause taxonomy and rewrite the raw causes into internal / system / external."),
        ("判断扰动到达过程更像均匀计数，还是已经出现 clustering 与 over-dispersion。", "Test whether the disruption process still looks like simple uniform counting or already shows clustering and over-dispersion."),
        ("判断分钟损失的主体形状和尾部结构是不是可以用同一种分布解释。", "Test whether the body and tail of minute losses can be explained by one common distribution."),
        ("把机场月度状态从“高风险月份识别”升级到“坏状态会不会延续”。", "Upgrade the question from identifying high-risk months to asking whether bad states persist."),
        ("把频率、严重度和依赖结构收束成 annual impact 的情景结果。", "Bring frequency, severity, and dependence together into annual impact scenarios."),
        ("最后把统计输出翻译成运营商可以提前做什么、重点管什么。", "Translate the statistical output into what operators should prepare earlier and manage first."),
        ("原始字段是数据库语言，不是管理语言。课程里的Risk Identification更看重责任边界、控制手段和管理动作，所以要把 cause 重新组织成 <b>internal / system / external</b>。", "The raw fields are database language, not management language. In the course framing, risk identification is more about management boundary, control levers, and operator action, so the causes need to be reorganized into <b>internal / system / external</b>."),
        ("例如 <b>Late Aircraft</b> 在原始表里是单独原因，但从运营链条看，它往往不是独立世界，而是前序航班延误、机组恢复、维修衔接失稳后的传导结果，因此更适合放回 internal disruption 的语义里。", "For example, <b>Late Aircraft</b> appears as its own cause in the raw table, but operationally it is often a propagated result of upstream delay, crew recovery pressure, and maintenance linkage, so it fits better inside the semantics of internal disruption."),
        ("这一步不是显著性检验，而是Risk Identification框架重写。", "This step is not a significance test; it is a rewrite of the risk identification framework."),
        ("Risk Identification热力图", "Risk Identification Heatmap"),
        ("最大延误原因仍是 Airline，占总延误分钟 36.2%。", "The largest delay cause is still Airline, accounting for 36.2% of total delay minutes."),
        ("均值与方差接近时的基础计数世界。", "The baseline count world in which mean and variance are close."),
        ("更适合事件聚集、方差大于均值的高波动世界。", "Better suited to clustered events and count behavior with variance above the mean."),
        ("右偏长尾，适合大多数月份不极端、少数月份拉长尾部。", "A right-skewed long-tail shape in which most months are moderate but a few stretch the tail."),
        ("形状更灵活，用来检验分钟损失是否属于另一类右偏结构。", "A more flexible shape used to test whether minute losses follow another right-skewed structure."),
        ("变量", "Variable"),
        ("最优模型", "Best model"),
        ("均值", "Mean"),
        ("方差", "Variance"),
        ("总延误班次", "Total delayed arrivals"),
        ("取消班次", "Cancelled arrivals"),
        ("备降班次", "Diverted arrivals"),
        ("内部运营扰动次数", "Internal disruption counts"),
        ("系统 / NAS 扰动次数", "System / NAS disruption counts"),
        ("外部冲击扰动次数", "External shock disruption counts"),
        ("总体平均延误分钟", "Overall average delay minutes"),
        ("内部扰动平均分钟", "Internal average delay minutes"),
        ("系统 / NAS 平均分钟", "System / NAS average delay minutes"),
        ("外部冲击平均分钟", "External shock average delay minutes"),
        ("样本量", "Sample size"),
        ("扰动态增量", "Increase in disrupted state"),
        ("分钟损失的工作模型随变量而变，说明形状层次确实不同。", "The working model for minute losses changes across variables, which shows the shape hierarchy really differs."),
        ("最清晰的变量是 <b>系统 / NAS 平均分钟</b>，ΔAIC 为 <b>117.1</b>。这不是让结论更混乱，而是让解释更贴近变量本身。", "The clearest case is <b>System / NAS average minutes</b>, with ΔAIC <b>117.1</b>. That does not make the result messier; it makes the interpretation closer to the variable itself."),
        ("如果所有 severity 变量都机械地落在同一种分布上，反而要怀疑是不是候选集太窄、或解释过度简化。当前结果更像真实世界：不同风险块的分钟损失尾部并不完全一样。", "If all severity variables mechanically fell into one distribution, that would actually suggest the candidate set was too narrow or the interpretation too simplified. The current result looks more like reality: the tails of minute losses differ across risk blocks."),
        ("计数型扰动不是均匀到达", "Count disruptions do not arrive uniformly"),
        ("6/6 个 frequency 变量全部选中 Negative Binomial", "All 6/6 frequency variables selected Negative Binomial"),
        ("说明方差显著高于均值，Poisson 过于理想化。", "This shows variance is far above the mean, so Poisson is too idealized."),
        ("主要风险确实来自内部运行链条", "The main risk truly comes from the internal operating chain"),
        ("与原始 cause 结构一致，不是为了好讲而重命名。", "This matches the raw cause structure rather than being a relabeling exercise for presentation."),
        ("坏状态有持续性", "Bad states show persistence"),
        ("说明系统进入扰动态后，恢复并不是自动完成。", "This means that once the system enters disruption, recovery is not automatic."),
        ("高峰情景更危险", "Peak scenarios are more dangerous"),
        ("节假日高峰情景 的 VaR95 最高", "The holiday-peak scenario has the highest VaR95"),
        ("与节假日和高压运行更脆弱的现实经验一致。", "This matches the operational intuition that holiday and high-pressure periods are more fragile."),
        ("状态影响差异", "State Impact Difference"),
        ("normal / disrupted 的影响差异是可以在图上直接看到的。", "The impact difference between normal and disrupted states is directly visible in the chart."),
        ("Negative Binomial 对应“方差远大于均值”的现实，internal 风险主导对应原始延误原因结构，holiday peak 风险最高符合高峰运行更脆弱的直觉，而 P(D→D) 高于 P(N→D) 则对应系统进入坏状态后恢复更慢的经验。", "Negative Binomial matches the reality of variance far above the mean, internal risk dominance matches the raw delay-cause structure, holiday-peak risk being highest matches the intuition that peak operations are more fragile, and P(D→D) exceeding P(N→D) matches the experience that recovery slows once the system enters a bad state."),
        ("这一页的作用，其实是告诉听众：这份项目不是“统计结果看起来漂亮”，而是“统计结果和现实运行逻辑相互印证”。", "The purpose of this page is to show that the project is not just statistically neat-looking; the statistical results and operational logic reinforce each other."),
        ("Frequency 和 severity 能告诉我们扰动出现得多不多、严重到什么程度，但它们默认看到的是“单个月份的分布形状”。", "Frequency and severity can tell us how often disruptions occur and how severe they are, but by default they still describe the distributional shape of single months."),
        ("对机场运营来说，这还不够。管理者更关心的是：如果这个月已经进入高压状态，下个月会不会更容易继续坏下去？如果答案是会，那么运营风险就不只是单点冲击，而是带有持续性和恢复压力的过程。", "For airport operations, that is not enough. Managers care more about whether a high-pressure month makes the next month more likely to remain bad. If the answer is yes, operational risk is not just a point shock but a process with persistence and recovery pressure."),
        ("所以 Markov 在这里的任务，是把问题从“高风险月份识别”升级为“状态会不会延续”。", "So the role of Markov here is to upgrade the question from identifying high-risk months to asking whether states persist."),
        ("只有先把每个月定义成 <b>normal</b> 或 <b>disrupted</b>，后面才谈得上转移。否则所谓的 Markov 只是术语，没有稳定的状态基础。", "Only after defining each month as <b>normal</b> or <b>disrupted</b> can transition analysis begin. Otherwise Markov is only a term with no stable state basis behind it."),
        ("这意味着 composite rule 更适合拿来做State Dependence分析，因为它更稳定，也更像综合运营压力。", "This means the composite rule is more suitable for state-dependence analysis because it is more stable and better reflects combined operating pressure."),
        ("如果转移结果显示 <b>P(D→D)</b> 和 <b>P(N→D)</b> 差不多，那么 Markov 只会告诉我们“坏状态没有明显惯性”，那它的附加价值就有限。", "If the transition results showed <b>P(D→D)</b> and <b>P(N→D)</b> to be similar, Markov would only tell us that bad states have little inertia, and its added value would be limited."),
        ("相反，如果进入 disrupted 之后继续留在 disrupted 的概率明显更高，就说明State Dependence确实存在，做 Markov 就不是多余的，而是在揭示前面分布模型看不到的时间结构。", "By contrast, if the probability of remaining disrupted after entering disruption is clearly higher, state dependence is real and Markov is not redundant; it is revealing temporal structure that the earlier distribution models cannot see."),
        ("也正因为如此，多年度样本扩展到 <b>72</b> 个 airport-month 后，Markov 才真正值得做。", "For exactly that reason, Markov only becomes truly worthwhile once the multi-year sample expands to <b>72</b> airport-month observations."),
        ("<b>P(N→D) = 0.185</b>，而 <b>P(D→D) = 0.471</b>。这说明坏状态不是随机独立地散落在时间线上，而是存在明显的延续倾向。", "<b>P(N→D) = 0.185</b>, while <b>P(D→D) = 0.471</b>. That means bad states are not scattered randomly and independently across time; they show a clear persistence tendency."),
        ("也就是说，Markov 在这里确实补充了新信息：如果只看前面的 frequency 和 severity，你知道风险高不高；但做了 Markov 之后，你还能知道系统一旦变坏，会不会更难立刻恢复。", "In other words, Markov genuinely adds new information here: frequency and severity tell you how high the risk is, but Markov tells you whether the system becomes harder to recover once it turns bad."),
        ("因此，这一步不是重复已有结论，而是在证明“State Dependence”本身值得被纳入项目框架。", "So this step is not repeating earlier findings; it is showing that state dependence itself deserves to be included in the project framework."),
        ("坏状态不是吸收态，因为 <b>P(D→N) = 0.529</b> 仍然高于 <b>P(D→D)</b>，说明系统仍有恢复能力；但 <b>P(D→D)</b> 明显高于 <b>P(N→D)</b>，又说明恢复并不是自动完成的。", "A bad state is not absorbing, because <b>P(D→N) = 0.529</b> is still higher than <b>P(D→D)</b>, which means the system retains recovery capacity; but <b>P(D→D)</b> being clearly above <b>P(N→D)</b> also shows that recovery is not automatic."),
        ("所以最稳妥的课堂口径是：<b>JFK 的月度运营状态存在有说服力的经验性State Dependence</b>。这不是过度夸大的强统计证明，但已经足以支持“坏月份会带来后续恢复压力”的管理解释。", "So the most defensible classroom wording is that <b>JFK’s monthly operating states show persuasive empirical state dependence</b>. This is not an overstated formal statistical proof, but it is strong enough to support the management interpretation that bad months create follow-on recovery pressure."),
        ("基准情景是参考线，天气冲击与节假日高峰代表外部压力，disruption stress 代表系统处于更脆弱状态时的上移风险。相对基准，风险最高的 <b>节假日高峰情景</b> 在 Expected 上高出 <b>305,324</b> 分钟，在 VaR95 上高出 <b>391,212</b> 分钟。", "The base scenario is the reference line, weather shock and holiday peak represent external pressure, and disruption stress represents an upward shift when the system is more fragile. Relative to the base case, the highest-risk <b>holiday-peak scenario</b> is higher by <b>305,324</b> minutes in Expected impact and <b>391,212</b> minutes in VaR95."),
        ("这一页真正想说明的是：项目最后没有停留在“哪个分布更好”，而是回到管理者会问的那句话上，<b>如果运行环境变差，年度风险会被抬高到什么程度</b>。", "What this page really shows is that the project does not end at asking which distribution is better; it returns to the manager’s question: <b>if the operating environment worsens, how far can annual risk be pushed upward?</b>"),
        ("情景", "Scenario"),
        ("节假日高峰情景", "Holiday peak"),
        ("扰动加压情景", "Disruption stress"),
        ("天气冲击情景", "Weather shock"),
        ("基准情景", "Base"),
        ("航司内部恢复", "Internal airline recovery"),
        ("加大 turnaround buffer，优先补强机组与维修衔接", "Increase turnaround buffers and prioritize crew / maintenance recovery linkage"),
        ("internal 风险占主导，且 top cause 是 Airline。", "Internal risk dominates, and the top cause remains Airline."),
        ("机场 / NAS 协同", "Airport / NAS coordination"),
        ("在高压月份提前做容量、时隙与地面资源协调", "Coordinate capacity, slots, and ground resources earlier in high-pressure months"),
        ("system 风险在 severity 上有明显抬升。", "System risk shows a clear upward shift in severity."),
        ("季节与天气应对", "Seasonal and weather response"),
        ("对 holiday / weather 情景配置 contingency plan", "Prepare contingency plans for holiday and weather scenarios"),
        ("aggregate risk 在这两类情景下显著上移。", "Aggregate risk shifts upward clearly in these two scenarios."),
        ("月度预警机制", "Monthly early-warning mechanism"),
        ("持续监控 delay rate、cancellation rate 与 impact composite", "Keep monitoring delay rate, cancellation rate, and the impact composite"),
        ("它们直接决定状态识别与坏状态延续判断。", "These indicators directly support state identification and persistence judgment."),
        ("它不是从模型出发，而是从真实运营问题出发；它不是只给结果，而是把为什么这样分类、为什么只比这四个分布、为什么用 AIC、为什么可以做 Markov 全部串成了一条可解释的方法链。", "It does not start from a model; it starts from a real operating problem. It does not only report results; it explains why the taxonomy was rebuilt, why only these four distributions were compared, why AIC was used, and why Markov was worth doing within one interpretable method chain."),
        ("你可以把它定义成：<b>一个从机场真实运行问题出发，经过Risk Identification、统计建模、State Dependence分析，再回到Operator Action的完整 operational risk project。</b>", "You can frame it as <b>a complete operational risk project that starts from a real airport operating problem, moves through risk identification, statistical modeling, and state-dependence analysis, and then returns to operator action.</b>"),
    ]
    for zh, en in replacements:
        html_output_en = html_output_en.replace(zh, en)
    return html_output_en


_build_showcase_html_en_legacy = build_showcase_html_en


def build_showcase_guide() -> str:
    return dedent(
        f"""
        # 中文展示版 HTML 导出说明

        ## 目标文件
        - HTML: `{SHOWCASE_HTML.relative_to(base.BASE_DIR)}`
        - 建议 PDF 文件名: `jfk_modeling_process_showcase_zh.pdf`

        ## 推荐浏览器
        - Microsoft Edge
        - Google Chrome

        ## 导出步骤
        1. 打开 `reports/dashboard/jfk_modeling_process_showcase_zh.html`。
        2. 等待所有图表和背景完全加载。
        3. 按 `Ctrl+P` 打开打印面板。
        4. 打印机选择“另存为 PDF”。
        5. 布局选择“横向”。
        6. 勾选“背景图形”。
        7. 缩放保持默认，或选择“适合页面”。
        8. 页边距使用“默认”或“最小”，并关闭页眉页脚。
        9. 在打印预览中确认每页单独分页，再导出 PDF。

        ## 导出前检查
        - 深色封面和章节页背景被保留。
        - 图表、公式和关键数字完整可见。
        - 标题与正文没有重叠。
        - 中文显示正常，没有乱码或裁切。

        ## 说明
        - 这份文件更偏答辩展示，而不是说明书式报告。
        - 页面按固定画布分页设计，可直接走浏览器打印为 PDF。
        - 如果后续要做自动 PDF 导出，再在这一版基础上补自动化即可。
        """
    ).strip() + "\n"


def build_showcase_guide_en() -> str:
    return dedent(
        f"""
        # English Showcase HTML Export Guide

        ## Target files
        - HTML: `{SHOWCASE_HTML_EN.relative_to(base.BASE_DIR)}`
        - Suggested PDF name: `jfk_modeling_process_showcase_en.pdf`

        ## Recommended browsers
        - Microsoft Edge
        - Google Chrome

        ## Export steps
        1. Open `reports/dashboard/jfk_modeling_process_showcase_en.html`.
        2. Wait until all charts and backgrounds are fully loaded.
        3. Press `Ctrl+P`.
        4. Choose `Save as PDF`.
        5. Set layout to `Landscape`.
        6. Enable `Background graphics`.
        7. Keep scaling at default or `Fit to page`.
        8. Use `Default` or `Minimum` margins and disable headers/footers.
        9. Confirm each page is separated correctly in print preview, then export.

        ## What to check before export
        - Dark cover and section backgrounds are preserved.
        - Charts, formulas, and key figures are fully visible.
        - Titles and body text do not overlap.
        - Page order is correct and no text is clipped.

        ## Notes
        - This file is the English counterpart of the final showcase report.
        - It is designed for browser-based print-to-PDF export.
        """
    ).strip() + "\n"


def build_showcase_html_en(html_output_zh: str) -> str:
    html_output_en = _build_showcase_html_en_legacy(html_output_zh)
    residual_replacements = [
        ("为什么飞机延误<br>", "Why flight delays<br>"),
        ("这是一份从 <b>problem motivation</b>、<b>Risk Identification</b>、<b>distribution choice</b>、", "This is a complete showcase that moves from <b>problem motivation</b>, <b>risk identification</b>, and <b>distribution choice</b>,"),
        ("到 <b>operator implications</b> 的完整项目展示版。", "to <b>operator implications</b> in one complete project showcase."),
        ("它同时具备real operating pressure、public reproducible data，以及a measurable loss proxy。", "It combines real operating pressure, public reproducible data, and a measurable loss proxy."),
        ("这意味着它既能讲现实问题，也能撑起完整方法链，而不只是做一页描述统计。", "That means it can address a real operating problem while supporting a full methodological chain, rather than stopping at descriptive statistics."),
        ("更重要的是，Delays, cancellations, and diversions naturally propagate into one another, which makes this topic especially suitable for linking frequency, severity, dependence, and aggregate risk into one coherent story.", "More importantly, delays, cancellations, and diversions naturally propagate into one another, which makes this topic especially suitable for linking frequency, severity, dependence, and aggregate risk into one coherent story."),
        ("<span>Total impact 14,856,758 分钟</span>", "<span>Total impact 14,856,758 minutes</span>"),
        ("<p><b>AIC = 2k - 2\\ln(L)</b>。它不是 p-value，不负责回答“显著不显著”，而是负责在同一候选集里平衡拟合质量与模型复杂度。</p>", "<p><b>AIC = 2k - 2\\ln(L)</b> is not a p-value test. Its job is not to say whether something is statistically significant, but to balance goodness of fit against model complexity within the same candidate set.</p>"),
        ("<p>所以 AIC 的逻辑不是“证明这个分布是真的”，而是“在当前候选集里，这个分布作为工作模型更合适”。这也是为什么报告里要反复强调 relative fit，而不是 absolute truth。</p>", "<p>So the logic of AIC is not to prove that a distribution is true; it is to decide which distribution is the more suitable working model within the current candidate set. That is why the report emphasizes relative fit rather than absolute truth.</p>"),
        ("<div class='formula-box'>候选模型拟合 → 参数估计 → 计算 AIC → 选择 AIC 更低者</div>", "<div class='formula-box'>Fit candidate models → estimate parameters → compute AIC → select the lower-AIC model</div>"),
        ("<p>我们不仅看谁最低，也看低了多少。比如 Frequency 里 <b>Total delayed arrivals</b> 的 ΔAIC 达到 <b>169,085.5</b>，说明优势不是边缘性的；Severity 里最清晰的是 <b>System / NAS average delay minutes</b>，ΔAIC 为 <b>117.1</b>。</p>", "<p>We do not just check which model is lowest; we also ask how much lower it is. In the frequency results, <b>Total delayed arrivals</b> has a ΔAIC of <b>169,085.5</b>, so the advantage is not marginal. In severity, the clearest case is <b>System / NAS average delay minutes</b>, with a ΔAIC of <b>117.1</b>.</p>"),
        ("<p>换句话说，AIC 让我们在“候选分布为什么这样定”和“最后为什么这样选”之间搭起了一座桥。没有这一页，后面的结果页很容易被读成黑箱。</p>", "<p>In other words, AIC creates the bridge between why these candidate distributions were chosen and why the final model was selected. Without this step, the later result pages would look much more like a black box.</p>"),
        ("<p>当Variance显著高于Mean，且 disrupted 状态下的计数整体抬升时，Poisson 会把真实世界压扁。Negative Binomial 更能容纳事件聚集与高波动。</p>", "<p>When variance is clearly above the mean and the counts shift upward in disrupted months, Poisson compresses the real world too aggressively. Negative Binomial can better accommodate clustering and high volatility.</p>"),
        ("<p>这背后的运营含义也很直接：延误不是均匀散落在每个月里，而是会在某些压力月份集中爆发。对管理者来说，这意味着风险不是“平均分摊”的，而是存在高压窗口和恢复压力。</p>", "<p>The operating implication is straightforward: delays are not spread evenly across months. They cluster in high-pressure periods, which means risk is not evenly shared across the year but concentrated in windows with recovery pressure.</p>"),
        ("Frequency AIC 对比", "Frequency AIC Comparison"),
        ("ΔAIC 越大，说明较优模型的优势越明确。", "The larger the ΔAIC, the clearer the advantage of the preferred model."),
        ("Severity 没有“一刀切”的统一答案。总体分钟损失和分风险块分钟损失的形状并不完全一样，", "Severity does not have a one-size-fits-all answer. The shape of overall minute losses is not identical to the shape of losses within each risk block,"),
        ("这反而说明项目不是在把所有Variable强塞进同一解释框架。", "which shows the project is not forcing every variable into the same explanatory frame."),
        ("对答辩来说，这一点反而很重要，因为它说明你们没有为了追求整齐而牺牲解释力。", "For the presentation, that matters because it shows you did not sacrifice explanatory power just to make the results look neat."),
        ("Severity AIC 对比", "Severity AIC Comparison"),
        ("分钟损失的工作模型随Variable而变，说明形状层次确实不同。", "The working model for minute losses changes across variables, which shows the underlying shape hierarchy really differs."),
        ("<p>最清晰的Variable是 <b>System / NAS average delay minutes</b>，ΔAIC 为 <b>117.1</b>。这不是让结论更混乱，而是让解释更贴近Variable本身。</p>", "<p>The clearest variable is <b>System / NAS average delay minutes</b>, with a ΔAIC of <b>117.1</b>. That does not make the result messier; it makes the explanation closer to the variable itself.</p>"),
        ("<p>如果所有 severity Variable都机械地落在同一种分布上，反而要怀疑是不是候选集太窄、或解释过度简化。当前结果更像真实世界：不同风险块的分钟损失尾部并不完全一样。</p>", "<p>If all severity variables mechanically landed on the same distribution, that would actually raise the suspicion that the candidate set was too narrow or the explanation too simplified. The current result looks more like reality: the tails of minute losses are not identical across risk blocks.</p>"),
        ("6/6 个 frequency Variable全部选中 Negative Binomial", "All 6/6 frequency variables selected Negative Binomial"),
        ("说明Variance显著高于Mean，Poisson 过于理想化。", "This shows variance is far above the mean, so Poisson is too idealized."),
        ("<p>Negative Binomial 对应“Variance远大于Mean”的现实，internal 风险主导对应原始延误原因结构，holiday peak 风险最高符合高峰运行更脆弱的直觉，而 P(D→D) 高于 P(N→D) 则对应系统进入坏状态后恢复更慢的经验。</p>", "<p>Negative Binomial matches the reality of variance far above the mean, internal risk dominance matches the raw delay-cause structure, holiday-peak risk being highest matches the intuition that peak operations are more fragile, and P(D→D) exceeding P(N→D) matches the experience that recovery slows once the system enters a bad state.</p>"),
        ("<p><b>delay_cancel_impact_composite</b> 与 <b>impact_q75</b> 的 disrupted share 都是 <b>25.0%</b>，但前者跨年波动更低：<b>0.152</b> vs <b>0.186</b>。This means the composite rule is more suitable for state-dependence analysis because it is more stable and better reflects combined operating pressure.</p>", "<p>Both <b>delay_cancel_impact_composite</b> and <b>impact_q75</b> classify <b>25.0%</b> of months as disrupted, but the former has lower cross-year volatility: <b>0.152</b> versus <b>0.186</b>. That makes the composite rule more suitable for state-dependence analysis because it is more stable and better reflects combined operating pressure.</p>"),
        ("节假日高峰Scenario", "Holiday peak"),
        ("扰动加压Scenario", "Disruption stress"),
        ("天气冲击Scenario", "Weather shock"),
        ("基准Scenario", "Base"),
        ("对 holiday / weather Scenario配置 contingency plan", "Prepare contingency plans for holiday and weather scenarios"),
        ("aggregate risk 在这两类Scenario下显著上移。", "Aggregate risk shifts upward clearly in these two scenarios."),
    ]
    for zh, en in residual_replacements:
        html_output_en = html_output_en.replace(zh, en)
    html_output_en = html_output_en.replace(
        "../charts/chart_8_frequency_aic_comparison.png",
        "../charts/chart_8_frequency_aic_comparison_en.png",
    )
    html_output_en = html_output_en.replace(
        "../charts/chart_9_severity_aic_comparison.png",
        "../charts/chart_9_severity_aic_comparison_en.png",
    )
    html_output_en = html_output_en.replace(
        "../charts/chart_10_state_rule_comparison.png",
        "../charts/chart_10_state_rule_comparison_en.png",
    )
    english_css = """
    html[lang="en"] {
      --font-body: "Aptos", "Segoe UI", "Helvetica Neue", Arial, sans-serif;
      --font-display: "Georgia", "Times New Roman", serif;
    }
    html[lang="en"] .ct { height: 648px; padding: 10px 26px 12px; }
    html[lang="en"] .chapter-tag { font-size: 9px; letter-spacing: 0.10em; }
    html[lang="en"] .page-title { font-size: 13px; }
    html[lang="en"] .stack { gap: 12px; }
    html[lang="en"] .stack.spacious { gap: 14px; }
    html[lang="en"] .panel,
    html[lang="en"] .visual-block { padding: 14px 16px; border-radius: 22px; }
    html[lang="en"] .panel-title { font-size: 10px; margin-bottom: 10px; }
    html[lang="en"] .panel-body { font-size: 11px; line-height: 1.54; }
    html[lang="en"] .panel-body p + p,
    html[lang="en"] .panel-body ul + p,
    html[lang="en"] .panel-body p + ul { margin-top: 6px; }
    html[lang="en"] .formula-box { font-size: 9px; line-height: 1.44; padding: 9px 11px; }
    html[lang="en"] .big-claim { font-size: 32px; line-height: 1.04; max-width: 16ch; }
    html[lang="en"] .claim-text { font-size: 12px; line-height: 1.56; max-width: 60ch; }
    html[lang="en"] .eyebrow,
    html[lang="en"] .section-kicker,
    html[lang="en"] .proof-title,
    html[lang="en"] .action-head,
    html[lang="en"] .closing-title { font-size: 10px; }
    html[lang="en"] .inline-proof span,
    html[lang="en"] .step-ribbon span,
    html[lang="en"] .capsule { min-height: 34px; padding: 0 10px; font-size: 10px; }
    html[lang="en"] .proof-highlight { font-size: 18px; }
    html[lang="en"] .proof-text,
    html[lang="en"] .action-note,
    html[lang="en"] .closing-band p,
    html[lang="en"] .stat-note,
    html[lang="en"] .visual-caption { font-size: 11px; line-height: 1.5; }
    html[lang="en"] .ratio-copy,
    html[lang="en"] .action-main { font-size: 13px; line-height: 1.52; }
    html[lang="en"] .stat-value { font-size: 24px; }
    html[lang="en"] .visual-medium { height: 190px; }
    html[lang="en"] .visual-tall { height: 334px; }
    html[lang="en"] .data-table thead th { font-size: 9px; }
    html[lang="en"] .data-table tbody td { font-size: 11px; padding: 6px 0; }
    """
    html_output_en = html_output_en.replace("</style>", english_css + "\n  </style>")
    return html_output_en


def _add_panel(ax, x: float, y: float, w: float, h: float, title: str, body: str, *,
               face: str = "#ffffff", edge: str = "#d7ddd9", title_color: str = "#11212b",
               body_color: str = "#33444d", body_size: int = 10, title_size: int = 12) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.0,
        edgecolor=edge,
        facecolor=face,
        transform=ax.transAxes,
        zorder=1,
    )
    ax.add_patch(patch)
    ax.text(
        x + 0.018,
        y + h - 0.04,
        title,
        transform=ax.transAxes,
        fontsize=title_size,
        fontweight="bold",
        color=title_color,
        va="top",
        zorder=2,
    )
    wrapped = "\n".join(fill(paragraph, width=72) for paragraph in body.split("\n"))
    ax.text(
        x + 0.018,
        y + h - 0.085,
        wrapped,
        transform=ax.transAxes,
        fontsize=body_size,
        color=body_color,
        va="top",
        linespacing=1.45,
        zorder=2,
    )


def _add_image_panel(ax, x: float, y: float, w: float, h: float, title: str, image_path, caption: str) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.0,
        edgecolor="#d7ddd9",
        facecolor="#ffffff",
        transform=ax.transAxes,
        zorder=1,
    )
    ax.add_patch(patch)
    ax.text(
        x + 0.018,
        y + h - 0.04,
        title,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        color="#11212b",
        va="top",
        zorder=2,
    )
    image_ax = ax.inset_axes([x + 0.02, y + 0.08, w - 0.04, h - 0.16], transform=ax.transAxes)
    image_ax.imshow(mpimg.imread(image_path))
    image_ax.axis("off")
    ax.text(
        x + 0.018,
        y + 0.028,
        fill(caption, width=84),
        transform=ax.transAxes,
        fontsize=9,
        color="#5f7079",
        va="bottom",
        zorder=2,
    )


def _new_pdf_page(section: str, title: str, subtitle: str, *, dark: bool = False):
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.patch.set_facecolor("#10202b" if dark else "#f7f5ef")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(
        0.055,
        0.955,
        section,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        color="#f0b996" if dark else "#7a5c47",
        va="top",
    )
    ax.text(
        0.055,
        0.91,
        title,
        transform=ax.transAxes,
        fontsize=24,
        fontweight="bold",
        color="#eef5f4" if dark else "#11212b",
        va="top",
    )
    ax.text(
        0.055,
        0.865,
        fill(subtitle, width=120),
        transform=ax.transAxes,
        fontsize=11,
        color="#d8e5e3" if dark else "#5f7079",
        va="top",
        linespacing=1.4,
    )
    return fig, ax


def build_english_pdf_report(
    freq_table: pd.DataFrame,
    sev_table: pd.DataFrame,
    state_rules: pd.DataFrame,
    transitions: pd.DataFrame,
    scenarios: pd.DataFrame,
    model_df: pd.DataFrame,
    airport_panel: pd.DataFrame,
    cause_summary: pd.DataFrame,
) -> None:
    years = sorted(int(year) for year in airport_panel["year"].unique())
    total_flights = float(airport_panel["total_arrival_flights"].sum())
    total_impact = float(airport_panel["disruption_impact_minutes"].sum())
    disrupted_months = int((airport_panel["operational_state"] == "disrupted").sum())
    cause_share_map = {
        str(row["cause"]): float(row["share_of_total_delay_minutes"])
        for row in cause_summary.to_dict(orient="records")
    }
    internal_share = cause_share_map.get("Airline", 0.0) + cause_share_map.get("Late Aircraft", 0.0)
    system_share = cause_share_map.get("NAS", 0.0)
    external_share = cause_share_map.get("Weather", 0.0) + cause_share_map.get("Security", 0.0)
    top_cause = cause_summary.sort_values("share_of_total_delay_minutes", ascending=False).iloc[0]
    freq_biggest_gap = freq_table.sort_values("delta_aic", ascending=False).iloc[0]
    sev_biggest_gap = sev_table.sort_values("delta_aic", ascending=False).iloc[0]
    selected_rule = state_rules.loc[state_rules["selected"]].iloc[0]
    alt_rule = state_rules.loc[~state_rules["selected"]].iloc[0]
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

    with PdfPages(SHOWCASE_PDF_EN) as pdf:
        info = pdf.infodict()
        info["Title"] = "JFK Operational Risk English Report"
        info["Author"] = "Codex"
        fig, ax = _new_pdf_page(
            "JFK Operational Risk Report",
            "English Modeling Process Report",
            "A presentation-ready English report built directly as PDF, summarizing problem motivation, risk classification, distribution choice, Markov dependence, and operator implications.",
            dark=True,
        )
        _add_panel(
            ax, 0.055, 0.58, 0.42, 0.19, "What this report is for",
            "This report is meant for the final English presentation. It does not only present results. It explains why flight delay is an operational-risk problem, why the taxonomy was rebuilt, why only four distributions were compared, why AIC was used, and why Markov dependence adds value.",
            face="#17313b", edge="#314d57", title_color="#eef5f4", body_color="#d8e5e3", body_size=10,
        )
        _add_panel(
            ax, 0.50, 0.58, 0.445, 0.19, "Core scope",
            f"Sample window: {years[0]}-{years[-1]}\nFlights covered: {base.format_int(total_flights)}\nAirport-month observations: {len(airport_panel)}\nDisrupted months: {disrupted_months}",
            face="#17313b", edge="#314d57", title_color="#eef5f4", body_color="#d8e5e3", body_size=10,
        )
        _add_panel(
            ax, 0.055, 0.28, 0.89, 0.22, "Executive takeaway",
            f"Internal operating disruption dominates total delay minutes at {base.format_pct(internal_share, 1)}. All six frequency variables prefer Negative Binomial, the clearest severity winner has Delta AIC {base.format_float(sev_biggest_gap.delta_aic, 1)}, and P(D->D)={base.format_float(p_dd,3)} exceeds P(N->D)={base.format_float(p_nd,3)}. The highest aggregate-risk scenario is {scenario_peak['label']}, which pushes annual VaR95 to {base.format_int(scenario_peak['var_95_minutes'])} delay-equivalent minutes.",
            face="#f4ede3", edge="#d8c5b0", title_color="#11212b", body_color="#33444d", body_size=11,
        )
        ax.text(0.055, 0.16, f"Total annualized impact proxy observed in the sample: {base.format_int(total_impact)} delay-equivalent minutes", transform=ax.transAxes, fontsize=14, color="#eef5f4", fontweight="bold")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, ax = _new_pdf_page(
            "Section 1",
            "Why Flight Delays Were Chosen",
            "Flight delays are a strong operational-risk topic because they are visible, measurable, recurrent, and naturally linked to cancellations, diversions, and recovery pressure.",
        )
        _add_panel(
            ax, 0.055, 0.54, 0.36, 0.28, "Why this problem matters",
            "A strong course project needs a real operating problem, a defensible loss proxy, and enough structure to support a full analytical chain. Flight delays meet all three conditions. They are operationally meaningful, observable in public BTS data, and naturally connect frequency, severity, dependence, and aggregate impact.",
        )
        _add_panel(
            ax, 0.055, 0.22, 0.36, 0.24, "Data structure",
            f"The project works with two levels of data. Airline-month observations ({len(model_df)}) support frequency and severity fitting. Airport-month observations ({len(airport_panel)}) support state definition and Markov dependence. This split is what makes the full method chain possible.",
        )
        _add_image_panel(
            ax, 0.445, 0.18, 0.50, 0.64, "Multi-Year Operating Variation",
            base.CHART_DIR / "chart_1_multiyear_monthly_trend.png",
            "The trend chart shows that disruption is not flat across time. The project therefore treats operational risk as a dynamic process instead of a one-off descriptive problem.",
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, ax = _new_pdf_page(
            "Section 2",
            "Why the Taxonomy Was Rebuilt",
            "The raw BTS categories are data fields, not management categories. The project rewrites them into internal, system, and external risk blocks so the later models can be interpreted in operational language.",
        )
        _add_panel(
            ax, 0.055, 0.53, 0.38, 0.29, "Classification logic",
            f"Internal = Airline + Late Aircraft ({base.format_pct(internal_share,1)} of delay minutes)\nSystem = NAS ({base.format_pct(system_share,1)})\nExternal = Weather + Security ({base.format_pct(external_share,1)})\n\nThis is not a significance test. It is a risk-identification rewrite so the project can speak in management terms rather than database labels.",
        )
        _add_panel(
            ax, 0.055, 0.22, 0.38, 0.24, "Why this matters",
            f"The dominant raw cause is still {top_cause['cause']}, accounting for {base.format_pct(top_cause['share_of_total_delay_minutes'],1)} of total delay minutes. The rebuilt taxonomy therefore stays anchored in the raw structure while making the results easier to explain in class and to operators.",
        )
        _add_image_panel(
            ax, 0.46, 0.46, 0.49, 0.36, "Cause Breakdown",
            base.CHART_DIR / "chart_2_delay_cause_breakdown.png",
            "The cause-breakdown chart supports the mapping from raw BTS causes into course-level risk blocks.",
        )
        _add_image_panel(
            ax, 0.46, 0.10, 0.49, 0.28, "Risk Identification Heatmap",
            base.CHART_DIR / "chart_3_risk_heatmap.png",
            "The heatmap is used as a risk-identification lens. It does not replace statistical modeling, but helps explain relative operating pressure across risk blocks.",
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, ax = _new_pdf_page(
            "Section 3",
            "Candidate Distributions and AIC",
            "The project deliberately keeps the candidate set small: Poisson vs Negative Binomial for frequency, and Lognormal vs Weibull for severity. AIC is used as a relative model-selection rule, not a significance test.",
        )
        _add_panel(
            ax, 0.055, 0.54, 0.34, 0.28, "Why these four distributions",
            "Poisson is the baseline count model when mean and variance are close. Negative Binomial is the natural alternative when counts cluster and variance exceeds the mean. Lognormal captures right-skewed severity with a long tail, while Weibull is a flexible alternative for different body-and-tail shapes.",
        )
        _add_panel(
            ax, 0.055, 0.22, 0.34, 0.24, "How AIC is used",
            f"AIC = 2k - 2ln(L).\nThe workflow is: fit candidate models -> estimate parameters -> compute AIC -> choose the lower-AIC model.\n\nThe strongest frequency gap is {freq_biggest_gap['label']} with Delta AIC {base.format_float(freq_biggest_gap.delta_aic,1)}.",
        )
        _add_image_panel(
            ax, 0.415, 0.18, 0.53, 0.64, "Frequency AIC Comparison",
            base.CHART_DIR / "chart_8_frequency_aic_comparison_en.png",
            "All six frequency variables prefer Negative Binomial, which supports the idea that disruption counts are over-dispersed rather than close to a simple Poisson process.",
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, ax = _new_pdf_page(
            "Section 4",
            "Severity Results",
            "Severity does not collapse into one universal shape. That is useful rather than problematic, because it shows that different risk blocks have meaningfully different minute-loss structures.",
        )
        _add_panel(
            ax, 0.055, 0.54, 0.34, 0.28, "Main reading",
            f"The clearest severity winner is {sev_biggest_gap['label']}, with Delta AIC {base.format_float(sev_biggest_gap.delta_aic,1)}. Some variables prefer Lognormal while others prefer Weibull. That means the project is not forcing all severity variables into one explanation just for symmetry.",
        )
        _add_panel(
            ax, 0.055, 0.22, 0.34, 0.24, "Interpretation",
            "This is a strength in a report setting. If every severity variable had selected the same distribution automatically, the explanation would look cleaner but less credible. The current result is closer to real operations: different disruption blocks have different tail behavior.",
        )
        _add_image_panel(
            ax, 0.415, 0.18, 0.53, 0.64, "Severity AIC Comparison",
            base.CHART_DIR / "chart_9_severity_aic_comparison_en.png",
            "The chart highlights where Lognormal or Weibull has a stronger relative fit advantage for average delay-minute variables.",
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, ax = _new_pdf_page(
            "Section 5",
            "Why Markov Was Worth Doing",
            "Frequency and severity describe how often and how severely disruption appears. Markov is used because managers also need to know whether a bad month makes the next month more likely to remain bad.",
        )
        _add_panel(
            ax, 0.055, 0.54, 0.34, 0.28, "State rule logic",
            f"The chosen rule is {selected_rule['label']}. It classifies {base.format_pct(selected_rule.disrupted_share,1)} of months as disrupted, the same share as {alt_rule['label']}, but with lower cross-year volatility ({base.format_float(selected_rule.yearly_disrupted_share_std,3)} vs {base.format_float(alt_rule.yearly_disrupted_share_std,3)}). That makes it more stable for dependence analysis.",
        )
        _add_panel(
            ax, 0.055, 0.22, 0.34, 0.24, "Why this step matters",
            "If P(D->D) had been similar to P(N->D), Markov would add little value. The method is only worthwhile if the results reveal persistence that the distribution models cannot show by themselves.",
        )
        _add_image_panel(
            ax, 0.415, 0.18, 0.53, 0.64, "State Rule Comparison",
            base.CHART_DIR / "chart_10_state_rule_comparison_en.png",
            "The selected composite rule is closer to a defensible state framework because it balances disrupted share and cross-year stability.",
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, ax = _new_pdf_page(
            "Section 6",
            "Markov Results and Meaning",
            "The Markov result matters because it shows that bad states are not scattered independently across time. They have inertia.",
        )
        _add_panel(
            ax, 0.055, 0.54, 0.34, 0.28, "Key probabilities",
            f"P(N->D) = {base.format_float(p_nd,3)}\nP(D->D) = {base.format_float(p_dd,3)}\nP(D->N) = {base.format_float(p_dn,3)}\nPersistence ratio = {base.format_float(transition_ratio,2)}x\n\nBecause P(D->D) is clearly above P(N->D), once the system turns bad it is materially harder to recover immediately.",
        )
        _add_panel(
            ax, 0.055, 0.22, 0.34, 0.24, "How to phrase the evidence",
            "The careful claim is that JFK monthly operations show persuasive empirical state dependence. This is not an overstated formal proof, but it is strong enough to support the management interpretation that bad months create recovery pressure in subsequent months.",
        )
        _add_image_panel(
            ax, 0.415, 0.18, 0.53, 0.64, "Markov Transition Matrix",
            base.CHART_DIR / "chart_7_markov_transition_matrix.png",
            "The transition matrix turns the dependence argument into an interpretable state-to-state probability structure.",
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, ax = _new_pdf_page(
            "Section 7",
            "Aggregate Annual Risk",
            "The last modeling step brings frequency, severity, and dependence together and asks a management question: how far does annual operational impact move under different scenarios?",
        )
        _add_panel(
            ax, 0.055, 0.54, 0.34, 0.28, "Scenario reading",
            f"The highest-risk scenario is {scenario_peak['label']}. Relative to the base case, expected annual impact increases by {base.format_int(scenario_peak['expected_impact_minutes'] - base_expected)} minutes and VaR95 increases by {base.format_int(scenario_peak['var_95_minutes'] - base_var95)} minutes. That is the final synthesis of the project.",
        )
        _add_panel(
            ax, 0.055, 0.22, 0.34, 0.24, "Why this is the final decision layer",
            "Managers do not ultimately choose between distributions. They choose where to prepare capacity, recovery buffers, and contingency plans. The scenario layer translates model structure into that kind of decision question.",
        )
        _add_image_panel(
            ax, 0.415, 0.18, 0.53, 0.64, "Annual Aggregate Operational Impact by Scenario",
            base.CHART_DIR / "chart_6_aggregate_risk_scenarios.png",
            "The scenario curves summarize how the annual delay-equivalent impact distribution shifts under more fragile operating environments.",
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, ax = _new_pdf_page(
            "Section 8",
            "Operator Implications",
            "The project becomes complete only when the statistical output is translated back into operator action.",
        )
        _add_panel(
            ax, 0.055, 0.56, 0.42, 0.24, "Action 1 and 2",
            "Internal airline recovery: increase turnaround buffers and strengthen crew-maintenance linkage, because internal disruption dominates the delay-minute structure.\n\nAirport / NAS coordination: prepare capacity, slot, and ground-resource coordination earlier in high-pressure months, because system risk shows a meaningful severity lift.",
        )
        _add_panel(
            ax, 0.055, 0.24, 0.42, 0.24, "Action 3 and 4",
            "Seasonal and weather response: prepare contingency plans for holiday-peak and weather-shock scenarios because annual aggregate risk shifts upward under both.\n\nMonthly early warning: keep monitoring delay rate, cancellation rate, and the impact composite, because these variables support state identification and persistence judgment.",
        )
        _add_panel(
            ax, 0.52, 0.48, 0.425, 0.32, "Final framing for presentation",
            "This can be framed as a complete operational-risk project that starts from a real airport problem, moves through risk identification, statistical modeling, and state dependence, and then returns to operator action. That framing shows depth, not just workload.",
            face="#f4ede3", edge="#d8c5b0", body_size=11,
        )
        _add_panel(
            ax, 0.52, 0.24, 0.425, 0.16, "Deliverables retained in the repo",
            "Chinese HTML showcase for discussion\nEnglish PDF report for final presentation\nEnglish comparison charts for AIC and state-rule pages",
            body_size=10,
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

def run_validations(html_output_zh: str) -> None:
    zh_page_count = html_output_zh.count('class="report-page')
    if zh_page_count != 12:
        raise ValueError(f"Expected 12 Chinese showcase pages, found {zh_page_count}.")
    if not SHOWCASE_HTML.exists():
        raise FileNotFoundError(f"Expected showcase HTML output not found: {SHOWCASE_HTML}")
    if not SHOWCASE_GUIDE.exists():
        raise FileNotFoundError(f"Expected showcase guide output not found: {SHOWCASE_GUIDE}")


def main() -> None:
    base.ensure_output_directories()
    base.apply_style()

    frequency_df = base.load_csv(base.FREQUENCY_SUMMARY)
    severity_df = base.load_csv(base.SEVERITY_SUMMARY)
    state_rule_df = base.load_csv(base.STATE_RULE_COMPARISON)
    transition_df = base.load_csv(base.STATE_TRANSITIONS)
    metrics_df = base.load_csv(base.SCENARIO_METRICS)
    model_df = base.load_csv(base.MODELING_INPUT)
    airport_panel = base.load_csv(base.AIRPORT_MONTH_PANEL)
    risk_heatmap = base.load_csv(base.RISK_HEATMAP)
    cause_summary = base.load_csv(base.CAUSE_SUMMARY)

    freq_table = base.build_frequency_table(frequency_df)
    sev_table = base.build_severity_table(severity_df)
    state_rules = base.build_state_rule_table(state_rule_df)
    transitions = base.build_transition_table(transition_df)
    scenarios = base.build_scenario_table(metrics_df)

    base.build_frequency_aic_chart(freq_table)
    base.build_severity_aic_chart(sev_table)
    base.build_state_rule_chart(state_rules)

    html_output = build_showcase_html(
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
    SHOWCASE_HTML.write_text(html_output, encoding="utf-8")
    SHOWCASE_GUIDE.write_text(build_showcase_guide(), encoding="utf-8")
    run_validations(html_output)

    print("Chinese showcase outputs written to:")
    print(f"  Chinese HTML report: {SHOWCASE_HTML}")
    print(f"  Chinese PDF guide: {SHOWCASE_GUIDE}")


if __name__ == "__main__":
    main()
