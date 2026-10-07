"""SMB Cash-Advance Repayment & Risk — Streamlit dashboard.

A portfolio piece by Leo Chan (github.com/GoldenCrate).

ALL DATA IS SYNTHETIC (see data_gen.py). This app demonstrates experimentation,
cohort analysis, risk segmentation, and stakeholder storytelling for an
embedded-fintech capital product — the kind of work a Data Scientist at a
company like Parafin does day to day.

Charts follow Cole Nussbaumer Knaflic's "Storytelling with Data":
takeaway titles, decluttered axes, direct labels, and restrained color.

Run:  streamlit run app.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import analytics as an
import data_gen

# --------------------------------------------------------------------------- #
# Palette — restrained. One accent (teal) for "signal", grey for everything
# else, a single alert red reserved for risk / loss.
# --------------------------------------------------------------------------- #
INK = "#1c2b33"
GREY = "#b7c2c7"
GREY_D = "#6b7a82"
ACCENT = "#0f8b8d"       # signal / primary series
ACCENT_L = "#7fc3c4"
ALERT = "#c0463b"        # loss / default
GRID = "#e8edef"

st.set_page_config(
    page_title="SMB Advance Repayment & Risk",
    page_icon="📊",
    layout="wide",
)


@st.cache_data
def load_data() -> dict[str, pd.DataFrame]:
    """Generate (seeded) synthetic data once per session."""
    return data_gen.generate()


DATA = load_data()
advances = DATA["advances"]
repayments = DATA["repayments"]
offers = DATA["offers"]


def usd(x: float, cents: bool = False) -> str:
    return f"${x:,.2f}" if cents else f"${x:,.0f}"


def usd_compact(x: float) -> str:
    if abs(x) >= 1e6:
        return f"${x / 1e6:.1f}M"
    if abs(x) >= 1e3:
        return f"${x / 1e3:.0f}K"
    return f"${x:,.0f}"


def base_layout(fig: go.Figure, title: str, subtitle: str = "") -> go.Figure:
    """Apply decluttered Storytelling-with-Data styling to a figure."""
    header = f"<b>{title}</b>"
    if subtitle:
        header += f"<br><span style='font-size:13px;color:{GREY_D}'>{subtitle}</span>"
    fig.update_layout(
        title=dict(text=header, x=0, xanchor="left", font=dict(size=19, color=INK)),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Inter, Segoe UI, sans-serif", color=INK, size=13),
        margin=dict(l=10, r=20, t=74, b=10),
        showlegend=False,
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor=GRID, ticks="outside",
                     tickcolor=GRID)
    fig.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False, linecolor="white")
    return fig


def so_what(text: str) -> None:
    """Render a consistent 'So what / recommendation' callout."""
    st.markdown(
        f"""<div style="border-left:4px solid {ACCENT};background:#f3f9f9;
        padding:14px 18px;border-radius:4px;margin-top:8px;">
        <span style="font-weight:700;color:{ACCENT};letter-spacing:.04em;
        font-size:12px;">SO WHAT / RECOMMENDATION</span><br>
        <span style="color:{INK};font-size:15px;">{text}</span></div>""",
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Sidebar navigation
# --------------------------------------------------------------------------- #
st.sidebar.title("SMB Advance\nRepayment & Risk")
st.sidebar.caption("Portfolio demo · Leo Chan · github.com/GoldenCrate")
view = st.sidebar.radio(
    "View",
    [
        "Portfolio KPIs",
        "Cohort Repayment Curves",
        "A/B Test: Offer Structure",
        "Risk by Segment & Band",
    ],
)
st.sidebar.markdown("---")
st.sidebar.warning("All data on this dashboard is **synthetic** and seeded for "
                   "reproducibility. No real merchants or advances are represented.")


# =========================================================================== #
# VIEW 1 — Portfolio KPIs
# =========================================================================== #
def view_kpis() -> None:
    st.title("Portfolio health at a glance")
    st.caption("Synthetic SMB cash-advance book · as-of 2025-06-30")

    k = an.compute_kpis(advances)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Advances originated", f"{k['n_advances']:,}")
    c2.metric("Total disbursed", usd_compact(k["total_disbursed"]))
    c3.metric("Repaid to date / owed", f"{k['repayment_rate']:.1%}",
              help="Dollars collected so far vs. total contractually owed, "
                   "across a book that still has many in-flight advances.")
    c4.metric("Charge-off rate", f"{k['default_rate_count']:.1%}",
              delta=f"{k['default_rate_dollars']:.1%} of $ disbursed",
              delta_color="off",
              help="Share of advances charged off (count), and the dollar-weighted rate.")
    c5.metric("Avg factor rate", f"{k['avg_factor_rate']:.3f}×",
              help="Average repay multiple. 1.18× ≈ an 18% fee on principal.")

    st.markdown("")

    # Status composition — a simple decluttered horizontal stacked bar.
    mix = advances["status"].value_counts()
    order = ["repaid", "active", "charged_off"]
    labels = {"repaid": "Repaid in full", "active": "Active / on-track",
              "charged_off": "Charged off"}
    colors = {"repaid": ACCENT, "active": GREY, "charged_off": ALERT}

    fig = go.Figure()
    for s in order:
        n = int(mix.get(s, 0))
        fig.add_bar(
            y=["Book"], x=[n], orientation="h", name=labels[s],
            marker_color=colors[s],
            text=[f"{labels[s]}<br>{n:,} ({n / k['n_advances']:.0%})"],
            textposition="inside", insidetextanchor="middle",
            textfont=dict(color="white", size=13),
            hovertemplate=f"{labels[s]}: %{{x:,}}<extra></extra>",
        )
    fig.update_layout(barmode="stack", height=230)
    fig = base_layout(
        fig,
        "Two-thirds of the book is still in flight — charge-offs are contained",
        "Advance count by current status",
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    st.plotly_chart(fig, width='stretch')

    so_what(
        f"The book has deployed <b>{usd_compact(k['total_disbursed'])}</b> across "
        f"<b>{k['n_advances']:,}</b> advances at a blended <b>{k['avg_factor_rate']:.3f}×</b> "
        f"factor rate (~{(k['avg_factor_rate'] - 1) * 100:.0f}% fee), implying a "
        f"<b>{usd_compact(k['gross_fee_pool'])}</b> gross-fee pool if everything repays. "
        f"A <b>{k['default_rate_count']:.1%}</b> charge-off rate sits comfortably below "
        f"that fee margin, so unit economics are positive — but ~68% of advances are still "
        f"active, so keep watching the young cohorts in the next view."
    )


# =========================================================================== #
# VIEW 2 — Cohort repayment curves
# =========================================================================== #
def view_cohorts() -> None:
    st.title("How fast does each origination cohort pay us back?")
    st.caption("Cumulative % of owed dollars collected, by months since origination")

    matrix = an.cohort_repayment_matrix(repayments)
    summary = an.cohort_maturity_summary(matrix)

    tab_lines, tab_triangle = st.tabs(["Repayment curves", "Repayment triangle"])

    # --- Line set: highlight oldest vs. newest, grey the rest -------------- #
    with tab_lines:
        cohorts = list(matrix.index)
        newest, oldest = cohorts[-1], cohorts[0]
        # pick a mid cohort to also highlight
        mid = cohorts[len(cohorts) // 2]
        highlight = {oldest: ACCENT, mid: ACCENT_L, newest: INK}

        fig = go.Figure()
        for cohort in cohorts:
            series = matrix.loc[cohort].dropna()
            if cohort in highlight:
                continue
            fig.add_trace(go.Scatter(
                x=series.index, y=series.values, mode="lines",
                line=dict(color=GREY, width=1), hoverinfo="skip",
                opacity=0.5,
            ))
        for cohort, color in highlight.items():
            series = matrix.loc[cohort].dropna()
            fig.add_trace(go.Scatter(
                x=series.index, y=series.values, mode="lines",
                line=dict(color=color, width=3),
                hovertemplate=f"{cohort}<br>month %{{x}}: %{{y:.0%}} repaid<extra></extra>",
            ))
            # direct end-of-line label
            fig.add_annotation(
                x=series.index.max(), y=series.values[-1],
                text=f"  {cohort}", showarrow=False, xanchor="left",
                font=dict(color=color, size=12, family="Inter"),
            )
        fig.update_layout(height=460)
        fig.update_xaxes(title="Months since origination", dtick=1)
        fig.update_yaxes(title="", tickformat=".0%", range=[0, 1])
        fig = base_layout(
            fig,
            "Cohorts converge on ~90%+ recovery by month 12",
            "Grey = individual cohorts · colored = oldest, mid, newest cohort",
        )
        st.plotly_chart(fig, width='stretch')

    # --- Triangle heatmap -------------------------------------------------- #
    with tab_triangle:
        z = matrix.values * 100
        fig = go.Figure(data=go.Heatmap(
            z=z,
            x=[f"m{c}" for c in matrix.columns],
            y=list(matrix.index),
            colorscale=[[0, "#eef5f5"], [0.5, ACCENT_L], [1, ACCENT]],
            hovertemplate="cohort %{y}<br>%{x}: %{z:.0f}% repaid<extra></extra>",
            colorbar=dict(title="% repaid", ticksuffix="%"),
            showscale=True,
        ))
        fig.update_layout(height=560, showlegend=False)
        fig.update_yaxes(autorange="reversed")
        fig = base_layout(
            fig,
            "The repayment triangle: each row ages out to the right",
            "Blank upper-right cells = cohorts not yet old enough to observe",
        )
        st.plotly_chart(fig, width='stretch')

    latest = summary.sort_index()
    mean_m6 = matrix[6].mean() if 6 in matrix.columns else np.nan
    so_what(
        f"Every cohort tracks the same saturating curve — roughly "
        f"<b>{mean_m6:.0%}</b> of owed dollars are collected by month 6 and "
        f"<b>~90%+</b> by month 12. Because the shape is so stable, month-3 repayment "
        f"is a reliable early signal: a new cohort landing below its peers at month 3 "
        f"should trigger a review of that vintage's underwriting rather than waiting a "
        f"full year to see the loss."
    )


# =========================================================================== #
# VIEW 3 — A/B test on offer structure
# =========================================================================== #
def view_ab() -> None:
    st.title("Does offer structure B lift take-rate without hurting repayment?")
    st.caption("Experiment: A = standard factor-rate offer · "
               "B = lower upfront fee, higher holdback %")

    # Primary metric: take-rate (acceptance). Guardrail: repayment among accepted.
    take = an.ab_test_from_offers(offers, "accepted")
    accepted_only = offers[offers["accepted"] == 1]
    guard = an.ab_test_from_offers(accepted_only, "repaid_in_full")

    st.subheader("Primary metric — take-rate (offer acceptance)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("A — take-rate", f"{take['p_a']:.1%}", help=f"{take['successes_a']:,} of {take['n_a']:,}")
    c2.metric("B — take-rate", f"{take['p_b']:.1%}",
              delta=f"+{take['abs_lift'] * 100:.1f} pts",
              help=f"{take['successes_b']:,} of {take['n_b']:,}")
    c3.metric("Relative lift", f"+{take['rel_lift']:.1%}")
    c4.metric("p-value", f"{take['p_value']:.4f}",
              delta="significant" if take["significant"] else "not significant",
              delta_color="normal" if take["significant"] else "off")

    # Bar chart with 95% CI on each arm's rate.
    def wald_ci(p, n):
        se = np.sqrt(p * (1 - p) / n)
        z = 1.959963985
        return z * se

    err_a = wald_ci(take["p_a"], take["n_a"])
    err_b = wald_ci(take["p_b"], take["n_b"])

    fig = go.Figure()
    fig.add_bar(
        x=["Offer A (control)", "Offer B (variant)"],
        y=[take["p_a"], take["p_b"]],
        marker_color=[GREY, ACCENT],
        error_y=dict(type="data", array=[err_a, err_b], color=GREY_D, thickness=1.5, width=8),
        text=[f"{take['p_a']:.1%}", f"{take['p_b']:.1%}"],
        textposition="outside", textfont=dict(size=15, color=INK),
        hovertemplate="%{x}<br>take-rate %{y:.2%}<extra></extra>",
        width=0.5,
    )
    fig.update_layout(height=380)
    fig.update_yaxes(tickformat=".0%", range=[0, max(take["p_a"], take["p_b"]) * 1.35])
    sig_txt = "statistically significant" if take["significant"] else "not significant"
    fig = base_layout(
        fig,
        f"Offer B lifts take-rate by {take['abs_lift'] * 100:.1f} pts "
        f"(+{take['rel_lift']:.0%}) — {sig_txt}",
        f"Two-proportion z-test · z = {take['z_stat']:.2f} · p = {take['p_value']:.4f} · "
        f"95% CI on lift: [{take['ci_low'] * 100:.1f}, {take['ci_high'] * 100:.1f}] pts · "
        "bars show 95% CI per arm",
    )
    st.plotly_chart(fig, width='stretch')

    st.subheader("Guardrail metric — repaid-in-full rate (among accepted)")
    g1, g2, g3, g4 = st.columns(4)
    g1.metric("A — repaid in full", f"{guard['p_a']:.1%}")
    g2.metric("B — repaid in full", f"{guard['p_b']:.1%}",
              delta=f"{guard['abs_lift'] * 100:+.1f} pts", delta_color="off")
    g3.metric("95% CI on diff",
              f"[{guard['ci_low'] * 100:+.1f}, {guard['ci_high'] * 100:+.1f}] pts")
    g4.metric("p-value", f"{guard['p_value']:.4f}",
              delta="no significant harm" if not guard["significant"] else "significant drop",
              delta_color="off" if not guard["significant"] else "inverse")

    guard_clause = (
        "the repayment guardrail shows no statistically significant degradation"
        if not guard["significant"]
        else f"BUT the guardrail flags a significant repayment drop (p={guard['p_value']:.3f}) — do not ship as-is"
    )
    decision = "Ship Offer B." if take["significant"] and not guard["significant"] else "Hold — investigate the guardrail."
    so_what(
        f"<b>{decision}</b> Offer B lifts take-rate from <b>{take['p_a']:.1%}</b> to "
        f"<b>{take['p_b']:.1%}</b> (+{take['rel_lift']:.0%} relative, p={take['p_value']:.4f}, "
        f"95% CI [{take['ci_low'] * 100:.1f}, {take['ci_high'] * 100:.1f}] pts), and "
        f"{guard_clause}. More accepted advances at equal repayment quality means more "
        f"fee revenue on the same funnel — roll B out, keeping the repaid-in-full rate "
        f"on a monitoring dashboard."
    )

    with st.expander("How the test is computed"):
        st.markdown(
            "- **Two-proportion z-test** (`scipy.stats.norm`) on acceptance counts per arm.\n"
            "- **Pooled** standard error for the test statistic / p-value (H₀: p_A = p_B); "
            "**unpooled** SE for the 95% CI on the observed difference.\n"
            "- Take-rate is the **primary** metric; repaid-in-full among accepted offers is a "
            "**guardrail** so a take-rate win can't mask a credit-quality loss.\n"
            "- All logic lives in `analytics.two_proportion_ztest` and is unit-tested."
        )


# =========================================================================== #
# VIEW 4 — Risk by segment & band
# =========================================================================== #
def view_risk() -> None:
    st.title("Where is the credit risk concentrated?")
    st.caption("Charge-off rates across merchant segments and underwriting risk bands")

    by_seg = an.default_rate_by(advances, "segment")
    by_band = an.default_rate_by(advances, "risk_band").reindex(["A", "B", "C", "D"])

    portfolio_rate = advances["status"].eq(an.STATUS_CHARGED_OFF).mean()

    col1, col2 = st.columns(2)

    # --- Segment bars: alert-red for the worst, grey for the rest --------- #
    with col1:
        seg = by_seg.sort_values("default_rate")
        worst = seg["default_rate"].idxmax()
        colors = [ALERT if idx == worst else GREY for idx in seg.index]
        fig = go.Figure()
        fig.add_bar(
            x=seg["default_rate"], y=seg.index, orientation="h",
            marker_color=colors,
            text=[f"{v:.1%}" for v in seg["default_rate"]],
            textposition="outside", textfont=dict(size=13, color=INK),
            hovertemplate="%{y}<br>charge-off %{x:.1%}<extra></extra>",
        )
        fig.add_vline(x=portfolio_rate, line=dict(color=GREY_D, width=1, dash="dot"))
        fig.add_annotation(x=portfolio_rate, y=-0.6, yref="y",
                           text=f"portfolio {portfolio_rate:.1%}",
                           showarrow=False, font=dict(color=GREY_D, size=11),
                           xanchor="center")
        fig.update_layout(height=420)
        fig.update_xaxes(tickformat=".0%", range=[0, seg["default_rate"].max() * 1.3])
        fig = base_layout(
            fig,
            "Restaurants charge off ~2× the safest segment",
            "Charge-off rate by merchant segment · dotted line = portfolio average",
        )
        st.plotly_chart(fig, width='stretch')

    # --- Risk band: monotonic gradient, the whole point is A<B<C<D -------- #
    with col2:
        band_colors = {"A": ACCENT, "B": ACCENT_L, "C": "#e3a44b", "D": ALERT}
        fig = go.Figure()
        fig.add_bar(
            x=by_band.index, y=by_band["default_rate"],
            marker_color=[band_colors[b] for b in by_band.index],
            text=[f"{v:.1%}" for v in by_band["default_rate"]],
            textposition="outside", textfont=dict(size=14, color=INK),
            hovertemplate="band %{x}<br>charge-off %{y:.1%}<br>"
                          "%{customdata:,} advances<extra></extra>",
            customdata=by_band["n_advances"],
            width=0.6,
        )
        fig.update_layout(height=420)
        fig.update_yaxes(tickformat=".0%", range=[0, by_band["default_rate"].max() * 1.3])
        fig.update_xaxes(title="Risk band (A = safest)")
        fig = base_layout(
            fig,
            "Risk bands rank-order default cleanly — underwriting works",
            "Charge-off rate by assigned risk band",
        )
        st.plotly_chart(fig, width='stretch')

    # Exposure table
    st.markdown("##### Dollar exposure by segment")
    tbl = by_seg.copy()
    tbl["disbursed"] = tbl["disbursed"].map(usd_compact)
    tbl["dollars_charged_off"] = tbl["dollars_charged_off"].map(usd_compact)
    tbl["default_rate"] = (tbl["default_rate"] * 100).round(1).astype(str) + "%"
    tbl["dollar_default_rate"] = (tbl["dollar_default_rate"] * 100).round(1).astype(str) + "%"
    tbl = tbl.rename(columns={
        "n_advances": "Advances", "n_defaults": "Charge-offs",
        "disbursed": "Disbursed", "dollars_charged_off": "$ charged off",
        "default_rate": "CO rate (ct)", "dollar_default_rate": "CO rate ($)",
    })
    st.dataframe(tbl, width='stretch')

    worst_seg = by_seg["default_rate"].idxmax()
    worst_rate = by_seg["default_rate"].max()
    so_what(
        f"The A→D risk bands rank-order charge-offs monotonically "
        f"({by_band.loc['A', 'default_rate']:.1%} → {by_band.loc['D', 'default_rate']:.1%}), "
        f"so the underwriting model is doing its job. The concentration is by segment: "
        f"<b>{worst_seg}</b> charge off at <b>{worst_rate:.1%}</b> vs. the "
        f"<b>{portfolio_rate:.1%}</b> portfolio average. Rather than cutting the segment, "
        f"nudge its pricing — a modestly higher factor rate or tighter Band-C/D limits for "
        f"{worst_seg} merchants restores risk-adjusted return without turning away good demand."
    )


# --------------------------------------------------------------------------- #
# Router
# --------------------------------------------------------------------------- #
if view == "Portfolio KPIs":
    view_kpis()
elif view == "Cohort Repayment Curves":
    view_cohorts()
elif view == "A/B Test: Offer Structure":
    view_ab()
else:
    view_risk()

st.markdown("---")
st.caption("Built by Leo Chan · github.com/GoldenCrate · synthetic-data portfolio demo · "
           "not affiliated with any company.")
