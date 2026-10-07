"""Framework-free business logic for the SMB Cash-Advance Repayment & Risk dashboard.

Every function here takes plain pandas objects / numbers and returns plain
pandas objects / dicts. There is NO Streamlit import, so this module can be
unit-tested in isolation (see tests/test_analytics.py).

All monetary figures are in USD. All data consumed here is SYNTHETIC
(see data_gen.py) and exists only to demonstrate analytical technique.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

# Statuses an advance can carry.
STATUS_REPAID = "repaid"
STATUS_ACTIVE = "active"
STATUS_CHARGED_OFF = "charged_off"


# --------------------------------------------------------------------------- #
# Headline KPIs
# --------------------------------------------------------------------------- #
def compute_kpis(advances: pd.DataFrame) -> dict:
    """Headline portfolio metrics computed from the advance ledger.

    Parameters
    ----------
    advances : DataFrame
        One row per originated advance. Required columns:
        advance_id, disbursed, factor_rate, expected_repayment,
        amount_repaid, status.

    Returns
    -------
    dict of scalar KPIs.
    """
    n = len(advances)
    if n == 0:
        raise ValueError("advances frame is empty; nothing to summarize")

    disbursed = advances["disbursed"].sum()
    expected = advances["expected_repayment"].sum()
    repaid = advances["amount_repaid"].sum()

    charged_off = advances["status"].eq(STATUS_CHARGED_OFF)
    n_charged_off = int(charged_off.sum())

    # Repayment rate: dollars collected vs. dollars contractually owed.
    repayment_rate = repaid / expected if expected else 0.0

    # Charge-off (default) rate, measured two honest ways.
    default_rate_count = n_charged_off / n
    default_rate_dollars = (
        advances.loc[charged_off, "disbursed"].sum() / disbursed if disbursed else 0.0
    )

    # Revenue = fees earned = expected repayment minus principal, for the
    # portion we actually collected. Simplified but directionally honest.
    avg_factor_rate = float(advances["factor_rate"].mean())
    gross_fee_pool = expected - disbursed  # total fees if everything repays

    return {
        "n_advances": int(n),
        "total_disbursed": float(disbursed),
        "total_expected_repayment": float(expected),
        "total_repaid": float(repaid),
        "repayment_rate": float(repayment_rate),
        "default_rate_count": float(default_rate_count),
        "default_rate_dollars": float(default_rate_dollars),
        "n_charged_off": n_charged_off,
        "avg_factor_rate": avg_factor_rate,
        "gross_fee_pool": float(gross_fee_pool),
    }


# --------------------------------------------------------------------------- #
# Cohort repayment curves ("repayment triangle")
# --------------------------------------------------------------------------- #
def cohort_repayment_matrix(repayments: pd.DataFrame) -> pd.DataFrame:
    """Build a cohort x months-since-origination matrix of cumulative % repaid.

    Parameters
    ----------
    repayments : DataFrame
        Long/tidy frame with columns:
        cohort (str 'YYYY-MM'), months_since (int >=0),
        cum_pct_repaid (float in [0, 1]).

    Returns
    -------
    DataFrame indexed by cohort, columns = months_since (ascending),
    values = mean cumulative fraction repaid across that cohort's advances.
    Cells that have not "aged" into existence are NaN, producing the classic
    lower-triangular repayment triangle.
    """
    matrix = repayments.pivot_table(
        index="cohort",
        columns="months_since",
        values="cum_pct_repaid",
        aggfunc="mean",
    )
    matrix = matrix.sort_index()
    matrix = matrix.reindex(sorted(matrix.columns), axis=1)
    return matrix


def cohort_maturity_summary(matrix: pd.DataFrame) -> pd.DataFrame:
    """For each cohort, report its latest observed cumulative repayment and age.

    Useful for a decluttered "where does each cohort stand now" readout.
    """
    rows = []
    for cohort, series in matrix.iterrows():
        observed = series.dropna()
        if observed.empty:
            continue
        latest_month = int(observed.index.max())
        rows.append(
            {
                "cohort": cohort,
                "months_observed": latest_month,
                "latest_cum_repaid": float(observed.loc[latest_month]),
            }
        )
    return pd.DataFrame(rows).set_index("cohort")


# --------------------------------------------------------------------------- #
# A/B test: two-proportion z-test
# --------------------------------------------------------------------------- #
def two_proportion_ztest(
    successes_a: int,
    n_a: int,
    successes_b: int,
    n_b: int,
    alpha: float = 0.05,
) -> dict:
    """Two-sided two-proportion z-test comparing conversion of B vs. A.

    Convention: A is the control/baseline, B is the variant. "lift" is the
    absolute percentage-point difference (p_b - p_a). A pooled-proportion
    standard error is used for the test statistic / p-value; an unpooled SE
    is used for the confidence interval around the observed difference (the
    standard textbook treatment).

    Returns a dict with both-arm rates, absolute & relative lift, z, p-value,
    and a 95% CI (by default) on the absolute difference.
    """
    if n_a <= 0 or n_b <= 0:
        raise ValueError("sample sizes must be positive")
    if not (0 <= successes_a <= n_a) or not (0 <= successes_b <= n_b):
        raise ValueError("successes must lie within [0, n] for each arm")

    p_a = successes_a / n_a
    p_b = successes_b / n_b
    diff = p_b - p_a

    # Pooled SE for the hypothesis test (H0: p_a == p_b).
    p_pool = (successes_a + successes_b) / (n_a + n_b)
    se_pool = np.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))

    if se_pool == 0:
        z = 0.0
        p_value = 1.0
    else:
        z = diff / se_pool
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    # Unpooled SE for the CI around the observed difference.
    se_unpooled = np.sqrt(p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    ci_low = diff - z_crit * se_unpooled
    ci_high = diff + z_crit * se_unpooled

    rel_lift = diff / p_a if p_a else np.nan

    return {
        "p_a": float(p_a),
        "p_b": float(p_b),
        "n_a": int(n_a),
        "n_b": int(n_b),
        "successes_a": int(successes_a),
        "successes_b": int(successes_b),
        "abs_lift": float(diff),
        "rel_lift": float(rel_lift),
        "z_stat": float(z),
        "p_value": float(p_value),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "alpha": float(alpha),
        "significant": bool(p_value < alpha),
    }


def ab_test_from_offers(offers: pd.DataFrame, metric: str = "accepted") -> dict:
    """Run the two-proportion z-test directly off the offers ledger.

    Parameters
    ----------
    offers : DataFrame
        One row per offer presented. Columns: offer_structure ('A'/'B'),
        and a 0/1 outcome column named by `metric` (default 'accepted' for
        take-rate; 'repaid_in_full' also supported).
    """
    if metric not in offers.columns:
        raise ValueError(f"metric '{metric}' not found in offers columns")

    arm_a = offers[offers["offer_structure"] == "A"]
    arm_b = offers[offers["offer_structure"] == "B"]

    return two_proportion_ztest(
        successes_a=int(arm_a[metric].sum()),
        n_a=len(arm_a),
        successes_b=int(arm_b[metric].sum()),
        n_b=len(arm_b),
    )


# --------------------------------------------------------------------------- #
# Risk / segment view
# --------------------------------------------------------------------------- #
def default_rate_by(advances: pd.DataFrame, dimension: str) -> pd.DataFrame:
    """Default (charge-off) rate and exposure aggregated by a categorical dim.

    Parameters
    ----------
    advances : DataFrame
        Must contain `dimension`, 'status', 'disbursed'.
    dimension : str
        Column to group by, e.g. 'segment' or 'risk_band'.

    Returns
    -------
    DataFrame indexed by the dimension with columns:
    n_advances, n_defaults, default_rate, disbursed, dollars_charged_off,
    dollar_default_rate — sorted by default_rate descending.
    """
    if dimension not in advances.columns:
        raise ValueError(f"dimension '{dimension}' not in advances")

    grp = advances.groupby(dimension)
    out = pd.DataFrame(
        {
            "n_advances": grp.size(),
            "n_defaults": grp["status"].apply(lambda s: (s == STATUS_CHARGED_OFF).sum()),
            "disbursed": grp["disbursed"].sum(),
            "dollars_charged_off": grp.apply(
                lambda d: d.loc[d["status"] == STATUS_CHARGED_OFF, "disbursed"].sum(),
                include_groups=False,
            ),
        }
    )
    out["default_rate"] = out["n_defaults"] / out["n_advances"]
    out["dollar_default_rate"] = np.where(
        out["disbursed"] > 0, out["dollars_charged_off"] / out["disbursed"], 0.0
    )
    return out.sort_values("default_rate", ascending=False)
