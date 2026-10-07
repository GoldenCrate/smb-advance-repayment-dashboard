"""Synthetic data generator for the SMB Cash-Advance Repayment & Risk dashboard.

Everything produced here is FAKE. There are no real merchants, no real
advances, and no real repayment behavior. The generator is seeded so the
numbers are stable across runs and reproducible for reviewers.

Entities
--------
merchants   : SMBs that could receive a cash advance (segment, platform, risk).
advances    : advances that were actually originated (one per merchant here).
repayments  : tidy cohort x months-since cumulative repayment observations.
offers      : the A/B experiment ledger — offers presented under structure A or B.

Run `python data_gen.py` to (re)generate CSVs into ./data and print a summary.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd

SEED = 42
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# Origination cohorts: 18 monthly cohorts ending at the "as-of" date.
AS_OF = pd.Timestamp("2025-06-30")
N_COHORTS = 18
MAX_TERM_MONTHS = 12  # nominal repayment term for a cash advance

SEGMENTS = ["Restaurants", "Retail", "E-commerce", "Health & Beauty", "Professional Svcs"]
# Baseline per-segment default propensity (synthetic, hand-set for realism).
SEGMENT_RISK = {
    "Restaurants": 0.135,
    "Retail": 0.095,
    "E-commerce": 0.070,
    "Health & Beauty": 0.080,
    "Professional Svcs": 0.055,
}
PLATFORMS = ["ShopFlow", "BistroOS", "MarketHub", "PayLink"]


def _risk_band(prob_default: float) -> str:
    """Map a continuous default probability to a letter risk band A-D."""
    if prob_default < 0.06:
        return "A"
    if prob_default < 0.10:
        return "B"
    if prob_default < 0.14:
        return "C"
    return "D"


def generate(seed: int = SEED) -> dict[str, pd.DataFrame]:
    """Generate and return all synthetic frames in a dict.

    Deterministic for a fixed seed. No disk I/O (that's `main`).
    """
    rng = np.random.default_rng(seed)

    cohorts = pd.period_range(end=AS_OF.to_period("M"), periods=N_COHORTS, freq="M")
    cohort_labels = [str(c) for c in cohorts]

    # ------------------------------------------------------------------ #
    # Merchants + advances (one advance per merchant for simplicity)
    # ------------------------------------------------------------------ #
    n_merchants = 2600
    seg = rng.choice(SEGMENTS, size=n_merchants, p=[0.26, 0.24, 0.20, 0.16, 0.14])
    platform = rng.choice(PLATFORMS, size=n_merchants)

    # Monthly card revenue drives advance size (lognormal, segment-shifted).
    base_rev = rng.lognormal(mean=10.4, sigma=0.55, size=n_merchants)
    monthly_revenue = np.clip(base_rev, 4_000, 400_000)

    # Each merchant's latent default probability = segment base + noise,
    # modulated slightly by (lower revenue -> higher risk).
    seg_base = np.array([SEGMENT_RISK[s] for s in seg])
    rev_factor = np.clip(1.25 - monthly_revenue / monthly_revenue.mean() * 0.25, 0.7, 1.6)
    prob_default = np.clip(seg_base * rev_factor + rng.normal(0, 0.015, n_merchants), 0.01, 0.45)
    risk_band = np.array([_risk_band(p) for p in prob_default])

    # Assign each advance to an origination cohort (recent months a bit heavier).
    cohort_weights = np.linspace(0.7, 1.3, N_COHORTS)
    cohort_weights /= cohort_weights.sum()
    cohort_idx = rng.choice(N_COHORTS, size=n_merchants, p=cohort_weights)
    origination_month = np.array(cohort_labels)[cohort_idx]

    # Advance amount ~ 0.8x-1.6x monthly revenue, rounded to nearest $500.
    advance_mult = rng.uniform(0.8, 1.6, n_merchants)
    disbursed = np.round(monthly_revenue * advance_mult / 500) * 500
    disbursed = np.clip(disbursed, 3_000, 350_000)

    # Factor rate: lower-risk merchants get cheaper capital (1.10 - 1.32).
    factor_rate = np.clip(1.10 + prob_default * 0.9 + rng.normal(0, 0.01, n_merchants), 1.10, 1.32)
    factor_rate = np.round(factor_rate, 3)
    expected_repayment = np.round(disbursed * factor_rate, 2)

    # How many months has this advance been aging as of AS_OF?
    age_months = (N_COHORTS - 1) - cohort_idx  # cohort 0 is oldest
    age_months = np.clip(age_months, 0, MAX_TERM_MONTHS)

    # Realized outcome. Draw a default event; timing scaled by age.
    defaulted = rng.random(n_merchants) < prob_default
    # Mature enough to be fully repaid? (reached term and didn't default)
    matured = age_months >= MAX_TERM_MONTHS

    status = np.full(n_merchants, "active", dtype=object)
    status[defaulted & (age_months >= 2)] = "charged_off"
    status[matured & ~defaulted] = "repaid"
    # (young, non-defaulted advances remain "active")

    # Amount repaid to date.
    # repaid advances -> full expected; charged_off -> partial; active -> pro-rata.
    frac_through_term = np.clip(age_months / MAX_TERM_MONTHS, 0, 1)
    amount_repaid = np.empty(n_merchants)
    for i in range(n_merchants):
        if status[i] == "repaid":
            amount_repaid[i] = expected_repayment[i]
        elif status[i] == "charged_off":
            # recovered somewhere between 20% and 65% of expected before default
            amount_repaid[i] = expected_repayment[i] * rng.uniform(0.20, 0.65)
        else:  # active, on-track with small noise
            on_track = frac_through_term[i] * (1 - 0.06 * rng.random())
            amount_repaid[i] = expected_repayment[i] * np.clip(on_track, 0, 1)
    amount_repaid = np.round(amount_repaid, 2)

    advances = pd.DataFrame(
        {
            "advance_id": [f"ADV{100000 + i}" for i in range(n_merchants)],
            "merchant_id": [f"M{200000 + i}" for i in range(n_merchants)],
            "segment": seg,
            "platform": platform,
            "risk_band": risk_band,
            "monthly_revenue": np.round(monthly_revenue, 2),
            "cohort": origination_month,
            "age_months": age_months,
            "disbursed": disbursed,
            "factor_rate": factor_rate,
            "expected_repayment": expected_repayment,
            "amount_repaid": amount_repaid,
            "status": status,
        }
    )
    merchants = advances[
        ["merchant_id", "segment", "platform", "risk_band", "monthly_revenue"]
    ].copy()

    # ------------------------------------------------------------------ #
    # Cohort repayment curves (tidy: cohort x months_since)
    # ------------------------------------------------------------------ #
    # For each cohort we model the AVERAGE cumulative fraction repaid as a
    # saturating curve, with older/healthier cohorts climbing a touch higher.
    repay_rows = []
    for ci, label in enumerate(cohort_labels):
        observed_age = int((N_COHORTS - 1) - ci)
        observed_age = min(observed_age, MAX_TERM_MONTHS)
        # Cohort-level asymptote: how much of expected it will ultimately collect.
        cohort_default = SEGMENT_RISK  # unused directly; use mean portfolio
        asymptote = np.clip(0.985 - rng.normal(0.0, 0.01) - 0.004 * (N_COHORTS - ci), 0.90, 0.99)
        rate_k = 0.38 + rng.normal(0, 0.02)  # curve steepness
        for m in range(0, observed_age + 1):
            cum = asymptote * (1 - np.exp(-rate_k * m))
            cum = float(np.clip(cum + rng.normal(0, 0.004), 0, 1))
            repay_rows.append(
                {"cohort": label, "months_since": m, "cum_pct_repaid": round(cum, 4)}
            )
    repayments = pd.DataFrame(repay_rows)

    # ------------------------------------------------------------------ #
    # A/B experiment ledger (offer structure A vs B)
    # ------------------------------------------------------------------ #
    # Experiment: structure A = standard factor-rate offer; structure B =
    # a "lower upfront fee, slightly higher holdback %" restructure. We test
    # whether B lifts the take-rate (acceptance) without hurting repayment.
    n_offers = 9000
    offer_structure = rng.choice(["A", "B"], size=n_offers, p=[0.5, 0.5])
    off_seg = rng.choice(SEGMENTS, size=n_offers, p=[0.26, 0.24, 0.20, 0.16, 0.14])

    # True (synthetic) acceptance probabilities: B genuinely lifts take-rate.
    base_accept = 0.188
    accept_p = np.where(offer_structure == "B", base_accept + 0.021, base_accept)
    accepted = (rng.random(n_offers) < accept_p).astype(int)

    # Among accepted offers, did they repay in full? B is designed to be
    # repayment-neutral (no meaningful degradation) — a guardrail metric.
    base_repay = 0.86
    repay_p = np.where(offer_structure == "B", base_repay - 0.004, base_repay)
    repaid_in_full = np.where(
        accepted == 1, (rng.random(n_offers) < repay_p).astype(int), 0
    )

    offers = pd.DataFrame(
        {
            "offer_id": [f"OF{300000 + i}" for i in range(n_offers)],
            "offer_structure": offer_structure,
            "segment": off_seg,
            "accepted": accepted,
            "repaid_in_full": repaid_in_full,
        }
    )

    return {
        "merchants": merchants,
        "advances": advances,
        "repayments": repayments,
        "offers": offers,
    }


def main() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    frames = generate()
    for name, df in frames.items():
        path = os.path.join(DATA_DIR, f"{name}.csv")
        df.to_csv(path, index=False)
        print(f"wrote {path:<60} rows={len(df):>6}  cols={df.shape[1]}")

    adv = frames["advances"]
    print("\nquick sanity check (SYNTHETIC data):")
    print(f"  advances originated   : {len(adv):,}")
    print(f"  total disbursed       : ${adv['disbursed'].sum():,.0f}")
    print(f"  charge-off rate (cnt) : {adv['status'].eq('charged_off').mean():.1%}")
    print(f"  status mix            : {adv['status'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
