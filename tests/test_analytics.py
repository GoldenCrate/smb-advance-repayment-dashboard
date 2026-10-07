"""Unit tests for analytics.py — runs without Streamlit.

    pytest            # from the project root
"""

import numpy as np
import pandas as pd
import pytest

import analytics as an
import data_gen


@pytest.fixture(scope="module")
def frames():
    return data_gen.generate(seed=42)


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
def test_generator_is_deterministic():
    a = data_gen.generate(seed=7)["advances"]
    b = data_gen.generate(seed=7)["advances"]
    pd.testing.assert_frame_equal(a, b)


# --------------------------------------------------------------------------- #
# KPIs
# --------------------------------------------------------------------------- #
def test_compute_kpis_shapes_and_bounds(frames):
    k = an.compute_kpis(frames["advances"])
    assert k["n_advances"] == len(frames["advances"])
    assert 0 <= k["repayment_rate"] <= 1
    assert 0 <= k["default_rate_count"] <= 1
    assert k["total_disbursed"] > 0
    assert k["avg_factor_rate"] > 1.0  # advances always cost more than principal


def test_compute_kpis_empty_raises():
    with pytest.raises(ValueError):
        an.compute_kpis(frames_empty())


def frames_empty():
    return pd.DataFrame(
        columns=["disbursed", "factor_rate", "expected_repayment",
                 "amount_repaid", "status"]
    )


# --------------------------------------------------------------------------- #
# Two-proportion z-test
# --------------------------------------------------------------------------- #
def test_ztest_detects_clear_difference():
    r = an.two_proportion_ztest(200, 1000, 300, 1000)
    assert r["p_b"] > r["p_a"]
    assert r["abs_lift"] == pytest.approx(0.10, abs=1e-9)
    assert r["p_value"] < 0.001
    assert r["significant"]
    # CI should exclude zero for a clear effect
    assert r["ci_low"] > 0


def test_ztest_null_when_equal():
    r = an.two_proportion_ztest(150, 1000, 150, 1000)
    assert r["abs_lift"] == pytest.approx(0.0, abs=1e-9)
    assert r["p_value"] == pytest.approx(1.0, abs=1e-9)
    assert not r["significant"]


def test_ztest_ci_brackets_point_estimate():
    r = an.two_proportion_ztest(120, 800, 170, 820)
    assert r["ci_low"] < r["abs_lift"] < r["ci_high"]


def test_ztest_matches_known_value():
    # Hand-checked against a two-proportion z calculation.
    r = an.two_proportion_ztest(50, 500, 80, 500)
    # p_a=.10, p_b=.16, pooled p=.13 -> SE=.02126 -> z≈2.822
    assert r["z_stat"] == pytest.approx(2.822, abs=0.01)
    assert r["significant"]


def test_ztest_invalid_inputs():
    with pytest.raises(ValueError):
        an.two_proportion_ztest(10, 0, 5, 100)
    with pytest.raises(ValueError):
        an.two_proportion_ztest(150, 100, 5, 100)  # successes > n


def test_ab_from_offers_take_rate_positive(frames):
    r = an.ab_test_from_offers(frames["offers"], "accepted")
    # Generator builds in a genuine lift for B.
    assert r["p_b"] > r["p_a"]
    assert r["significant"]


# --------------------------------------------------------------------------- #
# Cohort matrix
# --------------------------------------------------------------------------- #
def test_cohort_matrix_is_triangular(frames):
    m = an.cohort_repayment_matrix(frames["repayments"])
    # Columns should be sorted ascending months_since.
    assert list(m.columns) == sorted(m.columns)
    # Month 0 column fully observed; the newest cohort only reaches month 0.
    assert m[0].notna().all()
    # Values are fractions.
    vals = m.values[~np.isnan(m.values)]
    assert vals.min() >= 0 and vals.max() <= 1
    # Cumulative repayment is non-decreasing along each row (allow tiny noise).
    for _, row in m.iterrows():
        obs = row.dropna().values
        assert np.all(np.diff(obs) >= -0.02)


def test_cohort_maturity_summary(frames):
    m = an.cohort_repayment_matrix(frames["repayments"])
    s = an.cohort_maturity_summary(m)
    assert "latest_cum_repaid" in s.columns
    assert (s["months_observed"] >= 0).all()


# --------------------------------------------------------------------------- #
# Risk aggregation
# --------------------------------------------------------------------------- #
def test_default_rate_by_segment(frames):
    d = an.default_rate_by(frames["advances"], "segment")
    assert (d["default_rate"] >= 0).all() and (d["default_rate"] <= 1).all()
    assert d["n_defaults"].sum() == frames["advances"]["status"].eq("charged_off").sum()
    # sorted descending by default_rate
    assert d["default_rate"].is_monotonic_decreasing


def test_default_rate_by_band_rank_orders(frames):
    d = an.default_rate_by(frames["advances"], "risk_band").reindex(["A", "B", "C", "D"])
    # By construction A should be safest and D riskiest.
    assert d.loc["A", "default_rate"] < d.loc["D", "default_rate"]


def test_default_rate_bad_dimension(frames):
    with pytest.raises(ValueError):
        an.default_rate_by(frames["advances"], "not_a_column")
