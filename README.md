# SMB Cash-Advance Repayment & Risk — Dashboard

A Streamlit portfolio project modeling the analytics behind an **embedded-fintech
capital product**: short-term cash advances extended to small and medium
businesses (SMBs) through platform partners, repaid as a fixed multiple
(factor rate) of the amount disbursed.

It demonstrates the day-to-day toolkit of a data scientist on a capital team:
**experimentation (A/B testing), cohort/vintage analysis, risk segmentation,
and stakeholder storytelling** — each view ends with an explicit
*"So what / recommendation."*

> ⚠️ **All data in this project is 100% synthetic** and generated with a fixed
> random seed (`data_gen.py`). There are no real merchants, advances, or
> repayment records. It exists only to showcase analytical technique.

Author: **Leo Chan — [github.com/GoldenCrate](https://github.com/GoldenCrate)**

---

## Views

1. **Portfolio KPIs** — advances originated, total disbursed, repayment rate
   (collected vs. owed), charge-off / default rate (count and dollar-weighted),
   and average factor rate, with a status-composition bar.
2. **Cohort Repayment Curves** — monthly origination cohorts plotted as
   cumulative-% -repaid curves and as a classic lower-triangular **repayment
   triangle** heatmap; shows how quickly each vintage pays back.
3. **A/B Test: Offer Structure** — compares two offer structures (A = standard
   factor-rate offer, B = lower upfront fee / higher holdback) on **take-rate**
   using a **two-proportion z-test** (lift, z-stat, p-value, 95% CI), with
   **repaid-in-full** among accepted offers as a **guardrail** metric.
4. **Risk by Segment & Band** — charge-off rates by merchant segment and by
   underwriting risk band (A–D), with dollar exposure, to show where losses
   concentrate and whether the risk bands rank-order default correctly.

## Design

Charts follow Cole Nussbaumer Knaflic's *Storytelling with Data*: takeaway
sentence titles, decluttered axes and gridlines, direct labels instead of
legends, and restrained color (one teal accent for signal, grey for context,
a single red reserved for loss/default).

## Project layout

```
app.py                 # Streamlit multi-view app (presentation only)
analytics.py           # framework-free business logic (unit-testable, no Streamlit)
data_gen.py            # seeded synthetic data generator  ->  ./data/*.csv
tests/test_analytics.py# pytest unit tests for the analytics module
requirements.txt
README.md
.gitignore
```

The separation matters: **`analytics.py` imports no Streamlit**, so every
metric, the z-test, the cohort matrix, and the risk aggregations are tested in
isolation with `pytest`.

## Run it

```bash
# 1. install
pip install -r requirements.txt

# 2. (optional) regenerate the synthetic CSVs + print a sanity summary
python data_gen.py

# 3. run the app
streamlit run app.py
```

The app generates its data in-memory on startup (cached), so step 2 is only
needed if you want the CSV extracts in `./data/`.

## Tests

```bash
pytest
```

Covers generator determinism, KPI bounds, the two-proportion z-test (including
a hand-checked z value and null/invalid cases), the triangular cohort matrix,
and monotonic risk-band ordering.

## Notes on the numbers

- **Factor rate** is the repay multiple (e.g. `1.18×` ≈ an 18% fee on
  principal); `expected_repayment = disbursed × factor_rate`.
- **Repayment rate** on the KPI view is *dollars collected to date vs. total
  dollars owed* across a book with many still-active advances, so it reads
  lower than a matured-cohort recovery rate (see the cohort view, which reaches
  ~90%+ by month 12).
- **Charge-off / default rate** is reported both by advance count and
  dollar-weighted.

*Not affiliated with or endorsed by any company. Built independently as a
portfolio demonstration.*
