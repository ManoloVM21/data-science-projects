## Collaboration Context
This project was developed in collaboration with a university Data Science Society, which facilitated access to the donor data used below.

## My Contribution
- Data cleaning and preprocessing
- Feature engineering
- Model development (XGBoost, with a Ridge regression baseline)
- Model evaluation and interpretation

# Donor Pledge Amount Prediction

Regression model that forecasts each donor's annual pledge amount for the next fundraising year, and segments the full donor base into four actionable outreach quadrants.

Built using 10 years of proprietary donation data from a university athletics fundraising organization ("the Organization"), joined from six internal MySQL tables (~45K active donors).

---

## Problem

The Organization's fundraising agents manually prioritize donor outreach with limited analytical support. The goal was to replace gut-feel prioritization with a data-driven model that estimates how much each donor is likely to commit — and flags which donors are most worth the agent's time.

---

## Technical Approach

| Step | Detail |
|---|---|
| **Data** | Six MySQL tables joined by `FanID`: pledge/payment history, demographics, wealth scores (iWave), education, board membership |
| **Target** | `log1p(annual pledge amount)` — log transformation applied to handle extreme right skew (median ~$650, max $791K+) |
| **Split** | Temporal: Train 2014–2023 · Validation 2024 · Holdout 2025 — no random split to prevent data leakage |
| **Baseline** | Ridge Regression (handles multicollinearity between temporal features) |
| **Model** | XGBoost Regressor with RandomizedSearchCV (25 combinations, 3-fold CV) |
| **Evaluation** | MAE and RMSE in original dollar scale via `expm1` — MAE preferred because RMSE is dominated by mega-donors |

**Validation result:** MAE ≈ $1,146 (R² = 0.750 excluding major donors). On raw dollar MAE, this does *not* beat a naïve "repeat most recent pledge" baseline ($1,088 MAE) — persistence is a strong benchmark here since most donors renew at the same amount. The model's value is in producing a consistent ranked estimate for *every* donor, including new/lapsed donors a persistence rule can't handle, plus the SHAP-based explainability and segmentation a baseline can't supply. See the [full report](donor_prediction.html) for the honest comparison.

---

## Feature Engineering Highlights

- **Ordinal membership encoding** — 10 membership tiers mapped to integers (Legacy Chief=10 → Spirit=1)
- **Degree classification** — 93 unique degree strings collapsed to 5 ordinal levels via `classify_degree()`
- **Regional encoding** — Florida flag + 4 US regions (50-state OHE adds noise; FL represents 75% of donors)
- **Missingness flags** — `Age_null`, `Properties_null` preserve information in missing wealth/demographic data
- **Age imputation** — median computed on train set only, applied to validation and holdout

---

## Donor Segmentation

Final output: each donor assigned to one of four quadrants by crossing predicted pledge amount (p75 threshold) against historical payment reliability (binary: 100% commitment rate or not):

| Quadrant | Amount | Reliability | Strategy |
|---|---|---|---|
| VIP | High | 100% | Priority contact, relationship maintenance |
| Alto Potencial | High | < 100% | High value, needs follow-up |
| Base Estable | Low | 100% | Reliable, low-maintenance |
| Baja Prioridad | Low | < 100% | Lower agent investment |

Segmentation table exported to CSV for Tableau dashboards used by the fundraising team.

---

## Skills

`Python` · `XGBoost` · `scikit-learn` · `pandas` · `NumPy` · `Matplotlib` · `MySQL` · Feature Engineering · Temporal Cross-Validation · Regression · Donor Segmentation · Data Leakage Prevention

---

## Limitations

- Model scope limited to ~45K donors with pledge history out of 706K in the base table — new donor prospecting requires a separate model
- Holdout set (2025, 1,173 rows) has atypical distribution due to pre-registered small pledges; R² on this split is not representative
- `TotalPledgeAnual` dominates feature importance (~50%) — the model largely learns to carry forward current-year behavior, which is strong but not sufficient for detecting trend changes
