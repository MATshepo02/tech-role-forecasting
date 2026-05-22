# =============================================================================
# ITSFA4-14 Project 1 | Question 2 (35 Marks)
# Topic: Time Series Model Selection and Fitting
# Dataset: tech_roles_trends.csv
# Author: Matshepo Tshabangu
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
import itertools
import warnings
warnings.filterwarnings("ignore")

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "#f9f9f9",
    "axes.grid":        True,
    "grid.alpha":       0.4,
    "font.size":        11,
})

ROLES   = ["Data Analyst", "Software Engineer", "Cybersecurity Specialist"]
COLOURS = ["#2980b9", "#27ae60", "#e74c3c"]


# =============================================================================
# Load and prepare data (same as Q1 — run Q1 first or re-run setup here)
# =============================================================================

df = pd.read_csv("tech_roles_trends.csv")
date_col = df.columns[0]
df.rename(columns={date_col: "Date"}, inplace=True)
df["Date"] = pd.to_datetime(df["Date"])
df.sort_values("Date", inplace=True)
df.set_index("Date", inplace=True)
df.index.freq = pd.infer_freq(df.index)

print(f"Loaded: {df.shape[0]} weeks × {df.shape[1]} roles")
print(f"Date range: {df.index.min().date()} to {df.index.max().date()}\n")


# =============================================================================
# Q2 — Train / Test Split (2 Marks)
# =============================================================================

print("=" * 65)
print("TRAIN / TEST SPLIT")
print("=" * 65)

# Hold out last 12 weeks as test set (matches Q3 forecast horizon)
TEST_WEEKS = 12

splits = {}
for col in df.columns[:3]:
    series = df[col].dropna()
    train  = series.iloc[:-TEST_WEEKS]
    test   = series.iloc[-TEST_WEEKS:]
    splits[col] = {"train": train, "test": test, "full": series}
    print(f"{col}:")
    print(f"  Train: {len(train)} weeks  ({train.index[0].date()} → {train.index[-1].date()})")
    print(f"  Test : {len(test)}  weeks  ({test.index[0].date()} → {test.index[-1].date()})\n")


# =============================================================================
# Model Justification via ACF / PACF plots (supports parameter selection)
# =============================================================================

print("=" * 65)
print("ACF / PACF ANALYSIS — Model Parameter Identification")
print("=" * 65)

for col, colour in zip(df.columns[:3], COLOURS):
    series = splits[col]["train"]

    # Difference to make stationary (d=1 typical for weekly search data)
    series_diff = series.diff().dropna()

    fig, axes = plt.subplots(2, 2, figsize=(16, 8))

    # Raw
    axes[0, 0].plot(series.index, series.values, colour=colour, linewidth=1.2)
    axes[0, 0].set_title("Original Series (train)", fontweight="bold")
    axes[0, 0].set_ylabel("Interest Score")

    # Differenced
    axes[0, 1].plot(series_diff.index, series_diff.values,
                    colour="#2c3e50", linewidth=1.2)
    axes[0, 1].axhline(0, colour="red", linewidth=0.8, linestyle="--")
    axes[0, 1].set_title("1st-Order Differenced Series", fontweight="bold")

    # ACF — determines q (MA order): cuts off after lag q
    plot_acf(series_diff, lags=40, ax=axes[1, 0], colour=colour,
             title="ACF (differenced)")
    axes[1, 0].set_xlabel("Lag (weeks)")

    # PACF — determines p (AR order): cuts off after lag p
    plot_pacf(series_diff, lags=40, ax=axes[1, 1], colour=colour,
              title="PACF (differenced)")
    axes[1, 1].set_xlabel("Lag (weeks)")

    fig.suptitle(f"ACF / PACF Analysis — {col}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fname = f"plot_08_acf_pacf_{col.replace(' ', '_')}.png"
    plt.savefig(fname, dpi=150)
    plt.show()
    print(f"Saved: {fname}")


# =============================================================================
# Q2 — Model Selection with Justification (6 Marks)
# =============================================================================

print("\n" + "=" * 65)
print("MODEL SELECTION — SARIMA")
print("=" * 65)

# Justification for SARIMA:
# 1. Weekly search data shows trend (non-stationary → needs d≥1)
# 2. Annual seasonality observed in decomposition (period = 52 weeks → S=52)
# 3. ACF/PACF after differencing guide AR (p) and MA (q) terms
# 4. SARIMA(p,d,q)(P,D,Q)[52] generalises ARIMA to handle seasonal patterns
# 5. Auto-selection via AIC grid search to find optimal parameters objectively

print("""
MODEL CHOICE: SARIMA(p,d,q)(P,D,Q)[52]

Justification:
  - Weekly data with 52-week (annual) seasonality detected via decomposition
  - Trend non-stationarity confirmed by ADF/KPSS → d=1 (first differencing)
  - ACF/PACF on differenced series guide p and q selection
  - Seasonal ARIMA extends standard ARIMA to explicitly model annual cycles
  - AIC-based grid search used to identify optimal (p,d,q)(P,D,Q) per role
  - SARIMA is robust for labour market search data with recurring seasonal spikes
""")


# =============================================================================
# Q2 — Identifying Model Parameters via AIC Grid Search (12 Marks)
# =============================================================================

print("=" * 65)
print("PARAMETER SELECTION — AIC Grid Search")
print("=" * 65)

def sarima_grid_search(series, p_range, d_val, q_range,
                       P_range, D_val, Q_range, s=52):
    """
    Grid search over SARIMA (p,d,q)(P,D,Q)[s] parameter combinations.
    Returns a DataFrame ranked by AIC.
    """
    results = []
    param_grid = list(itertools.product(p_range, q_range, P_range, Q_range))
    total = len(param_grid)
    print(f"  Testing {total} parameter combinations (d={d_val}, D={D_val}, s={s})...")

    for i, (p, q, P, Q) in enumerate(param_grid):
        try:
            model = SARIMAX(
                series,
                order=(p, d_val, q),
                seasonal_order=(P, D_val, Q, s),
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            fit = model.fit(disp=False)
            results.append({
                "p": p, "d": d_val, "q": q,
                "P": P, "D": D_val, "Q": Q,
                "AIC": round(fit.aic, 2),
                "BIC": round(fit.bic, 2)
            })
        except Exception:
            pass

    results_df = pd.DataFrame(results).sort_values("AIC").reset_index(drop=True)
    return results_df


# Grid search ranges (kept small for runtime; extend if needed)
p_vals = range(0, 3)
q_vals = range(0, 3)
P_vals = range(0, 2)
Q_vals = range(0, 2)
d = 1
D = 1

best_params = {}
grid_results_all = {}

for col in df.columns[:3]:
    print(f"\n--- {col} ---")
    series = splits[col]["train"]

    grid_df = sarima_grid_search(series, p_vals, d, q_vals,
                                  P_vals, D, Q_vals, s=52)
    grid_results_all[col] = grid_df

    best = grid_df.iloc[0]
    best_params[col] = {
        "order":          (int(best.p), int(best.d), int(best.q)),
        "seasonal_order": (int(best.P), int(best.D), int(best.Q), 52)
    }
    print(f"\n  Top 5 models by AIC:\n{grid_df.head(5).to_string(index=False)}")
    print(f"\n  ★ Best model: SARIMA{best_params[col]['order']}"
          f"{best_params[col]['seasonal_order']}")
    print(f"    AIC = {best.AIC} | BIC = {best.BIC}")


# =============================================================================
# Q2 — Fit Best Model per Role (15 Marks)
# =============================================================================

print("\n" + "=" * 65)
print("MODEL FITTING")
print("=" * 65)

fitted_models = {}

for col, colour in zip(df.columns[:3], COLOURS):
    print(f"\n--- Fitting SARIMA for: {col} ---")
    series = splits[col]["train"]
    params = best_params[col]

    model = SARIMAX(
        series,
        order=params["order"],
        seasonal_order=params["seasonal_order"],
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    result = model.fit(disp=False)
    fitted_models[col] = result

    print(f"  SARIMA{params['order']}{params['seasonal_order']}")
    print(f"  AIC  : {result.aic:.2f}")
    print(f"  BIC  : {result.bic:.2f}")
    print(f"  Log-Likelihood: {result.llf:.2f}")
    print("\n  Model Summary:\n")
    print(result.summary())

    # --- In-sample fit plot ---
    fitted_vals = result.fittedvalues

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(series.index, series.values, colour=colour,
            label="Actual (train)", linewidth=1.2)
    ax.plot(fitted_vals.index, fitted_vals.values, colour="black",
            label="Fitted values", linewidth=1.2, linestyle="--")
    ax.set_title(f"In-Sample Fit — {col}\nSARIMA{params['order']}{params['seasonal_order']}",
                 fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Interest Score")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.tight_layout()
    fname = f"plot_09_fitted_{col.replace(' ', '_')}.png"
    plt.savefig(fname, dpi=150)
    plt.show()
    print(f"  Saved: {fname}")


# =============================================================================
# Save models & parameters for use in Q3
# =============================================================================

import pickle
with open("fitted_models.pkl", "wb") as f:
    pickle.dump(fitted_models, f)
with open("splits.pkl", "wb") as f:
    pickle.dump(splits, f)
with open("best_params.pkl", "wb") as f:
    pickle.dump(best_params, f)

print("\n✓ Models saved to fitted_models.pkl")
print("✓ Data splits saved to splits.pkl")
print("✓ Best parameters saved to best_params.pkl")
print("\nQ2 Complete ✓")
