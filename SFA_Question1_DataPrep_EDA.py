# =============================================================================
# ITSFA4-14 Project 1 | Question 1 (35 Marks)
# Topic: Time Series — Data Preparation, EDA, Decomposition, Stationarity
# Dataset: tech_roles_trends.csv
# Author: Matshepo Tshabangu
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
import warnings
warnings.filterwarnings("ignore")

# Plot styling
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
# Q1.1 — Prepare the dataset for time series analysis
# =============================================================================

print("=" * 65)
print("Q1.1 — DATA PREPARATION")
print("=" * 65)

# --- Load ---
df = pd.read_csv("tech_roles_trends.csv")
print(f"\nRaw shape: {df.shape[0]} rows × {df.shape[1]} columns")
print("\nFirst 5 rows:")
print(df.head())
print("\nColumn names:", df.columns.tolist())
print("\nData types:\n", df.dtypes)

# --- Rename columns for consistency ---
# Assumes columns: date/week, Data Analyst, Software Engineer, Cybersecurity Specialist
# Update column names below to match your actual CSV
date_col = df.columns[0]       # first column assumed to be date
df.rename(columns={date_col: "Date"}, inplace=True)

# --- Parse date as datetime and set as index ---
df["Date"] = pd.to_datetime(df["Date"])
df.sort_values("Date", inplace=True)
df.set_index("Date", inplace=True)

print(f"\nDate range: {df.index.min().date()} to {df.index.max().date()}")
print(f"Total weeks: {len(df)}")

# --- CHECK: Missing values ---
print("\n--- Missing Values ---")
missing = df.isnull().sum()
print(missing)
assert missing.sum() == 0, "WARNING: Missing values detected — handle before proceeding!"
print("✓ No missing values found.")

# --- CHECK: Duplicate observations ---
print("\n--- Duplicate Rows ---")
dups = df.duplicated()
print(f"Duplicate rows: {dups.sum()}")
assert dups.sum() == 0, "WARNING: Duplicate rows detected — drop before proceeding!"
print("✓ No duplicate observations found.")

# --- CHECK: Time index is ordered chronologically ---
print("\n--- Chronological Order Check ---")
is_sorted = df.index.is_monotonic_increasing
print(f"Index is monotonically increasing (chronological): {is_sorted}")
assert is_sorted, "WARNING: Dates are not in order — run df.sort_index()"
print("✓ Time index is correctly ordered chronologically.")

# --- CHECK: Regular weekly frequency ---
print("\n--- Frequency Check ---")
df.index.freq = pd.infer_freq(df.index)
print(f"Inferred frequency: {df.index.freq}")
print("\nClean dataset summary:")
print(df.describe().round(2))


# =============================================================================
# Q1.2 — Exploratory Analysis: visualisations, trends, seasonality
# =============================================================================

print("\n" + "=" * 65)
print("Q1.2 — EXPLORATORY ANALYSIS")
print("=" * 65)

# --- Plot 1: Raw time series for all three roles ---
fig, ax = plt.subplots(figsize=(14, 5))
for col, colour in zip(df.columns[:3], COLOURS):
    ax.plot(df.index, df[col], label=col, colour=colour, linewidth=1.5)

ax.set_title("Weekly Search Interest — Tech Job Roles (Apr 2019 – Apr 2024)",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Date")
ax.set_ylabel("Search Interest Score")
ax.legend(fontsize=10)
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
plt.tight_layout()
plt.savefig("plot_01_raw_series.png", dpi=150)
plt.show()
print("Plot saved: plot_01_raw_series.png")

# --- Plot 2: Individual subplots per role ---
fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
for ax, col, colour in zip(axes, df.columns[:3], COLOURS):
    ax.plot(df.index, df[col], colour=colour, linewidth=1.4)
    # Rolling 12-week mean trend line
    rolling_mean = df[col].rolling(window=12).mean()
    ax.plot(df.index, rolling_mean, colour="black", linewidth=2,
            linestyle="--", label="12-week rolling mean")
    ax.set_title(col, fontweight="bold")
    ax.set_ylabel("Interest Score")
    ax.legend(fontsize=9)

axes[-1].set_xlabel("Date")
axes[-1].xaxis.set_major_locator(mdates.YearLocator())
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
fig.suptitle("Individual Time Series with Rolling Mean Trend",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("plot_02_individual_series.png", dpi=150)
plt.show()
print("Plot saved: plot_02_individual_series.png")

# --- Plot 3: Correlation heatmap ---
fig, ax = plt.subplots(figsize=(6, 4))
corr = df[df.columns[:3]].corr()
sns.heatmap(corr, annot=True, fmt=".3f", cmap="coolwarm",
            vmin=-1, vmax=1, ax=ax, linewidths=0.5)
ax.set_title("Correlation Between Job Role Search Interests", fontweight="bold")
plt.tight_layout()
plt.savefig("plot_03_correlation_heatmap.png", dpi=150)
plt.show()
print("Plot saved: plot_03_correlation_heatmap.png")

# --- Plot 4: Annual seasonality — average search by month ---
df_monthly = df.copy()
df_monthly["Month"] = df_monthly.index.month

fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
month_names = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]
for ax, col, colour in zip(axes, df.columns[:3], COLOURS):
    monthly_avg = df_monthly.groupby("Month")[col].mean()
    ax.bar(monthly_avg.index, monthly_avg.values, color=colour, alpha=0.8)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(month_names, rotation=45, ha="right")
    ax.set_title(col, fontweight="bold")
    ax.set_ylabel("Avg Interest Score")

fig.suptitle("Average Monthly Search Interest — Seasonal Patterns",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("plot_04_seasonal_monthly.png", dpi=150)
plt.show()
print("Plot saved: plot_04_seasonal_monthly.png")

# --- Plot 5: Boxplots by year ---
df_yearly = df.copy()
df_yearly["Year"] = df_yearly.index.year

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, col, colour in zip(axes, df.columns[:3], COLOURS):
    groups = [df_yearly[df_yearly["Year"] == y][col].values
              for y in sorted(df_yearly["Year"].unique())]
    bp = ax.boxplot(groups, patch_artist=True,
                    labels=sorted(df_yearly["Year"].unique()))
    for patch in bp["boxes"]:
        patch.set_facecolor(colour)
        patch.set_alpha(0.6)
    ax.set_title(col, fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Interest Score")

fig.suptitle("Year-by-Year Distribution of Search Interest",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("plot_05_annual_boxplots.png", dpi=150)
plt.show()
print("Plot saved: plot_05_annual_boxplots.png")

print("\nEDA Interpretation:")
print("  - Observe whether scores are trending upward/downward over time")
print("  - Rolling mean reveals the underlying long-run trend, smoothing noise")
print("  - Monthly boxplot reveals seasonal spikes (e.g., Jan job search surge)")
print("  - Correlation heatmap shows how the three roles move together")


# =============================================================================
# Q1.2 (continued) — Seasonal Decomposition
# =============================================================================

print("\n--- Seasonal Decomposition ---")

# Assumes weekly data → period = 52 (annual seasonality)
PERIOD = 52

for col, colour in zip(df.columns[:3], COLOURS):
    series = df[col].dropna()

    decomp = seasonal_decompose(series, model="additive", period=PERIOD)

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    components = [
        (series,           "Observed",  colour),
        (decomp.trend,     "Trend",     "#2c3e50"),
        (decomp.seasonal,  "Seasonal",  "#8e44ad"),
        (decomp.resid,     "Residual",  "#95a5a6"),
    ]
    for ax, (data, label, c) in zip(axes, components):
        ax.plot(data.index, data.values, colour=c, linewidth=1.2)
        ax.set_ylabel(label, fontsize=10)
        if label == "Residual":
            ax.axhline(0, colour="black", linewidth=0.8, linestyle="--")

    axes[-1].set_xlabel("Date")
    axes[-1].xaxis.set_major_locator(mdates.YearLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.suptitle(f"Seasonal Decomposition — {col}  (period={PERIOD} weeks)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fname = f"plot_06_decomp_{col.replace(' ', '_')}.png"
    plt.savefig(fname, dpi=150)
    plt.show()
    print(f"Saved: {fname}")

    # Decomposition stats
    trend_strength = 1 - (decomp.resid.var() /
                          (decomp.trend.dropna() + decomp.resid.dropna()).var())
    seasonal_strength = 1 - (decomp.resid.var() /
                              (decomp.seasonal + decomp.resid).var())
    print(f"\n{col}:")
    print(f"  Trend strength   : {trend_strength:.4f}  (0=none, 1=strong)")
    print(f"  Seasonal strength: {seasonal_strength:.4f}  (0=none, 1=strong)")

print("\nDecomposition Interpretation:")
print("  - Trend component shows long-run direction (rising, falling, flat)")
print("  - Seasonal component shows repeating annual cycles")
print("  - Residual should be white noise (random) if model fits well")
print("  - High trend/seasonal strength values indicate dominant components")


# =============================================================================
# Q1.3 — Stationarity Testing and Transformation
# =============================================================================

print("\n" + "=" * 65)
print("Q1.3 — STATIONARITY TESTING & TRANSFORMATION")
print("=" * 65)

def adf_test(series, name="Series"):
    """Augmented Dickey-Fuller test. H0: series has unit root (non-stationary)."""
    result = adfuller(series.dropna(), autolag="AIC")
    p_value = result[1]
    stationary = p_value < 0.05
    print(f"\n  ADF Test — {name}")
    print(f"    ADF Statistic : {result[0]:.4f}")
    print(f"    p-value       : {p_value:.6f}")
    print(f"    Critical (5%) : {result[4]['5%']:.4f}")
    print(f"    Stationary    : {'✓ YES' if stationary else '✗ NO  (p > 0.05)'}")
    return stationary

def kpss_test(series, name="Series"):
    """KPSS test. H0: series IS stationary (opposite of ADF)."""
    result = kpss(series.dropna(), regression="c", nlags="auto")
    p_value = result[1]
    stationary = p_value > 0.05
    print(f"\n  KPSS Test — {name}")
    print(f"    KPSS Statistic: {result[0]:.4f}")
    print(f"    p-value       : {p_value:.4f}")
    print(f"    Critical (5%) : {result[3]['5%']:.4f}")
    print(f"    Stationary    : {'✓ YES' if stationary else '✗ NO  (p ≤ 0.05 → reject H0)'}")
    return stationary

stationarity_results = {}

for col in df.columns[:3]:
    print(f"\n{'─'*55}")
    print(f"ROLE: {col}")
    print(f"{'─'*55}")
    series = df[col].dropna()

    # --- Test on raw series ---
    print("\n[A] Original Series:")
    adf_ok   = adf_test(series, f"{col} (original)")
    kpss_ok  = kpss_test(series, f"{col} (original)")
    is_stationary = adf_ok and kpss_ok

    if is_stationary:
        print(f"\n  → {col} is STATIONARY. No transformation needed.")
        df[f"{col}_stationary"] = series
        stationarity_results[col] = {"method": "none", "d": 0}
    else:
        # --- 1st Order Differencing ---
        print(f"\n[B] 1st-Order Differencing (d=1):")
        diff1 = series.diff().dropna()
        adf_ok1  = adf_test(diff1, f"{col} (diff=1)")
        kpss_ok1 = kpss_test(diff1, f"{col} (diff=1)")

        if adf_ok1 and kpss_ok1:
            print(f"\n  → {col}: 1st differencing achieves stationarity. d=1")
            df[f"{col}_stationary"] = diff1
            stationarity_results[col] = {"method": "diff", "d": 1}
        else:
            # --- 2nd Order Differencing ---
            print(f"\n[C] 2nd-Order Differencing (d=2):")
            diff2 = series.diff().diff().dropna()
            adf_ok2  = adf_test(diff2, f"{col} (diff=2)")
            kpss_ok2 = kpss_test(diff2, f"{col} (diff=2)")
            print(f"\n  → {col}: 2nd differencing applied. d=2")
            df[f"{col}_stationary"] = diff2
            stationarity_results[col] = {"method": "diff", "d": 2}

# --- Plot: Original vs stationary series ---
fig, axes = plt.subplots(3, 2, figsize=(16, 12))
for i, (col, colour) in enumerate(zip(df.columns[:3], COLOURS)):
    stat_col = f"{col}_stationary"
    # Original
    axes[i, 0].plot(df.index, df[col], colour=colour, linewidth=1.2)
    axes[i, 0].set_title(f"{col} — Original", fontweight="bold")
    axes[i, 0].set_ylabel("Interest Score")
    # Stationary
    axes[i, 1].plot(df[stat_col].index, df[stat_col].values,
                    colour="#2c3e50", linewidth=1.2)
    axes[i, 1].axhline(0, colour="red", linewidth=0.8, linestyle="--")
    d_val = stationarity_results[col]["d"]
    axes[i, 1].set_title(f"{col} — Stationary (d={d_val})", fontweight="bold")
    axes[i, 1].set_ylabel("Differenced Value")

for ax in axes[-1]:
    ax.set_xlabel("Date")
fig.suptitle("Original vs Stationary Time Series (After Differencing)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("plot_07_stationarity_comparison.png", dpi=150)
plt.show()
print("\nSaved: plot_07_stationarity_comparison.png")

# --- Summary ---
print("\n" + "=" * 55)
print("STATIONARITY SUMMARY")
print("=" * 55)
for col, info in stationarity_results.items():
    print(f"  {col}: d = {info['d']}  ({info['method'] or 'no transformation'})")
print("\nThese d values will be used as the integration order in ARIMA models.")

print("\nQ1 Complete. All plots saved ✓")
