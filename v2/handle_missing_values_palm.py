import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# ==============================
# CONFIG
# ==============================
DATA_PATH = "./data/palm_oil_data.csv"
OUTPUT_PATH = "./data/palm_oil_data_cleaned.csv"

# ==============================
# LOAD DATA
# ==============================
df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.sort_values("date").set_index("date")

print("Initial missing values:\n", df.isna().sum(), "\n")

# ==============================
# DEFINE HELPERS
# ==============================
def fill_financial_time_series(df, col):
    """
    Handle missing values for each column using logic suited for its type.
    """
    series = df[col]

    # Strategy depends on column name pattern
    if "price" in col or "close" in col:
        # Price data — use time interpolation, then ffill/bfill
        filled = series.interpolate(method="time", limit_direction="both")
        filled = filled.ffill().bfill()

    elif "ret" in col or "basis" in col:
        # Return/basis — re-derive from price if possible, else fill with median
        filled = series.fillna(series.median())

    elif "volume" in col:
        # Volume — short gaps by interpolation, long gaps by rolling mean
        filled = series.interpolate(method="time", limit=3)
        rolling_mean = series.rolling(5, min_periods=1).mean()
        filled = filled.fillna(rolling_mean).ffill().bfill()

    elif "oer" in col or "production" in col or "ffb" in col:
        # Monthly or fundamental data — interpolate across months
        filled = series.interpolate(method="linear", limit_direction="both")
        filled = filled.ffill().bfill()

    else:
        # Default numeric strategy
        if np.issubdtype(series.dtype, np.number):
            filled = series.interpolate(method="linear", limit_direction="both")
            filled = filled.ffill().bfill()
        else:
            filled = series.ffill().bfill()

    return filled


def report_missing(df):
    """
    Show missing-value summary in a compact way.
    """
    total = len(df)
    return pd.DataFrame({
        "missing": df.isna().sum(),
        "missing_%": (df.isna().sum() / total * 100).round(2)
    }).sort_values("missing_%", ascending=False)


# ==============================
# COLUMN-BY-COLUMN HANDLING
# ==============================
summary_before = report_missing(df)
print("Missing summary BEFORE:\n", summary_before, "\n")

for col in df.columns:
    df[col] = fill_financial_time_series(df, col)
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    df[col] = df[col].fillna(method='ffill').fillna(method='bfill')

df = df.replace([np.inf, -np.inf], np.nan)
df = df.fillna(method='ffill').fillna(method='bfill')


summary_after = report_missing(df)
print("Missing summary AFTER:\n", summary_after, "\n")

# ==============================
# CHECK REMAINING GAPS
# ==============================
if summary_after["missing"].sum() == 0:
    print("✅ All missing values filled successfully!")
else:
    print("⚠️ Some columns still contain NaN. Consider manual review.")

# ==============================
# OPTIONAL VISUAL CHECK
# ==============================
plot_cols = [c for c in df.columns if "close" in c or "price" in c]
if plot_cols:
    df[plot_cols].plot(title="Key price columns after cleaning", figsize=(10, 4))
    plt.tight_layout()
    plt.show()

# ==============================
# SAVE CLEANED FILE
# ==============================
df.to_csv(OUTPUT_PATH)
print(f"Cleaned data saved to: {OUTPUT_PATH}")
