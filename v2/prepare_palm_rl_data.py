import pandas as pd
import numpy as np
import joblib, json, os
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# ==============================
# CONFIG
# ==============================
DATA_PATH = "./data/palm_oil_data_cleaned.csv"
OUTPUT_DIR = "./data/rl_ready"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================
# LOAD
# ==============================
df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.sort_values("date").set_index("date")

# Keep numeric columns only
df = df.select_dtypes(include=[np.number])

# ==============================
# HANDLE MISSING VALUES
# ==============================
df = df.interpolate(method="time").ffill().bfill()

# Replace remaining missing with column means
df = df.fillna(df.mean())

# ==============================
# FEATURE ENGINEERING
# ==============================
def add_features(df):
    out = df.copy()
    # Returns
    out["spot_ret"] = out["spot_close"].pct_change()
    out["fut_ret"] = out["fut_close"].pct_change()

    # Log returns
    out["log_ret_spot"] = np.log(out["spot_close"]).diff()
    out["log_ret_fut"] = np.log(out["fut_close"]).diff()

    # Basis (difference)
    out["basis"] = out["fut_close"] - out["spot_close"]
    out["basis_pct"] = out["basis"] / out["spot_close"]

    # Rolling stats
    for w in [5, 20, 60]:
        out[f"spot_ma_{w}"] = out["spot_close"].rolling(w).mean()
        out[f"fut_ma_{w}"] = out["fut_close"].rolling(w).mean()
        out[f"spot_std_{w}"] = out["spot_close"].rolling(w).std()
        out[f"fut_std_{w}"] = out["fut_close"].rolling(w).std()
        out[f"basis_mean_{w}"] = out["basis"].rolling(w).mean()
        out[f"basis_std_{w}"] = out["basis"].rolling(w).std()

    # Rolling correlation & hedge ratio
    for w in [5, 20, 60]:
        cov = out["spot_close"].rolling(w).cov(out["fut_close"])
        var_fut = out["fut_close"].rolling(w).var()
        out[f"hr_ols_{w}"] = cov / var_fut

    # Volume normalization
    if "volume" in out.columns:
        out["volume_z"] = (out["volume"] - out["volume"].rolling(20).mean()) / (out["volume"].rolling(20).std() + 1e-9)

    # Calendar features
    out["dayofweek"] = out.index.dayofweek
    out["month"] = out.index.month
    out["sin_doy"] = np.sin(2 * np.pi * out.index.dayofyear / 365.25)
    out["cos_doy"] = np.cos(2 * np.pi * out.index.dayofyear / 365.25)
    out["is_month_end"] = out.index.is_month_end.astype(int)

    return out

df = add_features(df)

# ==============================
# TARGET: Hedge Ratio (OLS baseline)
# ==============================
df["hedge_ratio_target"] = df["hr_ols_20"].ffill().bfill()

# ==============================
# CLEAN OUTLIERS (Winsorize)
# ==============================
for col in ["spot_ret", "fut_ret", "basis", "basis_pct"]:
    if col in df.columns:
        low, high = df[col].quantile([0.01, 0.99])
        df[col] = df[col].clip(low, high)

# ==============================
# DROP NA
# ==============================
df = df.dropna().copy()

# ==============================
# TRAIN / VAL / TEST SPLIT
# ==============================
train_size, val_size = 0.7, 0.15
n = len(df)
train_end = int(train_size * n)
val_end = int((train_size + val_size) * n)

train_df = df.iloc[:train_end]
val_df = df.iloc[train_end:val_end]
test_df = df.iloc[val_end:]

# ==============================
# NORMALIZE
# ==============================
scaler = StandardScaler()
obs_features = [c for c in df.columns if c not in ["hedge_ratio_target"]]

X = train_df[obs_features]

print("Any NaN:", X.isna().any().any())
print("Any inf:", np.isinf(X.to_numpy()).any())
print("Describe:")
print(X.describe())

train_df[obs_features] = train_df[obs_features].replace([np.inf, -np.inf], np.nan)
train_df[obs_features] = train_df[obs_features].fillna(method='ffill').fillna(method='bfill')

scaler.fit(train_df[obs_features])

X_train = scaler.transform(train_df[obs_features])
X_val = scaler.transform(val_df[obs_features])
X_test = scaler.transform(test_df[obs_features])

y_train = train_df["hedge_ratio_target"].values
y_val = val_df["hedge_ratio_target"].values
y_test = test_df["hedge_ratio_target"].values

# ==============================
# REWARD DESIGN
# ==============================
def compute_pnl(spot, fut, hedge_ratio):
    """PnL from 1 long spot, short hedge_ratio * futures."""
    return (spot.pct_change() - hedge_ratio * fut.pct_change()).fillna(0)

def reward_function(pnl, lam_vol=10.0, transaction_cost=1e-4, deltas=None):
    reward = pnl.mean() - lam_vol * pnl.std()
    if deltas is not None:
        reward -= transaction_cost * np.abs(deltas).mean()
    return reward

baseline_pnl = compute_pnl(df["spot_close"], df["fut_close"], df["hedge_ratio_target"])
baseline_reward = reward_function(baseline_pnl)

# ==============================
# SAVE OUTPUTS
# ==============================
train_df.to_csv(f"{OUTPUT_DIR}/train.csv")
val_df.to_csv(f"{OUTPUT_DIR}/val.csv")
test_df.to_csv(f"{OUTPUT_DIR}/test.csv")

np.save(f"{OUTPUT_DIR}/X_train.npy", X_train)
np.save(f"{OUTPUT_DIR}/X_val.npy", X_val)
np.save(f"{OUTPUT_DIR}/X_test.npy", X_test)
np.save(f"{OUTPUT_DIR}/y_train.npy", y_train)
np.save(f"{OUTPUT_DIR}/y_val.npy", y_val)
np.save(f"{OUTPUT_DIR}/y_test.npy", y_test)

joblib.dump({"scaler": scaler, "obs_features": obs_features}, f"{OUTPUT_DIR}/scalers_joblib.pkl")

meta = {
    "n_samples": len(df),
    "features": obs_features,
    "reward_baseline": baseline_reward,
    "train_ratio": train_size,
    "val_ratio": val_size,
    "test_ratio": 1 - train_size - val_size,
}
json.dump(meta, open(f"{OUTPUT_DIR}/rl_meta.json", "w"), indent=2)

print("✅ RL-ready data created!")
print(f"Samples: {len(df)}, Train/Val/Test split: {len(train_df)}, {len(val_df)}, {len(test_df)}")
print(f"Baseline reward (OLS hedge): {baseline_reward:.6f}")
print(f"Files saved in: {OUTPUT_DIR}")
