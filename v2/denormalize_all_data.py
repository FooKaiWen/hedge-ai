import pandas as pd
from datetime import datetime

# Load each file
ffb_daily = pd.read_csv('./data/ffb_daily.csv')
oer_monthly = pd.read_csv('./data/oer_monthly.csv')
ffb_monthly = pd.read_csv('./data/ffb_monthly.csv')
nextmonth_data = pd.read_csv('./data/nextmonth_data.csv')
fcpo_daily = pd.read_csv('./data/fcpo_daily.csv')
nextday_data = pd.read_csv('./data/nextday_data.csv')
full_cpo_daily = pd.read_csv('./data/full_cpo_daily_2005_2025_corrected.csv')
cpo_production_monthly = pd.read_csv('./data/cpo_production_monthly.csv')

# Rename columns for consistency
full_cpo_daily.rename(columns={'date': 'date', 'price': 'spot_close'}, inplace=True)
fcpo_daily.rename(columns={'datetime': 'date', 'close': 'fut_close'}, inplace=True)
nextday_data.rename(columns={'datetime': 'date', 'close_tmr': 'next_day_pred'}, inplace=True)
nextmonth_data.rename(columns={'datetime': 'date', 'close_one_month_after': 'next_month_pred'}, inplace=True)
oer_monthly.rename(columns={'Period': 'period', 'Value': 'oer'}, inplace=True)
cpo_production_monthly.rename(columns={'Period': 'period', 'Value': 'production'}, inplace=True)
ffb_daily.rename(columns={'Datetime': 'date', 'FFB_Price': 'ffb_price'}, inplace=True)
ffb_monthly.rename(columns={'Datetime': 'period', 'FFB_Price': 'ffb_monthly_price'}, inplace=True)

# Parse dates safely
for df in [full_cpo_daily, fcpo_daily, nextday_data, nextmonth_data, ffb_daily]:
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

oer_monthly['period'] = pd.to_datetime(oer_monthly['period'], format='%Y-%m', errors='coerce')
cpo_production_monthly['period'] = pd.to_datetime(cpo_production_monthly['period'], format='%Y-%m', errors='coerce')
ffb_monthly['period'] = pd.to_datetime(ffb_monthly['period'], format='%Y-%m', errors='coerce')

# --- Function to filter closing prices for intraday data (not applied to full_cpo_daily) ---
def filter_closing(df):
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

    if df['date'].dt.time.isnull().all():
        # If all dates have no time component, skip filtering
        return df

    df['time'] = df['date'].dt.time
    filtered = df[df['time'] == pd.to_datetime('21:00').time()].copy()
    drop_cols = ['time', 'symbol', 'open', 'high', 'low']
    existing = [col for col in drop_cols if col in filtered.columns]
    return filtered.drop(columns=existing)

# Apply filtering only to intraday datasets
fcpo_daily = filter_closing(fcpo_daily)[['date', 'fut_close', 'volume']]
nextday_data = filter_closing(nextday_data)[['date', 'next_day_pred']]
nextmonth_data = filter_closing(nextmonth_data)[['date', 'next_month_pred']]

# Create master date range
min_date = min(full_cpo_daily['date'].min(), fcpo_daily['date'].min())
max_date = max(full_cpo_daily['date'].max(), fcpo_daily['date'].max())
date_range = pd.date_range(start=min_date, end=max_date, freq='D')
master_df = pd.DataFrame({'date': date_range})

# # --- Normalize all to date-only before merging ---
# master_df['date'] = pd.to_datetime(master_df['date']).dt.date
# full_cpo_daily['date'] = pd.to_datetime(full_cpo_daily['date']).dt.date
# fcpo_daily['date'] = pd.to_datetime(fcpo_daily['date']).dt.date
# nextday_data['date'] = pd.to_datetime(nextday_data['date']).dt.date
# nextmonth_data['date'] = pd.to_datetime(nextmonth_data['date']).dt.date
# ffb_daily['date'] = pd.to_datetime(ffb_daily['date']).dt.date

# Normalize both to daily granularity before merge
full_cpo_daily['merge_date'] = pd.to_datetime(full_cpo_daily['date']).dt.floor('D')
master_df['merge_date'] = pd.to_datetime(master_df['date']).dt.floor('D')

master_df = master_df.merge(full_cpo_daily[['merge_date', 'spot_close']], on='merge_date', how='left')
master_df.drop(columns=['merge_date'], inplace=True)

print(master_df[['date', 'spot_close']].head(10))
print(master_df['spot_close'].isna().mean(), "NaN ratio after merge")

# Merge core data sources
# master_df = master_df.merge(full_cpo_daily, on='date', how='left')
master_df = master_df.merge(fcpo_daily, on='date', how='left')
master_df = master_df.merge(nextday_data, on='date', how='left')
master_df = master_df.merge(nextmonth_data, on='date', how='left')

# --- FFB Prices (daily + monthly fallback) ---
master_df = master_df.merge(ffb_daily, on='date', how='left')
master_df['month_year'] = master_df['date'].dt.to_period('M')
ffb_monthly['period'] = ffb_monthly['period'].dt.to_period('M')
master_df = master_df.merge(ffb_monthly[['period', 'ffb_monthly_price']], left_on='month_year', right_on='period', how='left')

# Fill FFB price gaps
master_df['ffb_price'] = master_df['ffb_price'].fillna(master_df['ffb_monthly_price'])
master_df['ffb_price'] = master_df['ffb_price'].fillna(master_df['ffb_price'].mean())
master_df.drop(columns=['month_year', 'period', 'ffb_monthly_price'], inplace=True)

# --- Add OER (interpolated monthly to daily) ---
oer_monthly['month_year'] = oer_monthly['period'].dt.to_period('M')
master_df['month_year'] = master_df['date'].dt.to_period('M')
master_df = master_df.merge(oer_monthly[['month_year', 'oer']], on='month_year', how='left')
master_df['oer'] = master_df['oer'].interpolate(method='linear').ffill().bfill()
master_df.drop(columns=['month_year'], inplace=True)

# --- Add Production (interpolated monthly to daily) ---
cpo_production_monthly['month_year'] = cpo_production_monthly['period'].dt.to_period('M')
master_df['month_year'] = master_df['date'].dt.to_period('M')
master_df = master_df.merge(cpo_production_monthly[['month_year', 'production']], on='month_year', how='left')
master_df['production'] = master_df['production'].interpolate(method='linear').ffill().bfill()
master_df.drop(columns=['month_year'], inplace=True)

# --- Handle Missing Values Smartly ---
numeric_cols = master_df.select_dtypes(include='number').columns
for col in numeric_cols:
    if master_df[col].isna().sum() > 0:
        master_df[col] = master_df[col].interpolate(method='linear').ffill().bfill()

# --- Save Final Dataset ---
master_df.to_csv('./data/palm_oil_data.csv', index=False)
print("✅ Merged data saved to palm_oil_data.csv (daily spot price preserved without time filtering).")
