import pandas as pd

# Load monthly data
monthly_df = pd.read_csv('./data/cleaned_cpo_monthly_2005_2025.csv', parse_dates=['date'])
monthly_df.set_index('date', inplace=True)

# Shift to month-ends for frequency compatibility
monthly_df.index = monthly_df.index + pd.offsets.MonthEnd(0)

# Debug: Check monthly after shift
print("Monthly after shift (head):")
print(monthly_df.head())

# Interpolate to daily with time method
daily_from_monthly = monthly_df.resample('D').interpolate(method='time')

# Debug: Check interpolated daily (sample)
print("Daily from monthly (sample 2005-2006):")
print(daily_from_monthly.loc['2005-09-01':'2006-01-01'].head(10))

# Load daily data
daily_df = pd.read_csv('./data/cleaned_cpo_daily_prices.csv', parse_dates=['date'])
daily_df.set_index('date', inplace=True)

# Combine preferring daily
combined_df = daily_from_monthly.combine_first(daily_df)

# Fill edges if needed
combined_df = combined_df.ffill().bfill()

# Save
combined_df.reset_index(inplace=True)
combined_df.to_csv('./data/full_cpo_daily_2005_2025_corrected.csv', index=False)

print("Corrected full daily CPO prices saved.")
print(combined_df.head(10))  # Should show variation from start