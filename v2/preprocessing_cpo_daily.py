import pandas as pd

# Load the raw CSV
input_file = './raw_data/cpo_daily_prices.csv'  # Replace with your actual file path
df = pd.read_csv(input_file, parse_dates=['date'])

# Replace 'PH' and 'NT' with NaN and convert price to float
df['price'] = pd.to_numeric(df['price'], errors='coerce')

# Sort by date to ensure chronological order
df = df.sort_values('date').reset_index(drop=True)

# Forward-fill missing prices (carries over last valid price for non-trading days)
df['price'] = df['price'].ffill()

# Back-fill any leading NaNs (if data starts with missing)
df['price'] = df['price'].bfill()

# Drop rows where price is still NaN (if any remain, though unlikely after ffill/bfill)
df = df.dropna(subset=['price'])

# Save the cleaned CSV
output_file = './data/cleaned_cpo_daily_prices.csv'
df.to_csv(output_file, index=False)

print(f"Cleaned data saved to {output_file}")
print(df.head(10))  # Preview the first 10 rows