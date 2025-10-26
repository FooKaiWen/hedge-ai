import pandas as pd

# Read the datasets
fcpo_df = pd.read_csv('/Users/Nextale/DS/P1/hedge-ai/data/fcpo_daily.csv')
agri_df = pd.read_csv('/Users/Nextale/DS/P1/hedge-ai/data/processed_agricultural_data.csv')

# Process FCPO data
fcpo_df['datetime'] = pd.to_datetime(fcpo_df['datetime'])
fcpo_df['Year'] = fcpo_df['datetime'].dt.year
fcpo_df['Month'] = fcpo_df['datetime'].dt.month
fcpo_monthly = fcpo_df.groupby(['Year', 'Month'])['close'].mean().reset_index()
fcpo_monthly = fcpo_monthly.rename(columns={'close': 'FCPO_Price'})

# Merge datasets
final_df = pd.merge(agri_df, fcpo_monthly, on=['Year', 'Month'], how='left')

# Save the final dataset
final_df.to_csv('/Users/Nextale/DS/P1/hedge-ai/data/final_dataset.csv', index=False)

print("Successfully merged datasets and saved to data/final_dataset.csv")
