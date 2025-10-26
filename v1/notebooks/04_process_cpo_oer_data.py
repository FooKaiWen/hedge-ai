import pandas as pd
import glob
import re

# --- Process CPO Production data ---
cpo_files = glob.glob('/Users/Nextale/DS/P1/hedge-ai/raw_data/PRODUCTION OF CRUDE PALM OIL*.xlsx')

all_cpo_data = []
for file in cpo_files:
    df = pd.read_excel(file, header=3) # Skip first 3 rows
    df = df.rename(columns={df.columns[0]: 'State'})
    df = df.dropna(subset=['State'])
    df = df[~df['State'].isin(['TOTAL', 'P.MALAYSIA', 'SABAH', 'SARAWAK'])]

    # Get year from filename
    year_match = re.search(r'(\d{4})', file)
    if year_match:
        year = int(year_match.group(1))
    else:
        print(f"Could not extract year from filename: {file}")
        continue
    df['Year'] = year

    # Select only month columns
    month_cols = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
    cols_to_melt = ['State', 'Year'] + [col for col in df.columns if col in month_cols]
    df = df[cols_to_melt]

    # Melt dataframe
    df = df.melt(id_vars=['State', 'Year'], var_name='Month', value_name='CPO_Production')
    
    # Convert month name to number
    df['Month'] = pd.to_datetime(df['Month'], format='%b').dt.month

    all_cpo_data.append(df)

if all_cpo_data:
    cpo_df = pd.concat(all_cpo_data, ignore_index=True)
    cpo_df.to_csv('/Users/Nextale/DS/P1/hedge-ai/data/cpo_monthly.csv', index=False)
    print("Successfully processed CPO data and saved to data/cpo_monthly.csv")
else:
    print("No CPO data was processed.")

# --- Process OER data ---
oer_files = glob.glob('/Users/Nextale/DS/P1/hedge-ai/raw_data/OIL EXTRACTION RATE FOR CRUDE PALM OIL*.xlsx')

all_oer_data = []
for file in oer_files:
    df = pd.read_excel(file, header=3) # Skip first 3 rows
    df = df.rename(columns={df.columns[0]: 'State'})
    df = df.dropna(subset=['State'])
    df = df[~df['State'].isin(['TOTAL', 'P.MALAYSIA', 'SABAH', 'SARAWAK'])]

    # Get year from filename
    year_match = re.search(r'(\d{4})', file)
    if year_match:
        year = int(year_match.group(1))
    else:
        print(f"Could not extract year from filename: {file}")
        continue
    df['Year'] = year

    # Select only month columns
    month_cols = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
    cols_to_melt = ['State', 'Year'] + [col for col in df.columns if col in month_cols]
    df = df[cols_to_melt]

    # Melt dataframe
    df = df.melt(id_vars=['State', 'Year'], var_name='Month', value_name='OER')
    
    # Convert month name to number
    df['Month'] = pd.to_datetime(df['Month'], format='%b').dt.month

    all_oer_data.append(df)

if all_oer_data:
    oer_df = pd.concat(all_oer_data, ignore_index=True)
    oer_df.to_csv('/Users/Nextale/DS/P1/hedge-ai/data/oer_monthly.csv', index=False)
    print("Successfully processed OER data and saved to data/oer_monthly.csv")
else:
    print("No OER data was processed.")

# --- Merge FFB, CPO and OER data ---
ffb_file_path = '/Users/Nextale/DS/P1/hedge-ai/data/ffb_monthly.csv'
cpo_file_path = '/Users/Nextale/DS/P1/hedge-ai/data/cpo_monthly.csv'
oer_file_path = '/Users/Nextale/DS/P1/hedge-ai/data/oer_monthly.csv'

try:
    ffb_df = pd.read_csv(ffb_file_path)
    cpo_df = pd.read_csv(cpo_file_path)
    oer_df = pd.read_csv(oer_file_path)

    # Clean up region/state names for merging
    cpo_df['State'] = cpo_df['State'].str.strip()
    oer_df['State'] = oer_df['State'].str.strip()
    ffb_df = ffb_df.rename(columns={'Region': 'State'}) # Rename Region to State
    ffb_df['State'] = ffb_df['State'].str.strip()


    # Merge CPO and OER
    cpo_oer_df = pd.merge(cpo_df, oer_df, on=['Year', 'Month', 'State'], how='left')

    # Merge with FFB
    final_df = pd.merge(cpo_oer_df, ffb_df, on=['Year', 'Month', 'State'], how='left')

    final_df.to_csv('/Users/Nextale/DS/P1/hedge-ai/data/processed_agricultural_data.csv', index=False)
    print("Successfully merged all data and saved to data/processed_agricultural_data.csv")

except FileNotFoundError as e:
    print(f"Error: {e}. Please make sure all required CSV files are present.")
