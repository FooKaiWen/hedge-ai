import pandas as pd
import glob
import re

def process_cpo_data(raw_data_path, output_path):
    """
    Processes CPO production excel files into a monthly CSV file.

    Args:
        raw_data_path (str): The path to the directory containing the raw CPO excel files.
        output_path (str): The path to the directory where the processed CSV file will be saved.
    """
    cpo_files = glob.glob(f'{raw_data_path}/PRODUCTION OF CRUDE PALM OIL*.xlsx')

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
        cpo_df.to_csv(f'{output_path}/cpo_monthly.csv', index=False)
        print(f"Successfully processed CPO data and saved to {output_path}/cpo_monthly.csv")
    else:
        print("No CPO data was processed.")

def process_oer_data(raw_data_path, output_path):
    """
    Processes OER excel files into a monthly CSV file.

    Args:
        raw_data_path (str): The path to the directory containing the raw OER excel files.
        output_path (str): The path to the directory where the processed CSV file will be saved.
    """
    oer_files = glob.glob(f'{raw_data_path}/OIL EXTRACTION RATE FOR CRUDE PALM OIL*.xlsx')

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
        oer_df.to_csv(f'{output_path}/oer_monthly.csv', index=False)
        print(f"Successfully processed OER data and saved to {output_path}/oer_monthly.csv")
    else:
        print("No OER data was processed.")

if __name__ == '__main__':
    # This part is for standalone execution and testing
    process_cpo_data('../raw_data', '../data')
    process_oer_data('../raw_data', '../data')
