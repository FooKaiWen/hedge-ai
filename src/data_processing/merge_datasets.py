import pandas as pd

def merge_data(data_path, output_path):
    """
    Merges the processed agricultural data with the FCPO data.

    Args:
        data_path (str): The path to the directory containing the processed data files.
        output_path (str): The path to the directory where the final dataset will be saved.
    """
    try:
        ffb_df = pd.read_csv(f'{data_path}/ffb_monthly.csv')
        cpo_df = pd.read_csv(f'{data_path}/cpo_monthly.csv')
        oer_df = pd.read_csv(f'{data_path}/oer_monthly.csv')
        fcpo_df = pd.read_csv(f'{data_path}/fcpo_daily.csv')

        # Process FCPO data
        fcpo_df['datetime'] = pd.to_datetime(fcpo_df['datetime'])
        fcpo_df['Year'] = fcpo_df['datetime'].dt.year
        fcpo_df['Month'] = fcpo_df['datetime'].dt.month
        fcpo_monthly = fcpo_df.groupby(['Year', 'Month'])['close'].mean().reset_index()
        fcpo_monthly = fcpo_monthly.rename(columns={'close': 'FCPO_Price'})

        # Merge agricultural data
        cpo_oer_df = pd.merge(cpo_df, oer_df, on=['Year', 'Month', 'State'], how='left')
        agri_df = pd.merge(cpo_oer_df, ffb_df, on=['Year', 'Month', 'State'], how='left')

        # Merge with FCPO data
        final_df = pd.merge(agri_df, fcpo_monthly, on=['Year', 'Month'], how='left')

        # Save the final dataset
        final_df.to_csv(f'{output_path}/final_dataset.csv', index=False)
        print(f"Successfully merged datasets and saved to {output_path}/final_dataset.csv")

    except FileNotFoundError as e:
        print(f"Error: {e}. Please make sure all required CSV files are present in {data_path}.")

if __name__ == '__main__':
    # This part is for standalone execution and testing
    merge_data('../data', '../data')
