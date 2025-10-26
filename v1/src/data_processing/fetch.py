from tvDatafeed import TvDatafeed, Interval
import pandas as pd

def fetch_fcpo_data(username, password, n_bars=5000):
    """
    Fetches daily FCPO data from TradingView.

    Args:
        username (str): TradingView username.
        password (str): TradingView password.
        n_bars (int): Number of bars to fetch.

    Returns:
        pd.DataFrame: A DataFrame containing the FCPO data, or None if an error occurs.
    """
    try:
        tv = TvDatafeed(username, password)
        fcpo_data = tv.get_hist(symbol='FCPO1!', exchange='MYX', interval=Interval.in_daily, n_bars=n_bars)
        return fcpo_data
    except Exception as e:
        print(f"An error occurred during data fetching: {e}")
        return None

def save_data(df, file_path):
    """
    Saves a DataFrame to a CSV file.

    Args:
        df (pd.DataFrame): The DataFrame to save.
        file_path (str): The path to the output CSV file.
    """
    if df is not None and not df.empty:
        df.to_csv(file_path)
        print(f"Successfully saved data to {file_path}")
    else:
        print("No data to save.")

if __name__ == '__main__':
    # This part is for standalone execution and testing
    # In a real application, you would import the functions and call them with your credentials
    # For security reasons, it's better to load credentials from a config file or environment variables
    # For example:
    # import os
    # username = os.environ.get('TV_USERNAME')
    # password = os.environ.get('TV_PASSWORD')

    # As a placeholder, I will use the credentials from the original script.
    # In a real-world scenario, you should replace this with a secure way of handling credentials.
    username = 'kwfoo'
    password = 'Vinandwen0613'

    data = fetch_fcpo_data(username, password)
    save_data(data, 'data/fcpo_daily.csv')
