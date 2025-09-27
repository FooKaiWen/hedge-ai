from tvDatafeed import TvDatafeed, Interval
import pandas as pd

try:
    tv = TvDatafeed()

    # Fetch last 5000 bars of daily FCPO data from MYX exchange
    fcpo_data = tv.get_hist(symbol='FCPO1!', exchange='MYX', interval=Interval.in_daily, n_bars=5000)

    # Save data to a CSV file
    if fcpo_data is not None and not fcpo_data.empty:
        fcpo_data.to_csv('data/fcpo_daily.csv')
        print("Successfully fetched and saved FCPO data.")
    else:
        print("No data was returned from TradingView.")

except Exception as e:
    print(f"An error occurred: {e}")
