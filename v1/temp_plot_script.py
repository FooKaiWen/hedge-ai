import pandas as pd
import matplotlib.pyplot as plt
import os

# Define file paths
fcpo_path = r'E:\FKW\MDS\P1\Outcome\hedge-ai\data\fcpo_daily.csv'
nextday_path = r'E:\FKW\MDS\P1\Outcome\hedge-ai\data\nextday_data.csv'
output_dir = r'E:\FKW\MDS\P1\Outcome\hedge-ai\outputs'
output_path = os.path.join(output_dir, 'fcpo_vs_nextday_chart.png')

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

try:
    # Read and prepare the first dataset
    fcpo_df = pd.read_csv(fcpo_path, parse_dates=['datetime'])
    fcpo_df.set_index('datetime', inplace=True)

    # Read and prepare the second dataset, explicitly handling the date format
    nextday_df = pd.read_csv(nextday_path)
    nextday_df['datetime'] = pd.to_datetime(nextday_df['datetime'], errors='coerce')
    nextday_df.dropna(subset=['datetime'], inplace=True)
    nextday_df.set_index('datetime', inplace=True)

    # Align the dataframes on their index to ensure they match up
    df1, df2 = fcpo_df.align(nextday_df, join='inner', axis=0)

    # Create the plot
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(15, 7))

    # Plot aligned data
    plt.plot(df1.index, df1['close'], label='FCPO Daily Close', color='blue', linewidth=1.5)
    plt.plot(df2.index, df2['close_tmr'], label='Next Day Close Forecast', color='orange', linestyle='--', linewidth=1.5)

    # Add titles and labels
    plt.title('FCPO Daily Close vs. Next Day Forecast', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price', fontsize=12)
    plt.legend(fontsize=10)
    plt.tight_layout()

    # Save the figure
    plt.savefig(output_path)

    print(f"Chart successfully generated and saved to: {output_path}")

except FileNotFoundError as e:
    print(f"Error: The file was not found - {e.filename}")
except Exception as e:
    print(f"An error occurred: {e}")