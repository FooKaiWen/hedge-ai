import pandas as pd
import glob

# Get all daily FFB excel files
ffb_files = glob.glob('/Users/Nextale/DS/P1/hedge-ai/raw_data/daily_ffb_*.xlsx')

# Process each file and store in a list
all_ffb_data = []
for file in ffb_files:
    # Read the excel file, using the second row as header
    df = pd.read_excel(file, header=1)
    
    # Rename the date column
    df = df.rename(columns={df.columns[0]: 'Date'})

    # Get the year and month from the file name
    try:
        year = int(file.split('_')[-1].split('.')[0][4:])
        month = int(file.split('_')[-1].split('.')[0][2:4])
    except (ValueError, IndexError):
        print(f"Could not extract year and month from filename: {file}")
        continue

    # Add Year and Month columns
    df['Year'] = year
    df['Month'] = month

    # Melt the dataframe to long format
    df = df.melt(id_vars=['Year', 'Month', 'Date'], var_name='Region', value_name='FFB_Price')

    all_ffb_data.append(df)

# Concatenate all dataframes
if all_ffb_data:
    ffb_df = pd.concat(all_ffb_data, ignore_index=True)

    # Clean up Region names
    ffb_df['Region'] = ffb_df['Region'].str.replace('*', '', regex=False).str.strip()
    ffb_df = ffb_df.replace({'P.MASIA': 'Peninsular Malaysia', 'P.MALAYSIA': 'Peninsular Malaysia'})

    # Convert FFB_Price to numeric, coercing errors to NaN
    ffb_df['FFB_Price'] = pd.to_numeric(ffb_df['FFB_Price'], errors='coerce')

    # Drop rows with NaN in FFB_Price
    ffb_df = ffb_df.dropna(subset=['FFB_Price'])

    # Calculate monthly average FFB price
    ffb_monthly = ffb_df.groupby(['Year', 'Month', 'Region'])['FFB_Price'].mean().reset_index()

    # Save to csv
    ffb_monthly.to_csv('/Users/Nextale/DS/P1/hedge-ai/data/ffb_monthly.csv', index=False)

    print("Successfully processed FFB data and saved to data/ffb_monthly.csv")
else:
    print("No FFB data was processed.")