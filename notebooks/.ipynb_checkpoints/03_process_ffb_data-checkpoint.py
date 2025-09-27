import pandas as pd
import glob

# Get all daily FFB excel files
ffb_files = glob.glob('../raw_data/daily_ffb_*.xlsx')
all_ffb_data = []
for file in ffb_files:
    df = pd.read_excel(file, header=1)

    # Convert numeric columns safely
    df = df.apply(pd.to_numeric, errors="coerce")
    
    # Extract year & month from filename (e.g., ..._MMYYYY.xlsx)
    try:
        fname = file.split('_')[-1].split('.')[0]
        year, month = int(fname[4:]), int(fname[2:4])
    except (ValueError, IndexError):
        raise ValueError(f"Could not extract year and month from filename: {file}")
    
    # Build tidy DataFrame
    df = (
        df.rename(columns={df.columns[0]: "Day"})
          .assign(
              Year=year,
              Month=month,
              FFB_Price=lambda d: d[
                  ["North / Utara", "South / Selatan", "Central / Tengah", 
                   "East Coast / Timur", "Sabah", "Sarawak*"]
              ].mean(axis=1),
              Datetime=lambda d: pd.to_datetime(d[["Year", "Month", "Day"]])
          )
          .set_index("Datetime")[["FFB_Price"]]
          .sort_index()
    )
    all_ffb_data.append(df)

# Concatenate all dataframes
if all_ffb_data:
    ffb_df = pd.concat(all_ffb_data)
    ffb_df = ffb_df.sort_index()
    ffb_df = ffb_df.dropna(subset=['FFB_Price'])
    ffb_df_monthly = ffb_df.resample("ME").mean()
    ffb_df_monthly.index = ffb_df_monthly.index.to_period("M")
    ffb_df.to_csv('../data/ffb_daily.csv')
    ffb_df_monthly.to_csv('../data/ffb_monthly.csv')