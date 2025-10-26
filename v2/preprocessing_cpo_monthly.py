import pandas as pd
from datetime import datetime, timedelta

# Load the Excel file
file_path = './raw_data/raw_CPO_monthly_2005_2025.xlsx'  # Replace with your file path
df = pd.read_excel(file_path, sheet_name='Sheet1', engine='openpyxl')

# Function to convert Excel serial date to Python date
def excel_serial_to_date(serial):
    print(serial)
    # Excel serial starts from 1900-01-01, but offset for compatibility
    return datetime(1899, 12, 30) + timedelta(days=serial)

# Apply conversion to the 'Month' column
df['Month'] = pd.to_datetime(df['Month'], format='%b-%y')

# Rename columns for clarity and drop 'Change' if not needed
df = df[['Month', 'Price']]
df.columns = ['date', 'price']

# Sort by date (if not already)
df = df.sort_values('date').reset_index(drop=True)

# Save to CSV
output_path = './data/cleaned_cpo_monthly_2005_2025.csv'
df.to_csv(output_path, index=False)

print(f"Preprocessed data saved to {output_path}")
print(df.head())  # Preview