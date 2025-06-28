import pandas as pd
import os
import numpy as np # Import numpy for calculations

# --- 1. SETUP AND DATA LOADING (Your original code) ---

# Your directory path for the data file
DATA_DIRECTORY = r'C:\Users\james\ICAIF_25\Current_Code\Data'
# The name of your file
FILE_NAME = 'all_data_consolidated.parquet'

# Construct the full path to the file
full_path = os.path.join(DATA_DIRECTORY, FILE_NAME)

# Check if the file exists before trying to load it
if not os.path.exists(full_path):
    print(f"Error: File not found at {full_path}")
    # Exit the script if the file doesn't exist
    exit()

print(f"Attempting to load file from: {full_path}")

# Load the data
try:
    df = pd.read_parquet(full_path, engine='pyarrow')
    print("\nFile loaded successfully!")
    print("Original DataFrame shape:", df.shape)
    print("Original memory usage:")
    df.info(memory_usage='deep')
except Exception as e:
    print(f"An error occurred while loading the Parquet file: {e}")
    exit()


# --- 2. DATA CLEANING ---
print("\n--- Starting Data Cleaning ---")

# Drop the 'Unnamed: 0' column, which is an artifact from a previous save (e.g., df.to_csv())
if 'Unnamed: 0' in df.columns:
    df.drop(columns=['Unnamed: 0'], inplace=True)
    print("Dropped 'Unnamed: 0' column.")

# Based on your inspection, these columns seem to be entirely or mostly NaN.
# We will drop them to save memory and simplify the DataFrame.
cols_to_drop = ['volume_notional', 'mddv21', 'rhov', 'dhl']
df.drop(columns=cols_to_drop, inplace=True)
print(f"Dropped empty/superfluous columns: {cols_to_drop}")

# Ensure the 'date' column is in datetime format (it already is, but this is best practice)
df['date'] = pd.to_datetime(df['date'])
print("Ensured 'date' column is in datetime format.")

# Check for and drop any fully duplicate rows (same ticker, same date, same values)
initial_rows = len(df)
df.drop_duplicates(subset=['ticker', 'date'], keep='first', inplace=True)
print(f"Removed {initial_rows - len(df)} duplicate rows based on ticker and date.")

# Sort the DataFrame by ticker and then by date. This is crucial for time-series calculations.
df.sort_values(by=['ticker', 'date'], inplace=True)
print("Sorted DataFrame by 'ticker' and 'date'.")


# --- 3. DATA STRUCTURING: SETTING A MULTI-INDEX ---
print("\n--- Structuring Data with a Multi-Index ---")

# Set a MultiIndex of (ticker, date). This makes slicing and analysis much more efficient.
# For example, you can easily select all data for 'SPY' using df.loc['SPY']
df.set_index(['ticker', 'date'], inplace=True)
print("Set ('ticker', 'date') as the DataFrame index.")


# --- 4. FEATURE ENGINEERING ---
print("\n--- Performing Feature Engineering ---")

# Calculate Market Capitalization (requires 'sharesOut' column)
# This is a fundamental metric for company size.
if 'sharesOut' in df.columns and 'close' in df.columns:
    df['market_cap'] = df['close'] * df['sharesOut']
    print("Calculated 'market_cap' (close * sharesOut).")

# Calculate rolling moving averages. We use .groupby('ticker') to ensure
# the calculation restarts for each new stock and doesn't mix data between tickers.
# .transform() is used to return a Series with the same index as the original df.
print("Calculating 50-day and 200-day moving averages...")
df['ma_50'] = df.groupby(level='ticker')['close'].transform(lambda x: x.rolling(window=50, min_periods=10).mean())
df['ma_200'] = df.groupby(level='ticker')['close'].transform(lambda x: x.rolling(window=200, min_periods=50).mean())
print("Calculated 'ma_50' and 'ma_200'.")

# Calculate rolling volatility (21-day standard deviation of daily returns)
# 21 trading days is approximately one month.
print("Calculating 21-day rolling volatility...")
# Using `pct_change` is a convenient way to calculate returns.
daily_returns = df.groupby(level='ticker')['close'].pct_change()
df['volatility_21d'] = daily_returns.groupby(level='ticker').transform(lambda x: x.rolling(window=21, min_periods=10).std())
print("Calculated 'volatility_21d'.")


# --- 5. FINAL INSPECTION AND ANALYSIS EXAMPLE ---
print("\n--- Final Data Inspection and Analysis Example ---")

print("Shape of the processed data:", df.shape)
print("\nFirst 5 rows of the processed DataFrame:")
print(df.head())

print("\nLast 5 rows of the processed DataFrame:")
print(df.tail())

print("\nData types and memory usage of the processed DataFrame:")
df.info(memory_usage='deep')

# --- Analysis Example: Analyze SPY ---
print("\n--- Analysis Example: Retrieving and Describing SPY Data ---")
try:
    # Use .loc to select all data for the 'SPY' ticker from the index
    spy_data = df.loc['SPY']
    print("Successfully selected data for ticker 'SPY'.")
    print("SPY data from 2021-12-27 onwards:")
    print(spy_data.loc['2021-12-27':]) # Example of date-based slicing

    print("\nDescriptive statistics for SPY:")
    # We select a few interesting columns to describe
    print(spy_data[['close', 'volume', 'market_cap', 'ma_50', 'volatility_21d']].describe())
except KeyError:
    print("Could not find 'SPY' in the data index.")


# --- 6. SAVING THE PROCESSED DATA (Optional but Recommended) ---
print("\n--- Saving Processed Data ---")

# It's good practice to save your cleaned and feature-engineered DataFrame
# so you don't have to repeat these steps every time.
PROCESSED_FILE_NAME = 'all_data_processed.parquet'
processed_full_path = os.path.join(DATA_DIRECTORY, PROCESSED_FILE_NAME)

try:
    df.to_parquet(processed_full_path)
    print(f"Successfully saved the processed DataFrame to:\n{processed_full_path}")
except Exception as e:
    print(f"An error occurred while saving the processed file: {e}")