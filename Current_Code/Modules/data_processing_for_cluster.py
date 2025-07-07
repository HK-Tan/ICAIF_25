import pandas as pd
import os
import numpy as np

# --- 1. SETUP AND DATA LOADING ---

# Directory path for the data file
DATA_DIRECTORY = r'C:\Users\james\Downloads'
# Use the processed file for speed and cleanliness, if it exists
PROCESSED_FILE_NAME = 'all_data_processed.parquet'
ORIGINAL_FILE_NAME = 'all_data_consolidated.parquet'

processed_path = os.path.join(DATA_DIRECTORY, PROCESSED_FILE_NAME)
original_path = os.path.join(DATA_DIRECTORY, ORIGINAL_FILE_NAME)

df = None

# Prioritize loading the processed file created by the previous script
if os.path.exists(processed_path):
    print(f"Loading pre-processed data from: {processed_path}")
    df = pd.read_parquet(processed_path)
    # The processed file already has the correct index and is sorted
else:
    print(f"Processed file not found. Loading original data from: {original_path}")
    if os.path.exists(original_path):
        df = pd.read_parquet(original_path)
        print("Performing initial cleaning on original file...")
        # Perform minimal required cleaning for this task
        df.drop(columns=['Unnamed: 0'], inplace=True, errors='ignore')
        df['date'] = pd.to_datetime(df['date'])
        df.sort_values(by=['ticker', 'date'], inplace=True)
        df.set_index(['ticker', 'date'], inplace=True)
    else:
        print(f"Error: Neither processed nor original file found.")
        exit()

print("\nData loaded successfully.")
print("Original DataFrame shape (from loaded file):", df.shape)


# --- 2. CALCULATE LOG RETURNS ---
# The formula for log return is: ln(Price_t / Price_{t-1})
# A more direct way in pandas is to take the log of the prices and then find the difference.

print("\nCalculating daily log returns for each ticker...")

# We use groupby(level='ticker') to ensure the calculation is contained within each stock.
# This prevents calculating a return using the last price of 'ZTS' for the first price of 'ZUO'.
# .diff() calculates the difference from the previous row within each group.
df['log_return_otc'] = np.log(df['close'] / df['open'])

print("Calculation complete.")


# --- 3. RESHAPE THE DATAFRAME ---
# We want to transform the data from a "long" format to a "wide" format.
# Current Format (Long):
# Index (ticker, date) | close | log_return
# ------------------------------------------
# ('SPY', '2020-01-03') | 145.4 | 0.01
# ('SPY', '2020-01-04') | 146.1 | 0.005
# ('XLF', '2020-01-03') | 22.8  | -0.02
#
# Target Format (Wide):
# Index (date) | SPY    | XLF
# ------------------------------
# '2020-01-03' | 0.01   | -0.02
# '2020-01-04' | 0.005  | ...

print("\nPivoting data to create the 'date by ticker' DataFrame...")

# We only need the 'log_return' column for the values.
# .unstack(level='ticker') is the command that moves the 'ticker' part of the
# MultiIndex to become the new columns of the DataFrame.
log_returns_df = df['log_return_otc'].unstack(level='ticker')

# The first entry for each ticker will be NaN, as there's no previous day to compute a return from.
# This is expected and correct behavior. We can fill these if a model requires it, but for now, we keep them.
# log_returns_df.fillna(0, inplace=True) # Optional: fill NaNs with 0

print("Pivoting complete.")



# # --- 2. INSPECT NANS BEFORE PROCESSING ---

# print("\n--- Inspecting Data Before NaN Processing ---")
# print("Shape of the data:", log_returns_df.shape)
# print("Total number of NaN values:", log_returns_df.isna().sum().sum())

# # Let's look at a stock that exists from the start (SPY) and one that IPO'd later (ZTS)
# # This will clearly show the pre-IPO NaNs
# tickers_to_inspect = [t for t in ['SPY', 'ZTS'] if t in log_returns_df.columns]
# if tickers_to_inspect:
#     print("\nLog returns for 'SPY' and 'ZTS' at the beginning of the dataset:")
#     # .head() will show the initial NaN for SPY and the pre-IPO NaNs for ZTS
#     print(log_returns_df[tickers_to_inspect].head())


# # --- 3. PROCESS NAN VALUES ---

# print("\n--- Processing NaN Values ---")
# print("Filling all NaN values with 0. This handles:")
# print("1. Pre-IPO periods (no stock exists, so 0 return).")
# print("2. Post-delisting periods (stock no longer trades, so 0 return).")
# print("3. Mid-series non-trading days (price is held constant, so 0 return).")

# # The .fillna() method is perfect for this. We use inplace=True to modify the DataFrame directly.
# log_returns_df.fillna(0, inplace=True)

# print("Processing complete.")


# --- 4. VERIFY THE CHANGES ---

print("\n--- Verifying Data After NaN Processing ---")
print("Total number of NaN values after processing:", log_returns_df.isna().sum().sum())

# if tickers_to_inspect:
#     print("\nLog returns for 'SPY' and 'ZTS' after processing:")
#     # Now, the same .head() call will show 0s instead of NaNs
#     print(log_returns_df[tickers_to_inspect].head())

print("\nFinal DataFrame Head:")
print(log_returns_df.head())


# --- 4. FINAL INSPECTION AND SAVING ---

print("\n--- Final Log Returns DataFrame ---")
print("Shape of the final log returns data (dates, tickers):", log_returns_df.shape)

print("\nFirst 5 rows:")
print(log_returns_df.head())

print("\nLast 5 rows:")
print(log_returns_df.tail())

# Example: Check the returns for a few specific tickers towards the end
print("\nExample Data for tickers 'AAPL', 'MSFT', 'TSLA', 'SPY':")
# Using .get() with a default empty list handles cases where a ticker might not exist
tickers_to_show = [t for t in ['AAPL', 'MSFT', 'TSLA', 'SPY'] if t in log_returns_df.columns]
if tickers_to_show:
    print(log_returns_df[tickers_to_show].tail())
else:
    print("None of the example tickers were found in the data.")


# Save this new, useful DataFrame for future analysis
LOG_RETURNS_FILE_NAME = 'log_returns_w_NAs.parquet'
log_returns_full_path = os.path.join(DATA_DIRECTORY, LOG_RETURNS_FILE_NAME)

try:
    log_returns_df.to_parquet(log_returns_full_path)
    print(f"\nSuccessfully saved the log returns DataFrame to:\n{log_returns_full_path}")
except Exception as e:
    print(f"An error occurred while saving the log returns file: {e}")