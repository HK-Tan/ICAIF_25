import pandas as pd
import os
import glob
from datetime import datetime

# --- All of your functions (load_yearly_data, load_all_data) remain unchanged ---
def load_yearly_data(year: int, base_path: str) -> pd.DataFrame:
    # ... (code from your original script) ...
    print(f"--- Starting to load data for the year: {year} ---")
    year_str = str(year)
    year_path = os.path.join(base_path, year_str)
    if not os.path.isdir(year_path):
        print(f"Error: Directory not found at '{year_path}'")
        return pd.DataFrame()
    file_pattern = os.path.join(year_path, f"{year_str}*.gz")
    gz_files = glob.glob(file_pattern)
    csv_pattern = os.path.join(year_path, f"{year_str}*.csv")
    csv_files = glob.glob(csv_pattern)
    all_files = sorted(list(set(gz_files + csv_files)))
    if not all_files:
        print(f"No data files found in '{year_path}'")
        return pd.DataFrame()
    daily_dataframes = []
    total_files = len(all_files)
    print(f"Found {total_files} daily data files to process for {year}.")
    for i, file_path in enumerate(all_files):
        filename = os.path.basename(file_path)
        date_str = filename.split('.')[0]
        if daily_dataframes and daily_dataframes[-1]['date'].iloc[0] == pd.to_datetime(date_str, format='%Y%m%d'):
            continue
        try:
            daily_df = pd.read_csv(file_path, compression='infer')
            daily_df['date'] = pd.to_datetime(date_str, format='%Y%m%d')
            daily_dataframes.append(daily_df)
        except Exception as e:
            print(f"    Could not read or process file {filename}. Error: {e}")
    if not daily_dataframes:
        print(f"No data was successfully loaded for {year}.")
        return pd.DataFrame()
    print(f"Combining daily files for {year} into a yearly DataFrame...")
    yearly_df = pd.concat(daily_dataframes, ignore_index=True)
    print(f"--- Successfully loaded {len(yearly_df):,} rows for {year}. ---")
    return yearly_df

def load_all_data(base_path: str) -> pd.DataFrame:
    # ... (code from your original script) ...
    print(f"=== Starting to load all data from base directory: '{base_path}' ===")
    if not os.path.isdir(base_path):
        print(f"Error: Base directory not found at '{base_path}'")
        return pd.DataFrame()
    try:
        year_dirs = sorted([
            d.name for d in os.scandir(base_path)
            if d.is_dir() and d.name.isdigit() and len(d.name) == 4
        ])
    except FileNotFoundError:
        print(f"Error: Cannot access directory '{base_path}'.")
        return pd.DataFrame()
    if not year_dirs:
        print(f"No year directories (e.g., '2020', '2021') found in '{base_path}'.")
        return pd.DataFrame()
    print(f"Found year directories: {', '.join(year_dirs)}")
    all_dataframes = []
    for year_str in year_dirs:
        year = int(year_str)
        yearly_df = load_yearly_data(year=year, base_path=base_path)
        if not yearly_df.empty:
            all_dataframes.append(yearly_df)
        else:
            print(f"Warning: No data was loaded for the year {year}. Skipping.")
    if not all_dataframes:
        print("No data could be loaded from any of the year directories.")
        return pd.DataFrame()
    print("\n>>> Combining all yearly DataFrames into a single master DataFrame...")
    master_df = pd.concat(all_dataframes, ignore_index=True)
    print(f"=== Successfully loaded a total of {len(master_df):,} rows from {len(year_dirs)} years. ===")
    return master_df


if __name__ == '__main__':
    DATA_DIRECTORY = r'C:\Users\james\Downloads\Yearly'
    PARQUET_OUTPUT_PATH = os.path.join(DATA_DIRECTORY, 'all_data_consolidated.parquet')

    all_data = load_all_data(base_path=DATA_DIRECTORY)

    if not all_data.empty:
        print("\n--- Data Inspection (Before Cleaning) ---")
        all_data.info()

        # =================================================================
        # 4. CLEAN THE DATA before saving
        # =================================================================
        print("\n--- Cleaning data for Parquet compatibility ---")

        # Check if the problematic column exists before trying to clean it
        if 'SICCD' in all_data.columns:
            print("Cleaning 'SICCD' column...")
            # Step 1: Convert to numeric, forcing errors to become 'NaN'
            all_data['SICCD'] = pd.to_numeric(all_data['SICCD'], errors='coerce')

            # Step 2: Convert the column to a nullable integer type.
            # This preserves the numbers as integers while properly handling missing values.
            all_data['SICCD'] = all_data['SICCD'].astype('Int64')
            print("'SICCD' column converted to Int64.")
        else:
            print("Column 'SICCD' not found, skipping cleaning step.")

        print("\n--- Data Inspection (After Cleaning) ---")
        # Notice the change in dtype for SICCD from 'object' to 'Int64'
        all_data.info()

        # =================================================================
        # 5. Save the consolidated DataFrame to a Parquet file
        # =================================================================
        print(f"\n--- Saving data to Parquet file ---")
        try:
            print(f"Saving {len(all_data):,} rows to '{PARQUET_OUTPUT_PATH}'...")
            all_data.to_parquet(PARQUET_OUTPUT_PATH, engine='pyarrow', index=False)
            print(f"Successfully saved the data.")
        except ImportError:
            print("\nError: Could not save to Parquet.")
            print("To enable this, please install the 'pyarrow' library by running: pip install pyarrow")
        except Exception as e:
            print(f"\nAn error occurred while saving the Parquet file: {e}")

    else:
        print("\nNo data was loaded, so no file will be saved.")