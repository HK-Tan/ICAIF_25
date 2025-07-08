import pandas as pd
import os
import random
from pathlib import Path

# Global variables for clustering parameters
NUM_CLUSTERS = 5
CLUSTERING_PARAMS = {
    'random_state': 42,
    'n_init': 10,
    'max_iter': 300
}

# Configuration: set to True for labels, False for clustered returns
RETURN_LABELS = True

# Data folder path
DATA_FOLDER = "Current_Code/Data/Cluster_Labels"

def get_parquet_files(data_folder):
    """Get all parquet files from the specified data folder."""
    data_path = Path(data_folder)
    if not data_path.exists():
        print(f"Data folder '{data_folder}' does not exist.")
        return []
    
    parquet_files = list(data_path.glob("*.parquet"))
    return parquet_files

def read_parquet_headers(parquet_files):
    """Read headers of parquet files and return as dataframes (10 random columns only)."""
    dataframes = {}
    
    for file_path in parquet_files:
        try:
            # First, read the full file to get column names
            df_full = pd.read_parquet(file_path, engine='pyarrow')
            all_columns = list(df_full.columns)
            
            # Select up to 10 random columns
            num_cols_to_select = min(10, len(all_columns))
            selected_columns = random.sample(all_columns, num_cols_to_select)
            
            # Create dataframe with only selected columns
            df = df_full[selected_columns]
            
            print(f"\nFile: {file_path}")
            print(f"Total columns in file: {len(all_columns)}")
            print(f"Selected {num_cols_to_select} random columns")
            print(f"Shape: {df.shape}")
            print(f"Selected columns: {selected_columns}")
            print(f"Data types:\n{df.dtypes}")
            print("-" * 50)
            
            # Store the dataframe
            dataframes[file_path.name] = df
            
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    
    return dataframes

def main():
    """Main function to process parquet files."""
    print(f"Cluster Query Tool")
    print(f"Number of clusters: {NUM_CLUSTERS}")
    print(f"Clustering parameters: {CLUSTERING_PARAMS}")
    print(f"Return labels: {RETURN_LABELS}")
    print("=" * 50)
    
    # Get parquet files
    parquet_files = get_parquet_files(DATA_FOLDER)
    
    if not parquet_files:
        print(f"No parquet files found in '{DATA_FOLDER}' folder.")
        return
    
    print(f"Found {len(parquet_files)} parquet file(s):")
    for file_path in parquet_files:
        print(f"  - {file_path}")
    
    # Read headers and create dataframes
    dataframes = read_parquet_headers(parquet_files)
    
    print(f"\nLoaded {len(dataframes)} dataframe(s) successfully.")
    
    # Return the dataframes for further processing
    return dataframes

if __name__ == "__main__":
    dfs = main()