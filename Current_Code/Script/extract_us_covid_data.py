import pandas as pd
import numpy as np

def extract_us_covid_data(input_file, output_file, chunk_size=10000):
    """
    Extract US COVID data from the large CSV file and reformat it to match 
    the structure of the financial CSV files (long format with observation_date).
    
    Args:
        input_file (str): Path to the input COVID CSV file
        output_file (str): Path to the output CSV file
        chunk_size (int): Size of chunks to process at a time
    """
    # Selected features to extract (modify as needed)
    selected_features = [
        'ConfirmedCases',
        'ConfirmedDeaths',
        'PopulationVaccinated',
        'StringencyIndex_Average',
        'GovernmentResponseIndex_Average',
        'ContainmentHealthIndex_Average',
        'EconomicSupportIndex',
        "C1M_School closing",
        "C2M_Workplace closing",
        "C3M_Cancel public events",
        "C4M_Restrictions on gatherings",
        "C6M_Stay at home requirements",
        "C7M_Restrictions on internal movement",
        "C8EV_International travel controls",
        "E1_Income support",
        "E2_Debt/contract relief",
        "E3_Fiscal measures",
        "E4_International support",
        "H1_Public information campaigns",
        "H2_Testing policy",
        "H3_Contact tracing",
        "H4_Emergency investment in healthcare",
        "H5_Investment in vaccines",
        "H6M_Facial Coverings",
        "H7_Vaccination policy",
        "H8M_Protection of elderly people",
        "V1_Vaccine Prioritisation (summary)",
        "V2A_Vaccine Availability (summary)",
        "V2B_Vaccine age eligibility/availability age floor (general population summary)",
        "V2C_Vaccine age eligibility/availability age floor (at risk summary)",
        "V3_Vaccine Financial Support (summary)",
        "V4_Mandatory Vaccination (summary)"
    ]
    print("Starting US COVID data extraction...")
    
    # Initialize list to store US data
    us_data = []
    
    # Process file in chunks
    chunk_count = 0
    for chunk in pd.read_csv(input_file, chunksize=chunk_size):
        chunk_count += 1
        print(f"Processing chunk {chunk_count}...")
        
        # Filter out invalid rows (like the #country row) and get US data
        chunk = chunk[chunk['CountryName'] != '#country']
        us_chunk = chunk[chunk['CountryName'] == 'United States'].copy()
        
        if not us_chunk.empty:
            # Select only the date column and selected features
            columns_to_keep = ['Date'] + [col for col in selected_features if col in us_chunk.columns]
            us_chunk = us_chunk[columns_to_keep]
            us_data.append(us_chunk)
    
    if not us_data:
        print("No US data found!")
        return
    
    # Combine all US data
    print("Combining US data...")
    us_df = pd.concat(us_data, ignore_index=True)
    
    # Sort by date - parse dates in YYYYMMDD format
    us_df['Date'] = pd.to_datetime(us_df['Date'], format='%Y%m%d', errors='coerce')
    us_df = us_df.sort_values('Date')
    
    # Remove rows with invalid dates (NaT)
    us_df = us_df.dropna(subset=['Date'])
    
    # Rename date column to match financial CSV format
    us_df.rename(columns={'Date': 'observation_date'}, inplace=True)
    
    # Convert date to string format (YYYY-MM-DD)
    us_df['observation_date'] = us_df['observation_date'].dt.strftime('%Y-%m-%d')
    
    # Keep all features in a single dataframe
    print(f"Reformatting data...")
    
    # Remove any columns that weren't in our selected features (except observation_date)
    columns_to_keep = ['observation_date'] + [col for col in selected_features if col in us_df.columns]
    us_df = us_df[columns_to_keep]
    
    # Remove rows where all feature columns are NaN
    feature_columns = [col for col in us_df.columns if col != 'observation_date']
    us_df = us_df.dropna(subset=feature_columns, how='all')
    
    if not us_df.empty:
        # Save all features to CSV
        print(f"Saving to {output_file}...")
        us_df.to_csv(output_file, index=False)
        
        print(f"Successfully extracted US COVID data!")
        print(f"Shape: {us_df.shape}")
        print(f"Features saved: {feature_columns}")
        print(f"Date range: {us_df['observation_date'].min()} to {us_df['observation_date'].max()}")
        
        # Show summary of each feature
        print(f"\nFeature summary:")
        for col in feature_columns:
            non_null_count = us_df[col].notna().sum()
            print(f"  {col}: {non_null_count} non-null values")
    else:
        print("No valid feature data found!")

if __name__ == "__main__":
    input_file = "Current_Code/Data/COVID_DATA.csv" 
    # This is the csv file obtained from https://data.humdata.org/dataset/oxford-covid-19-government-response-tracker
    output_file = "Current_Code/Data/USCOVIDDATA_20200101_20221231.csv"

    extract_us_covid_data(input_file, output_file)