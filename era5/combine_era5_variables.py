import pandas as pd
from pathlib import Path
import re
import tqdm

def extract_date_from_filename(filename):
    """Extracts the date string from a filename."""
    pattern = re.compile(r"_(\d{4})-(\d{2})-(\d{2})")
    match = pattern.search(filename)
    if match:
        return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
    return None

def get_data(filepath):
    """Loads CSV data."""
    return pd.read_csv(filepath, sep='\t')

def gather_daily_data(processed_files_dir):
    """Gathers data for each unique date from all CSV files."""
    files = Path(processed_files_dir).glob('**/*.csv')
    print(f"Gathering {len(files)} files")
    
    daily_data = {}
    for file in tqdm(files):
        date_str = extract_date_from_filename(file.name)
        if date_str:
            if date_str not in daily_data:
                daily_data[date_str] = []
            daily_data[date_str].append(get_data(file))
    return daily_data

def pivot_and_save_daily_data(daily_data, output_dir_path):
    """Pivots data to have variables as columns and saves the daily data to CSV files."""
    for date_str, dfs in tqdm(daily_data.items()):
        # Initialize the combined DataFrame with the first DataFrame in the list
        combined_df = dfs[0]
        # Iteratively merge the rest of the DataFrames on 'Date', 'Latitude', and 'Longitude' (because merge() only works on two DataFrames at a time)
        for df in dfs[1:]:
            combined_df = combined_df.merge(df, on=['Date', 'Latitude', 'Longitude'], how='outer')
        
        # Save combined DataFrame to CSV
        output_filename = output_dir_path / f"daily_1x1_{date_str}.csv"
        combined_df.to_csv(output_filename, sep='\t', index=False)
        print(f"Data saved to {output_filename}")

def process_era5_files(processed_files_dir, output_dir):
    """Main function to orchestrate the processing of ERA5 files."""
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    daily_data = gather_daily_data(processed_files_dir)
    pivot_and_save_daily_data(daily_data, output_dir_path)

# Example usage
processed_files_dir = '/data/pdonnelly/era5/processed_files_test'
output_dir = '/data/pdonnelly/era5/daily_combined'
process_era5_files(processed_files_dir, output_dir)
