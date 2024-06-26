import pandas as pd
from pathlib import Path
import re
from tqdm import tqdm

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
    
    daily_data = {}
    for file in tqdm(files):
        date_str = extract_date_from_filename(file.name)
        if date_str:
            if date_str not in daily_data:
                daily_data[date_str] = []
            daily_data[date_str].append(get_data(file))
    return daily_data

def custom_sort(col):
    # Basic columns that don't need sorting
    if col in ['Date', 'Latitude', 'Longitude']:
        return (col, 0)

    # Extract variable and level from column name
    variable, level = col.rsplit('_', 1)
    # Convert level to integer for proper numeric sorting
    level = int(level)
    # Return a tuple that first sorts by level, then by variable name
    return (variable, level)

def save_daily_data(daily_data, output_dir_path):
    """Saves the daily data to CSV files."""
    for date_str, dfs in tqdm(daily_data.items()):
        # Initialize the combined DataFrame with the first DataFrame in the list
        combined_df = dfs[0]
        # Iteratively merge the rest of the DataFrames on 'Date', 'Latitude', and 'Longitude' (because merge() only works on two DataFrames at a time)
        for df in dfs[1:]:
            combined_df = combined_df.merge(df, on=['Date', 'Latitude', 'Longitude'], how='outer')
        
        # Sort columns using the custom sort key
        sorted_columns = sorted(combined_df.columns, key=custom_sort)

        # Reindex DataFrame with sorted columns
        combined_df_sorted = combined_df[sorted_columns]

        # Save combined DataFrame to CSV
        output_filename = output_dir_path / f"daily_1x1_{date_str}.csv"
        combined_df_sorted.to_csv(output_filename, sep='\t', index=False)

def process_era5_files(processed_files_dir, output_dir):
    """Main function to orchestrate the processing of ERA5 files."""
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    daily_data = gather_daily_data(processed_files_dir)
    save_daily_data(daily_data, output_dir_path)

# Example usage
processed_files_dir = '/data/pdonnelly/era5/processed_files'
output_dir = '/data/pdonnelly/era5/daily_combined'
process_era5_files(processed_files_dir, output_dir)
