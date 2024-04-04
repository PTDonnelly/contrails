import pandas as pd
from pathlib import Path
import re

def extract_date_from_filename(filename):
    """Extracts the date string from a filename."""
    pattern = re.compile(r"_(\d{4})-(\d{2})-(\d{2})")
    match = pattern.search(filename)
    if match:
        return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
    return None

def load_and_tag_data(filepath):
    """Loads CSV data and tags it with the variable name."""
    df = pd.read_csv(filepath, sep='\t')
    print(df.head())
    variable_name = filepath.name.split('_')[0]
    df['variable'] = variable_name
    print(df.head())
    return df

def gather_daily_data(processed_files_dir):
    """Gathers data for each unique date from all CSV files."""
    daily_data = {}
    for file in Path(processed_files_dir).glob('**/*.csv'):
        date_str = extract_date_from_filename(file.name)
        if date_str:
            if date_str not in daily_data:
                daily_data[date_str] = []
            daily_data[date_str].append(load_and_tag_data(file))
            input()
    exit()
    return daily_data

def pivot_and_save_daily_data(daily_data, output_dir_path):
    """Pivots data to have variables as columns and saves the daily data to CSV files."""
    for date_str, dfs in daily_data.items():
        day_df = pd.concat(dfs)
        print(day_df.head())
        pivoted_df = day_df.pivot_table(index=['latitude', 'longitude', 'date'], columns='variable', values='value').reset_index()
        output_filename = output_dir_path / f"daily_1x1_{date_str}.csv"
        pivoted_df.to_csv(output_filename, index=False)
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
