import pandas as pd
import os
from datetime import datetime, timedelta

def convert_longitude(lon):
    """
    Convert longitude from -180 to 180 range to 0 to 360 range.
    """
    return lon % 360

def read_and_combine_daily_data(base_path, start_date, end_date):
    # Create data paths
    iasi_path = f"{base_path}iasi/binned_olr/"
        
    combined_data = []

    # Generate date range
    current_date = start_date
    while current_date <= end_date:
        print(current_date)
        
        year = current_date.strftime('%Y')
        month = current_date.strftime('%m')
        day = current_date.strftime('%d')
        date_str = current_date.strftime('%Y-%m-%d')

        # Construct file paths
        iasi_file_path = os.path.join(iasi_path, year, month, day, 'spectra_and_cloud_products_binned.csv')

        if os.path.exists(iasi_file_path):
            # Read data
            iasi_df = pd.read_csv(iasi_file_path, sep='\t')

            # Convert IASI longitudes to 0-360 range
            iasi_df['Longitude'] = iasi_df['Longitude'].apply(convert_longitude)
            combined_data.append(iasi_df)

        current_date += timedelta(days=1)

    combined_df = pd.concat(combined_data, ignore_index=True) if combined_data else pd.DataFrame()
    combined_df.to_csv(f"{base_path}machine_learning/all_iasi_combined.csv", sep='\t', index=False)
    return

def main():
    base_path = 'G:/My Drive/Research/Postdoc_2_CNRS_LATMOS/data/'
    start_date = datetime(2013, 3, 1)
    end_date = datetime(2023, 5, 31)
    
    # Define training and testing sets
    read_and_combine_daily_data(base_path, start_date, end_date)

if __name__ == "__main__":
    main()