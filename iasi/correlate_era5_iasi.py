import pandas as pd
import os
from datetime import datetime, timedelta
import seaborn as sns
import matplotlib.pyplot as plt

def read_and_combine_daily_data(iasi_base_path, era5_base_path, start_date, end_date):
    combined_data = []

    # Generate date range
    current_date = start_date
    while current_date <= end_date:
        year = current_date.strftime('%Y')
        month = current_date.strftime('%m')
        day = current_date.strftime('%d')
        date_str = current_date.strftime('%Y-%m-%d')

        # Construct file paths
        iasi_file_path = os.path.join(iasi_base_path, year, month, day, 'spectra_and_cloud_products.csv')
        era5_file_path = os.path.join(era5_base_path, f'daily_1x1_{date_str}.csv')

        if os.path.exists(iasi_file_path) and os.path.exists(era5_file_path):
            # Read data
            iasi_df = pd.read_csv(iasi_file_path)
            era5_df = pd.read_csv(era5_file_path)

            # Combine data based on latitude and longitude
            combined_df = pd.merge(iasi_df, era5_df, on=['latitude', 'longitude'])
            combined_data.append(combined_df)

        current_date += timedelta(days=1)

    return pd.concat(combined_data, ignore_index=True) if combined_data else pd.DataFrame()

def plot_era5_fields_by_olr_quintiles(data):
    # Calculate quintiles for 'OLR_mean'
    data['OLR_quintile'] = pd.qcut(data['OLR_mean'], 5, labels=False)

    # Select ERA5 fields - assuming these are all columns except for lat, lon, OLR_mean, and OLR_quintile
    era5_columns = [col for col in data.columns if col not in ['latitude', 'longitude', 'OLR_mean', 'OLR_quintile']]

    # Create pair plot
    pair_plot = sns.pairplot(data, vars=era5_columns, hue='OLR_quintile', palette='coolwarm')
    plt.show()

# Example usage
iasi_base_path = 'G:/My Drive/Research/Postdoc_2_CNRS_LATMOS/data/iasi/binned_olr'
era5_base_path = 'G:/My Drive/Research/Postdoc_2_CNRS_LATMOS/data/era5/daily_combined'
start_date = datetime(2018, 3, 1)
end_date = datetime(2018, 3, 31)
combined_data = read_and_combine_daily_data(iasi_base_path, era5_base_path, start_date, end_date)
plot_era5_fields_by_olr_quintiles(combined_data)
