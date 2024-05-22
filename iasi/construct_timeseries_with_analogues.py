import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

def downscale_map(data):
    lat_bins = np.arange(30, 61, 5)
    lon_bins = np.arange(-60, 1, 5)
    data['lat_bin'] = pd.cut(data['Latitude'], bins=lat_bins, labels=lat_bins[:-1])
    data['lon_bin'] = pd.cut(data['Longitude'], bins=lon_bins, labels=lon_bins[:-1])
    data_5x5 = data.groupby(['lat_bin', 'lon_bin'], observed=True)['OLR_mean'].mean().reset_index()
    return data_5x5

def adjust_longitude(lon):
    return (lon + 180) % 360 - 180

def extract_analogues(dates_file, cor_file):
    dates_df = pd.read_csv(dates_file, header=None, sep="  ")
    cor_df = pd.read_csv(cor_file, header=None, sep="  ")
    
    lon_bins = dates_df.iloc[:, 0].apply(adjust_longitude)
    lat_bins = dates_df.iloc[:, 1]
    dates = dates_df.iloc[:, 2:]
    cor = cor_df.iloc[:, 2:]

    best_analogues = pd.Series(index=dates.index, dtype=object)
    
    for i in dates.index:
        # Find the column with the highest correlation (minus 2 to account for the removal of lat and lon columns)
        best_idx = cor.iloc[i].idxmax() - 2
        best_analogues.iloc[i] = dates.iloc[i, best_idx]
    
    best_analogues.index = zip(lat_bins, lon_bins)
    return best_analogues

def plot_maps(original_df, reconstructed_df, target_date):
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    print(original_df.head())
    print(reconstructed_df.head())
    
    original_pivot = original_df.pivot(index='lat_bin', columns='lon_bin', values='OLR')
    reconstructed_pivot = reconstructed_df.pivot(index='lat_bin', columns='lon_bin', values='OLR')
    
    original_plot = axes[0].imshow(original_pivot, cmap='cividis', aspect='auto', origin='lower')
    axes[0].set_title(f'Original Map: {target_date}')
    fig.colorbar(original_plot, ax=axes[0])
    
    reconstructed_plot = axes[1].imshow(reconstructed_pivot, cmap='cividis', aspect='auto', origin='lower')
    axes[1].set_title(f'Reconstructed Map: {target_date}')
    fig.colorbar(reconstructed_plot, ax=axes[1])
    
    plt.tight_layout()
    plt.show()

def construct_daily_map(target_date, iasi_path, dates_file_path, cor_file_path):
    # Read the OLR data for the target date
    year, month, day = target_date.split('-')
    iasi_file_path = os.path.join(iasi_path, year, month.zfill(2), day.zfill(2), 'spectra_and_cloud_products_binned.csv')
    if os.path.exists(iasi_file_path):
        iasi_df = pd.read_csv(iasi_file_path, sep="\t", engine='python')
    else:
        return None
    iasi_scaled_df = downscale_map(iasi_df)

    # Read the analogous dates and correlations
    best_analogues = extract_analogues(dates_file_path, cor_file_path)

    reconstructed_scaled_iasi_df = pd.DataFrame()

    for (lat_bin, lon_bin), best_date in best_analogues.items():
        analogue_year, analogue_month, analogue_day = best_date.split('-')
        analogue_file_path = os.path.join(iasi_path, analogue_year, analogue_month.zfill(2), analogue_day.zfill(2), 'spectra_and_cloud_products_binned.csv')
        if os.path.exists(analogue_file_path):
            analogue_df = pd.read_csv(analogue_file_path, sep="\t", engine='python')
            analogue_scaled_df = downscale_map(analogue_df)
            print(lat_bin, lon_bin) 
            # print(analogue_scaled_df.head())

            matched_data = analogue_scaled_df[(analogue_scaled_df['lat_bin'] == lat_bin) & (analogue_scaled_df['lon_bin'] == lon_bin)]
            reconstructed_scaled_iasi_df = pd.concat([reconstructed_scaled_iasi_df, matched_data])

            # print(matched_data.head())
        # input()
    exit()

    reconstructed_scaled_iasi_df['target_date'] = target_date
    
    plot_maps(iasi_scaled_df, reconstructed_scaled_iasi_df, target_date)
    
    return reconstructed_scaled_iasi_df

def main():
    # Base paths for data
    iasi_path = 'G:\\My Drive\\Research\\Postdoc_2_CNRS_LATMOS\\data\\iasi\\binned_olr'
    analogues_path = "G:\\My Drive\\Research\\Postdoc_2_CNRS_LATMOS\\data\\circulation_analogues"
    
    # Date range for Spring 2020
    spring_2020_dates = pd.date_range(start='2020-03-16', end='2020-04-13')
    years = spring_2020_dates.year
    months = spring_2020_dates.month
    days = spring_2020_dates.day

    # Construct the new time series
    new_iasi_timeseries = pd.DataFrame()

    for year, month, day in zip(years, months, days):
        target_date = f"{year}-{month:02d}-{day:02d}"
        
        # iasi_file_path = os.path.join(iasi_path, f"{year}", f"{month:02d}", f"{day:02d}", "spectra_and_cloud_products_binned.csv")
        dates_file_path = os.path.join(analogues_path, f"an2000.dates.50.{target_date}")
        cor_file_path = os.path.join(analogues_path, f"an2000.cor.50.{target_date}.")

        # Check if both files exist
        if os.path.exists(dates_file_path) and os.path.exists(cor_file_path):
            daily_map = construct_daily_map(target_date, iasi_path, dates_file_path, cor_file_path)
            new_iasi_timeseries = pd.concat([new_iasi_timeseries, daily_map])
    
    # Save the new time series to a CSV file
    new_iasi_timeseries.to_csv('new_iasi_timeseries.csv', index=False)

if __name__ == "__main__":
    main()
