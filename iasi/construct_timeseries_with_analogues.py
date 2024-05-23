import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import date2num

lat_bins = np.arange(30, 60, 5)
lon_bins = np.arange(-60, 0, 5)

def find_and_print_duplicates(df):
    # Identify duplicates based on Latitude and Longitude
    duplicates = df[df.duplicated(subset=['Latitude', 'Longitude'], keep=False)]
    if not duplicates.empty:
        print("Duplicate rows based on Latitude and Longitude:")
        print(duplicates)
    else:
        print("No duplicate rows found.")
    return

def downscale_map(data):
    lat_bins = np.arange(30, 66, 5)
    lon_bins = np.arange(-60, 6, 5)
    
    # Use pd.cut to assign bins
    data['lat_bin'] = pd.cut(data['Latitude'], bins=lat_bins, labels=lat_bins[:-1], include_lowest=True, right=False)
    data['lon_bin'] = pd.cut(data['Longitude'], bins=lon_bins, labels=lon_bins[:-1], include_lowest=True, right=False)
    return data

def adjust_longitude(lon):
    return ((lon + 180) % 360 - 180)

def extract_analogues_and_correlations(dates_file, cor_file):
    dates_df = pd.read_csv(dates_file, header=None, sep="  ", engine='python')
    cor_df = pd.read_csv(cor_file, header=None, sep="  ", engine='python')
    
    lon_bins = dates_df.iloc[:, 0].apply(adjust_longitude).astype(int)
    lat_bins = dates_df.iloc[:, 1].astype(int)
    dates = dates_df.iloc[:, 2:]
    cor = cor_df.iloc[:, 2:]

     # Reset the column headers to start at 0
    dates.columns = range(dates.shape[1])
    cor.columns = range(cor.shape[1])

    sorted_dates = []
    sorted_cors = []

    for i in dates.index:
        sorted_indices = cor.iloc[i].sort_values(ascending=False, na_position='last').index
        sorted_dates.append(dates.iloc[i, sorted_indices].values)
        sorted_cors.append(cor.iloc[i, sorted_indices].values)
    
    date_columns = pd.DataFrame(sorted_dates, columns=[f'date_{i}' for i in range(dates.shape[1])])
    cor_columns = pd.DataFrame(sorted_cors, columns=[f'cor_{i}' for i in range(cor.shape[1])])

    combined_df = pd.concat([pd.DataFrame({'lat_bin': lat_bins, 'lon_bin': lon_bins}), date_columns, cor_columns], axis=1)

    # # Remove duplicate boundary point
    # combined_df = combined_df[combined_df['lon_bin'] != 0]

    return combined_df

def remove_duplicates_from_data(analogue_df, analogue_file_path, best_date):
    # Convert the Date column to datetime format for comparison
    analogue_df['Date'] = pd.to_datetime(analogue_df['Date'])
    file_date = pd.to_datetime(best_date)
    
    # Find duplicates based on Latitude and Longitude
    duplicates = analogue_df[analogue_df.duplicated(subset=['Latitude', 'Longitude'], keep=False)]
    
    if not duplicates.empty:
        print(f"Found duplicates in file {analogue_file_path}. Fixing...")
        
        # Keep the rows where the Date matches the file date
        correct_rows = duplicates[duplicates['Date'] == file_date]
        
        # Drop the duplicates and keep the correct rows
        analogue_df = analogue_df.drop(duplicates.index)
        analogue_df = pd.concat([analogue_df, correct_rows]).drop_duplicates(subset=['Latitude', 'Longitude'])

        # Overwrite the file with the cleaned DataFrame
        analogue_df.to_csv(analogue_file_path, sep="\t", index=False)
        print(f"Fixed duplicates and saved cleaned data back to {analogue_file_path}")
    return analogue_df

def plot_maps(iasi_path, original_df, reconstructed_df, target_date):
    fig, axes = plt.subplots(2, 1, figsize=(6, 6))

    # Pivot original data set
    original_pivot = original_df.pivot(index='Latitude', columns='Longitude', values='OLR_mean')    
    # # Drop the column where Longitude = 0 and the row where Latitude = 60
    # original_pivot = original_pivot.drop(columns=0, errors='ignore')
    # original_pivot = original_pivot.drop(index=60, errors='ignore')

    # Pivot reconstructed data set
    reconstructed_pivot = reconstructed_df.pivot(index='Latitude', columns='Longitude', values='OLR_mean')
    # Reindex the reconstructed pivot to match the original
    reconstructed_pivot = reconstructed_pivot.reindex(index=original_pivot.index, columns=original_pivot.columns)
    # Fill NaNs with the column mean
    reconstructed_pivot = reconstructed_pivot.apply(lambda x: x.fillna(np.nan), axis=0)
    # Convert the Date column to datetime format
    reconstructed_df['Date'] = pd.to_datetime(reconstructed_df['Date'])
    # Convert the dates to numerical format for coloring
    date_nums = date2num(reconstructed_df['Date'])

    lat_bins = np.arange(30, 61, 1)
    lon_bins = np.arange(-60, 1, 1)

    # Original map
    # im0 = axes[0].pcolormesh(lon_bins, lat_bins, original_pivot.values, cmap='cividis', shading='auto')
    # axes[0].set_title(f'Original Map: {target_date}')
    # fig.colorbar(im0, ax=axes[0], shrink=0.75)

    # # Add 5-degree gridlines
    # axes[0].set_xticks(np.arange(-60, 1, 5))
    # axes[0].set_yticks(np.arange(30, 61, 5))
    # axes[0].grid(which='both', color='gray', linestyle='--', linewidth=0.5)
    # axes[0].set_aspect('equal', 'box')

    # Reconstructed map
    im1 = axes[0].pcolormesh(lon_bins, lat_bins, reconstructed_pivot.values, cmap='cividis', shading='auto')
    axes[0].set_title(f'Reconstructed IIR: Analogues from 2013-2019')
    fig.colorbar(im1, ax=axes[0], shrink=0.75)

    # Add 5-degree gridlines
    axes[0].set_xticks(np.arange(-60, 1, 5))
    axes[0].set_yticks(np.arange(30, 61, 5))
    axes[0].grid(which='both', color='gray', linestyle='--', linewidth=0.5)
    axes[0].set_aspect('equal', 'box')

    # Scatter plot of reconstructed data
    scatter = axes[1].scatter(reconstructed_df['Longitude'], reconstructed_df['Latitude'], 
                              c=date_nums, cmap='jet', marker='o', s=10)
    axes[1].set_title('Reconstructed IIR: Analogues from 2013-2019')
    fig.colorbar(scatter, ax=axes[1], label='Date', format=mdates.DateFormatter('%Y'))
    axes[1].set_xticks(np.arange(-60, 1, 5))
    axes[1].set_yticks(np.arange(30, 61, 5))
    axes[1].grid(which='both', color='gray', linestyle='--', linewidth=0.5)
    axes[1].set_aspect('equal', 'box')
    # Set the limits for the scatter plot
    axes[1].set_xlim(lon_bins.min(), lon_bins.max())
    axes[1].set_ylim(lat_bins.min(), lat_bins.max())

    plt.tight_layout()
    plt.savefig(os.path.join(iasi_path, f"reconstructed_IIR_{target_date}.png"), bbox_inches='tight', dpi=300)
    plt.close()

def construct_daily_mean(target_date, iasi_path, combined_df):
    # Read the OLR data for the target date
    year, month, day = target_date.split('-')
    iasi_file_path = os.path.join(iasi_path, year, month.zfill(2), day.zfill(2), 'spectra_and_cloud_products_binned.csv')
    
    if os.path.exists(iasi_file_path):
        iasi_df = pd.read_csv(iasi_file_path, sep="\t", engine='python')
    else:
        return None
    iasi_df = downscale_map(iasi_df)

    reconstructed_iasi_df = pd.DataFrame()

    for index, row in combined_df.iterrows():
        lat_bin = row['lat_bin']
        lon_bin = row['lon_bin']

        if (lat_bin in lat_bins) and (lon_bin in lon_bins):
            for i in range(50):
                best_date = row[f'date_{i}']
                analogue_year, analogue_month, analogue_day = best_date.split('-')
                analogue_file_path = os.path.join(iasi_path, analogue_year, analogue_month.zfill(2), analogue_day.zfill(2), 'spectra_and_cloud_products_binned.csv')

                if os.path.exists(analogue_file_path):
                    analogue_df = pd.read_csv(analogue_file_path, sep="\t", engine='python')
                    analogue_df = remove_duplicates_from_data(analogue_df, analogue_file_path, best_date)
                    analogue_df = downscale_map(analogue_df)

                    # Extract 1x1 degree data for the 5x5 degree grid cell
                    matched_df = analogue_df[(analogue_df['lat_bin'] == lat_bin) &
                                            (analogue_df['lon_bin'] == lon_bin)]
                    
                    if not matched_df.empty:
                        reconstructed_iasi_df = pd.concat([reconstructed_iasi_df, matched_df])
                        break  # Exit the loop once a valid file is found
    
    # Plot for posterity
    plot_maps(iasi_path, iasi_df, reconstructed_iasi_df, target_date)
    
    # Calculate the average OLR for the reconstructed data
    reconstructed_daily_average = reconstructed_iasi_df['OLR_mean'].mean()
    return reconstructed_daily_average

def main():
    # Base paths for data
    iasi_path = 'G:\\My Drive\\Research\\Postdoc_2_CNRS_LATMOS\\data\\iasi\\binned_olr'
    analogues_path = "G:\\My Drive\\Research\\Postdoc_2_CNRS_LATMOS\\data\\circulation_analogues"
    output_path = "G:\\My Drive\\Research\\Postdoc_2_CNRS_LATMOS\\data\\machine_learning"
    
    # Date range for Spring 2020
    spring_2020_dates = pd.date_range(start='2020-03-16', end='2020-04-13')
    years = spring_2020_dates.year
    months = spring_2020_dates.month
    days = spring_2020_dates.day

    # Construct the new time series
    daily_average_list = []

    for year, month, day in zip(years, months, days):
        target_date = f"{year}-{month:02d}-{day:02d}"
        
        dates_file_path = os.path.join(analogues_path, f"an2000.dates.50.{target_date}")
        cor_file_path = os.path.join(analogues_path, f"an2000.cor.50.{target_date}")

        # Check if both files exist
        if os.path.exists(dates_file_path) and os.path.exists(cor_file_path):
            combined_df = extract_analogues_and_correlations(dates_file_path, cor_file_path)
            daily_average = construct_daily_mean(target_date, iasi_path, combined_df)
            daily_average_list.append({'Date': target_date, 'OLR_mean': daily_average})
    
    # Convert daily_average_list to a DataFrame and save
    daily_average_df = pd.DataFrame(daily_average_list)

    # Read in the era5_iasi_combined file
    era5_iasi_combined_df = pd.read_csv(os.path.join(output_path, "era5_iasi_combined.csv"), sep='\t')

    # Merge the new daily averages into the existing DataFrame based on the Date column
    era5_iasi_combined_df['Date'] = pd.to_datetime(era5_iasi_combined_df['Date'])
    daily_average_df['Date'] = pd.to_datetime(daily_average_df['Date'])
    
    # Identify the dates that need updating
    dates_to_update = daily_average_df['Date']

    # Update only the rows that need changing
    mask = era5_iasi_combined_df['Date'].isin(dates_to_update)
    era5_iasi_combined_df.loc[mask, 'OLR_mean'] = era5_iasi_combined_df.loc[mask, 'Date'].map(daily_average_df.set_index('Date')['OLR_mean'])

    # Save the updated DataFrame to a new file
    era5_iasi_combined_df.to_csv(os.path.join(output_path, "era5_iasi_combined_corrected.csv"), sep='\t', index=False)

if __name__ == "__main__":
    main()