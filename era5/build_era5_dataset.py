import dask
import logging
import netCDF4 as nc
import numpy as np
import os
import pandas as pd
from pathlib import Path
import xarray as xr

import snoop

# Set up logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# # Set Dask to use the 'processes' scheduler globally
# dask.config.set(scheduler='processes')

def adjust_longitude_bounds(longitudes, lon_bounds):
    if np.any(longitudes > 180):  # Assuming longitudes are in 0 to 360
        lon_bounds = [(lon + 360) % 360 for lon in lon_bounds]  # Adjust bounds to 0-360
    return longitudes, lon_bounds

def extract_data_slice(dataset, variable_name, time_idx, target_level, latitudes, longitudes, lat_bounds, lon_bounds):
  
    # Find indices for latitude and longitude bounds
    lat_indices = np.where((latitudes >= lat_bounds[0]) & (latitudes <= lat_bounds[1]))[0]
    lon_indices = np.where((longitudes >= lon_bounds[0]) & (longitudes <= lon_bounds[1]))[0]
    
    # Assume level index is already determined outside this function
    level_index = np.where(dataset.variables['level'][:] == target_level)[0][0]
    
    print(dataset.variables[variable_name].shape)
    
    # Extract the slice
    variable_slice = dataset.variables[variable_name][time_idx, level_index, lat_indices[0]:lat_indices[-1]+1, lon_indices[0]:lon_indices[-1]+1]
    
    return variable_slice

def custom_regrid(data_slice, lat, lon, target_resolution=1):
    """
    Aggregate data to a coarser grid.

    Args:
        data_slice (np.array): The 2D array of data for a specific time and level.
        lat (np.array): 1D array of latitude values corresponding to data_slice.
        lon (np.array): 1D array of longitude values corresponding to data_slice.
        target_resolution (int): The target resolution in degrees for regridding.

    Returns:
        np.array: The regridded 2D array.
    """   
    
    # Determine new grid size
    lat_bins = np.arange(np.floor(lat.min()), np.ceil(lat.max()), target_resolution)
    lon_bins = np.arange(np.floor(lon.min()), np.ceil(lon.max()), target_resolution)

    # Bin the latitudes and longitudes
    lat_idxs = np.digitize(lat, bins=lat_bins) - 1
    lon_idxs = np.digitize(lon, bins=lon_bins) - 1

    # Create an empty array for the coarser grid
    coarse_grid = np.zeros((len(lat_bins) - 1, len(lon_bins) - 1))

    # Aggregate data into the coarser grid
    for i in range(len(lat_bins) - 1):
        for j in range(len(lon_bins) - 1):
            # Find data points within the current coarse grid cell
            mask = (lat_idxs == i) & (lon_idxs == j)
            # Average these data points
            coarse_grid[i, j] = np.mean(data_slice[mask])

    return coarse_grid




def save_slice_to_csv(time_value, level_value, latitudes, longitudes, variable_slice, variable_name, output_file):
    """
    Converts a data slice to a DataFrame and appends it to a CSV file.
    """
    # Meshgrid for the slice
    Lat, Lon = np.meshgrid(latitudes, longitudes, indexing='ij')

    # Flatten the data for the DataFrame
    flat_data = variable_slice.flatten()
    Lat_flat = Lat.flatten()
    Lon_flat = Lon.flatten()

    # DataFrame creation
    df = pd.DataFrame({
        'date': np.repeat(time_value, len(flat_data)),
        'pressure': np.repeat(level_value, len(flat_data)),
        'latitude': Lat_flat,
        'longitude': Lon_flat,
        variable_name: flat_data
    })

    # Append to CSV file
    header = not Path(output_file).exists()
    df.to_csv(output_file, mode='a', header=header, index=False)
    logging.info(f"Appended data to {output_file}")

def save_daily_averages_to_csv(daily_averages, days, latitudes, longitudes, output_csv, variable_name):
    # Assuming daily_averages shape: (days, latitudes, longitudes)
    # Flatten latitude and longitude for DataFrame format
    Lat, Lon = np.meshgrid(latitudes, longitudes, indexing='ij')
    Lat_flat = Lat.flatten()
    Lon_flat = Lon.flatten()
    
    with open(output_csv, 'w') as csvfile:
        # Write header
        csvfile.write("date,pressure,latitude,longitude,{}\n".format(variable_name))
        
        for day, daily_avg in zip(days, daily_averages):
            date_str = day.strftime('%Y-%m-%d')
            for lat, lon, value in zip(Lat_flat, Lon_flat, daily_avg.flatten()):
                csvfile.write("{},{},{},{},{}\n".format(date_str, target_level, lat, lon, value))

def daily_averaging(data, times):
    """
    Compute daily averages from hourly data.

    Args:
        data (np.array): The 3D array of data (time, lat, lon).
        times (list): List of datetime objects corresponding to each time step in data.

    Returns:
        np.array: Daily averaged data.
    """
    # Convert times to pandas Series to utilize resample functionality
    time_series = pd.Series(times)

    # Find unique days
    days = time_series.dt.floor('D').unique()

    # Initialize an empty list to hold daily averages
    daily_avg_data = []

    for day in days:
        # Find indices for the current day
        idxs = time_series.dt.floor('D') == day
        # Average the data for the current day
        daily_avg = np.mean(data[idxs], axis=0)
        daily_avg_data.append(daily_avg)

    return np.array(daily_avg_data), days


def process_and_aggregate(input_file, output_file, variable_name, target_level=250, lat_bounds=(30, 60), lon_bounds=(300, 360)):
    
    # Open the NetCDF dataset
    dataset = nc.Dataset(input_file, 'r')
    
    # Define your geographic bounds (for the North Atlantic Ocean in this example)
    lat_bounds = (30, 60)
    lon_bounds = (-60, 0)

    # Get latitude and longitude arrays
    latitudes = dataset.variables['latitude'][:]
    longitudes = dataset.variables['longitude'][:]

    # Adjust longitude bounds if your dataset uses a different convention (e.g., 0 to 360)
    longitudes, lon_bounds = adjust_longitude_bounds(longitudes, lon_bounds)

    # Process each time slice
    regridded_slices = []
    times = []
    for time_idx in range(dataset.dimensions['time'].size):
        # Extract the slice for the target level and geographic region
        variable_slice = extract_data_slice(dataset, variable_name, time_idx, target_level, latitudes, longitudes, lat_bounds, lon_bounds)
        
        # Apply regridding to the slice
        regridded_slice = custom_regrid(variable_slice, latitudes, longitudes, target_resolution=1)
        regridded_slices.append(regridded_slice)
        
        # Keep track of times for daily averaging
        time_value = nc.num2date(dataset.variables['time'][time_idx], dataset.variables['time'].units)
        times.append(time_value)
    
    # Apply daily averaging
    daily_averages, days = daily_averaging(np.array(regridded_slices), times)

    # Convert daily averages to DataFrame and save to CSV
    save_daily_averages_to_csv(daily_averages, days, latitudes, longitudes, output_file, variable_name)

    # Cleanup
    dataset.close()
    
def process_era5_files(variables_dict, start_year, end_year, start_month, end_month, output_directory='/data/pdonnelly/era5/processed_files'):
    base_path = Path(f"/bdd/ECMWF/ERA5/NETCDF/GLOBAL_025/hourly/AN_PL/{start_year}")
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    
    for short_name in variables_dict.values():
        for year in range(start_year, end_year + 1):
            for month in range(start_month, end_month + 1):
                # Define filenames
                input_file = base_path / f"{short_name}.{year}{month:02d}.ap1e5.GLOBAL_025.nc"
                output_file = output_directory / f"{short_name}_daily_{year}{month:02d}_1x1.csv"
                    
                if input_file.exists():
                    process_and_aggregate(input_file, output_file, short_name)

                    logging.info(f"Processed {output_file}")
                    
                else:
                    logging.info(f"File does not exist: {input_file}")

                exit()

# Define ERA5 variables
variables_dict = {
    "cloud cover": "cc",
    "temperature": "ta",
    "specific humidity": "q",
    "relative humidity": "r",
    "geopotential": "geopt",
    "eastward wind": "u",
    "northward wind": "v",
    "ozone mass mixing ratio": "o3",
}

# Execute on specified date range
process_era5_files(variables_dict, 2018, 2018, 3, 3)
