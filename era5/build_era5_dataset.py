import logging
import netCDF4 as nc
import numpy as np
import os
import pandas as pd
from pathlib import Path
from scipy.interpolate import griddata

import snoop

# Set up logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# # Set Dask to use the 'processes' scheduler globally
# dask.config.set(scheduler='processes')

def extract_data_slice(dataset, variable_name, time_idx, target_level, latitudes, lat_bounds, longitudes, lon_bounds):
    # Find indices for latitude and longitude bounds
    lat_indices = np.where((latitudes >= lat_bounds[0]) & (latitudes <= lat_bounds[1]))[0]
    lon_indices = np.where((longitudes >= lon_bounds[0]) & (longitudes <= lon_bounds[1]))[0]
    
    # Assume level index is already determined outside this function
    level_index = np.where(dataset.variables['level'][:] == target_level)[0][0]
    
    # Extract the slice
    variable_slice = dataset.variables[variable_name][time_idx, level_index, lat_indices[0]:lat_indices[-1]+1, lon_indices[0]:lon_indices[-1]+1]
    
    return variable_slice

def create_target_grid(latitudes, longitudes, target_resolution):
    """
    Generate target latitude and longitude arrays based on target resolution.

    Args:
    - latitudes (numpy array): Array of original latitude values.
    - longitudes (numpy array): Array of original longitude values.
    - target_resolution (float): The desired resolution for the target grid.

    Returns:
    - (numpy array, numpy array): Target latitude and longitude arrays.
    """
    lat_min, lat_max = np.min(latitudes), np.max(latitudes)
    lon_min, lon_max = np.min(longitudes), np.max(longitudes)

    target_lats = np.arange(lat_min, lat_max + target_resolution, target_resolution)
    target_lons = np.arange(lon_min, lon_max + target_resolution, target_resolution)
    
    return target_lats, target_lons

def regrid_data(data, latitudes, longitudes, target_resolution, method='linear'):
    """
    Regrid data using scipy's griddata interpolation.

    Args:
    - data (numpy array): 2D array of the original data to regrid.
    - lat (numpy array): 1D array of latitude values for the original data.
    - lon (numpy array): 1D array of longitude values for the original data.
    - target_resolution (float): The target resolution in degrees.
    - method (str): Interpolation method ('linear', 'nearest', 'cubic').

    Returns:
    - numpy array: The regridded data on the target grid.
    """
    # # Generate the target grid
    # target_lat, target_lon = create_target_grid(latitudes, longitudes, target_resolution)

    # # Create a meshgrid for the original coordinates
    # lon_mesh, lat_mesh = np.meshgrid(longitudes, latitudes)
    
    # # Flatten the meshgrid for interpolation
    # points = np.column_stack((lat_mesh.ravel(), lon_mesh.ravel()))
    # values = data.ravel()
    
    # # Create a meshgrid for the target coordinates
    # target_lon_mesh, target_lat_mesh = np.meshgrid(target_lon, target_lat)

    # Create a meshgrid for the original coordinates
    lon_mesh, lat_mesh = np.meshgrid(longitudes, latitudes)
    
    # Flatten the meshgrid for interpolation
    points = np.column_stack((lat_mesh.ravel(), lon_mesh.ravel()))
    values = data.ravel()
    
    # Generate the target grid
    target_lat = np.arange(latitudes.min(), latitudes.max(), target_resolution)
    target_lon = np.arange(longitudes.min(), longitudes.max(), target_resolution)
    target_lon_mesh, target_lat_mesh = np.meshgrid(target_lon, target_lat)


    
    # Interpolate to the new grid
    regridded_data = griddata(points, values, (target_lat_mesh, target_lon_mesh), method=method)
    
    return regridded_data



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


def process_and_aggregate(input_file, output_file, variable_name, target_level, lat_bounds, lon_bounds):
    
    # Open the NetCDF dataset
    dataset = nc.Dataset(input_file, 'r')
    
    # Get latitude and longitude arrays
    latitudes = dataset.variables['latitude'][:]
    longitudes = dataset.variables['longitude'][:]

    # Process each time slice
    regridded_slices = []
    times = []
    for time_idx in range(dataset.dimensions['time'].size):
        # Extract the slice for the target level and geographic region
        variable_slice = extract_data_slice(dataset, variable_name, time_idx, target_level, latitudes, lat_bounds, longitudes, lon_bounds)
        
        # Apply regridding to the slice
        regridded_slice = regrid_data(variable_slice, latitudes, longitudes, target_resolution=1)
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
                    process_and_aggregate(input_file, output_file, short_name, target_level=250, lat_bounds=(30, 60), lon_bounds=(300, 360))

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
