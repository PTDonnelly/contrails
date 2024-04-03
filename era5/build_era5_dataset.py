import datetime as dt
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

def prepare_dataset(dataset, target_level, lat_bounds, lon_bounds):    
    # Get latitude and longitude arrays
    latitudes = dataset.variables['latitude'][:]
    longitudes = dataset.variables['longitude'][:]

    # Get index location of data slices
    level_index, slice_lats, slice_lons = find_data_slice(dataset, target_level, latitudes, lat_bounds, longitudes, lon_bounds)
    return level_index, slice_lats, slice_lons

def find_data_slice(dataset, target_level, latitudes, lat_bounds, longitudes, lon_bounds):
    # Find indices for latitude and longitude bounds
    lat_indices = np.where((latitudes >= lat_bounds[0]) & (latitudes <= lat_bounds[1]))[0]
    lon_indices = np.where((longitudes >= lon_bounds[0]) & (longitudes <= lon_bounds[1]))[0]
    
    # Assume level index is already determined outside this function
    level_index = np.where(dataset.variables['level'][:] == target_level)[0][0]
    
    # Extract latitude and longitude slices
    slice_lats = dataset.variables['latitude'][lat_indices[0]:lat_indices[-1]+1]
    slice_lons = dataset.variables['longitude'][lon_indices[0]:lon_indices[-1]+1]
    return level_index, slice_lats, slice_lons

def extract_data_slice(dataset, variable_name, time_index, level_index, slice_lats, slice_lons, lat_bounds, lon_bounds):
    # Find indices for latitude and longitude bounds
    lat_indices = np.where((slice_lats >= lat_bounds[0]) & (slice_lats <= lat_bounds[1]))[0]
    lon_indices = np.where((slice_lons >= lon_bounds[0]) & (slice_lons <= lon_bounds[1]))[0]
    
    # Extract the data slice for the given variable
    slice_data = dataset.variables[variable_name][time_index, level_index, lat_indices[0]:lat_indices[-1]+1, lon_indices[0]:lon_indices[-1]+1]
    return slice_data

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

def regrid_data(slice_data, slice_lats, slice_lons, target_resolution, method='linear'):
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
    # Create a meshgrid for the original coordinates
    lon_mesh, lat_mesh = np.meshgrid(slice_lons, slice_lats)
    
    # Flatten the meshgrid for interpolation
    points = np.column_stack((lat_mesh.ravel(), lon_mesh.ravel()))
    values = slice_data.ravel()
    
    # Generate the target grid
    target_lat = np.arange(slice_lats.min(), slice_lats.max(), target_resolution)
    target_lon = np.arange(slice_lons.min(), slice_lons.max(), target_resolution)
    target_lon_mesh, target_lat_mesh = np.meshgrid(target_lon, target_lat)

    # Interpolate to the new grid
    regridded_data = griddata(points, values, (target_lat_mesh, target_lon_mesh), method=method)
    return regridded_data

# def save_daily_averages_to_csv(daily_averages, days, latitudes, longitudes, output_csv, variable_name):
#     # Assuming daily_averages shape: (days, latitudes, longitudes)
#     # Flatten latitude and longitude for DataFrame format
#     Lat, Lon = np.meshgrid(latitudes, longitudes, indexing='ij')
#     Lat_flat = Lat.flatten()
#     Lon_flat = Lon.flatten()
    
#     with open(output_csv, 'w') as csvfile:
#         # Write header
#         csvfile.write("date,pressure,latitude,longitude,{}\n".format(variable_name))
        
#         for day, daily_avg in zip(days, daily_averages):
#             date_str = day.strftime('%Y-%m-%d')
#             for lat, lon, value in zip(Lat_flat, Lon_flat, daily_avg.flatten()):
#                 csvfile.write("{},{},{},{},{}\n".format(date_str, target_level, lat, lon, value))

@snoop
def process_dataset(dataset, variable_name, level_index, slice_lats, slice_lons, lat_bounds, lon_bounds, target_resolution):
    # Convert time variable to datetime objects
    times = nc.num2date(dataset.variables['time'][:], dataset.variables['time'].units)

    # Convert cftime DatetimeGregorian objects to datetime.date
    dates = np.unique([dt.date(time.year, time.month, time.day) for time in times])

    daily_averages = []
    for date in dates:
        # print(f"Day: {date}")
        
        # Find time indices for the current day
        time_indices = [i for i, time in enumerate(times) if dt.datetime(time.year, time.month, time.day).date() == date]
    
        # Process each time slice for the day
        day_slices = []
        for time_index in time_indices:
            # print(f"Hour: {time_index}")
            # Extract slice (assuming a function that handles the extraction)
            slice_data = extract_data_slice(dataset, variable_name, time_index, level_index, slice_lats, slice_lons, lat_bounds, lon_bounds)
            # Regrid the slice
            regridded_slice = regrid_data(slice_data, slice_lats, slice_lons, target_resolution)
            day_slices.append(regridded_slice)
        
        # Compute the daily average
        daily_average = np.mean(day_slices, axis=0)
        daily_averages.append(daily_average)
    
    return daily_averages

    # Convert daily averages to DataFrame and save to CSV
    save_daily_averages_to_csv(daily_averages, days, latitudes, longitudes, output_file, variable_name)

    # Cleanup
    dataset.close()
    
def process_era5_files(variables_dict, start_year, end_year, start_month, end_month, output_directory='/data/pdonnelly/era5/processed_files'):
    base_path = Path(f"/bdd/ECMWF/ERA5/NETCDF/GLOBAL_025/hourly/AN_PL/{start_year}")
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    target_level = 250
    lat_bounds = (30, 60)
    lon_bounds = (300, 360)
    target_resolution = 1
    
    for short_name in variables_dict.values():
        for year in range(start_year, end_year + 1):
            for month in range(start_month, end_month + 1):
                # Define filenames
                input_file = base_path / f"{short_name}.{year}{month:02d}.ap1e5.GLOBAL_025.nc"
                output_file = output_directory / f"{short_name}_daily_{year}{month:02d}_1x1.csv"
                    
                if input_file.exists():
                    # Open the NetCDF dataset
                    dataset = nc.Dataset(input_file, 'r')

                    level_index, slice_lats, slice_lons = prepare_dataset(dataset, target_level, lat_bounds, lon_bounds)
                    
                    daily_averages = process_dataset(dataset, short_name, level_index, slice_lats, slice_lons, lat_bounds, lon_bounds, target_resolution)
                    
                    print(len(daily_averages))
                    # save_daily_averages_to_csv()

                    logging.info(f"Processed {output_file}")
                    
                    # Cleanup
                    dataset.close()
                else:
                    logging.info(f"File does not exist: {input_file}")

                exit()

def main():
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

if __name__ == "__main__":
    main()