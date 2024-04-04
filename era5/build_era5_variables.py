import datetime as dt
import logging
from multiprocessing import Pool
import netCDF4 as nc
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import griddata

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

def extract_data_slice(dataset, variable_name, time_index, level_index, lat_indices, lon_indices):    
    # Extract the data slice for the given variable
    return dataset.variables[variable_name][time_index, level_index, lat_indices[0]:lat_indices[-1]+1, lon_indices[0]:lon_indices[-1]+1]

def get_spatial_indices(slice_lats, slice_lons, lat_bounds, lon_bounds):
    # Find indices for latitude and longitude bounds
    lat_indices = np.where((slice_lats >= lat_bounds[0]) & (slice_lats <= lat_bounds[1]))[0]
    lon_indices = np.where((slice_lons >= lon_bounds[0]) & (slice_lons <= lon_bounds[1]))[0]
    return lat_indices, lon_indices

def create_target_grid(slice_data, slice_lats, slice_lons, target_resolution):
    # Create a meshgrid for the original coordinates
    lon_mesh, lat_mesh = np.meshgrid(slice_lons, slice_lats)
    # Flatten the meshgrid for interpolation
    points = np.column_stack((lat_mesh.ravel(), lon_mesh.ravel()))
    values = slice_data.ravel()

    # Generate the target grid
    target_lat = np.arange(slice_lats.min(), slice_lats.max(), target_resolution)
    target_lon = np.arange(slice_lons.min(), slice_lons.max(), target_resolution)
    target_lon_mesh, target_lat_mesh = np.meshgrid(target_lon, target_lat)
    return points, values,target_lat_mesh, target_lon_mesh

def regrid_data(points, values, target_lon_mesh, target_lat_mesh, method='nearest'):
        # Interpolate to the new grid
    regridded_data = griddata(points, values, (target_lon_mesh, target_lat_mesh),method=method)
    return regridded_data

def save_daily_average_to_csv(daily_average, target_lon_mesh, target_lat_mesh, variable_name, target_level, date, output_file):
    # Flatten the latitude and longitude grids
    lat_flat = target_lat_mesh.ravel()
    lon_flat = target_lon_mesh.ravel()
    
    # Flatten the daily average data
    data_flat = daily_average.ravel()
    
    # Tag the variabel with its pressure level
    variable_column_header = f"{variable_name}_{target_level}"
    
    # Create a DataFrame
    df = pd.DataFrame({
        'Date': [date] * len(data_flat),  # Repeat the date for each row
        'Latitude': lat_flat,
        'Longitude': lon_flat,
        variable_column_header: data_flat  # Use the variable name as the column name for data
    })
    
    # Save the DataFrame to CSV
    file_name = f"{output_file}_{date}.csv"
    df.to_csv(file_name, sep='\t', index=False)
    logging.info(f"Data saved to {file_name}")

def create_daily_average_dataset(dataset, variable_name, output_file, target_level, level_index, slice_lats, slice_lons, lat_bounds, lon_bounds, target_resolution):
    # Get indices of geographic region
    lat_indices, lon_indices = get_spatial_indices(slice_lats, slice_lons, lat_bounds, lon_bounds)

    # Convert time variable to datetime objects
    times = nc.num2date(dataset.variables['time'][:], dataset.variables['time'].units)

    # Convert cftime DatetimeGregorian objects to datetime.date
    dates = np.unique([dt.date(time.year, time.month, time.day) for time in times])

    for date in dates:
        # Find time indices for the current day (only considering daytime values)
        time_indices = [i for i, time in enumerate(times) if dt.datetime(time.year, time.month, time.day).date() == date and 6 <= time.hour <= 18]
    
        # Process each time slice for the day
        day_slices = []
        for time_index in time_indices:            
            # Extract slice (assuming a function that handles the extraction)
            slice_data = extract_data_slice(dataset, variable_name, time_index, level_index, lat_indices, lon_indices)

            # Downscale data in slice on to lower-resolution grid
            points, values, target_lat_mesh, target_lon_mesh = create_target_grid(slice_data, slice_lats, slice_lons, target_resolution)
            regridded_slice = regrid_data(points, values, target_lat_mesh, target_lon_mesh)
            day_slices.append(regridded_slice)

        # Compute the daily average
        daily_average = np.mean(day_slices, axis=0)
        
        # Convert daily averages to DataFrame and save to CSV
        save_daily_average_to_csv(daily_average, target_lon_mesh, target_lat_mesh, variable_name, target_level, date, output_file)
    return

def process_single_variable(args):
    variable_name, year, month, output_directory, target_level, lat_bounds, lon_bounds, target_resolution = args
    base_path = Path(f"/bdd/ECMWF/ERA5/NETCDF/GLOBAL_025/hourly/AN_PL/{year}")
    input_file = base_path / f"{variable_name}.{year}{month:02d}.ap1e5.GLOBAL_025.nc"
    output_file = output_directory / f"{variable_name}_{target_level}_daily_1x1"

    if input_file.exists():
        dataset = nc.Dataset(input_file, 'r')
        level_index, slice_lats, slice_lons = prepare_dataset(dataset, target_level, lat_bounds, lon_bounds)
        create_daily_average_dataset(dataset, variable_name, output_file, target_level, level_index, slice_lats, slice_lons, lat_bounds, lon_bounds, target_resolution)
        dataset.close()
    else:
        logging.info(f"File does not exist: {input_file}")

def process_era5_files(variables_dict, start_year, end_year, start_month, end_month, output_directory='/data/pdonnelly/era5/processed_files'):
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    lat_bounds = (30, 60)
    lon_bounds = (300, 360)
    target_resolution = 1

    tasks = []
    for values in variables_dict.values():
        short_name = values['short_name']
        target_levels = values['target_level']
        for target_level in target_levels:
            for year in range(start_year, end_year + 1):
                for month in range(start_month, end_month + 1):
                    tasks.append((short_name, year, month, output_directory, target_level, lat_bounds, lon_bounds, target_resolution))

    with Pool() as pool:
        pool.map(process_single_variable, tasks)

# Define ERA5 variables
variables_dict = {
    "cloud cover": {"short_name": "cc", "target_level": [200, 300]},
    "temperature": {"short_name": "ta", "target_level": [200, 300]},
    "specific humidity": {"short_name": "q", "target_level": [200, 300]},
    "relative humidity": {"short_name": "r", "target_level": [200, 300]},
    "geopotential": {"short_name": "geopt", "target_level": [250, 500]},
    "eastward wind": {"short_name": "u", "target_level": [200, 300]},
    "northward wind": {"short_name": "v", "target_level": [200, 300]},
    "ozone mass mixing ratio": {"short_name": "o3", "target_level": [200, 300]},
}

if __name__ == '__main__':
    process_era5_files(variables_dict, 2018, 2023, 3, 5)
