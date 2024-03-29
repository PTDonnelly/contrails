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

# def reduce_fields(input_file, short_name):
#     ds = xr.open_dataset(input_file, chunks={})#, 'level':10, 'longitude':360, 'latitude':180})
    
#     # Select upper-tropospheric pressures where contrails form and focus on the North Atlantic Ocean (NAO)
#     ds_selected = ds[short_name].sel(level=[250], latitude=slice(60, 30), longitude=slice(300, 360), drop=True)
#     logging.info("Windowing geographic region")
    
#     # Regrid to 1x1 degree using interpolation or nearest-neighbor method
#     ds_coarse = ds_selected.coarsen(latitude=4, longitude=4, boundary='trim').mean()
#     logging.info("Downscaling spatial grid")

#     # Create daily averages
#     ds_daily = ds_coarse.resample(time='1D').mean()
#     logging.info("Computing daily average")
    
#     return ds_daily

# @snoop
# def convert_slice_to_dataframe(time_value, level_value, latitude, longitude, variable_slice, variable_name):
#     """
#     Converts a slice of the dataset for a specific time and level to a DataFrame.

#     Parameters:
#     - time_value: The specific time value of the slice.
#     - level_value: The specific level value of the slice.
#     - latitude: 1D array of latitude values.
#     - longitude: 1D array of longitude values.
#     - variable_slice: 2D array (latitude x longitude) of the variable values for the slice.
#     - variable_name: Name of the variable being processed.

#     Returns:
#     - df: A pandas DataFrame representing the slice.
#     """
#     # Create meshgrids for latitude and longitude, resulting in arrays
#     Lat, Lon = np.meshgrid(latitude, longitude, indexing='ij')
    
#     # Flatten the meshgrids and the variable slice to create 1D arrays
#     Lat_flat = Lat.flatten()
#     Lon_flat = Lon.flatten()
#     variable_flat = variable_slice.flatten()

#     # Create a DataFrame from the flattened arrays
#     df = pd.DataFrame({
#         'date': np.repeat(time_value, len(Lat_flat)),
#         'pressure': np.repeat(level_value, len(Lon_flat)),
#         'latitude': Lat_flat,
#         'longitude': Lon_flat,
#         variable_name: variable_flat
#     })

#     return df


# def save_reduced_fields_to_netcdf(output_file, ds=None):
#     # Write to new NetCDF file
#     logging.info(f"Saving: {output_file}.nc")
#     logging.info(ds.shape)
#     ds.to_netcdf(f"{output_file}.nc")

# def save_reduced_fields_to_csv(output_file, df):
#     logging.info(df.shape)

#     # Convert to DataFrame and write to a CSV file
#     logging.info(f"Saving: {output_file}.csv")
#     df.to_csv(f"{output_file}.csv", mode='a', sep='\t', index=False)

# def process_and_save_to_csv(ds, variable_name, output_file):
#     logging.info("Processing and saving dataset to CSV")

#     # Ensure no existing file conflicts
#     output_csv = f"{output_file}.csv"
#     if os.path.exists(output_csv):
#         os.remove(output_csv)

#     for time_value in ds['time'].values:
#         for level_value in ds['level'].values:
#             # Select the slice for the current time and level
#             variable_slice = ds.sel(time=time_value, level=level_value).values

#             # Convert the slice to a DataFrame
#             df_slice = convert_slice_to_dataframe(time_value, level_value, ds['latitude'].values, ds['longitude'].values, variable_slice, variable_name)

#             # Append this slice to the CSV file
#             save_reduced_fields_to_csv(output_csv, df_slice)

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

def process_variable(input_file, output_file, variable_name, target_level=250):
    """
    Processes a specified variable for a given pressure level across all time steps.
    """
    with nc.Dataset(input_file, 'r') as dataset:
        # Extract dimensions
        times = dataset.variables['time'][:]
        latitudes = dataset.variables['latitude'][:]
        longitudes = dataset.variables['longitude'][:]
        levels = dataset.variables['level'][:]
        
        # Convert target level to index
        level_index = np.where(levels == target_level)[0]
        if not level_index:
            logging.warning(f"Level {target_level} not found in the dataset.")
            return

        time_unit = dataset.variables['time'].units

        for time_idx in range(len(times)):
            time_value = nc.num2date(times[time_idx], units=time_unit)
            variable_slice = dataset.variables[variable_name][time_idx, level_index, :, :].squeeze()

            # Save each slice to CSV
            save_slice_to_csv(time_value, target_level, latitudes, longitudes, variable_slice, variable_name, output_file)
    
def process_era5_files(variables_dict, start_year, end_year, start_month, end_month, output_directory='/data/pdonnelly/era5/processed_files'):
    base_path = Path(f"/bdd/ECMWF/ERA5/NETCDF/GLOBAL_025/hourly/AN_PL/{start_year}")
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    
    for short_name in variables_dict.values():
        for year in range(start_year, end_year + 1):
            for month in range(start_month, end_month + 1):
                # Define filenames
                input_file = base_path / f"{short_name}.{year}{month:02d}.ap1e5.GLOBAL_025.nc"
                output_file = output_directory / f"{short_name}_daily_{year}{month:02d}_1x1"
                    

                if input_file.exists():
                    # Read and reduce atmospheric data, store in xarray DataSet
                    # ds = reduce_fields(input_file, short_name)
                    # # save_reduced_fields_to_netcdf(output_file, ds)

                    # # Convert xarray Dataset into pandas DataFrame
                    # # df = convert_dataset_to_dataframe(ds, short_name)
                    # # save_reduced_fields_to_csv(output_file, df)

                    # process_and_save_to_csv(ds, short_name, output_file)

                    process_variable(input_file, output_file, short_name, target_level=250)

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
