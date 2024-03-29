import dask
import logging
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

def reduce_fields(input_file, short_name):
    ds = xr.open_dataset(input_file, chunks={})#, 'level':10, 'longitude':360, 'latitude':180})
    
    # Select upper-tropospheric pressures where contrails form and focus on the North Atlantic Ocean (NAO)
    ds_selected = ds[short_name].sel(level=[250], latitude=slice(60, 30), longitude=slice(300, 360), drop=True)
    logging.info("Windowing geographic region")
    
    # Regrid to 1x1 degree using interpolation or nearest-neighbor method
    ds_coarse = ds_selected.coarsen(latitude=4, longitude=4, boundary='trim').mean()
    logging.info("Downscaling spatial grid")

    # Create daily averages
    ds_daily = ds_coarse.resample(time='1D').mean()
    logging.info("Computing daily average")
    
    return ds_daily

@snoop
def convert_dataset_to_dataframe(ds, short_name):
    logging.info("Converting xarray DataSet to pandas DataFrame")

    # Assuming the Dataset has dimensions ('time', 'level', 'latitude', 'longitude')
    # and that 'time' and 'level' are coordinate variables
    time = ds['time'].values
    level = ds['level'].values
    latitude = ds['latitude'].values
    longitude = ds['longitude'].values

    print(ds.variables)

    exit()

    # Get the variable of interest as a numpy array
    variable_data = ds[short_name].values

    # Create meshgrids for each pair of coordinates, resulting in arrays
    # that have the same shape as the variable_data
    T, L, Lat, Lon = np.meshgrid(time, level, latitude, longitude, indexing='ij')

    # Flatten the meshgrids and the data array to create 1D arrays
    T_flat = T.flatten()
    L_flat = L.flatten()
    Lat_flat = Lat.flatten()
    Lon_flat = Lon.flatten()
    variable_flat = variable_data.flatten()

    # Create a DataFrame from the flattened arrays
    df = pd.DataFrame({
        'date': T_flat,
        'pressure': L_flat,
        'latitude': Lat_flat,
        'longitude': Lon_flat,
        short_name: variable_flat
    })

    return df

def save_reduced_fields_to_netcdf(output_file, ds=None):
    # Write to new NetCDF file
    logging.info(f"Saving: {output_file}.nc")
    logging.info(ds.shape)
    ds.to_netcdf(f"{output_file}.nc")

def save_reduced_fields_to_csv(output_file, df):
    logging.info(df.shape)

    # Convert to DataFrame and write to a CSV file
    logging.info(f"Saving: {output_file}.csv")
    df.to_csv(f"{output_file}.csv", sep='\t', index=False)
    
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
                    ds = reduce_fields(input_file, short_name)
                    # save_reduced_fields_to_netcdf(output_file, ds)

                    # Convert xarray Dataset into pandas DataFrame
                    df = convert_dataset_to_dataframe(ds, short_name)
                    save_reduced_fields_to_csv(output_file, df)
                        
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
