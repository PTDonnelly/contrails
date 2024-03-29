import dask.dataframe as dd
import logging
import numpy as np
import os
import pandas as pd
from pathlib import Path
import xarray as xr

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def reduce_fields(input_file, short_name):
    ds = xr.open_dataset(input_file, chunks={'time':1, 'level':10, 'longitude':360, 'latitude':180})
    
    # Select upper-tropospheric pressures where contrails form and focus on the North Atlantic Ocean (NAO)
    ds_selected = ds[short_name].sel(level=[200, 250, 300], latitude=slice(60, 30), longitude=slice(300, 360), drop=True)
    logging.info("Windowing geographic region")
    
    # Regrid to 1x1 degree using interpolation or nearest-neighbor method
    ds_coarse = ds_selected.coarsen(latitude=4, longitude=4, boundary='trim').mean()
    logging.info("Downscaling spatial grid")

    # Create daily averages
    ds_daily = ds_coarse.resample(time='1D').mean()
    logging.info("Computing daily average")
    
    return ds_daily

def save_reduced_fields_to_netcdf(output_file, ds=None):
    # Write to new NetCDF file
    logging.info(f"Saving: {output_file}.nc")
    logging.info(ds.shape)
    ds.to_netcdf(f"{output_file}.nc")

def save_reduced_fields_to_csv(output_file, ds=None):
    if not ds:
        # Read the saved NetCDF file
        ds = xr.open_dataset(f"{output_file}.nc")
    logging.info(ds.shape)
    
    # Convert to DataFrame and write to a CSV file
    df = ds.to_dataframe().reset_index()
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
                    # Read and reduce atmospheric data
                    ds_reduced = reduce_fields(input_file, short_name)

                    save_reduced_fields_to_netcdf(output_file, ds_reduced)
                    save_reduced_fields_to_csv(output_file, ds_reduced)
                        
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
