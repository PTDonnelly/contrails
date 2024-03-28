import dask.dataframe as dd
import numpy as np
import os
import pandas as pd
from pathlib import Path
import xarray as xr

def process_era5_files(variables_dict, start_year, end_year, start_month, end_month, output_directory='/data/pdonnelly/era5/processed_files'):
    base_path = Path(f"/bdd/ECMWF/ERA5/NETCDF/GLOBAL_025/hourly/AN_PL/{start_year}")
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    
    for long_name, short_name in variables_dict.items():
        for year in range(start_year, end_year + 1):
            for month in range(start_month, end_month + 1):
                # Define filenames
                input_file = base_path / f"{short_name}.{year}{month:02d}.ap1e5.GLOBAL_025.nc"
                output_file = output_directory / f"{short_name}_daily_{year}{month:02d}_1x1"
                    

                if input_file.exists():
                    ds = xr.open_dataset(input_file, chunks={'time':1, 'level':5, 'longitude':200, 'latitude':100})
                    
                    # Select upper-tropospheric pressures where contrails form and focus on the North Atlantic Ocean (NAO)
                    ds_selected = ds[short_name].sel(level=[200, 250, 300],
                                                     latitude=slice(60, 30),
                                                     longitude=slice(300, 360),
                                                     drop=True)
                    print(ds_selected.shape)
                    
                    # Regrid to 1x1 degree using interpolation or nearest-neighbor method
                    ds_coarse = ds_selected.coarsen(latitude=4,
                                                    longitude=4,
                                                    boundary='trim').mean()
                    print(ds_coarse.shape)

                    # Create daily averages
                    ds_daily = ds_coarse.resample(time='1D').mean()
                    print(ds_daily.shape)

                    # Write to new NetCDF file
                    ds_daily.to_netcdf(f"{output_file}.nc")

                    # Read the saved NetCDF file
                    ds_reduced = xr.open_dataset(f"{output_file}.nc")
                    
                    # Convert to DataFrame and write to a CSV file
                    df_reduced = ds_reduced.to_dataframe().reset_index()
                    df_reduced.to_csv(f"{output_file}.csv", sep='\t', index=False)
                    
                    print(f"Processed {output_file}")
                    
                else:
                    print(f"File does not exist: {input_file}")

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
