import numpy as np
import os
import pandas as pd
from pathlib import Path
import xarray as xr

def process_era5_files(variables_dict, start_year, end_year, start_month, end_month, output_directory):
    base_path = Path(f"/bdd/ECMWF/ERA5/NETCDF/GLOBAL_025/hourly/AN_PL/{start_year}")
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    
    for long_name, short_name in variables_dict.items():
        for year in range(start_year, end_year + 1):
            for month in range(start_month, end_month + 1):
                file_path = base_path / f"{short_name}.{year}{month:02d}.ap1e5.GLOBAL_025.nc"
                
                if file_path.exists():
                    ds = xr.open_dataset(file_path, chunks={})
                    
                    print("Dataset dimensions before slicing:", ds.dims)
                    print("Coordinate values before slicing:")
                    print("Latitude range:", ds['latitude'].min().data, "to", ds['latitude'].max().data)
                    print("Longitude range:", ds['longitude'].min().data, "to", ds['longitude'].max().data)

                    input()
                    # Select upper-tropospheric pressures where contrails form and focus on the North Atlantic Ocean (NAO)
                    ds_selected = ds[short_name].sel(level=[200, 250, 300],
                                         latitude=slice(30, 60),
                                         longitude=slice(-60, 0))
                    print(ds_selected.shape)
                    input()
                    # Regrid to 1x1 degree using interpolation or nearest-neighbor method
                    ds_coarse = ds_selected.coarsen(latitude=4,
                                                    longitude=4,
                                                    boundary='trim').mean()  # Example coarsening
                    
                    print(ds_coarse.shape)
                    input()

                    # Create daily averages
                    ds_daily = ds_coarse.resample(time='1D').mean()
                    
                    print(ds_daily.shape)
                    input()

                    # Convert all data points to csv format
                    df_daily = ds_daily.to_dataframe().reset_index()

                    # Write to new NetCDF file
                    output_file = output_directory / f"{short_name}_daily_{year}{month:02d}_1x1.csv"
                    
                    print(df_daily.columns)
                    input()
                    df_daily.to_csv(output_file, index=False)
                    
                    print(f"Processed {output_file}")
                    
                else:
                    print(f"File does not exist: {file_path}")

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
process_era5_files(variables_dict, 2018, 2018, 3, 3, './processed_files')
