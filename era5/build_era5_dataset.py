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
                    
                    # Select upper-tropospheric pressures where contrails form
                    pressure_levels = [200, 250, 300]
                    ds_selected = ds.sel(level=pressure_levels)
                    
                    # Regrid to 1x1 degree using interpolation or nearest-neighbor method
                    ds_coarse = ds_selected.coarsen(latitude=4, longitude=4, boundary='trim').mean()  # Example coarsening
                    
                    # Create daily averages
                    ds_daily = ds_coarse.resample(time='1D').mean()
                    
                    # Write to new NetCDF file
                    output_file = output_directory / f"{short_name}_daily_{year}{month:02d}_1x1.nc"
                    
                    print(np.shape(ds_daily))
                    input()
                    ds_daily.to_netcdf(output_file)
                    
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
