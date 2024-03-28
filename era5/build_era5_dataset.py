import numpy as np
import os
import pandas as pd
from pathlib import Path
import xarray as xr

def process_era5_files(variable_name, start_year, end_year, start_month, end_month, output_directory):
    base_path = Path(f"/bdd/ERA5/NETCDF/GLOBAL_025/hourly/AN_PL/{start_year}")
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    
    for year in range(start_year, end_year + 1):
        for month in range(start_month, end_month + 1):
            file_path = base_path / f"{variable_name}.{year}{month:02d}.aphe5.GLOBAL_025.nc"
            
            if file_path.exists():
                ds = xr.open_dataset(file_path)
                
                # Select the 10th altitude index
                ds_selected = ds.isel(level=9)  # Indexing starts at 0
                
                # Regrid to 1x1 degree using interpolation or nearest-neighbor method
                ds_coarse = ds_selected.coarsen(lat=4, lon=4, boundary='trim').mean()  # Example coarsening
                
                # Create daily averages
                ds_daily = ds_coarse.resample(time='1D').mean()
                
                # Write to new NetCDF file
                output_file = os.path.join(output_directory, f"{variable_name}_daily_{year}{month:02d}_1x1.nc")
                ds_daily.to_netcdf(output_file)
                
                print(f"Processed {output_file}")
                
            else:
                print(f"File does not exist: {file_path}")

# Execute on specified date range
process_era5_files('t', 2018, 2018, 3, 3, './processed_files')
