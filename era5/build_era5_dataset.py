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
                # Define filenames
                input_file = base_path / f"{short_name}.{year}{month:02d}.ap1e5.GLOBAL_025.nc"
                output_file = output_directory / f"{short_name}_daily_{year}{month:02d}_1x1"
                    

                if input_file.exists():
                    ds = xr.open_dataset(input_file, chunks={})
                    
                    # Select upper-tropospheric pressures where contrails form and focus on the North Atlantic Ocean (NAO)
                    ds_selected = ds[short_name].sel(level=[200, 250, 300],
                                                     latitude=slice(60, 30),
                                                     longitude=slice(300, 360))
                    
                    # Regrid to 1x1 degree using interpolation or nearest-neighbor method
                    ds_coarse = ds_selected.coarsen(latitude=4,
                                                    longitude=4,
                                                    boundary='trim').mean()
                    
                    
                    # Create daily averages
                    ds_daily = ds_coarse.resample(time='1D').mean()
                    
                    for time_slice in ds_daily.time:
                        ds_slice = ds_daily.sel(time=time_slice)
                        print(ds_slice.shape)
                        df_slice = ds_slice.to_dataframe().reset_index()
                        # Append to CSV
                        with open(f"{output_file}.csv", 'a') as f:
                            df_slice.to_csv(f, header=f.tell()==0, index=False)

                    # # Write to new NetCDF file
                    # # ds_daily.to_netcdf(f"{output_file}.nc")

                    # # # Read the saved NetCDF file
                    # # ds_reduced = xr.open_dataset(f"{output_file}.nc", chunks={})
                    # # Convert to DataFrame
                    # df_reduced = ds_reduced.to_dataframe().reset_index()
                    # # Write the DataFrame to a CSV file
                    # df_reduced.to_csv(f"{output_file}.csv", index=False)
                    
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
process_era5_files(variables_dict, 2018, 2018, 3, 3, './processed_files')
