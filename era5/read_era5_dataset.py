import xarray as xr
import pandas as pd
import numpy as np

# Open the NetCDF file
file_path = "C:\\Users\\donnelly\\Documents\\projects\\data\\20130301_atlantic.nc"
ds = xr.open_dataset(file_path)

# Extract the temperature
temperature = ds['t']

print(temperature)

exit()