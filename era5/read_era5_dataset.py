import xarray as xr
import pandas as pd
import numpy as np

# Open the NetCDF file
file_path = "C:\\Users\\donnelly\\Documents\\projects\\data\\20130301_atlantic.nc"
ds = xr.open_dataset(file_path)

print(ds)

exit()

# Extract the temperature at the 10th altitude index
# Select the temperature at the 10th altitude index
temperature = ds['temperature'].isel(altitude=9)

# Group by latitude and longitude bins and calculate the mean for each bin
lat_bins = np.arange(temperature.latitude.min(), temperature.latitude.max() + 1, 1)
lon_bins = np.arange(temperature.longitude.min(), temperature.longitude.max() + 1, 1)

grouped = temperature.groupby_bins("latitude", lat_bins).groupby_bins("longitude", lon_bins).mean()