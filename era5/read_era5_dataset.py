import xarray as xr

# Replace the path below with the path to your NetCDF file
file_path = "C:\\Users\\donnelly\\Documents\\projects\\data\\r.201303.aphe5.GLOBAL_1DD.nc"

# Open the NetCDF file
data = xr.open_dataset(file_path)

# Print the data to get an overview of the dataset's structure and variables
print(data)

