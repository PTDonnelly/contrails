import numpy as np
import pandas as pd
import os

# Define the radii and temperatures
radii = [1, 2, 3, 4, 5]#, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
temperatures = [160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270]

# Assuming each file has the same number of data points, 
# we'll determine this number by reading the first file
sample_file = f"ice_sphere_{temperatures[0]}K_{radii[0]:03}um.dat"
sample_path = os.path.join("C:\\Users\\padra\\Documents\\Research\\projects\\contrails\\scattering_calculations\\scattering_data\\", sample_file)

sample_data = pd.read_csv(sample_path, delimiter='\t', skiprows=1, header=None)
n_lambda = sample_data.shape[0]
n_params = sample_data.shape[1]

# Initialize the 3D array with NaNs to allow for missing files/data
data = np.full((len(radii), len(temperatures), n_lambda, n_params), np.nan)

# Fill the 3D array with data
for i, radius in enumerate(radii):
    for j, temp in enumerate(temperatures):
        file_name = f"ice_sphere_{temp}K_{radius:03}um.dat"
        file_path = os.path.join("C:\\Users\\padra\\Documents\\Research\\projects\\contrails\\scattering_calculations\\scattering_data\\", file_name)
                
        if os.path.exists(file_path):
            # Read the file and nsert the data into the array
            data[i, j, :, :] = pd.read_csv(file_path, delimiter='\t', skiprows=1, header=None)    
        else:
            print(f"File {file_name} not found.")

