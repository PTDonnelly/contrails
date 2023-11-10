import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import logging
import json

def read_dat_file(filepath, end_line=None):
    data = []
    with open(filepath, 'r') as file:
        lines = file.readlines()
        # Start reading from line 18, end at `end_line` if specified
        content_lines = lines[17:end_line] if end_line else lines[17:]

        for line in content_lines:
            # Check if line starts with '#' and is not just '#'
            if line.strip().startswith('#') and line.strip() != '#':
                # Remove the '#' and leading/trailing whitespaces
                clean_line = line.strip()[1:].strip()
                # Split the line by whitespace and convert to floats
                float_values = list(map(float, clean_line.split()))
                data.append(float_values)

    return np.transpose(np.array(data))

def read_baum_data(filename):
    """Function to read the Baum data from .dat file, cut off phase matrices beyond line 78"""
    data = read_dat_file(filename, end_line=78)
    data[0, :] = np.divide(1e4, data[0, :]) # Convert to wavenumbers
    return data

def read_ice_data(base_path, temperature_range, radius_range):
    # Number of temperatures and radii
    num_temperatures = len(temperature_range)
    num_radii = len(radius_range)
    # Shape of each scattering array
    num_columns, num_rows = read_dat_file(os.path.join(base_path, f"aerosols_con_160_01.dat")).shape

    # Pre-allocate a 4D numpy array
    ice_data_4d = np.empty((num_columns, num_rows, num_temperatures, num_radii))

    for it, temperature in enumerate(temperature_range):
        for ir, radius in enumerate(radius_range):
            filename = f"aerosols_con_{temperature:03}_{radius:02}.dat"
            filepath = os.path.join(base_path, filename)
            try:
                # Read the file into a 2D array
                data = read_dat_file(filepath)
                data[0, :] = np.divide(1e4, data[0, :]) # Convert to wavenumbers
                
                # Check if the file has the expected shape
                if data.shape == (num_columns, num_rows):
                    ice_data_4d[:, :, it, ir] = data
                else:
                    raise ValueError(f"Data shape mismatch in file {filename}")
                
            except (FileNotFoundError, ValueError) as e:
                print(e)
                # Fill with NaNs if the file is not found or shape mismatch
                ice_data_4d[:, :, it, ir] = np.nan

    return ice_data_4d

# Constants
CONFIG_FILE = "aerosol_config.json"

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_config():
    """Load configuration from a JSON file."""
    with open(CONFIG_FILE, 'r') as f:
        return json.load(f)

config = get_config()

# Read the Baum data
baum_data = read_baum_data(f"{config.get('aerosol_scattering_directory')}aerosols_baum00.dat")

# Read custom ice data
temperature_range = range(160, 271, 10)
radius_range = [1, 2, 5]#, 10]
ice_data = read_ice_data(config.get('aerosol_scattering_directory'), temperature_range, radius_range)
n_temperatures, n_radii = ice_data.shape[2], ice_data.shape[3]

# Plotting setup
fig = plt.figure(figsize=(3, 7))
gs = gridspec.GridSpec(6, 1, height_ratios=[1]*6)

# Create a colormap object
cmap = plt.cm.cividis

# Generate a list of colors from the colormap, one for each temperature
temperature_color_values = [cmap(x) for x in np.linspace(0, 1, n_temperatures)]

# Define y-axis labels
labels = [r'Wavenumnber (cm$^{-1}$)', r'$\sigma_{e}$',  r'$\sigma_{s}$',  r'$\sigma_{a}$',  r'$ssa$',  r'$g$',  r'$\sigma_{e,n}$',  r'$n$',  r'$k$']  

# Loop through and plot each property
for i in range(6):
    ax = plt.subplot(gs[i])

    # Baum data
    ax.plot(baum_data[0, :], baum_data[i+1, :], color='r')

    # Loop through the ice data and plot
    for temperature in range(n_temperatures):
        for radius in range(n_radii):
            # Use the color and alpha for the current temperature and radius
            color = temperature_color_values[temperature]
            # Plotting the i-th property across all wavelengths for a specific temperature and radius
            ax.plot(ice_data[0, :, temperature, radius], ice_data[i+1, :, temperature, radius], color=color, alpha=0.75)

    # Hide x-axis labels and ticks for all but the bottom subplot
    if i < 5:
        ax.tick_params(labelbottom=False, bottom=False, top=False)  # Also hides the ticks themselves
    elif i == 5:
        ax.set_xlabel(labels[0])
        # plt.xticks(np.arange(7, 13.1, 1))
 
    ax.set_ylabel(labels[i+1])
    ax.set_xlim(1200, 800)

plt.subplots_adjust(hspace=0)
plt.tight_layout()

# Save figure
plt.savefig(f"{config.get('aerosol_scattering_directory')}xsc_comparison.png", dpi=300, bbox_inches='tight')