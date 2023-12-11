import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import lognorm
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

def read_ice_data(base_path, temperature_range, wavelength_range):
    base_path += "efficiency_vs_radius_as_function_of_wavelength\\"
    # Number of temperatures and radii
    num_temperatures = len(temperature_range)
    num_radii = len(wavelength_range)
    # Shape of each scattering array
    num_columns, num_rows = read_dat_file(os.path.join(base_path, f"aerosols_con_230_w7.1.dat")).shape

    # Pre-allocate a 4D numpy array
    ice_data_4d = np.empty((num_columns, num_rows, num_temperatures, num_radii))

    for it, temperature in enumerate(temperature_range):
        for ir, wavelength in enumerate(wavelength_range):
            filename = f"aerosols_con_{temperature:03}_w{wavelength:02}.dat"
            filepath = os.path.join(base_path, filename)
            try:
                # Read the file into a 2D array
                data = read_dat_file(filepath)
                
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

# Read custom ice data
temperature_range = config.get('temperatures')
wavelength_range = [7.1, 8.1, 9.1, 10.1, 11.1, 12.1, 13.1] #config.get('wavelengths')
ice_data = read_ice_data(config.get('aerosol_scattering_directory'), temperature_range, wavelength_range)
n_temperatures, n_wavelengths = ice_data.shape[2], ice_data.shape[3]


# Plotting setup
fig = plt.figure(figsize=(3, 9))
nplots = 3
gs = gridspec.GridSpec(nplots, 1, height_ratios=[1, 1, 1])
fig.suptitle('Scattering Efficiencies of\nWater-Ice Mie Particles', ha='center')

# Generate a list of colors from the colormap, one for each temperature
wavelength_color_values = [plt.cm.jet(x) for x in np.linspace(0, 1, n_wavelengths)]
temperature_color_values = [plt.cm.cividis(x) for x in np.linspace(0, 1, n_temperatures)]

# Define y-axis labels
labels = [r'Particle Size ($\mu$m)', r'$Q_{ext}$',  r'$Q_{sca}$',  r'$Q_{abs}$']

# Loop through and plot each property
for i in range(nplots):
    ax = plt.subplot(gs[i])
   
    # Loop through the ice data and plot
    for itemperature in range(n_temperatures):
        for iwavelength, wavelength in enumerate(wavelength_range):
            # Use the color and alpha for the current temperature and wavelength
            # color = temperature_color_values[itemperature]
            color = wavelength_color_values[iwavelength]

            # Get radius range
            xdata = ice_data[0, :, itemperature, iwavelength]

            # Get scattering properties
            ydata = ice_data[i+1, :, itemperature, iwavelength]

            # Convert scattering cross-section into efficiency
            D = 2 * xdata
            ydata = (4 * ydata ) / (np.pi * D**2)

            # Plotting the i-th property across all wavelengths for a specific temperature and wavelength
            ax.plot(xdata, ydata, color=color, alpha=0.75, label=fr"$\lambda$={int(wavelength-0.1)} $\mu$m")

    # Hide x-axis labels and ticks for all but the bottom subplot
    if i < nplots-1:
        if 2 <= i <= 4:
            ax.set_yscale('log')
        ax.tick_params(labelbottom=False, bottom=False, top=False)  # Also hides the ticks themselves
    elif i == nplots-1:
        ax.set_xlabel(labels[0])
    ax.set_xlim(1, 15)
    ax.set_xscale('log')
    ax.set_ylabel(labels[i+1])
    if i == 0:
        ax.legend(fontsize=7, loc='lower right')
        

plt.subplots_adjust(hspace=0.1)
# plt.tight_layout()

# Save figure
plt.savefig(f"{config.get('aerosol_scattering_directory')}efficiency_comparison.png", dpi=300, bbox_inches='tight')