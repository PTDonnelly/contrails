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
                floats = list(map(float, clean_line.split()))
                data.append(floats)

    return np.transpose(np.array(data))

def read_ice_data(base_path, temperature_range, radius_range, shape_range, AR_range):
    # Number of temperatures and radii
    num_temperatures = len(temperature_range)
    num_radii = len(radius_range)
    num_shapes = len(shape_range)
    num_ARs = len(AR_range)

    # Shape of each scattering array
    num_columns, num_rows = read_dat_file(os.path.join(base_path, f"aerosols_con_230_SHAPE_SPHEROID_AR_1.0.dat")).shape

    # Pre-allocate a 4D numpy array
    ice_data = np.empty((num_columns, num_rows, num_temperatures, num_shapes, num_ARs))

    for it, temperature in enumerate(temperature_range):
        for ishape, shape in enumerate(shape_range):
            for iar, AR in enumerate(AR_range):
                filename = f"aerosols_con_{temperature:03}_{shape}_AR_{AR}.dat"
                filepath = os.path.join(base_path, filename)
                try:
                    # Read the file into a 2D array
                    data = read_dat_file(filepath)
                    
                    # Check if the file has the expected shape
                    if data.shape == (num_columns, num_rows):
                        ice_data[:, :, it, ishape, iar] = data
                    else:
                        raise ValueError(f"Data shape mismatch in file {filename}")
                except (FileNotFoundError, ValueError) as e:
                    print(e)
                    # Fill with NaNs if the file is not found or shape mismatch
                    ice_data[:, :, it, ishape, iar] = np.nan
    return ice_data

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
radius_range = config.get('radii')
shape_range = config.get('shape_ids')
AR_range = config.get('axis_ratios')
ice_data = read_ice_data(config.get('aerosol_scattering_directory'), temperature_range, radius_range, shape_range, AR_range)
n_temperatures, n_shapes, n_ARs = ice_data.shape[2], ice_data.shape[3], ice_data.shape[4]


# Plotting setup
fig = plt.figure(figsize=(3, 9))
nplots = 3
gs = gridspec.GridSpec(nplots, 1, height_ratios=[1, 1, 1])
fig.suptitle('Scattering Properties of\nWater-Ice Mie Particles at 8 $\\mu$m', y=0.95, ha='center')

# Generate a list of colors from the colormap, one for each temperature
# temperature_colors = [plt.cm.cividis(x) for x in np.linspace(0, 1, n_temperatures)]
shape_linestyles = ['-', ':']
AR_colors = ['royalblue', 'green', 'orange']

# Define y-axis labels
labels = [r'Mean Radius ($\mu$m)', r'$Q_{ext}$',  r'$Q_{abs}$', '$g$']

# Loop through and plot each property
for i in range(nplots):
    ax = plt.subplot(gs[i])
   
    # Loop through the ice data and plot
    for itemperature in range(n_temperatures):
        # color = temperature_colors[itemperature]

        for ishape, shape in enumerate(shape_range):
            linestyle = shape_linestyles[ishape]
            shape_name = shape.split('_')
            shape_name = shape_name[-1].lower()

            for iAR, AR in enumerate(AR_range):
                color = AR_colors[iAR]

                # Get radius range
                xdata = ice_data[0, :, itemperature, ishape, iAR]

                if 0 <= i <= 1:
                    # Get scattering properties
                    if i == 0:
                        ydata = ice_data[1, :, itemperature, ishape, iAR]
                        ax.set_ylim(0, 4)
                    elif i == 1:
                        ydata = ice_data[3, :, itemperature, ishape, iAR]
                        ax.set_ylim(0, 1.5)
                    # Convert scattering cross-section into efficiency
                    D = 2 * xdata
                    ydata = (4 * ydata) / (np.pi * D**2)
                elif i == 2:
                    ydata = ice_data[5, :, itemperature, ishape, iAR]
                    ax.set_ylim(0, 1)

                # Plotting the i-th property across all wavelengths for a specific temperature and wavelength
                ax.plot(xdata, ydata, ls=linestyle, color=color, label=f"{shape_name}, AR={AR}")

    # Hide x-axis labels and ticks for all but the bottom subplot
    if i < nplots-1:
        ax.tick_params(labelbottom=False, bottom=False, top=False)  # Also hides the ticks themselves
    elif i == nplots-1:
        ax.set_xlabel(labels[0])
        ax.legend(fontsize=8, loc='lower right')
    ax.set_xlim(1, 1e4)
    ax.set_xscale('log')
    ax.set_ylabel(labels[i+1])

plt.xticks((1e0, 1e1, 1e2, 1e3, 1e4))
        

plt.subplots_adjust(hspace=0.1)
# plt.tight_layout()

# Save figure
plt.savefig(f"{config.get('aerosol_scattering_directory')}efficiency_comparison.png", dpi=300, bbox_inches='tight')