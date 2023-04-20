import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Directory containing scattering data
scattering_dir = "scattering_data/"
shapes = ["spheroid", "cylinder"]
# Define the oblateness/prolateness of the particles
axis_ratios = [0.5, 1.0, 1.5]

# List of radii of the particles in microns
radii = [0.5, 1.0, 2.0, 3.0]
n_sizes = len(radii)

# Get wavelength and parameter grid
grid = np.loadtxt(f"spectral_grid.txt")
n_wavelengths = len(grid)

for shape in shapes:
    for axis_ratio in axis_ratios:
        
        # Initialize the figure and the subplots
        fig, axes = plt.subplots(6, 1, figsize=(6, 8), sharex=True, dpi=300)

        # Set the labels for each subplot
        labels = ["I_scat", "LDR", "x_scat", "x_ext", "ssa", "asym"]
        n_params = len(labels)

        # Set up the colormap
        colors = cm.jet(np.linspace(0, 1, n_params))

        # Create empty array to store data
        data = np.empty((n_sizes, n_wavelengths, n_params+1))

        # Iterate through the list of radii
        for idx, radius in enumerate(radii):
            data[idx, :, :] = np.loadtxt(f"{scattering_dir}{shape}_radius_{radius}_AR_{axis_ratio}.txt", skiprows=1)

        for iax, (ax, label) in enumerate(zip(axes, labels)):
            # Extract the columns
            wavelength = data[0, :, 0]
            properties = [data[isize, :, iax] for isize in range(n_sizes)]

            # Plot the data in each subplot
            for prop, color in zip(properties, colors):
                ax.plot(wavelength, prop, label=f"Radius: {radius} μm", color=color)
                ax.set_ylabel(label)

        # Set the x-axis label for the last subplot
        axes[-1].set_xlabel("Wavelength (μm)")

        # # Add a legend to the last subplot
        # axes[-1].legend()

        # Adjust the layout and display the figure
        plt.tight_layout()
        plt.xlim(7, 15)
        plt.savefig(f"{scattering_dir}{shape}_AR_{axis_ratio}_scattering_properties.png", dpi=300)