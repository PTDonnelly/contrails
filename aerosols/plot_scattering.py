import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Directory containing scattering data
scattering_dir = "C:\\Users\\padra\\Documents\\Research\\projects\\contrails\\scattering_calculations\\scattering_data\\"

def plot_scattering_vs_temperature():

    # List of particle shapes
    shapes = ["spheroid"]
    # List of temperature regimes (from Iwabuchi and Yang, 2011)
    temperatures = [(160 + i * 10) for i in range(12)]
    n_temps = len(temperatures)
    # List of oblateness/prolateness of the particles
    axis_ratios = [1.0]
    # List of radii of the particles in microns
    radii = [5]
    # Get wavelength and parameter grid
    wavelengths = np.loadtxt(f"spectral_grid.txt")
    n_wavelengths = len(wavelengths)

    for shape in shapes:
        for axis_ratio in axis_ratios:
            for radius in radii:
                # Initialize the figure and the subplots
                fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True, dpi=300)

                # Set the labels for each subplot
                labels = [r"$Q_{ext}$", "$\omega$", "g"]
                n_params = len(labels)

                # Set up the colormap
                colors = cm.seismic(np.linspace(0, 1, n_temps))

                # Create empty array to store data
                data = np.empty((n_temps, n_wavelengths, n_params))

                # Iterate through the list of radii
                for idx, temperature in enumerate(temperatures):
                    contents = np.loadtxt(f"{scattering_dir}{shape}_T_{temperature}_AR_{axis_ratio}_radius_{radius}.txt", skiprows=1)
                    data[idx, :, :] = contents[:, 4:]
                
                # Extract and plot each column
                for iax, (ax, label) in enumerate(zip(axes, labels)):
                    properties = [data[isize, :, iax] for isize in range(n_temps)]

                    # Plot the data in each subplot
                    for temp, prop, color in zip(temperatures, properties, colors):
                        ax.plot(wavelengths, prop, label=f"T = {temp} K", color=color)
                        ax.set_ylabel(label)
                        ax.grid(axis='both', markevery=1, color='gray', lw=0.5, ls='--')

                # Set the x-axis label for the last subplot
                axes[-1].set_xlabel("Wavelength (μm)")

                # Add a legend
                plt.legend(loc='center left', fontsize=7, bbox_to_anchor=(1, 0.56*n_params))

                # Adjust the layout and display the figure
                # plt.tight_layout()
                plt.xlim(7, 15)
                plt.savefig(f"{scattering_dir}{shape}_AR_{axis_ratio}_radius_{radius}_thermal_scattering_properties.png", dpi=300, bbox_inches='tight')

def plot_scattering_vs_radius():#

    
    shapes = ["spheroid"]#, "cylinder"]
    # Define the oblateness/prolateness of the particles
    axis_ratios = [0.3, 1.0, 3.0]

    # List of radii of the particles in microns
    radii = [float(i+1) for i in range(10)]
    n_sizes = len(radii)

    # Get wavelength and parameter grid
    wavelengths = np.loadtxt(f"spectral_grid.txt")
    n_wavelengths = len(wavelengths)

    for shape in shapes:
        for axis_ratio in axis_ratios:
            
            # Initialize the figure and the subplots
            fig, axes = plt.subplots(6, 1, figsize=(6, 8), sharex=True, dpi=300)

            # Set the labels for each subplot
            labels = ["I_scat", "LDR", "x_scat", "x_ext", "ssa", "asym"]
            n_params = len(labels)

            # Set up the colormap
            colors = cm.turbo(np.linspace(0, 1, n_sizes))

            # Create empty array to store data
            data = np.empty((n_sizes, n_wavelengths, n_params))

            # Iterate through the list of radii
            for idx, radius in enumerate(radii):
                contents = np.loadtxt(f"{scattering_dir}{shape}_AR_{axis_ratio}_radius_{radius}.txt", skiprows=1)
                data[idx, :, :] = contents[:, 1:]
            
            # Extract and plot each column
            for iax, (ax, label) in enumerate(zip(axes, labels)):
                properties = [data[isize, :, iax] for isize in range(n_sizes)]

                # Plot the data in each subplot
                for prop, color in zip(properties, colors):
                    ax.plot(wavelengths, prop, label=f"Radius: {radius} μm", color=color)
                    ax.set_ylabel(label)
                    ax.grid(axis='both', markevery=1, color='gray', lw=0.5, ls='--')

            # Set the x-axis label for the last subplot
            axes[-1].set_xlabel("Wavelength (μm)")

            # # Add a legend to the last subplot
            # axes[-1].legend()

            # Adjust the layout and display the figure
            plt.tight_layout()
            plt.xlim(7, 15)
            plt.savefig(f"{scattering_dir}{shape}_AR_{axis_ratio}_scattering_properties.png", dpi=300)


plot_scattering_vs_temperature()