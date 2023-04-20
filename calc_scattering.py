import gc
import numpy as np
import os
import pytmatrix.tmatrix as tmatrix
import pytmatrix.scatter as scatter
from scipy.interpolate import interp1d
import snoop
from sorcery import dict_of

def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory

def make_spectral_grid():
    """Define the wavelength grid for scattering calculations (minimum, maximum, and interval in microns)"""
    grid = 7, 15, 0.1
    return np.arange(grid[0], grid[1], grid[2], dtype=np.float64)

def save_spectral_grid(wavelengths):
    """Open a file to save the scattering properties"""
    with open(f"spectral_grid.txt", "w") as f:
        for wavelength in wavelengths:
            f.write(f"{wavelength:.4f}\n")
    return

def get_spectral_grid():
    """Define the spectral grid for scattering calculations. If it doesn't exist, create and save it."""
    wavelengths = make_spectral_grid()
    if not os.path.exists("spectral_grid.txt"):
        save_spectral_grid(wavelengths)
    return wavelengths

def get_particle():
    
    # Define the properties of the spherical particle
    radii = [0.5, 1.0, 2.0, 3.0]  # List of radii of the particles in microns
    
    # Define paricle shapes
    shapes = ["spheroid", "cylinder"]
    shape_ids = [tmatrix.Scatterer.SHAPE_SPHEROID, tmatrix.Scatterer.SHAPE_CYLINDER]

    # Define the oblateness/prolateness of the particles
    axis_ratios = [0.5, 1.0, 1.5]

    return dict_of(radii, shapes, shape_ids, axis_ratios)

def get_refractive_index(ref_wavelength: float):
    """Manually set or read from file the complex refractive index of the particle."""
    
    # Read the data into a numpy array
    data_array = np.loadtxt("water_ice_refractive_index.csv", delimiter=',', skiprows=1)

    # Extract the columns
    wavelength = data_array[:, 0]
    refractive_index = data_array[:, 1]
    extinction_cross_section = data_array[:, 2]

    # Create interpolation objects for refractive_index, n, and extinction_cross_section, k
    n_interp = interp1d(wavelength, refractive_index)
    k_interp = interp1d(wavelength, extinction_cross_section)
    
    real = n_interp(ref_wavelength)
    imaginary = k_interp(ref_wavelength)
    return [complex(r, i) for r, i in zip(real, imaginary)]

def get_scattering_parameters(scatterer: object, filename: object, wavelength: float):
                        
    # Calculate the scattering and absorption cross-sections
    sca_intensity = scatter.sca_intensity(scatterer)  # Scattering intensity (phase function)
    ldr = scatter.ldr(scatterer)  # Linear depolarizarion ratio
    sca_xsect = scatter.sca_xsect(scatterer)  # Scattering cross-section
    ext_xsect = scatter.ext_xsect(scatterer)  # Extinction cross-section
    ssa = scatter.ssa(scatterer)  # Single-scattering albedo
    asym = scatter.asym(scatterer)  # Asymmetry parameter

    # Write the scattering properties to the file
    filename.write(f"{wavelength:.8e}\t{sca_intensity:.8e}\t{ldr:.8e}\t{sca_xsect:.8e}\t{ext_xsect:.8e}\t{ssa:.8e}\t{asym:.8e}\n")

def main():
    
    # Define the spectral grid for scattering calculations
    wavelengths = get_spectral_grid()

    # Define particle parameters for scattering calculations
    particle = get_particle()

    # Create the complex refractive index look-up table 
    refractive_indices = get_refractive_index(wavelengths)

    # Create directory to store scattering data
    scattering_dir = create_dir("scattering_data/")

    # Iterate over the particle shapes
    for shape, shape_id in zip(particle["shapes"], particle["shape_ids"]):
        
        # Iterate over the particle axis_ratios
        for axis_ratio in particle["axis_ratios"]:
            
            # Iterate over the list of radii
            for radius in particle["radii"]:

                # Open a file to save the scattering properties
                outfile = f"{scattering_dir}{shape}_radius_{radius}_AR_{axis_ratio}.txt"
                with open(outfile, "w") as f:
                    
                    # Write a header for each column
                    f.write("lambda\tI_scat\tLDR\tx_scat\tx_ext\tssa\tasym\n")

                    # Iterate over the range of wavelengths
                    for wavelength, m in zip(wavelengths, refractive_indices):

                        # Create a Tmatrix object for the spherical particle
                        scatterer = tmatrix.Scatterer(radius=radius, wavelength=wavelength, m=m, shape=shape_id)

                        # Calculate scattering properties and write to file
                        get_scattering_parameters(scatterer=scatterer, filename=f, wavelength=wavelength)

                print(f"Scattering properties saved to {outfile}")

if __name__ == "__main__":
    main()