import gc
import numpy as np
# import numpy.typing as npt
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
    """Define the physical properties of the test particle"""
    
    # Particle radius in microns
    # radii = [float(i+1) for i in range(10)]
    radii = [5]
    
    # Particle shape
    shapes = ["spheroid"]#, "cylinder"]
    shape_ids = [tmatrix.Scatterer.SHAPE_SPHEROID]#, tmatrix.Scatterer.SHAPE_CYLINDER]

    # Oblateness/prolateness of the particles
    axis_ratios = [1.0]

    return dict_of(radii, shapes, shape_ids, axis_ratios)

def load_optical_data() -> list:
    """Read and decompose optical constants from Iwaguchi and Yang (2011)"""
    
    optical_dir = "C:\\Users\\padra\\Documents\\Research\\projects\\contrails\\scattering_calculations\\optical_data\\"
    filename = 'iwabuchi_optical_properties'
    data = np.loadtxt(f"{optical_dir}{filename}.txt", comments='#', delimiter=None, unpack=True)
    return data[0, :], data[1:13, :], data[13:, :]

def get_refractive_index(ref_wavelengths: np.ndarray):
    """Calculate array of complex refractive indices from ancillary optical data."""

    # Read in wavelength grid and refractive indices
    wavelengths, real, imaginary = load_optical_data()

    # Create arrays to hold interpolation objects for real and imaginary refractive index
    complex_new = np.empty((real.shape[0], ref_wavelengths.size), dtype=complex)

    # Interpolate optical data on to native scattering grid
    real_new = np.array([np.interp(ref_wavelengths, wavelengths, row) for row in real])
    imaginary_new = np.array([np.interp(ref_wavelengths, wavelengths, row) for row in imaginary])

    # Store complex refractive index array
    complex_new = real_new + 1j * imaginary_new

    return complex_new

def get_scattering_parameters(scatterer: object, filename: object):
                        
    # Calculate the scattering and absorption cross-sections
    sca_intensity = scatter.sca_intensity(scatterer)  # Scattering intensity (phase function)
    ldr = scatter.ldr(scatterer)  # Linear depolarizarion ratio
    sca_xsect = scatter.sca_xsect(scatterer)  # Scattering cross-section
    ext_xsect = scatter.ext_xsect(scatterer)  # Extinction cross-section
    ssa = scatter.ssa(scatterer)  # Single-scattering albedo
    asym = scatter.asym(scatterer)  # Asymmetry parameter

    # Write the scattering properties to the file
    if (ssa < 0) or (ssa > 1):
        filename.write(f"{scatterer.wavelength:.8e}\t{np.nan:<14}\t{np.nan:<14}\t{np.nan:<14}\t{np.nan:<14}\t{np.nan:<14}\t{np.nan:<14}\n")
    else:
        filename.write(f"{scatterer.wavelength:.8e}\t{sca_intensity:.8e}\t{ldr:.8e}\t{sca_xsect:.8e}\t{ext_xsect:.8e}\t{ssa:.8e}\t{asym:.8e}\n")

def main():
    
    # Define the spectral grid for scattering calculations
    wavelengths = get_spectral_grid()

    # Define particle parameters for scattering calculations
    particle = get_particle()

    # Create the complex refractive index look-up table 
    refractive_indices = get_refractive_index(wavelengths)

    # Define the temperature regimes of the refractive indices in Iwabuchi and Yang (2011)
    temperatures = [(160 + i * 10) for i in range(refractive_indices.shape[0])]

    # Create directory to store scattering data
    scattering_dir = create_dir("C:\\Users\\padra\\Documents\\Research\\projects\\contrails\\scattering_calculations\\scattering_data\\")

    # Instantiate the Tmatrix object
    scatterer = tmatrix.Scatterer()
    
    # Iterate over the particle shapes
    for shape, shape_id in zip(particle["shapes"], particle["shape_ids"]):      
        
        # Iterate over temperature regimes
        for temperature_idx, temperature in enumerate(temperatures):
            
            # Iterate over the particle axis_ratios
            for axis_ratio in particle["axis_ratios"]:
                
                # Iterate over the list of radii
                for radius in particle["radii"]:

                    # Open a file to save the scattering properties
                    outfile = f"{shape}_T_{temperature}_AR_{axis_ratio}_radius_{radius}.dat"
                    outpath = os.path.join(scattering_dir, outfile)
                    print(outpath)
                    with open(outpath, "w") as f:
                        
                        # Write a header for each column
                        f.write(f"{'lambda':<14}\t{'I_scat':<14}\t{'LDR':<14}\t{'x_scat':<14}\t{'x_ext':<14}\t{'ssa':<14}\t{'asym':<14}\n")

                        # Iterate over the range of wavelengths
                        for wavelength, m in zip(wavelengths, refractive_indices[temperature_idx, :]):
                            
                            # Update the scatterer attributes
                            scatterer.radius = radius
                            scatterer.wavelength = wavelength
                            scatterer.m = m
                            scatterer.axis_ratio = axis_ratio
                            scatterer.shape = shape_id

                            # Calculate scattering properties and write to file
                            get_scattering_parameters(scatterer=scatterer, filename=f)

if __name__ == "__main__":
    main()