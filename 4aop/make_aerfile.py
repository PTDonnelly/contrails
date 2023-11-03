import itertools
import os
import numpy as np
from pytmatrix.tmatrix import Scatterer
import pytmatrix.scatter as scatter

class SpectralGrid:
    def __init__(self, min_wavelength=7, max_wavelength=15, interval=0.1):
        self.min_wavelength = min_wavelength
        self.max_wavelength = max_wavelength
        self.interval = interval
        self.wavelengths = np.arange(self.min_wavelength, self.max_wavelength, self.interval, dtype=np.float64)
    
    def save_to_file(self, directory, filename="spectral_grid.txt"):
        if not os.path.exists(directory):
            os.makedirs(directory)
        filepath = os.path.join(directory, filename)
        with open(filepath, "w") as f:
            for wavelength in self.wavelengths:
                f.write(f"{wavelength:.4f}\n")


class ParticleProperties:
    def __init__(self, radius=5, shape=Scatterer.SHAPE_SPHEROID, axis_ratio=1.0):
        self.radius = radius
        self.shape = shape
        self.axis_ratio = axis_ratio


class OpticalDataLoader:
    def __init__(self, optical_data_path, filename='iwabuchi_optical_properties'):
        self.optical_data_path = optical_data_path
        self.filename = filename
    
    def load_data(self):
        full_path = os.path.join(self.optical_data_path, f"{self.filename}.txt")
        data = np.loadtxt(full_path, comments='#', delimiter=None, unpack=True)
        return data[0, :], data[1:13, :], data[13:, :]
    
    def interpolate_refractive_indices(self, wavelengths):
        wl_grid, real, imaginary = self.load_data()
        interpolated_real = np.array([np.interp(wavelengths, wl_grid, row) for row in real])
        interpolated_imaginary = np.array([np.interp(wavelengths, wl_grid, row) for row in imaginary])
        return interpolated_real + 1j * interpolated_imaginary


class ScatteringCalculator:
    def __init__(self, spectral_grid, output_directory):
        self.spectral_grid = spectral_grid
        self.output_directory = output_directory

    def run(self, f, wavelength, scatterer):
        # Do T-Matrix calculation
        sca_intensity = scatter.sca_intensity(scatterer)
        ldr = scatter.ldr(scatterer)
        sca_xsect = scatter.sca_xsect(scatterer)
        ext_xsect = scatter.ext_xsect(scatterer)
        ssa = scatter.ssa(scatterer)
        asym = scatter.asym(scatterer)

        if ssa < 0 or ssa > 1:
            properties = [np.nan] * 6
        else:
            properties = [sca_intensity, ldr, sca_xsect, ext_xsect, ssa, asym]
        
        return f"{wavelength:.8e}\t" + "\t".join(f"{prop:.8e}" for prop in properties) + "\n"


class ScatteringConfigurer:
    def __init__(self, optical_data_directory, output_directory):
        self.optical_data_loader = OpticalDataLoader(optical_data_directory)
        self.spectral_grid = SpectralGrid()
        self.output_directory = output_directory
        self.calculator = ScatteringCalculator(self.spectral_grid, output_directory)

    def run(self, shapes, shape_ids, radii, temperatures, axis_ratios):
        parameter_combinations = itertools.product(shapes, shape_ids, radii, temperatures, axis_ratios)

        for shape, shape_id, radius, temperature, axis_ratio in parameter_combinations:

            # Extract each temperature-dependent profile of refractive index
            ms = self.optical_data_loader.interpolate_refractive_indices(self.spectral_grid.wavelengths)[temperatures.index(temperature)]
            
            filename = f"ice_{shape}_{temperature}K_{radius:03}um.dat"
            filepath = os.path.join(self.output_directory, filename)

            with open(filepath, "w") as f:

                # Write header for scattering properties
                header = f"{'lambda':<16}{'I_scat':<16}{'LDR':<16}{'x_scat':<16}{'x_ext':<16}{'ssa':<16}{'asym':<16}\n"
                f.write(header)

                # Do calculation and write to file
                for wavelength, m in zip(self.spectral_grid.wavelengths, ms):
                    scatterer = Scatterer(radius=radius, wavelength=wavelength, m=m, axis_ratio=axis_ratio, shape=shape_id)
                    f.write(self.calculator.run(f, wavelength, scatterer))

            print(f"Done: {filepath}")


def main():
    # Assume these are lists of your properties
    shapes = ["sphere"]
    shape_ids = [Scatterer.SHAPE_SPHEROID]
    radii = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    temperatures = [160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270]
    axis_ratios = [1.0]
    
    optical_data_directory = "C:\\Users\\padra\\Documents\\Research\\projects\\contrails\\scattering_calculations\\optical_data\\"
    output_directory = "C:\\Users\\padra\\Documents\\Research\\projects\\contrails\\scattering_calculations\\scattering_data\\"
    
    
    # Assuming that shapes, radii, temperatures, and axis_ratios are defined lists of parameters
    config = ScatteringConfigurer(optical_data_directory, output_directory)
    config.run(shapes, shape_ids, radii, temperatures, axis_ratios)

if __name__ == "__main__":
    main()
