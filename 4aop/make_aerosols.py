import itertools
import os
import numpy as np
import pandas as pd
from pytmatrix.tmatrix import Scatterer
import pytmatrix.scatter as scatter

class SpectralGrid:
    def __init__(self, min_wavelength=7.999, max_wavelength=12.999, interval=0.5):
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
        
    
    def retrieve(self, scatterer, refractive_index):
        # Do T-Matrix calculation
        # sca_intensity = scatter.sca_intensity(scatterer) # Not used for 4A/OP ice
        # ldr = scatter.ldr(scatterer) # Not used for 4A/OP ice
        ext_xsect = scatter.ext_xsect(scatterer)
        sca_xsect = scatter.sca_xsect(scatterer) # Not used for 4A/OP ice
        abs_xsect = np.subtract(ext_xsect, sca_xsect) # Not used for 4A/OP ice
        ssa = scatter.ssa(scatterer)
        asym = scatter.asym(scatterer)
        ext_norm = np.divide(ext_xsect, np.max(ext_xsect)) # Not used for 4A/OP ice
        m_real = refractive_index.real # Not used for 4A/OP ice
        m_imag = refractive_index.imag # Not used for 4A/OP ice

        # Check if ssa is within the valid range [0, 1]
        if not 0 <= ssa <= 1:
            # Use a consistent number of properties; fill with np.nan if ssa is out of bounds
            properties = [np.nan] * 8
        else:
            # All good; prepare the properties list
            properties = [ext_xsect, sca_xsect, abs_xsect, ssa, asym, ext_norm, m_real, m_imag]
        
        # Return the formatted string
        return properties


class ScatteringConfigurer:
    def __init__(self, optical_data_directory, aer_directory, xsc_directory):
        self.aer_directory = aer_directory
        self.xsc_directory = xsc_directory
        self.optical_data_loader = OpticalDataLoader(optical_data_directory)
        self.spectral_grid = SpectralGrid()
        self.calculator = ScatteringCalculator(self.spectral_grid, self.xsc_directory)

    @staticmethod
    def aerosol_header(radius, temperature):
       # Define the header
        header = (
            f"# Water ice at T = {temperature} K, with r_eff = {radius} um derived from optical properties of Iwabuchi et al. (2011), and processed with PyTMatrix (wrapper for Mishchenko's T-Matrix FORTRAN code)\n"
            "#\n"
            "# size distribution: Standard Gamma\n"
            "# ------------------\n"
            "#\n"
            "#   minimum radius, [um]:      0.000E+00\n"
            "#   maximum radius, [um]:      0.000E+00\n"
            "#                  sigma:      0.000E+00\n"
            "#       Rmod (wet), [um]:      0.000E+00\n"
            "#       Rmod (dry), [um]:      7.000E+01\n"
            "#\n"
            "# optical parameters:\n"
            "# -------------------\n"
            "#\n"
            "# wavelength ext.coef  sca.coef  abs.coef  si.sc.alb  asym.par  ext.nor  m_real  m_imag  \n"
            "#     [um]     [1/km]    [1/km]    [1/km]\n"
            "#\n"
        )
        return header

    @staticmethod
    def format_properties(wavelength, properties):
        """Function to format properties into a string"""
        formatted_properties = " ".join(f"{prop:9.3E}" for prop in properties)
        # print(f"# {wavelength:10.3E} {formatted_properties}\n")
        return f"# {wavelength:10.3E} {formatted_properties}\n"
    
    
    def run(self, shapes, shape_ids, radii, temperatures, axis_ratios):
        parameter_combinations = itertools.product(shapes, shape_ids, radii, temperatures, axis_ratios)

        for shape, shape_id, radius, temperature, axis_ratio in parameter_combinations:

            # Extract each temperature-dependent profile of refractive index
            refractive_indices = self.optical_data_loader.interpolate_refractive_indices(self.spectral_grid.wavelengths)[temperatures.index(temperature)]
            
            filename = f"aerosols_con_{temperature:03}_{radius:02}.dat"
            filepath = os.path.join(self.xsc_directory, filename)
            
            with open(filepath, "w") as f:
                # Write header for scattering properties file
                f.write(ScatteringConfigurer.aerosol_header(radius, temperature))

                # Do calculation and write to file
                for wavelength, refractive_index in zip(self.spectral_grid.wavelengths, refractive_indices):
                    print(f"{wavelength} um")
                    scatterer = Scatterer(radius=radius, wavelength=wavelength, m=refractive_index, axis_ratio=axis_ratio, shape=shape_id)
                    properties = self.calculator.retrieve(scatterer, refractive_index)
                    f.write(ScatteringConfigurer.format_properties(wavelength, properties))

            # Create the corresponding 4A/OP aerosol configuration file
            self.create_aerfile(filepath)

            print(f"Done: {filepath}")


    def get_column_formats(self):
            """Function to get column formats of .dsf file"""
            return {
                0: "{:0>7.2f}".format,  # Leading zeros, two decimal places
                1: "{:0>7.2f}".format,  # Leading zeros, two decimal places
                2: "{:.2E}".format,     # Scientific notation with 2 decimal places
                3: "{}".format,         # String, as is
                4: "{:0>4.2f}".format,  # Leading zeros, two decimal places
                5: "{:0>4.2f}".format,  # Leading zeros, two decimal places
                6: "{:0>4.2f}".format,  # Leading zeros, two decimal places
                7: "{}".format          # String, as is
            }


    def format_dataframe(self, df, column_formats):
        """Function to format dataframe"""
        formatted_df = df.copy()
        for col, fmt in column_formats.items():
            formatted_df[col] = df[col].apply(fmt)
        return formatted_df


    def create_aerfile(self, xscfile):

        # Read the template .aer file into a DataFrame
        reference_aerfile = os.path.join(self.aer_directory, 'aer4atest_baum.dsf')
        df = pd.read_csv(reference_aerfile, delim_whitespace=True, header=None, skiprows=1)

        # Extract header from the template .aer file
        with open(reference_aerfile, 'r') as file:
            header = file.readline().strip()

        # Get the scattering file name from the xsc_file_path
        scatterer = os.path.basename(xscfile).split('aerosols_')[1].split('.dat')[0]

        # Modify the scattering scheme column
        df.iloc[:, 3] = scatterer

        # Apply column formats
        formatted_df = self.format_dataframe(df, self.get_column_formats())

        # Write to new aer file
        new_aerfile = f"{self.aer_directory}aer4atest_{scatterer}.dsf"
        with open(new_aerfile, 'w') as f:
            f.write(header + '\n')
        formatted_df.to_csv(new_aerfile, mode='a', sep=' ', index=False, header=False)
        
def main():
    # Assume these are lists of your properties
    shapes = ["sphere"]
    shape_ids = [Scatterer.SHAPE_SPHEROID]
    radii = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 40, 50]
    temperatures = [160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270]
    axis_ratios = [1.0]
    
    # Directories
    optical_data_directory = "C:\\Users\\padra\\Documents\\Research\\projects\\contrails\\scattering_calculations\\optical_data\\"
    aer_directory = "C:\\Users\\padra\\Documents\\Research\\projects\\contrails\\4aop\\datatm\\"
    xsc_directory = "C:\\Users\\padra\\Documents\\Research\\projects\\contrails\\4aop\\datscat\\"
    
    # Assuming that shapes, radii, temperatures, and axis_ratios are defined lists of parameters
    config = ScatteringConfigurer(optical_data_directory, aer_directory, xsc_directory)
    config.run(shapes, shape_ids, radii, temperatures, axis_ratios)

if __name__ == "__main__":
    main()
