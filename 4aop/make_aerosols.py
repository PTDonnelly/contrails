import itertools
import os
import numpy as np
import pandas as pd
from pytmatrix.tmatrix import Scatterer
import pytmatrix.scatter as scatter
from pytmatrix.psd import PSDIntegrator
from pytmatrix.psd import LognormalPSD
import logging
import json
import snoop

# Constants
CONFIG_FILE = "aerosol_config.json"

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_config():
    """Load configuration from a JSON file."""
    with open(CONFIG_FILE, 'r') as f:
        return json.load(f)

class SpectralGrid:
    """
    Manages spectral grid data for scattering calculations.
    
    Attributes:
        min_wavelength (float): Minimum wavelength of the grid.
        max_wavelength (float): Maximum wavelength of the grid.
        interval (float): The interval between wavelengths for a custom grid.
        wavelengths (numpy.ndarray): The array of wavelengths.
    """
    def __init__(self, min_wavelength=7, max_wavelength=13, interval=0.5):
        self.min_wavelength = min_wavelength
        self.max_wavelength = max_wavelength
        self.interval = interval
        self.wavelengths = None

    def set_custom_grid(self):
        """
        Sets up a custom spectral grid based on the min, max wavelengths, and interval.
        """
        num_points = int((self.max_wavelength - self.min_wavelength) / self.interval) + 1
        self.wavelengths = np.linspace(self.min_wavelength, self.max_wavelength, num_points)

    def set_model_grid(self, model_grid_path):
        """
        Sets up a spectral grid based on a model file.
        
        Args:
            model_grid_path (str): The file path to the model spectral grid.
        """
        try:
            wavelengths = np.loadtxt(model_grid_path)
            self.wavelengths = wavelengths[(wavelengths >= self.min_wavelength) & (wavelengths <= self.max_wavelength)]
        except IOError as e:
            raise ValueError(f"Error reading model grid file: {e}")


class OpticalDataLoader:
    def __init__(self, optical_data_path):
        self.optical_data_path = optical_data_path
        self.iwabuchi_grid = None
        self.real = None
        self.imaginary = None

    def get_data(self):
        data = np.loadtxt(self.optical_data_path, comments='#', delimiter=None, unpack=True)
        self.iwabuchi_grid = data[0, :]
        self.real = data[1:13, :]
        self.imaginary = data[13:, :]

    def interpolate(self, wavelengths):
        """Interpolate real and imaginary refractive indices onto the 4A/OP model grid"""
        # Make sure the data is loaded
        if self.iwabuchi_grid is None or self.real is None or self.imaginary is None:
            self.get_data()
        
        interpolated_real = np.array([np.interp(wavelengths, self.iwabuchi_grid, row) for row in self.real])
        interpolated_imaginary = np.array([np.interp(wavelengths, self.iwabuchi_grid, row) for row in self.imaginary])
        return interpolated_real + 1j * interpolated_imaginary


class ScatteringModel:
    def __init__(self, config, spectral_grid, optical_data):
        self.config = config
        self.spectral_grid = spectral_grid
        self.optical_data = optical_data
        self.wavelengths = None
        self.refractive_indices = None


    def get_refractive_indices(self, temperature):
        return self.refractive_indices[self.config.get('temperatures').index(temperature)]


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


    def make_aerfile(self, xsc_filename):

        # Read the template .aer file into a DataFrame
        reference_aerfile = os.path.join(self.config.get('aerosol_profile_directory'), 'aer4atest_baum.dsf')
        df = pd.read_csv(reference_aerfile, delim_whitespace=True, header=None, skiprows=1)

        # Extract header from the template .aer file
        with open(reference_aerfile, 'r') as file:
            header = file.readline().strip()

        # Get the scattering file name from the xsc_file_path
        scatterer = os.path.basename(xsc_filename).split('aerosols_')[1].split('.dat')[0]

        # Modify the scattering scheme column
        df.iloc[:, 3] = scatterer

        # Apply column formats
        formatted_df = self.format_dataframe(df, self.get_column_formats())

        # Write to new aer file
        new_aerfile = os.path.join(self.config.get('aerosol_profile_directory'), f'aer4atest_{scatterer}.dsf')
        with open(new_aerfile, 'w') as f:
            f.write(header + '\n')
        formatted_df.to_csv(new_aerfile, mode='a', sep=' ', index=False, header=False)


    def set_aerosol_header(self, f, radius, temperature):
        # Read the template .xsc file
        reference_xscfile = os.path.join(self.config.get('aerosol_scattering_directory'), 'aerosols_baum00.dat')
        with open(reference_xscfile, 'r') as file:
            lines = file.readlines()
            header = lines[0:17]
            header[0] = f"# Water ice: T = {temperature} K, r_eff = {radius} um, Iwabuchi et al. (2011), PyTMatrix\n"
        
        for line in header:
            f.write(line)
        
        return header


    def get_scattering_properties(self, scatterer, refractive_index):
        # Do T-Matrix calculation
        ext_xsect = scatter.ext_xsect(scatterer) # in units of wavelength squared
        sca_xsect = scatter.sca_xsect(scatterer) # in units of wavelength squared (Not used for 4A/OP ice)
        abs_xsect = np.subtract(ext_xsect, sca_xsect) # in units of wavelength squared (Not used for 4A/OP ice)
        ssa = scatter.ssa(scatterer)
        asym = scatter.asym(scatterer)
        ext_norm = 1 # np.divide(ext_xsect, ext_xsect[np.where(ext_xsect == 0.75)]) # Not used for 4A/OP ice
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


    def format_properties(self, wavelength, properties):
        """Function to format properties into a string"""
        formatted_properties = " ".join(f"{prop:9.3E}" for prop in properties)
        return f"# {wavelength:10.3E} {formatted_properties}\n"


    def create_particle_distribution(self, D0, s, D_max):
        """
        s = Standard deviation of the lognormal distribution
        D0 Median diameter in microns (naming consistent with PyTmatrix)
        """
        # Initialise the function
        psd_function = LognormalPSD(D0, s, D_max)

        # Initialise integrator
        psd_integrator = PSDIntegrator(D_max=psd_function.D_max, num_points=50)
        return psd_function, psd_integrator


    def calculate_properties_vs_wavelength(self, shape_id, radius, temperature, axis_ratio, geometric_std_dev):
        # Set the sepctral grid for these calculations
        self.wavelengths = self.spectral_grid.wavelengths

        # Gather the temperature-dependent refractive indices at this T
        refractive_indices_at_T = self.get_refractive_indices(temperature)
        xsc_filename = f"aerosols_con_{temperature:03}_r{radius:02}.dat"
        xsc_filepath = os.path.join(self.config.get('aerosol_scattering_directory'), xsc_filename)

        # Make corresponding aerosol input file for 4A/OP
        self.make_aerfile(xsc_filename)

        with open(xsc_filepath, "w") as f:
            self.set_aerosol_header(f, radius, temperature)
            
            # Initialise particle distribution integrator
            D_max = 11  # Maximum diameter in microns
            psd_function, psd_integrator = self.create_particle_distribution(radius, geometric_std_dev, D_max)

            for wavelength, refractive_index in zip(self.wavelengths, refractive_indices_at_T):

                # Initialise the scattering calculations
                scatterer = Scatterer(radius=radius, wavelength=wavelength, m=refractive_index, axis_ratio=axis_ratio, shape=getattr(Scatterer, shape_id))
                scatterer.psd_integrator = psd_integrator
                scatterer.psd = psd_function

                # Calculate scattering matrix
                psd_integrator.init_scatter_table(tm=scatterer, angular_integration=True, verbose=False)
                
                # Extract scattering properties from matrix
                properties = self.get_scattering_properties(scatterer, refractive_index)
                f.write(self.format_properties(wavelength, properties))

                print(wavelength, refractive_index, properties)

        logging.info(f"Completed: {xsc_filepath}\n")


    def calculate_properties_vs_radius(self, shape_id, wavelength, temperature, axis_ratio, geometric_std_dev):
        # Gather the temperature-dependent refractive indices at this T
        self.wavelengths = wavelength
        self.refractive_indices = self.optical_data.interpolate(self.wavelengths)
        refractive_indices_at_T = self.get_refractive_indices(temperature)

        xsc_filename = f"aerosols_con_{temperature:03}_w{wavelength:02}.dat"
        xsc_filepath = os.path.join(self.config.get('aerosol_scattering_directory'), xsc_filename)

        # Make corresponding aerosol input file for 4A/OP
        self.make_aerfile(xsc_filename)

        with open(xsc_filepath, "w") as f:
            self.set_aerosol_header(f, wavelength, temperature)

            radii = np.arange(1, 11, 0.5)
            for radius in radii:

                # Initialise particle distribution integrator
                # D_max = 11  # Maximum diameter in microns
                # psd_function, psd_integrator = self.create_particle_distribution(radius, geometric_std_dev, D_max)

                # Initialise the scattering calculations
                scatterer = Scatterer(radius=radius, wavelength=wavelength, m=refractive_indices_at_T, axis_ratio=axis_ratio, shape=getattr(Scatterer, shape_id))
                # scatterer.psd_integrator = psd_integrator
                # scatterer.psd = psd_function

                # # Calculate scattering matrix
                # psd_integrator.init_scatter_table(tm=scatterer, angular_integration=True, verbose=False)
                
                # Extract scattering properties from matrix
                properties = self.get_scattering_properties(scatterer, refractive_indices_at_T)
                f.write(self.format_properties(radius, properties))

                print(radius, properties)

        logging.info(f"Completed: {xsc_filepath}\n")


    def set_parameter_combinations(self):
        """Return an iterator based on the configuration"""
        return itertools.product(self.config.get('shape_ids'),
                                self.config.get('wavelengths'),
                                self.config.get('radii'),
                                self.config.get('temperatures'),
                                self.config.get('axis_ratios'),
                                self.config.get('geometric_standard_deviations')
                                )


    def run_vs_wavelength(self):
        for shape_id, radius, temperature, axis_ratio, geometric_std_dev in self.set_parameter_combinations():
            logging.info(f"Processing: Shape ID: {shape_id}, Radius: {radius}, Temperature: {temperature}, Axis Ratio: {axis_ratio}, Standard Deviation: {geometric_std_dev}")
            try:
                self.calculate_properties_vs_wavelength(shape_id, radius, temperature, axis_ratio, geometric_std_dev)
            except Exception as e:
                logging.error(f"Error processing {shape_id} at {temperature}K and {radius}um: {e}")

    def run_vs_radius(self):
        for shape_id, wavelength, radius, temperature, axis_ratio, geometric_std_dev in self.set_parameter_combinations():
            logging.info(f"Processing: Shape ID: {shape_id}, Wavelength: {wavelength}, Temperature: {temperature}, Axis Ratio: {axis_ratio}, Standard Deviation: {geometric_std_dev}")
            try:
                self.calculate_properties_vs_radius(shape_id, wavelength, temperature, axis_ratio, geometric_std_dev)
            except Exception as e:
                logging.error(f"Error processing {shape_id} at {temperature}K and {wavelength}um: {e}")
    

def main():
    
    # Prepare the parameters of particle distributions to be iterated over
    config = get_config()
    
    # Read in the reference optical data for water ice
    optical_data = OpticalDataLoader(config.get('optical_data_path'))
    
    # Generate the spectral grid (default: 4A/OP optimised scattering grid, otherwise: custom)
    spectral_grid = SpectralGrid()
    spectral_grid.set_model_grid(config.get('model_grid_path'))
    
    # PyTmatrix with these inputs
    model = ScatteringModel(config, spectral_grid, optical_data)
    model.run_vs_radius()

if __name__ == "__main__":
    main()
