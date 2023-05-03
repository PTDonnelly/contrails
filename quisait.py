import jax.numpy as jnp
from jax import jacfwd
import netCDF4 as nc
import numpy as np
import pyOptimalEstimation as poe
import subprocess
from typing import Tuple

class ForwardModel:
    def __init__(self, config: dict):
        """
        Initialize the forward model.

        Args:
            config (dict): Configuration dictionary for the forward model.
        """
        self.config = config
        self.state_vector = None
        self.synthetic_spectrum = None

    def update_state(self, state_vector: np.ndarray) -> None:
        """
        Update the state vector for the forward model.

        Args:
            state_vector (np.ndarray): The state vector to be used in the forward model.
        """
        self.state_vector = state_vector

    def update_input_file(self) -> None:
        """
        Update the 4A/OP input file with the new state_vector.
        This method depends on your specific 4A/OP configuration and input file format.
        """
        pass

    def launch(self) -> None:
        """
        Run the 4A/OP executable with the updated input file.
        Raises an exception if the simulation fails.
        """
        try:
            subprocess.run(["4AOP_executable", "input_file"], check=True)
        except subprocess.CalledProcessError:
            raise Exception("4A/OP simulation failed")

    def parse_output_file(self) -> None:
        """
        Parse the 4A/OP output file and store the simulated measurements.
        This method depends on your specific 4A/OP configuration and output file format.
        """
        self.synthetic_spectrum = np.array((0, 1))
        return

    def run(self) -> None:
        """
        Run the forward model, which includes updating the 4A/OP input file,
        launching the simulation, and parsing the output file.
        """
        self.update_input_file()
        self.launch()
        self.parse_output_file()

    def get_measurements(self) -> np.ndarray:
        """
        Return the synthetic spectrum.

        Returns:
            np.ndarray: The synthetic spectrum from the forward model.
        """
        return self.synthetic_spectrum

class Retrieval:
    def __init__(self, config: dict):
        """
        Initialize the retrieval object.

        Args:
            config (dict): Configuration dictionary for the forward model.
        """
        self.fouraop = ForwardModel(config)

    def forward_model(self, state_vector: np.ndarray) -> np.ndarray:
        """
        Run the forward model with the given state vector.

        Args:
            state_vector (np.ndarray): The state vector to be used in the forward model.

        Returns:
            np.ndarray: The simulated measurements from the forward model.
        """
        self.fouraop.update_state(state_vector)
        self.fouraop.run()
        return self.fouraop.get_measurements()

    def jacobian(self, state_vector: np.ndarray) -> np.ndarray:
        """
        Compute the Jacobian matrix using JAX's forward-mode automatic differentiation.

        Args:
            state_vector (np.ndarray): The state vector to be used in the forward model.

        Returns:
            np.ndarray: The Jacobian matrix.
        """
        state_vector_jax = jnp.array(state_vector)
        jacobian_func = jacfwd(self.forward_model)
        jac = jacobian_func(state_vector_jax)
        return jac

    def run_retrieval(self, x_a: np.ndarray, y_obs: np.ndarray, s_a: np.ndarray, s_y: np.ndarray) -> poe.OE:
        """
        Run the retrieval using pyOptimalEstimation (poe) with the given input parameters.

        Args:
            x_a (np.ndarray): The a priori state vector.
            y_obs (np.ndarray): The observation vector (measurements).
            s_a (np.ndarray): The a priori covariance matrix.
            s_y (np.ndarray): The measurement covariance matrix.

        Returns:
            poe.OE: The retrieval result object from pyOptimalEstimation.
        """
        oe = poe(
            x_a=x_a,
            forward_model=self.forward_model,
            jacobian=self.jacobian,
            y_obs=y_obs,
            s_a=s_a,
            s_y=s_y
            )
        
        return oe.retrieve()

class Inputs:
    def __init__(self, filepath: str):
        """
        Initialize the Inputs class with the given filepath.
        
        Args:
            filepath (str): Path to the IASI Level 2 product file (netCDF format).
        """
        self.filepath = filepath
        self.wavenumbers = None
        self.radiance = None

    def read_iasi_data(self) -> None:
        """
        Read the IASI data from the netCDF file and extract relevant variables.
        """
        # Open the NetCDF file
        with nc.Dataset(self.filepath, mode='r') as f:
            # Extract the desired variables
            self.latitude = f.variables['latitude'][:]
            self.longitude = f.variables['longitude'][:]
            self.brightness_temperature = f.variables['brightness_temperature'][:]

    @staticmethod
    def brightness_temp_to_radiance(brightness_temperature: np.ndarray, wavenumbers: np.ndarray) -> np.ndarray:
        """
        Convert brightness temperature to radiance.
        
        Args:
            brightness_temperature (np.ndarray): The brightness temperature array.
            wavenumbers (np.ndarray): The wavenumber array.
        
        Returns:
            np.ndarray: The radiance array.
        """
        c1 = 1.1910429723971886e-8  # W m^2 sr^-1 cm^-4
        c2 = 1.438776877372602e-2  # K cm

        radiance = (c1 * wavenumbers**3) / (np.exp(c2 * wavenumbers / brightness_temperature) - 1)
        return radiance

    def get_a_priori_state_vector(self) -> np.ndarray:
        """
        Extract the a priori state vector from the IASI dataset.

        Returns:
            np.ndarray: The a priori state vector.
        """
        # Adapt this method based on the actual structure of the IASI product
        state_vector = np.array(self.dataset.variables["apriori_state_vector"][:])
        return state_vector
    
    def get_measurement_vector(self) -> np.ndarray:
        """
        Extract the measurement spectrum (radiance) from the IASI dataset.

        Returns:
            np.ndarray: The measurement spectrum (radiance).
        """
        self.read_iasi_data()
        
        # Define the wavenumbers corresponding to the brightness temperatures
        self.wavenumbers = np.linspace(650, 2750, self.brightness_temperature.shape[1])

        # Convert brightness temperature to radiance
        self.radiance = self.brightness_temp_to_radiance(self.brightness_temperature, self.wavenumbers)
        return self.radiance

    def get_a_priori_covariance_matrix(self) -> np.ndarray:
        """
        Calculate the a priori covariance matrix from the IASI dataset.

        Returns:
            np.ndarray: The a priori covariance matrix.
        """
        uncertainties = {}
        for variable in self.dataset.variables:
            var = self.dataset.variables[variable]
            if hasattr(var, 'apriori_uncertainty'):
                uncertainties[variable] = var.apriori_uncertainty

        num_params = len(uncertainties)
        covariance_matrix = np.zeros((num_params, num_params))

        # Fill the diagonal elements with the square of the uncertainties
        for i, key in enumerate(uncertainties):
            covariance_matrix[i, i] = uncertainties[key] ** 2

        return covariance_matrix
    
    def get_measurement_covariance_matrix(self) -> np.ndarray:
        """
        Calculate the measurement covariance matrix from the IASI dataset.

        Returns:
            np.ndarray: The measurement covariance matrix.
        """
        # Extract the measurement uncertainties from the IASI dataset
        # Adapt this method based on the actual structure of the IASI product
        uncertainties = np.array(self.dataset.variables["measurement_uncertainties"][:])

        num_params = len(uncertainties)
        covariance_matrix = np.zeros((num_params, num_params))

        # Fill the diagonal elements with the square of the uncertainties
        for i, uncertainty in enumerate(uncertainties):
            covariance_matrix[i, i] = uncertainty ** 2

        return covariance_matrix

def main():
    """
    Main function to run the retrieval process.
    QUISAIT: Quick IASI Spectral Analysis and Inversion Tool. "Quick" here refers to
    the simplicity of design and intended reability. In reality, forward modelling and 
    optimal estimation retrieval can become quite computationally expensive. Nevertheless, 
    this code tries where possible to use highly-optimised libraries when doing computations
    in the Python interpreter. 
    """
    # Specify configuration for retrieval run
    forward_model_config = {}
    
    # Path to the IASI Level 2 product file (netCDF format)
    filepath = "path/to/iasi_level2_product.nc"

    # Instantiate the Inputs class to calculate retrieval parameters
    inputs = Inputs(filepath)

    # Construct retrieval parameters
    x_a = inputs.get_a_priori_state_vector()
    y_obs = inputs.get_measurement_vector()
    s_a = inputs.get_a_priori_covariance_matrix()
    s_y = inputs.get_measurement_covariance_matrix()

    # Create Retrieval object
    retrieval = Retrieval(forward_model_config)

    # Run the retrieval
    result = retrieval.run_retrieval(x_a, y_obs, s_a, s_y)

    print("Retrieved state vector:", result.x)
    print("A posteriori covariance matrix:", result.s)

if __name__ == "__main__":
    main()