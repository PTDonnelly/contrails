# Atmospheric Retrieval and Aerosol Scatter

This repository contains Python implementations of atmospheric retrieval methods and aerosol scattering calculations.

## Overview: Quick IASI Spectral Analysis and Inversion Tool (QUISAIT)

This script contains a Python implementation of atmospheric retrieval from Infrared Atmospheric Sounding Interferometer (IASI) measurements using the 4A/OP forward model and optimal estimation. "Quick" here refers to the simplicity of design and intended reability. In reality, forward modelling and optimal estimation retrieval can become quite computationally expensive. Nevertheless, this code tries where possible to use highly-optimised libraries when doing computations in the Python interpreter.

The code is organized into the following classes and functions:

1. `ForwardModel`: Class representing the 4A/OP forward model.
2. `Retrieval`: Class that handles the atmospheric retrieval process.
3. `Inputs`: Class for reading the IASI Level 1C calibrated spectra and Level 2 products and calculating retrieval parameters.
4. `main()`: The main function that initializes the inputs, runs the retrieval, and displays the results.

## Overview: Aerosol Scattering Calculator

This script contains a Python implementation of the scattering calculations of particles using the T-matrix method. It is particularly focused on the scattering properties of contrails, with optical constants provided by Iwabuchi and Yang (2011). 

The code is organized into the following classes and functions:

## Dependencies

QUISAIT:

- Python 3.7+
- JAX: For efficient computation and automatic differentiation.
- NumPy: For numerical operations.
- netCDF4: For reading IASI Level 2 product files.
- pyOptimalEstimation: For implementing the optimal estimation method.
- subprocess: For running the 4A/OP executable.

Scattering:

- Python 3.7+
- NumPy
- SciPy
- PyTMatrix (for Scattering Properties Calculator)

<!-- ## Installation

1. Clone the repository:

   ```
   git clone https://github.com/PTDonnelly/quisait.git
   ```

2. Navigate to the repository folder:

   ```
   cd quisait
   ```

3. Install the required dependencies:

   ```
   pip install -r requirements.txt
   ```

4. Run the retrieval script:

   ```
   python main.py
   ```

## Usage

To use this code for your own retrieval process, follow these steps:

1. Modify the `filepath` variable in the `main()` function to point to your IASI Level 2 product file.
2. Update the `ForwardModel` and `Inputs` classes to adapt to your specific 4A/OP configuration and IASI product structure.
3. Run the `main.py` script to perform the retrieval and display the results.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details. -->