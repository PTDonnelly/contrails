# Quick IASI Spectral Analysis and Inversion Tool (QUISAIT)

This repository contains a Python implementation of atmospheric retrieval from Infrared Atmospheric Sounding Interferometer (IASI) measurements using the 4A/OP forward model and optimal estimation.

## Overview

The code is organized into the following classes and functions:

1. `ForwardModel`: Class representing the 4A/OP forward model.
2. `Retrieval`: Class that handles the atmospheric retrieval process.
3. `Inputs`: Class for reading the IASI Level 1C calibrated spectra and Level 2 products and calculating retrieval parameters.
4. `main()`: The main function that initializes the inputs, runs the retrieval, and displays the results.

## Dependencies

- JAX: For efficient computation and automatic differentiation.
- NumPy: For numerical operations.
- netCDF4: For reading IASI Level 2 product files.
- pyOptimalEstimation: For implementing the optimal estimation method.
- subprocess: For running the 4A/OP executable.

## Installation

1. Clone the repository:

   ```
   git clone https://github.com/yourusername/quisait.git
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

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.