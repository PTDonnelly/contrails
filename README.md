# Package for IASI Spectra and Cloud Observations (PISCO)

PISCO is a Python package designed to facilitate the extraction, processing and analysis of Infrared Atmospheric Sounding Interferometer (IASI) spectra and retrieved cloud products.

## Features

- Extracts data from raw binary files using optimised C scripts developed by the IASI team.
- Processes data into conveniently-formatted spatio-temporal data of IASI products: Level 1C calibrated spectra or Level 2 cloud products.
- Supports correlation between Level 1C spectra and Level 2 cloud products.
- Provides configurable parameters via a config class.

## Installation

To install the package, clone this repository to your local machine by using the following command:

```bash
git clone https://github.com/PTDonnelly/Pisco.git
```


## Usage Instructions for the Config Module

The `Config` module is designed to handle user-specified settings for the processing of IASI satellite data.

### Step 1: Import the Module (by default this is done when initialising the Extractor class)
```python
from Config import Config
```

### Step 2: Create a Configuration Object

Use your own JSON configuration file to initialize a `Config` object.

```python
config = Config("path_to_your_config_file.json")
```

### JSON Configuration File

The configuration file is a JSON file that allows the user to specify parameters such as processing mode, year, month, day, output data path, and many more. The JSON keys correspond to attributes of the `Config` class and they are directly assigned on class instantiation. Here is a description of the parameters in the JSON configuration file:

- **mode (str):** The processing mode. Options are 'Process' or 'Correlate'.
- **L1C (bool):** Specifies whether Level 1C data should be processed.
- **L2 (bool):** Specifies whether Level 2 data should be processed.
- **year_list (List[int]):** List of years for extraction.
- **month_list (List[int]):** List of months for extraction.
- **day_list (List[int] or str):** List of specific days or "all" to scan all calendar days in the month.
- **days_in_months (List[int]):** Number of calendar days in each month.
- **datapath_out (str):** The user-defined data path for the processed output files.
- **targets (List[str]):** List of target IASI L1C products.
- **latitude_range (List[int]):** The spatial range for binning latitudes.
- **longitude_range (List[int]):** The spatial range for binning longitudes.
- **cloud_phase (int):** The desired cloud phase from the L2 products.

### Methods

The `Config` class has two methods to set further attributes and perform checks:

- **set_channels():** This method sets the list of IASI channel indices. It returns a list with integers from 1 to n (where n is set to 5 in the provided code).
- **check_mode_and_data_level():** This method checks if the execution mode and data level inputs agree before execution. If the mode is set to "Process", either 'L1C' or 'L2' must be set to true. If both or none are set to true, a ValueError will be raised.


## Usage


The main functionality of the package is provided through the `Extractor` class (configured by the `Config` class). The Extractor object uses these configurations to access, extract and process raw binary files of IASI data. (N.B.: a future version of Pisco will de-couple the directory and file paths from the Extractor module so that there is greater cohesion between the other modules.)

A default example is found in the module-level code `run_pisco.py`:

Remember to replace "path_to_your_config_file.json" with the actual path to your JSON configuration file if it's located somewhere else.

It scans through each specified year, month, and day in the configuration. For each day, it will either process Level 1C or Level 2 data, or correlate both Level 1C and Level 2 data, depending on your settings in the configuration file.

The `process_l1c(ex)`, `process_l2(ex)`, and `correlate_l1c_l2(ex)` are functions imported from the scripts.process_iasi module, and they accept the Extractor class as an argument. Make sure these scripts are available and properly defined in your project. 

## Summary 

The Pisco project is designed to facilitate the efficient extraction and processing of cloud product data from the Infrared Atmospheric Sounding Interferometer (IASI) for spectrum analysis, along with correlating L1C spectra with L2 cloud products.

The main components of the Pisco project include the `Config`, `Extractor`, `L1CProcessor`, `L2Processor`, and `Correlator` classes.

1. **Config Class**: This class manages the setting of configuration parameters required for data extraction and processing. These parameters include date ranges, data level, data paths, IASI L1C target products, IASI channel indices, spatial ranges for binning, and desired cloud phase from L2 products. **Change these values for each analysis.**

2. **Extractor Class**: This class is the key driver of the data extraction process. It utilises the `Config` class to set the parameters for the extraction process and manages the flow of extraction and processing.

3. **L1CProcessor Class**: This class handles the Level 1C (L1C) IASI data product, i.e., the calibrated spectra. It provides methods for loading, processing, saving, and preparing the L1C data for correlation with L2 products.

4. **L2Processor Class**: This class handles the the Level 2 (L2) IASI data product, specifically cloud products. It offers methods for loading, processing, saving the L2 data, and preparing it for correlation with L1C products.

5. **Correlator Class**: This class handles correlating the L1C spectra with L2 cloud products. It achieves this by taking the resulting files from `L1CProcessor` and `L2Processor` and providing methods for correlating the data between these products. It looks for observations with a specific cloud phase and finds the corresponding spectra, then deletes the products of `L1CProcessor` and `L2Processor`.

The `main` function executes the entire process. It starts by creating an instance of the `Config` class to set the necessary parameters for the data extraction. Then, an `Extractor` is instantiated with this configuration, running the data extraction and processing sequence. The extractor iterates through the specified date ranges, extracting and processing the data accordingly. `L1CProcessor` uses a C script (OBR tool designed by IASI team) to read binary data, extracts variables to an intermediate binary file, extracts that to CSV or HDF5, and deletes the intermediate file. `L2Processor` uses a C script (unofficial BUFR reader designed by IASI team) to read binary data, and save to CSV (there is no intermediate step with this reader). The `Correlator` class then correlates the processed L1C and L2 data.

The `mode` and `data_level` attributes of the `Config` class can be set in the JSON file to process each data level. `mode` can be set equal to "Process" or "Correlate". "Process is non-destructive, it extracts binary data and saves spectra (L1C) and cloud products (L2) in CSV format. If `mode` is set to "Correlate" then the "Process" functionality can still apply, but when the spectra are filtered based on the outputs of the cloud products, a final CSV file is produced and the products of the "Process" mode are deleted. "Correlate" can also be used if there are existing L1C and L2 files (no need to run the Extractor of Processors) as logn as the file naming is consistent.


Thus, if `mode` is set to "Process" then either L1C is True or L2 is True, both can't be True and both can't be false.  If `mode` is set to "Correlate" then L1C and L2 must both be true (run Extractor, Processors, then Correlator) or both be False (run Extractor only for paths and then Correlator). The order of execution in the former case is: `L2Processor` is executed first and generates a an output file for a given date containing the cloud phase observations. If no observations of the given cloud phase were found, no file is created and `L1CProcessor` does not execute.