# Python IASI Cloud Extraction and Processing for Spectrum Analysis (PyICECAPS)

PyICECAPS is a Python package designed to facilitate the extraction, processing and analysis of Infrared Atmospheric Sounding Interferometer (IASI) cloud data.

## Features

- Extracts data from raw binary files using optimised C scripts developed by the IASI team.
- Processes data into conveniently-formatted spatio-temporal data of IASI products: Level 1C calibrated spectra or Level 2 cloud products.
- Supports correlation between Level 1C spectra and Level 2 cloud products.
- Provides configurable parameters via a config class.

## Installation

To install the package, clone this repository to your local machine by using the following command:

```bash
git clone https://github.com/PTDonnelly/PyICECAPS.git
```

## Usage

The main functionality of the package is provided through the `IASIExtractor` class (configured by the `Config` class). An example usage is as follows:


```python
from iasi_config import Config
from iasi_extractor import IASIExtractor

config = Config()

# Customise parameters
config.year_list = [2020, 2021]
config.month_list = [1, 2, 3]
config.data_level = "L2"

config.set_parameters()

ex = IASIExtractor(config)

# Specify the dates for which data should be processed
ex.config.year_list = [2023]
ex.config.month_list = [1]

# Process the data
ex.get_datapaths()
ex.process_files()
ex.rename_files()

# Correlate Level 1C spectra and Level 2 cloud products
ex.correlate_l1c_l2()
```

## Summary 

The PyICECAPS project is designed to facilitate the efficient extraction and processing of cloud product data from the Infrared Atmospheric Sounding Interferometer (IASI) for spectrum analysis, along with correlating L1C spectra with L2 cloud products.

The main components of the PyICECAPS project include the `Config`, `IASIExtractor`, `L1CProcessor`, `L2Processor`, and `Correlator` classes.

1. **Config Class**: This class manages the setting of configuration parameters required for data extraction and processing. These parameters include date ranges, data level, data paths, IASI L1C target products, IASI channel indices, spatial ranges for binning, and desired cloud phase from L2 products. **Change these values for each analysis.**

2. **IASIExtractor Class**: This class is the key driver of the data extraction process. It utilises the `Config` class to set the parameters for the extraction process and manages the flow of extraction and processing.

3. **L1CProcessor Class**: This class handles the Level 1C (L1C) IASI data product, i.e., the calibrated spectra. It provides methods for loading, processing, saving, and preparing the L1C data for correlation with L2 products.

4. **L2Processor Class**: This class handles the the Level 2 (L2) IASI data product, specifically cloud products. It offers methods for loading, processing, saving the L2 data, and preparing it for correlation with L1C products.

5. **Correlator Class**: This class handles correlating the L1C spectra with L2 cloud products. It achieves this by taking the resulting files from `L1CProcessor` and `L2Processor` and providing methods for correlating the data between these products. It looks for observations with a specific cloud phase and finds the corresponding spectra, then deletes the products of `L1CProcessor` and `L2Processor`.

The `main` function executes the entire process. It starts by creating an instance of the `Config` class to set the necessary parameters for the data extraction. Then, an `IASIExtractor` is instantiated with this configuration, running the data extraction and processing sequence. The extractor iterates through the specified date ranges, extracting and processing the data accordingly. `L1CProcessor` uses a C script (OBR tool designed by IASI team) to read binary data, extracts variables to an intermediate binary file, extracts that to CSV or HDF5, and deletes the intermediate file. `L2Processor` uses a C script (unofficial BUFR reader designed by IASI team) to read binary data, and save to CSV (there is no intermediate step with this reader). The `Correlator` class then correlates the processed L1C and L2 data.

The `mode` and `data_level` attributes of the `Config` class can be set to process each data level. If `mode` is set to "Process" then `data_level` must be either "l1c" or "l2".  If `mode` is set to "Correlate" then `data_level` must euqal ["l2", "l1c"]. The indvidual cases run as described above. The order of execution in the latter case is: `L2Processor` is executed first and generates a an output file for a given date containing the cloud phase observations. If no observations of the given cloud phase were found, no file is created and `L1CProcessor` does not execute. It is possible to run simply one data level by supplying a list containing that single string (e.g., ["l1c"])