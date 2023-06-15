import os

from pisco import L1CProcessor, L2Processor, L1C_L2_Correlator

def process_l1c(ex: object):
    """
    Process level 1C IASI data.

    Extracts and processes IASI spectral data from intermediate binary files,
    applies quality control and saves the output.

    The result is a HDF5 file containing all good spectra from this intermediate file.
    """
    # Preprocess IASI Level 1C data
    ex.data_level = "l1c"
    ex.get_datapaths()
    ex.preprocess()
    ex.rename_files()
    
    # Process IASI Level 1C data
    if ex.intermediate_file_check:
        print(ex.datapath_out, ex.datafile_out)
        # Process extracted IASI data from intermediate binary files
        with L1CProcessor(ex.intermediate_file, ex.config.targets) as file:
            file.extract_spectra(ex.datapath_out, ex.datafile_out, ex.year, ex.month, ex.day)
    return

def process_l2(ex: object):
    """
    Process level 2 IASI data.

    Extracts and processes IASI cloud products from intermediate csv files and
    stores all data points with Cloud Phase == 2 (ice).

    The result is a HDF5 file containing all locations of ice cloud from this intermediate file.
    """
    # Preprocess IASI Level 2 data
    ex.data_level = "l2"
    ex.get_datapaths()
    for datafile_in in os.scandir(ex.datapath_in):
        # Check that entry is a file
        if datafile_in.is_file():
            # Set the current input file
            ex.datafile_in = datafile_in.name
            ex.preprocess()
    ex.rename_files()
    
    # Process IASI Level 2 data
    if ex.intermediate_file_check:
        # Process extracted IASI data from intermediate binary files
        with L2Processor(ex.intermediate_file, ex.config.latitude_range, ex.config.longitude_range, ex.config.cloud_phase) as file:
            file.extract_ice_clouds() 
    return

def correlate_l1c_l2(ex: object):
    """
    Correlate level 1C spectra and level 2 cloud products.

    Compares processed IASI products from csv files and
    stores all spectra co-located with instances of a given Cloud Phase.

    The result is a csv file containing all spectra at those locations and times.
    """
    # Preprocess IASI Level 2 data
    with L1C_L2_Correlator(ex.config.datapath_out, ex.year, ex.month, ex.day, ex.config.cloud_phase) as file:
        file.filter_spectra()
    return