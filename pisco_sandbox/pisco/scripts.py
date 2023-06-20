import os

from .extractor import Extractor
from .processor import L1CProcessor, L2Processor
from .correlator import L1C_L2_Correlator

def process_l1c(ex: Extractor):
    """
    Process level 1C IASI data.

    Extracts and processes IASI spectral data from intermediate binary files,
    applies quality control and saves the output.

    Args:
        Instance of the Extractor class 
    Result:
        A HDF5 file containing all good spectra from this intermediate file.
    """
    # Preprocess IASI Level 1C data
    ex.data_level = "l1c"
    ex.get_datapaths()
    ex.preprocess()
    ex.rename_files()
    
    # Process IASI Level 1C data
    if ex.intermediate_file_check:
        # Process extracted IASI data from intermediate binary files and save to CSV
        processor = L1CProcessor(ex.intermediate_file)

        processor.read_binary_file()
        processor.read_record_fields()
        processor.close_binary_file()
    return

def process_l2(ex: Extractor):
    """
    Process level 2 IASI data.

    Extracts and processes IASI spectral data from intermediate binary files,
    applies quality control and saves the output.

    Args:
        Instance of the Extractor class 
    Result:
        A HDF5 file containing all good spectra from this intermediate file.
    """
    # Preprocess IASI Level 2 data
    ex.data_level = "l2"
    ex.get_datapaths()
    ex.preprocess()
    ex.rename_files()
    
    # Process IASI Level 1C data
    if ex.intermediate_file_check:
        # Process extracted IASI data from intermediate binary files and save to CSV
        processor = L2Processor(ex.intermediate_file)

        processor.read_binary_file()
        processor.read_record_fields()
        processor.close_binary_file()
        # processor.extract_data(ex.datapath_out, ex.datafile_out, ex.year, ex.month, ex.day)
    return

def correlate_l1c_l2(ex: Extractor):
    """
    Correlate level 1C spectra and level 2 cloud products.

    Compares processed IASI products from CSV files and
    stores all spectra co-located with instances of a given Cloud Phase.

    Args:
        Instance of the Extractor class 
    
    Result:
        A CSV file containing all spectra at those locations and times.
    """  
    co = L1C_L2_Correlator(ex.config.datapath_out, ex.year, ex.month, ex.day, ex.config.cloud_phase)

    # Concatenate all L2 CSV files into a single coud products file
    co.gather_files()
    
    # Load IASI spectra and cloud products
    co.load_data()      
    
    # Correlates measurements, keep matching locations and times of observation
    co.correlate_measurements()
    
    # Saves the merged data, and deletes the original data.
    co.save_merged_data()

    co.preview_merged_data()
    return