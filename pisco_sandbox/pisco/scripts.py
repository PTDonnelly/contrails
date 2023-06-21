from .extraction import Extractor
from .preprocessing import Preprocessor
from .processing import Processor

def process_iasi(ex: Extractor, data_level: str):
    """
    This function is used to process IASI (Infrared Atmospheric Sounding Interferometer) data 
    by extracting raw binary files and preprocessing them into pandas DataFrames.

    It first uses an Extractor object to fetch the raw data files for a specific data level, 
    and then performs extraction. 

    If the extraction is successful, the function creates a Preprocessor object using the path 
    to the extracted intermediate file and the data level. This Preprocessor object preprocesses 
    the files to create a DataFrame for the given date.

    Parameters:
    ex (Extractor): An Extractor object which contains methods and attributes for data extraction.
    data_level (str): The data level string, which determines the level of data to be extracted 
                      and preprocessed.

    Returns:
    None: The function performs extraction and preprocessing operations but does not return a value.
    """
    # Use OBR to extract IASI data from raw binary files
    ex.data_level = data_level
    ex.get_datapaths()
    ex.extract_files()

    # If IASI data was successfully extracted
    if ex.intermediate_file_check:
        # Preprocess the data into pandas DataFrames
        p = Preprocessor(ex.intermediate_file, ex.data_level)
        p.preprocess_files(ex.year, ex.month, ex.day)
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
    p = Processor(ex.config.datapath_out, ex.year, ex.month, ex.day, ex.config.cloud_phase)

    # Concatenate all L2 CSV files into a single coud products file
    p.gather_files()
    
    # Load IASI spectra and cloud products
    p.load_data()      
    
    # Correlates measurements, keep matching locations and times of observation
    p.correlate_measurements()
    
    # Saves the merged data, and deletes the original data.
    p.save_merged_data()

    p.preview_merged_data()
    return