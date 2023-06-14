import os 

from pisco import Extractor, L1CProcessor, L2Processor, L1C_L2_Correlator

def process_l1c(ex: object):
    """
    Process level 1C IASI data.

    Extracts and processes IASI data from intermediate binary files,
    applies quality control and saves the output.

    The result is a HDF5 file containing all good spectra from this intermediate file.
    """
    # Preprocess IASI Level 1C data
    ex.get_datapaths()
    ex.preprocess()
    ex.rename_files()
    
    # Process IASI Level 1C data
    if ex.intermediate_file_check:
        # Process extracted IASI data from intermediate binary files
        with L1CProcessor(ex.intermediate_file, ex.config.targets) as file:
            file.extract_spectra(ex.datapath_out, ex.datafile_out, ex.year, ex.month, ex.day)

def process_l2(ex: object):
    """
    Process level 2 IASI data.

    Extracts and processes IASI cloud products from intermediate csv files and
    stores all data points with Cloud Phase == 2 (ice).

    The result is a HDF5 file containing all locations of ice cloud from this intermediate file.
    """
    # Preprocess IASI Level 2 data
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

def correlate_l1c_l2(ex: object):
    with L1C_L2_Correlator(ex.datapath_out, ex.datafile_out, ex.config.cloud_phase) as file:
        file.filter_spectra()
    return

def main():
    """PISCO: Package for IASI Spectra and Cloud Observations

    For each date specified, open raw binary files, reduce into intermediate files using optimised C scripts
    developed by IASI team, then produce conveniently-formatted spatio-temporal data
    of IASI products: L1C calibrated spectra or L2 cloud products.
    """
    # Instantiate a Pisco class to get data from raw binary files
    ex = Extractor()

    # Scan years, months, days (specific days or all calendar days, dependent on Config attributes)
    for year in ex.config.year_list:
        ex.year = f"{year:04d}"
        for im, month in enumerate(ex.config.month_list):
            ex.month = f"{month:02d}"
            day_range = ex.config.day_list if (not ex.config.day_list == "all") else range(1, ex.config.days_in_months[im-1] + 1)
            for day in day_range:
                ex.day = f"{day:02d}"
                
                
                if (ex.config.mode == "Process") and (ex.config.L1C == True):
                    process_l1c(ex)
                elif (ex.config.mode == "Process") and (ex.config.L2 == True):
                    process_l2(ex)
                elif (ex.config.mode == "Correlate") and (ex.config.L1C == True) and (ex.config.L2 == True):
                    process_l2(ex)
                    process_l1c(ex)
                    correlate_l1c_l2(ex)
                elif (ex.config.mode == "Correlate") and (ex.config.L1C == False) and (ex.config.L2 == False):
                    correlate_l1c_l2(ex)

if __name__ == "__main__":
    main()
