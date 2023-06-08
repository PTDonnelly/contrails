from iasi_config import Config
from iasi_extractor import IASIExtractor as extractor
from iasi_correlator import IASICorrelator as correlator

def main():
    """pyICECAPS: Python IASI Cloud Extraction and Processing for Spectrum Analysis.
    
    For each date specified, open raw binary files, reduce into intermediate files using optimised C scripts
    developed by IASI team, then produce conveniently-formatted spatio-temporal data
    of IASI products: L1C calibrated spectra or L2 cloud products.
    """
    # Instantiate the IASIConfig for the data reduction run
    config = Config()
    config.set_parameters()

    # Instantiate an IASIExtractor to get data from raw binary files
    ex = extractor(config)
    
    for year in ex.config.year_list:
        ex.year = f"{year:04d}"
        
        for im, month in enumerate(ex.config.month_list):
            ex.month = f"{month:02d}"
            
            for day in range(1, ex.config.days_in_months[im-1] + 1):
                ex.day = f"{day:02d}"
                
                ex.get_datapaths()
                ex.process_files()
                ex.rename_files()

                # Correlate of ice cloud locations and spectra
                ex.filter_spectra()
                
                # # Delete intermediate CSV/HDF files
                # co.delete_intermediate()

if __name__ == "__main__":
    main()
