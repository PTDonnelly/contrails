from iasi_config import Config
from iasi_extractor import IASIExtractor as extractor

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
                
                if config.mode == "Process":
                    # Process only one IASI data level at a time (spectra or cloud products)
                        ex.get_datapaths()
                        ex.process_files()
                        ex.rename_files()
                elif config.mode == "Correlate":
                    # Process both IASI data levels (spectra or cloud products) and save correlated observations
                    config.data_level = ["l2", "l1c"]
                    for level in config.data_level:
                        ex.data_level = level
                        ex.get_datapaths()
                        print(ex.datapath_in)
                        print(ex.datapath_out)
                        print(ex.datafile_l1c)
                        print(ex.datafile_l2)
                        input()
                        # ex.process_files()
                        # ex.rename_files()

                        # Correlate L1C spectra and L2 cloud products
                        # ex.correlate_l1c_l2()
                

if __name__ == "__main__":
    main()
