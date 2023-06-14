from pisco.pisco import Pisco

def main():
    """PISCO: Package for IASI Spectra and Cloud Observations

    For each date specified, open raw binary files, reduce into intermediate files using optimised C scripts
    developed by IASI team, then produce conveniently-formatted spatio-temporal data
    of IASI products: L1C calibrated spectra or L2 cloud products.
    """
    # Instantiate a Pisco class to get data from raw binary files
    extractor = Pisco()

    # Scan years
    for year in extractor.config.year_list:
        extractor.year = f"{year:04d}"
        
        # Scan months
        for im, month in enumerate(extractor.config.month_list):
            extractor.month = f"{month:02d}"
            
            # Scan days (specific days or all calendar days, dependent on Config attributes)
            day_range = extractor.config.day_list if (not extractor.config.day_list == "all") else range(1, extractor.config.days_in_months[im-1] + 1)
            for day in day_range:
                extractor.day = f"{day:02d}"
                
                # Process desired IASI data level
                # if extractor.config.mode == "Process": process only one IASI data level at a time (spectra or cloud products)
                # if extractor.config.mode == "Correlate": process both IASI data levels (spectra or cloud products) and save correlated observations
                for level in extractor.config.data_level:
                    extractor.data_level = level
                    
                    # extractor.config.mode == "Process": process only one IASI data level at a time (spectra or cloud products)
                    # extractor.config.mode == "Correlate": process both IASI data levels (spectra or cloud products) and save correlated observations
                    extractor.get_datapaths()
                    extractor.process_files()
                    extractor.rename_files()
                    if extractor.config.mode == "Correlate":
                        # Correlate L1C spectra and L2 cloud products
                        extractor.correlate_l1c_l2()
                

if __name__ == "__main__":
    main()
