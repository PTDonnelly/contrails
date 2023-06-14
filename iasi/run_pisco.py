from pisco import Pisco

def main():
    """PISCO: Package for IASI Spectra and Cloud Observations

    For each date specified, open raw binary files, reduce into intermediate files using optimised C scripts
    developed by IASI team, then produce conveniently-formatted spatio-temporal data
    of IASI products: L1C calibrated spectra or L2 cloud products.
    """
    # Instantiate a Pisco class to get data from raw binary files
    pisco = Pisco()

    # Scan years
    for year in pisco.config.year_list:
        pisco.year = f"{year:04d}"
        
        # Scan months
        for im, month in enumerate(pisco.config.month_list):
            pisco.month = f"{month:02d}"
            
            # Scan days (specific days or all calendar days, dependent on Config attributes)
            day_range = pisco.config.day_list if (not pisco.config.day_list == "all") else range(1, pisco.config.days_in_months[im-1] + 1)
            for day in day_range:
                pisco.day = f"{day:02d}"
                
                # Process desired IASI data level
                # if pisco.config.mode == "Process": process only one IASI data level at a time (spectra or cloud products)
                # if pisco.config.mode == "Correlate": process both IASI data levels (spectra or cloud products) and save correlated observations
                for level in pisco.config.data_level:
                    pisco.data_level = level
                    
                    # pisco.config.mode == "Process": process only one IASI data level at a time (spectra or cloud products)
                    # pisco.config.mode == "Correlate": process both IASI data levels (spectra or cloud products) and save correlated observations
                    pisco.get_datapaths()
                    pisco.process_files()
                    pisco.rename_files()
                    if pisco.config.mode == "Correlate":
                        # Correlate L1C spectra and L2 cloud products
                        pisco.correlate_l1c_l2()
                

if __name__ == "__main__":
    main()
