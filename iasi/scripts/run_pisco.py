from pisco import Extractor


# def _delete_intermediate_reduction_data(self, intermediate_file: str):
#         # Delete intermediate binary file (after extracting spectra and metadata)
#         os.remove(intermediate_file)
#         pass
    
#     def _process_l2(self, intermediate_file: str):
#         """
#         Process level 2 IASI data.

#         Extracts and processes IASI cloud products from intermediate csv files and
#         stores all data points with Cloud Phase == 2 (ice).

#         The result is a HDF5 file containing all locations of ice cloud from this intermediate file.
#         """
#         with L2Processor(intermediate_file, self.config.latitude_range, self.config.longitude_range, self.config.cloud_phase) as file:
#             file.extract_ice_clouds()
#         return

#     def _process_l1c(self, intermediate_file: str) -> None:
#         """
#         Process level 1C IASI data.

#         Extracts and processes IASI data from intermediate binary files,
#         applies quality control and saves the output.

#         The result is a HDF5 file containing all good spectra from this intermediate file.
#         """
#         # Process extracted IASI data from intermediate binary files
#         with L1CProcessor(intermediate_file, self.config.targets) as file:
#             file.extract_spectra(self.datapath_out, self.datafile_out, self.year, self.month, self.day)
#         return

#     def process(self, intermediate_file: str) -> None:
#         """
#         Runs separate processors for the IASI data based on its level, because
#         each intermediate file is different. 

#         Raises:
#             ValueError: If the data level is neither 'l1c' nor 'l2'.
#         """
#         # Choose the processing function based on the data level
#         if self.data_level == 'l1c':
#             self._process_l1c(intermediate_file)
#         elif self.data_level == 'l2':
#             self._process_l2(intermediate_file)
#         else:
#             # If the data level is not 'l1c' or 'l2', raise an error
#             raise ValueError("Invalid data path type. Accepts 'l1c' or 'l2'.")


#     def _get_suffix(self):
#         old_suffix=".bin"
#         if self.data_level == 'l1c':
#             new_suffix=".bin"
#         elif self.data_level == 'l2':
#             new_suffix=".out"
#         else:
#             raise ValueError("Invalid data path type. Accepts 'l1c' or 'l2'.")
#         return old_suffix, new_suffix

#     def rename_files(self):
#         old_suffix, new_suffix = self._get_suffix()
#         if os.path.isdir(self.datapath_out):
#             for filename in os.scandir(self.datapath_out):
#                 if filename.name.endswith(old_suffix):
#                     new_filename = f"{filename.name[:-len(old_suffix)]}{new_suffix}"
#                     os.rename(filename.path, os.path.join(self.datapath_out, new_filename))


#     def correlate_l1c_l2(self):
#         with Correlator(self.datapath_out, self.datafile_out, self.config.cloud_phase) as file:
#             file.filter_spectra()
#         return


def main():
    """PISCO: Package for IASI Spectra and Cloud Observations

    For each date specified, open raw binary files, reduce into intermediate files using optimised C scripts
    developed by IASI team, then produce conveniently-formatted spatio-temporal data
    of IASI products: L1C calibrated spectra or L2 cloud products.
    """
    # Instantiate a Pisco class to get data from raw binary files
    extractor = Extractor()

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
                for level in extractor.config.data_level:
                    extractor.data_level = level
                    extractor.get_datapaths()
                    extractor.preprocess_files()
                    
                    


                    extractor.rename_files()
                    if extractor.config.mode == "Correlate":
                        # Correlate L1C spectra and L2 cloud products
                        extractor.correlate_l1c_l2()
                

if __name__ == "__main__":
    main()
