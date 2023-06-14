import os
import pandas as pd
from typing import Optional

class L1C_L2_Correlator:
    def __init__(self, datapath_out: str, datafile_out: str, cloud_phase: int):
        self.datapath_out: str = datapath_out
        self.datafile_out: str = datafile_out
        self.cloud_phase: int = cloud_phase
        self.df_l1c: object = None
        self.df_l2: object = None


    def __enter__(self) -> 'L1C_L2_Correlator':
        """
        Opens two DataFrames loaded from the intermediate analysis data files.
        
        Returns:
        self: The L1C_L2_Correlator object itself.
        """
        # Open csv file
        print("Loading L1C spectra and L2 cloud products:")
        self._get_intermediate_analysis_data_paths()
        self.df_l1c, self.df_l2 = pd.read_csv(self.datafile_l1c), pd.read_csv(self.datafile_l2)
        return self


    def __exit__(self, type, value, traceback) -> None:
        """
        Ensure the file is closed when exiting the context.

        Args:
            type (Any): The exception type.
            value (Any): The exception value.
            traceback (Any): The traceback object.
        """
        # self.df_l1c.close()
        # self.df_l2.close()
        pass


    def _get_intermediate_analysis_data_paths(self) -> None:
        """
        Defines the paths to the intermediate analysis data files.
        """
        self.datafile_l1c = f"{self.datapath_out}L1C_test.csv"
        self.datafile_l2 = f"{self.datapath_out}L2_test.csv"


    def _delete_intermediate_analysis_data(self) -> None:
        """
        Delete the intermediate analysis data files used for correlating spectra and clouds.
        """
        os.remove(self.datafile_l1c)
        os.remove(self.datafile_l2)


    def _get_cloud_phase(self) -> Optional[str]:
        """
        Returns the cloud phase as a string based on the cloud phase value.
        If the retrieved cloud phase is unknown or uncertain, returns None.
        """
        cloud_phase_dictionary = {1: "aqueous", 2: "icy", 3: "mixed", 4: "clear"}
        return cloud_phase_dictionary.get(self.cloud_phase)


    def _build_output_directory_path(self) -> Optional[str]:
        """
        Returns the output directory path based on the cloud phase.
        If the cloud phase is unknown, returns None.
        """
        cloud_phase = self._get_cloud_phase()
        return None if cloud_phase is None else f"{self.datapath_out}{cloud_phase}/"


    def _save_merged_data(self, merged_df: pd.DataFrame) -> None:
        """
        Save the merged DataFrame to a CSV file in the output directory.
        If the output directory is unknown (because the cloud phase is unknown), print a message and return.
        """
        datapath_out = self._build_output_directory_path()
        if datapath_out is None:
            print("Cloud_phase is unknown or uncertain, skipping data.")
        else:
            final_file = f"{datapath_out}{self.datafile_out}.csv"
            print(f"Saving: {final_file}")
            merged_df.to_csv(final_file, index=False)
        return


    def _correlate_measurements(self) -> pd.DataFrame:
        """
        Merge two DataFrames based on latitude, longitude and datetime. 
        The latitude and longitude values are rounded to 2 decimal places.
        Rows from df_l1c that do not have a corresponding row in df_l2 are dropped.
        """
        decimal_places = 2
        self.df_l1c[['Latitude', 'Longitude']] = self.df_l1c[['Latitude', 'Longitude']].round(decimal_places)
        self.df_l2[['Latitude', 'Longitude']] = self.df_l2[['Latitude', 'Longitude']].round(decimal_places)

        merged_df = pd.merge(self.df_l1c, self.df_l2, on=['Latitude', 'Longitude', 'Datetime'], how='inner')
        return merged_df.dropna()


    def filter_spectra(self) -> None:
        """
        Loads the data, correlates measurements, saves the merged data, and deletes the original data.
        """
        merged_df = self._correlate_measurements()
        self._save_merged_data(merged_df)
        # self._delete_intermediate_analysis_data()