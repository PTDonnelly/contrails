import os
import pandas as pd
from typing import Optional

class L1C_L2_Correlator:
    def __init__(self, datapath_out: str, year: str, month: str, day: str, cloud_phase: int):
        self.datapath_out: str = datapath_out
        self.date: str = f"{year}/{month}/{day}/"
        # self.datafile_out: str = datafile_out
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
        self.datafile_l1c = f"{self.datapath_out}l1c/{self.date}extracted_spectra.csv"
        self.datafile_l2 = f"{self.datapath_out}l2/{self.date}cloud_products.csv"


    def _delete_intermediate_analysis_data(self) -> None:
        """
        Delete the intermediate analysis data files used for correlating spectra and clouds.
        """
        os.remove(self.datafile_l1c)
        # os.remove(self.datafile_l2)


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


    def _save_merged_data(self, merged_df_day: pd.DataFrame, merged_df_night: pd.DataFrame) -> None:
        """
        Save the merged DataFrame to a CSV file in the output directory.
        If the output directory is unknown (because the cloud phase is unknown), print a message and return.
        """
        datapath_out = self._build_output_directory_path()
        if datapath_out is None:
            print("Cloud_phase is unknown or uncertain, skipping data.")
        else:
            print(f"Saving final spectra for {datapath_out}")
            # final_file = f"{datapath_out}extracted_spectra.csv"
            # merged_df.to_csv(final_file, index=False)
            # # Save the DataFrame to a file in csv format, split by local time
            # df.to_hdf(f"{datapath_out}{datafile_out}.h5", key='df', mode='w')
            merged_df_day.to_csv(f"{datapath_out}day_extracted_spectra.csv", index=False, mode='w')
            merged_df_night.to_csv(f"{datapath_out}night_extracted_spectra.csv", index=False, mode='w')
        return


    def _correlate_measurements(self) -> pd.DataFrame:
        """
        Merge two DataFrames based on latitude, longitude and datetime. 
        The latitude and longitude values are rounded to 2 decimal places.
        Rows from df_l1c that do not have a corresponding row in df_l2 are dropped.

        Then separate into day and night observations
        """
        print(self.df_l1c.columns)
        print(self.df_l2.columns)
        decimal_places = 2
        self.df_l1c['Latitude', 'Longitude'] = self.df_l1c['Latitude', 'Longitude'].round(decimal_places)
        self.df_l2['Latitude', 'Longitude'] = self.df_l2['Latitude', 'Longitude'].round(decimal_places)

        merged_df = pd.merge(self.df_l1c, self.df_l2, on=['Latitude', 'Longitude', 'Datetime'], how='inner')


        # Convert the DataFrame 'Local Time' column (np.array) to boolean values
        merged_df['Local Time'] = merged_df['Local Time'].astype(bool)
        # Split the DataFrame into two based on 'Local Time' column
        merged_df_day = merged_df[merged_df['Local Time'] == True]
        merged_df_night = merged_df[merged_df['Local Time'] == False]
        # # Drop the 'Local Time' column from both DataFrames
        # merged_df_day = merged_df_day.drop(columns=['Local Time'])
        # merged_df_night = merged_df_night.drop(columns=['Local Time'])
        # # Remove 'Local Time' from the header list
        # header.remove('Local Time')
        
        return merged_df_day.dropna(), merged_df_night.dropna()


    def filter_spectra(self) -> None:
        """
        Loads the data, correlates measurements, saves the merged data, and deletes the original data.
        """
        merged_df_day, merged_df_night = self._correlate_measurements()
        self._save_merged_data(merged_df_day, merged_df_night)
        self._delete_intermediate_analysis_data()