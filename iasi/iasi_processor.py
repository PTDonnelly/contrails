from datetime import datetime
import glob
import os
import numpy as np
import pandas as pd
from typing import Tuple, List, Union, Optional

import numpy as np


class L1CProcessor:
    """
    Processor for the intermediate binary file of IASI L1C products output by OBR script.

    Attributes:
    filepath (str): Path to the binary file.
    targets (List[str]): List of target variables to extract.
    """
    def __init__(self, filepath: str, config: object):
        self.filepath = filepath
        self.targets = config.targets


    def __enter__(self) -> 'L1CProcessor':
        """
        Opens the binary file and prepares the preprocessing.
        
        Returns:
        self: The L1CProcessor object itself.
        """
        # Open binary file
        print("Loading intermediate L1C file:")
        self.f = open(self.filepath, 'rb')
        
        # Get structure of file header and data record
        self.header_size, self.number_of_channels, self.channel_IDs = self._read_header()
        self.record_size = self._read_record_size()
        self.number_of_measurements = 5#self.count_measurements()
        self._print_metadata()

        # Get fields information and prepare to store extracted data
        self.fields = self._get_fields()
        self.field_data = {}
        return self


    def __exit__(self, type, value, traceback) -> None:
        """
        Ensure the file is closed when exiting the context.

        Args:
            type (Any): The exception type.
            value (Any): The exception value.
            traceback (Any): The traceback object.
        """
        self.f.close()


    def _print_metadata(self):
        print(f"Header  : {self.header_size} bytes")
        print(f"Record  : {self.record_size} bytes")
        print(f"Spectrum: {self.number_of_channels} channels")
        print(f"Data    : {self.number_of_measurements} measurements")
        return
    

    def _read_header(self) -> Tuple[int, int]:
        """
        Reads the header of the binary file to obtain the header size and number of channels.

        Returns:
        Tuple[int, int]: A tuple containing the header size and number of channels.
        """
        # Read header size
        header_size = np.fromfile(self.f, dtype='uint32', count=1)[0]

        # Skip 1 byte
        self.f.seek(1, 1)

        # Skip 3 int32 values
        np.fromfile(self.f, dtype='uint32', count=3)

        # Skip 1 boolean value
        np.fromfile(self.f, dtype='bool', count=1)

        # Read number of channels
        number_of_channels = np.fromfile(self.f, dtype='uint32', count=1)[0]
        
        # Read channel ID numbers
        channel_IDs = np.fromfile(self.f, dtype='uint32', count=number_of_channels)[0]
        
        # Check this is a list of strings !!!
        print(f"CHANNEL IDs: {channel_IDs}")

        # Verify the header size
        self.verify_header(header_size)
        return header_size, number_of_channels, channel_IDs


    def _verify_header(self, header_size: int) -> None:
        """
        Verifies the header size by comparing it with the header size at the end of the header.

        Args:
        header_size (int): The header size read at the beginning of the header.
        """
        # Reset file pointer to the beginning
        self.f.seek(0)

        # Skip first int32 value
        np.fromfile(self.f, dtype='uint32', count=1)

        # Skip header content
        np.fromfile(self.f, dtype='uint8', count=header_size)

        # Read header size at the end of the header
        header_size_check = np.fromfile(self.f, dtype='uint32', count=1)[0]

        # Check if header sizes match
        assert header_size == header_size_check, "Header size mismatch"


    def _read_record_size(self) -> Union[int, None]:
        self.f.seek(self.header_size + 8)
        record_size = np.fromfile(self.f, dtype='uint32', count=1)[0]
        return None if record_size == 0 else record_size


    def _get_fields(self) -> List[tuple]:
        # Format of fields in binary file (field_name, data_type, data_size, cumulative_data_size)
        fields = [
            ('year', 'uint16', 2, 2),
            ('month', 'uint8', 1, 3),
            ('day', 'uint8', 1, 4),
            ('hour', 'uint8', 1, 5),
            ('minute', 'uint8', 1, 6),
            ('millisecond', 'uint32', 4, 10),
            ('latitude', 'float32', 4, 14),
            ('longitude', 'float32', 4, 18),
            ('satellite_zenith_angle', 'float32', 4, 22),
            ('bearing', 'float32', 4, 26),
            ('solar_zentih_angle', 'float32', 4, 30),
            ('solar_azimuth', 'float32', 4, 34),
            ('field_of_view_number', 'uint32', 4, 38),
            ('orbit_number', 'uint32', 4, 42),
            ('scan_line_number', 'uint32', 4, 46),
            ('height_of_station', 'float32', 4, 50),
            ('day_version', 'uint16', 2, 52),
            ('start_channel_1', 'uint32', 4, 56),
            ('end_channel_1', 'uint32', 4, 60),
            ('quality_flag_1', 'uint32', 4, 64),
            ('start_channel_2', 'uint32', 4, 68),
            ('end_channel_2', 'uint32', 4, 72),
            ('quality_flag_2', 'uint32', 4, 76),
            ('start_channel_3', 'uint32', 4, 80),
            ('end_channel_3', 'uint32', 4, 84),
            ('quality_flag_3', 'uint32', 4, 88),
            ('cloud_fraction', 'uint32', 4, 92),
            ('surface_type', 'uint8', 1, 93)]
        return fields
        

    def _count_measurements(self) -> int:
        """
        Calculate the number of measurements in the binary file based on its size, 
        header size and record size.

        Returns:
        int: The number of measurements.
        """
        # Get the total size of the file
        file_size = self.f.seek(0, 2)

        # Calculate the number of measurements
        return (file_size - self.header_size - 8) // (self.record_size + 8)


    def _read_field_data(self) -> None:
        """
        Reads the data of each field from the binary file and store it in the field_data dictionary.

        This function only extracts the first 8 fields and the ones listed in the targets attribute.
        """
        # Iterate over each field
        for i, (field, dtype, dtype_size, cumsize) in enumerate(self.fields):
            if (i < 8) or (field in self.targets):
                print(f"Extracting: {field}")

                # Move the file pointer to the starting position of the current field
                self.f.seek(self.header_size + 12 + cumsize, 0)

                # Calculate the byte offset to the next measurement
                byte_offset = self.record_size + 8 - dtype_size

                # Prepare an empty array to store the data of the current field
                data = np.empty(self.number_of_measurements)

                # Read the data of each measurement
                for measurement in range(self.number_of_measurements):
                    value = np.fromfile(self.f, dtype=dtype, count=1, sep='', offset=byte_offset)
                    data[measurement] = np.nan if len(value) == 0 else value[0]

                # Store the data in the field_data dictionary
                self.field_data[field] = data

        # Store datetime components field at the end of dictionary for later construction
        self.field_data["datetime"] = [int(self.field_data['year']),
                                        int(self.field_data['month']),
                                        int(self.field_data['day']),
                                        int(self.field_data['hour']),
                                        int(self.field_data['minute']),
                                        int(self.field_data['millisecond']/10000)]
        

    def _calculate_local_time(self) -> np.ndarray:
        """
        Calculate the local time (in hours, UTC) that determines whether it is day or night at a specific longitude.

        Returns:
        np.ndarray: Local time (in hours, UTC) within a 24 hour range, used to determine day (6-18) or night (0-6, 18-23).
        """

        # Retrieve the necessary field data
        hour, minute, millisecond, longitude = self.field_data['hour'], self.field_data['minute'], self.field_data['millisecond'], self.field_data['longitude']

        # Calculate the total time in hours, minutes, and milliseconds
        total_time = (hour * 1e4) + (minute * 1e2) + (millisecond / 1e3)

        # Extract the hour, minute and second components from total_time
        hour_component = np.floor(total_time / 10000)
        minute_component = np.mod((total_time - np.mod(total_time, 100)) / 100, 100)
        second_component_in_minutes = np.mod(total_time, 100) / 60

        # Calculate the total time in hours
        total_time_in_hours = (hour_component + minute_component + second_component_in_minutes) / 60

        # Convert longitude to time in hours and add it to the total time
        total_time_with_longitude = total_time_in_hours + (longitude / 15)

        # Add 24 hours to the total time to ensure it is always positive
        total_time_positive = total_time_with_longitude + 24

        # Take the modulus of the total time by 24 to get the time in the range of 0 to 23 hours
        time_in_range = np.mod(total_time_positive, 24)

        # Subtract 6 hours from the total time, shifting the reference for day and night (so that 6 AM becomes 0)
        time_shifted = time_in_range - 6

        # Take the modulus again to ensure the time is within the 0 to 23 hours range
        return np.mod(time_shifted, 24)
    

    def _store_space_time_coordinates(self) -> List:
        """
        Stores the space-time coordinates and a Boolean indicating whether the current time is day or night.

        Returns:
            List: A list containing longitude, latitude and a Boolean indicating day or night. 
        """
        # Calculate the local time
        local_time = self._calculate_local_time()

        # Return the longitude, latitude, and a Boolean indicating day (True) or night (False)
        return self.field_data['latitude'], self.field_data['longitude'], (6 < local_time < 18)


    def _store_spectral_radiance(self) -> np.ndarray:
        """
        Extracts and stores the spectral radiance measurements from the binary file.

        Returns:
            np.ndarray: An array of spectral radiance measurements for each channel.
        """
        print("Extracting: radiance")

        # Determine the position of the anchor point for spectral radiance data in the binary file
        last_field_end = self.fields[-1][-1] # End of the surface_type field

        # Go to spectral radiance data (skip header and previous record data, "12"s are related to reading )
        start_read_position = self.header_size + 12 + last_field_end + (4 * self.number_of_channels) #12
        self.f.seek(start_read_position, 0)
        
        # Calculate the offset to skip to the next measurement
        byte_offset = self.record_size + 8 - (4 * self.number_of_channels)
        
        # Initialize an empty numpy array to store the spectral radiance data
        data = np.empty((self.number_of_channels, self.number_of_measurements))

        # Iterate over each measurement and extract the spectral radiance data
        for measurement in range(self.number_of_measurements):
            value = np.fromfile(self.f, dtype='float32', count=self.number_of_channels, sep='', offset=byte_offset)
            data[:, measurement] = np.nan if len(value) == 0 else value

        # Return the array of spectral radiance data
        return data


    def _store_target_parameters(self) -> List:
        """
        Stores the target parameters from the field data.

        Returns:
            List: A list of the target parameters from the field data.
        """
        target_parameter_names = [field for field, _ in self.field_data.items() if (field in self.targets)]
        target_parameters =  [data for field, data in self.field_data.items() if (field in self.targets)]
        return target_parameter_names, target_parameters


    def _store_datetime_components(self) -> List:
        """
        Stores the datetime components from the datetime field data.

        Returns:
            List: A list of the datetime components from the field data (formatted like the outputs of the L2 reader)
        """
        return [f"{d[0]}{d[1]}{d[2]}.{d[3]}{d[4]}{d[5]}" for d in self.field_data['datetime']]


    def _build_header(self, target_parameter_names: List[str]):
        # Add the main data columns to the header
        header = ["Latitude", "Longitude", "Datetime", "Local Time"]
        # Add the IASI channel IDs: List[str]
        header.extend(self.channel_IDs)
        # Add the L1C target parameter names: List[str]
        header.extend(target_parameter_names)
        return header
    

    def extract_data(self) -> Tuple[List[int], np.array]:
        """
        Extracts relevant data from the observation dataset, processes it, and consolidates it into a 2D array.

        Returns:
            Tuple[List[int], np.array]: A tuple containing a header (which represents lengths of various components of data)
            and a 2D numpy array of consolidated data.

        """
        # Extract and process binary data
        self._read_field_data()
        latitude, longitude, local_time = self._store_space_time_coordinates()
        radiances = self._store_spectral_radiance()
        target_parameter_names, target_parameters = self._store_target_parameters()
        datetimes = self._store_datetime_components()

        # Concatenate processed observations into a single 2D array (number of parameters x number of measurements).
        data = np.concatenate((latitude, longitude, datetimes, local_time, radiances, target_parameters), axis=0)

        # Construct a header that contains the name of each data column
        header = self._build_header(target_parameter_names)
        return header, data


    def filter_bad_observations(self, data: np.array, date: object) -> np.array:
        """
        Filters bad observations based on IASI L1 data quality flags and date.
        
        Args:
            data (np.ndarray): A numpy array containing the observation data.
            date (object): The date for filtering the observations.
            
        Returns:
            np.ndarray: A filtered numpy array containing the good observations.
        """
        if date <= datetime(2012, 2, 8):
            check_quality_flag = data[-3, :] == 0
            check_data = np.sum(data[2:-4, :], axis=1) > 0
            good_flag = np.logical_and(check_quality_flag, check_data)
        else:
            check_quality_flags = np.logical_and(self.field_data['quality_flag_1'][:] == 0,
                                                self.field_data['quality_flag_2'][:] == 0,
                                                self.field_data['quality_flag_3'][:] == 0)
            # check_quality_flags = np.logical_and(data[-11, :] == 0, data[-10, :] == 0, data[-9, :] == 0)
            # check_data = np.sum(data[0:-6, :], axis=1) > 0
        good_flag = check_quality_flags #np.logical_and(check_quality_flags)#, check_data)
        
        good_data = data[:, good_flag]
        print(f"{np.round((data.shape[1] / good_data.shape[1]) * 100, 2)} % good data of {data.shape[1]} observations")
        return good_data

    @staticmethod
    def save_observations(datapath_out: str, datafile_out: str, header: list, data: np.array) -> None:
        """
        Saves the observation data to a file.
        
        Args:
            datapath_out (str): The path to save the file.
            datafile_out (str): The name of the output file.
            header (list): A list of integers to be written as the first line.
            data (np.ndarray): A numpy array containing the observation data.
            
        Returns:
            None
        """
        # # Transpose the data to match the previous structure
        # data = np.transpose(data)
        
        # Create a DataFrame with the transposed data
        df = pd.DataFrame(data, columns=header)
        
        # Save the DataFrame to a file in HDF5 format
        # df.to_hdf(f"{datapath_out}{datafile_out}.h5", key='df', mode='w')
        df.to_csv(f"{datapath_out}{datafile_out}.csv", columns=header, index=False, mode='w')
        return

    
    def extract_spectra(self, datapath_out: str, year: str, month: str, day: str):
        # Extract and process binary data
        header, all_data = self.extract_data()
        
        # Check observation quality and filter out bad observations
        good_data = self.filter_bad_observations(all_data, date=datetime(self.year, self.month, self.day))

        # Define the output filename and save outputs
        # datafile_out = f"IASI_L1C_{year}_{month}_{day}"
        self.save_observations(datapath_out, datafile_out, header, good_data)


class L2Processor:
    """
    Processor for the intermediate binary file of IASI L2 products output by OBR script.

    Attributes:
    filepath (str): Path to the binary file.
    """
    def __init__(self, filepath: str, config: object):
        self.filepath = filepath
        self.lat_range = config.latitude_range
        self.lon_range = config.longitude_range
        self.extracted_columns = None
        self.filtered_data = None
        self.cloud_phase = config.cloud_phase


    def __enter__(self) -> 'L2Processor':
        """
        Opens the csv file and prepares the preprocessing.
        
        Returns:
        self: The L2Processor object itself.
        """
        # Open csv file
        print("Loading intermediate L2 file:")
        self.df = pd.read_csv(self.filepath, sep='\s+', header=None)
        self.df.columns = ['Latitude', 'Longitude', 'Datetime', 'Orbit Number','Scanline Number', 'Pixel Number',
                        'Cloud Fraction', 'Cloud-top Temperature', 'Cloud-top Pressure', 'Cloud Phase',
                        'Column11', 'Column12', 'Column13', 'Column14', 'Column15', 'Column16', 'Column17', 'Column18']
        return self


    def __exit__(self, type, value, traceback) -> None:
        """
        Ensure the file is closed when exiting the context.

        Args:
            type (Any): The exception type.
            value (Any): The exception value.
            traceback (Any): The traceback object.
        """
        self.df.close()
    

    def _save_data(self):
        # Save the data to a file
        outfile = self.filepath.split('.')[0]
        self.filtered_data.to_csv(f"{outfile}.csv", columns=self.extracted_columns, index=False, mode='w')
        # self.filtered_data.to_hdf(f"{outfile}.h5", key='df', mode='w')


    def _filter_data(self):
        # Specify the columns that are to be extracted
        self.extracted_columns = ['Latitude', 'Longitude', 'Datetime', 'Cloud Fraction', 'Cloud-top Temperature', 'Cloud-top Pressure', 'Cloud Phase']
        data = []
        for _, row in self.df.iterrows():
            if (self.lat_range[0] <= row['Latitude'] <= self.lat_range[1]) and (self.lon_range[0] <= row['Longitude'] <= self.lon_range[1]):
                if (row['Cloud Phase'] == self.cloud_phase) and np.isfinite(row['Cloud-top Temperature']): 
                    data.append([row[column] for column in self.extracted_columns])
        self.filtered_data =  pd.DataFrame(data, columns=self.extracted_columns)

    
    def extract_ice_clouds(self):
        self._filter_data()
        self._save_data()


class Correlator:
    def __init__(self, datapath_out: str, datafile_out: str, cloud_phase: int):
        self.datapath_out: str = datapath_out
        self.datafile_out: str = datafile_out
        self.cloud_phase: int = cloud_phase
        self.df_l1c: object = None
        self.df_l2: object = None


    def __enter__(self) -> 'Correlator':
        """
        Opens two DataFrames loaded from the intermediate analysis data files.
        
        Returns:
        self: The Correlator object itself.
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