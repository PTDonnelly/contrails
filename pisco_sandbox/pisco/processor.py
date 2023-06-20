from datetime import datetime
import os
import numpy as np
import pandas as pd
from typing import Tuple, List, Union

import numpy as np

from .extractor import Extractor as ex


class L1CProcessor:
    """
    Processor for the intermediate binary file of IASI L1C products output by OBR script.

    Attributes:
    filepath (str): Path to the binary file.
    targets (List[str]): List of target variables to extract.
    """
    def __init__(self, filepath: str, targets: List[str]):
        self.filepath = filepath
        self.targets = targets


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
        self.skip_measurements = 100
        self.number_of_measurements = 1000#self._count_measurements()# // self.skip_measurements
        self._print_metadata()

        # Get fields information and prepare to store extracted data in an empty DataFrame
        self.fields = self._get_fields()
        self.field_df = pd.DataFrame()
        return self


    def __exit__(self, type, value, traceback) -> None:
        """
        Ensure the original binary file is closed when exiting the context, and delete it.

        Args:
            type (Any): The exception type.
            value (Any): The exception value.
            traceback (Any): The traceback object.
        """
        self.f.close()
        os.remove(self.filepath)


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
        channel_IDs = np.fromfile(self.f, dtype='uint32', count=number_of_channels)

        # Verify the header size
        self._verify_header(header_size)
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
                field_start = self.header_size + 12 + cumsize
                self.f.seek(field_start, 0)

                # Calculate the byte offset to the next measurement
                byte_offset = self.record_size + 8 - dtype_size

                # Prepare an empty array to store the data of the current field
                data = np.empty(self.number_of_measurements)
                
                # Read the data of each measurement
                for measurement in range(self.number_of_measurements):
                    # # Move the file pointer to the starting position of the current field
                    # self.f.seek(field_start * self.skip_measurements * measurement, 0)
                    
                    # Read bytes
                    value = np.fromfile(self.f, dtype=dtype, count=1, sep='', offset=byte_offset)
                    data[measurement] = np.nan if len(value) == 0 else value[0]

                # Store the data in the DataFrame
                self.field_df[field] = data
        print(self.field_df.head())

    def _calculate_local_time(self) -> np.ndarray:
        """
        Calculate the local time (in hours, UTC) that determines whether it is day or night at a specific longitude.

        Returns:
        np.ndarray: Local time (in hours, UTC) within a 24 hour range, used to determine day (6-18) or night (0-6, 18-23).
        """

        # Retrieve the necessary field data
        hour, minute, millisecond, longitude = self.field_df['hour'], self.field_df['minute'], self.field_df['millisecond'], self.field_df['longitude']

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

    def _store_space_time_coordinates(self) -> None:
        """
        Stores the local time Boolean indicating whether the current time is day or night.
        """
        # Calculate the local time
        local_time = self._calculate_local_time()

        # Store the Boolean indicating day (True) or night (False) in the DataFrame
        self.field_df['Local Time'] = (6 < local_time) & (local_time < 18)
        return

    def _read_spectral_radiance(self) -> None:
        """
        Extracts and stores the spectral radiance measurements from the binary file.
        """
        print("Extracting: radiance")

        # Determine the position of the anchor point for spectral radiance data in the binary file
        last_field_end = self.fields[-1][-1] # End of the surface_type field for the last measurement

        # Go to spectral radiance data (skip header and previous record data, "12"s are related to reading)
        spectrum_start = self.header_size + 12 + last_field_end + (4 * self.number_of_channels) 
        self.f.seek(spectrum_start, 0)

        # Calculate the offset to skip to the next measurement
        byte_offset = self.record_size + 8 - (4 * self.number_of_channels)
        
        # Initialize an empty numpy array to store the spectral radiance data
        data = np.empty((self.number_of_channels, self.number_of_measurements))

        # Iterate over each measurement and extract the spectral radiance data
        for measurement in range(self.number_of_measurements):
            # Move the file pointer to the starting position of the current field
            # self.f.seek(spectrum_start * self.skip_measurements * measurement, 0)

            # Read bytes
            value = np.fromfile(self.f, dtype='float32', count=self.number_of_channels, sep='', offset=byte_offset)
            data[:, measurement] = np.nan if len(value) == 0 else value

        # Assign channel IDs and values to DataFrame
        for i, id in enumerate(self.channel_IDs):
            self.field_df[f'Channel {id}'] = data[i, :]
        return

    def _store_datetime_components(self) -> List:
        """
        Stores the datetime components from the datetime field data.

        Returns:
            np.Array: An array of the datetime components from the transposed field data (formatted like the outputs of the L2 reader)
        """
        # Create 'Datetime' column
        self.field_df['Datetime'] = self.field_df['year'].apply(lambda x: f'{int(x):04d}') + \
                                    self.field_df['month'].apply(lambda x: f'{int(x):02d}') + \
                                    self.field_df['day'].apply(lambda x: f'{int(x):02d}') + '.' + \
                                    self.field_df['hour'].apply(lambda x: f'{int(x):02d}') + \
                                    self.field_df['minute'].apply(lambda x: f'{int(x):02d}') + \
                                    self.field_df['millisecond'].apply(lambda x: f'{int(x/10000):02d}')

        # Drop original time element columns
        self.field_df = self.field_df.drop(columns=['year', 'month', 'day', 'hour', 'minute', 'millisecond'])
        return  

    def filter_bad_observations(self, date: object) -> pd.DataFrame:
        """
        Filters bad observations based on IASI L1 data quality flags and date.
        
        Args:
            data (pd.DataFrame): A DataFrame containing the observation data.
            date (object): The date for filtering the observations.
            
        Returns:
            pd.DataFrame: A filtered DataFrame containing the good observations.
        """
        print("Filtering spectra:")

        if date <= datetime(2012, 2, 8):
            check_quality_flag = self.field_df['quality_flag'] == 0
            check_data = self.field_df.drop(['quality_flag', 'date_column'], axis=1).sum(axis=1) > 0
            good_flag = check_quality_flag & check_data
        else:
            check_quality_flags = (self.field_df['quality_flag_1'] == 0) & (self.field_df['quality_flag_2'] == 0) & (self.field_df['quality_flag_3'] == 0)
            good_flag = check_quality_flags

        # Throw away bad data, return the good
        good_df = self.field_df[good_flag]
        print(f"{np.round((len(good_df) / len(self.field_df)) * 100, 2)} % good data of {len(self.field_df)} observations")
        return good_df

    @staticmethod
    def save_observations(datapath_out: str, datafile_out: str, good_df: pd.DataFrame) -> None:
        """
        Saves the observation data to a file.
        
        Args:
            datapath_out (str): The path to save the file.
            datafile_out (str): The name of the output file.
            good_data (pd.DataFrame): A pandas DataFrame containing the observation data.
        """        
        # Save the DataFrame to a file in HDF5 format
        outfile = f"{datapath_out}{datafile_out}".split(".")[0]
        # good_df.to_hdf(f"{datapath_out}{datafile_out}.h5", key='df', mode='w')
        good_df.to_csv(f"{outfile}.csv", index=False, mode='w')
        return

    def extract_data(self, datapath_out: str, datafile_out: str, year: str, month: str, day: str):
        # Extract and process binary data
        self._read_field_data()
        self._store_space_time_coordinates()
        self._read_spectral_radiance()
        self._store_datetime_components()

        # print the DataFrame
        print(self.field_df.head())
        
        # Check observation quality and filter out bad observations
        date = datetime(int(year), int(month), int(day))
        good_df = self.filter_bad_observations(date=date)

        # Define the output filename and save outputs
        self.save_observations(datapath_out, datafile_out, good_df)


class IASIMetadata:
    """
    """
    def __init__(self, file: object):
        self.f: object = file
        self.header_size: int = None
        self.header_size : int = None 
        self.byte_order : int = None 
        self.format_version : int = None 
        self.satellite_identifier : int = None 
        self.record_header_size : int = None 
        self.brightness_temperature_brilliance : int = None 
        self.number_of_channels : int = None 
        self.channel_IDs : int = None
        self.AVHRR_brilliance : int = None 
        self.number_of_L2_sections : int = None 
        self.table_of_L2_sections : int = None
        self.record_size: int = None
        self.number_of_measurements: int = None

    def count_measurements(self) -> int:
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

    def read_record_size(self) -> Union[int, None]:
        self.f.seek(self.header_size + 8)
        record_size = np.fromfile(self.f, dtype='uint32', count=1)[0]
        return None if record_size == 0 else record_size

    def _verify_header(self) -> None:
        """
        Verifies the header size by comparing it with the header size at the end of the header.
        """
        # Read header size at the end of the header
        header_size_check = np.fromfile(self.f, dtype='uint32', count=1)[0]

        # Check if header sizes match
        assert self.header_size == header_size_check, "Header size mismatch"

    def read_iasi_common_header_metadata(self) -> Tuple[int, int]:
        """
        Reads the header of the binary file to obtain the header size and number of channels.

        Returns:
        Tuple[int, int]: A tuple containing the header size and number of channels.
        """
        # Read header entries
        self.header_size = np.fromfile(self.f, dtype='uint32', count=1)[0]
        self.byte_order = np.fromfile(self.f, dtype='uint8', count=1)[0]
        self.format_version = np.fromfile(self.f, dtype='uint32', count=1)[0]
        self.satellite_identifier = np.fromfile(self.f, dtype='uint32', count=1)[0]
        self.record_header_size = np.fromfile(self.f, dtype='uint32', count=1)[0]
        self.brightness_temperature_brilliance = np.fromfile(self.f, dtype='bool', count=1)[0]
        self.number_of_channels = np.fromfile(self.f, dtype='uint32', count=1)[0]
        self.channel_IDs = np.fromfile(self.f, dtype='uint32', count=number_of_channels)
        self.AVHRR_brilliance = np.fromfile(self.f, dtype='bool', count=1)[0]
        self.number_of_L2_sections = np.fromfile(self.f, dtype='uint16', count=1)[0]
        self.table_of_L2_sections = np.fromfile(self.f, dtype='uint32', count=number_of_L2_sections)[0]

        # Read header size at the end of the header, check for a match
        self._verify_header()       
        return
    
    def get_iasi_common_header(self):
        self.header_size, self.number_of_channels, self.channel_IDs = self.read_iasi_common_header_metadata()
        self.record_size = self.read_record_size()
        self.number_of_measurements = self.count_measurements()

    def get_iasi_common_record_fields(self) -> List[tuple]:
        # Format of fields in binary file (field_name, data_type, data_size, cumulative_data_size)
        common_fields = [
            ('year', 'uint16', 2),
            ('month', 'uint8', 1),
            ('day', 'uint8', 1),
            ('hour', 'uint8', 1),
            ('minute', 'uint8', 1),
            ('millisecond', 'uint32', 4,),
            ('latitude', 'float32', 4,),
            ('longitude', 'float32', 4,),
            ('satellite_zenith_angle', 'float32', 4,),
            ('bearing', 'float32', 4,),
            ('solar_zentih_angle', 'float32', 4,),
            ('solar_azimuth', 'float32', 4,),
            ('field_of_view_number', 'uint32', 4,),
            ('orbit_number', 'uint32', 4,),
            ('scan_line_number', 'uint32', 4,),
            ('height_of_station', 'float32', 4,)]
        return common_fields
    

    def get_iasi_l1c_record_fields(self) -> List[tuple]:
        # Format of fields in binary file (field_name, data_type, data_size, cumulative_data_size)
        l1c_fields = [
            ('day_version', 'uint16', 2),
            ('start_channel_1', 'uint32', 4),
            ('end_channel_1', 'uint32', 4),
            ('quality_flag_1', 'uint32', 4),
            ('start_channel_2', 'uint32', 4),
            ('end_channel_2', 'uint32', 4),
            ('quality_flag_2', 'uint32', 4),
            ('start_channel_3', 'uint32', 4),
            ('end_channel_3', 'uint32', 4),
            ('quality_flag_3', 'uint32', 4),
            ('cloud_fraction', 'uint32', 4),
            ('surface_type', 'uint8', 1)]
        return l1c_fields
    
    def get_iasi_l2_record_fields(self) -> List[tuple]:
        # Format of fields in binary file (field_name, data_type, data_size, cumulative_data_size)
        l2_fields = [
            ('superadiabatic_indicator', 'uint8', 1),
            ('land_sea_qualifier', 'uint8', 1),
            ('day_nght_qualifier', 'uint8', 1),
            ('processing_technique', 'uint32', 4),
            ('sun_glint_indicator', 'uint8', 1),
            ('cloud_formation_and_height_assignment', 'uint32', 4),
            ('instrument_detecting_clouds', 'uint32', 4),
            ('validation_flag_for_IASI_L1_product', 'uint32', 4),
            ('quality_completeness_of_retrieval', 'uint32', 4),
            ('retrieval_choice_indicator', 'uint32', 4),
            ('satellite_maoeuvre_indicator', 'uint32', 4),]
        return l2_fields
    
    def get_ozo_record_fields(self):
        pass

    def get_trg_record_fields(self):
        pass
    
    def get_clp_record_fields(self):
        pass

    def get_twt_record_fields(self):
        pass

    def get_ems_record_fields(self):
        pass

class L2Processor:
    """
    Processor for the intermediate binary file of IASI L2 products output by OBR script.

    Attributes:
    filepath (str): Path to the binary file.
    targets (List[str]): List of target variables to extract.
    """
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.f: object = None
        self.header: IASIMetadata = None
        self.field_df = pd.DataFrame

      
    def read_binary_file(self):
        # Open binary file
        print("Loading intermediate L2 file:")
        self.f = open(self.filepath, 'rb')
        
        # Get structure of file header and data record
        self.header = IASIMetadata(self.f)
        self.header.get_iasi_common_header()


    def read_record_fields(self) -> None:
        """
        Reads the data of each field from the binary file and store it in the field_data dictionary.

        This function only extracts the first 8 fields and the ones listed in the targets attribute.
        """
        # Read in binary field table
        fields = self.header.get_iasi_l2_record_fields()

        # Iterate over each field
        for field, dtype, dtype_size in fields:
            print(f"Extracting: {field}")
            
            # Start counting number of bytes passed
            cumsize += dtype_size

            # Move the file pointer to the starting position of the current field
            field_start = self.header.header_size + 12 + cumsize
            self.f.seek(field_start, 0)

            # Calculate the byte offset to the next measurement
            byte_offset = self.header.record_size + 8 - dtype_size

            # Prepare an empty array to store the data of the current field
            data = np.empty(self.header.number_of_measurements)
            
            # Read the data of each measurement
            for measurement in range(self.header.number_of_measurements):
                value = np.fromfile(self.f, dtype=dtype, count=1, sep='', offset=byte_offset)
                data[measurement] = np.nan if len(value) == 0 else value[0]

            # Store the data in the DataFrame
            self.field_df[field] = data
        print(self.field_df.head())




    def close_binary_file(self):
        self.f.close()


    # def filter_bad_observations(self, date: object) -> pd.DataFrame:
    #     """
    #     Filters bad observations based on IASI L1 data quality flags and date.
        
    #     Args:
    #         data (pd.DataFrame): A DataFrame containing the observation data.
    #         date (object): The date for filtering the observations.
            
    #     Returns:
    #         pd.DataFrame: A filtered DataFrame containing the good observations.
    #     """
    #     print("Filtering spectra:")

    #     if date <= datetime(2012, 2, 8):
    #         check_quality_flag = self.field_df['quality_flag'] == 0
    #         check_data = self.field_df.drop(['quality_flag', 'date_column'], axis=1).sum(axis=1) > 0
    #         good_flag = check_quality_flag & check_data
    #     else:
    #         check_quality_flags = (self.field_df['quality_flag_1'] == 0) & (self.field_df['quality_flag_2'] == 0) & (self.field_df['quality_flag_3'] == 0)
    #         good_flag = check_quality_flags

    #     # Throw away bad data, return the good
    #     good_df = self.field_df[good_flag]
    #     print(f"{np.round((len(good_df) / len(self.field_df)) * 100, 2)} % good data of {len(self.field_df)} observations")
    #     return good_df

    # @staticmethod
    # def save_observations(datapath_out: str, datafile_out: str, good_df: pd.DataFrame) -> None:
    #     """
    #     Saves the observation data to a file.
        
    #     Args:
    #         datapath_out (str): The path to save the file.
    #         datafile_out (str): The name of the output file.
    #         good_data (pd.DataFrame): A pandas DataFrame containing the observation data.
    #     """        
    #     # Save the DataFrame to a file in HDF5 format
    #     outfile = f"{datapath_out}{datafile_out}".split(".")[0]
    #     # good_df.to_hdf(f"{datapath_out}{datafile_out}.h5", key='df', mode='w')
    #     good_df.to_csv(f"{outfile}.csv", index=False, mode='w')
    #     return

    
    # def extract_spectra(self, datapath_out: str, datafile_out: str, year: str, month: str, day: str):
    #     # Extract and process binary data
    #     self._read_field_data()
    #     # self._store_space_time_coordinates()
    #     # self._read_spectral_radiance()
    #     # # self._store_target_parameters()
    #     # self._store_datetime_components()

    #     # print the DataFrame
    #     print(self.field_df.head())
        
    #     # # Check observation quality and filter out bad observations
    #     # date = datetime(int(year), int(month), int(day))
    #     # good_df = self.filter_bad_observations(date=date)

    #     # # Define the output filename and save outputs
    #     # self.save_observations(datapath_out, datafile_out, good_df)