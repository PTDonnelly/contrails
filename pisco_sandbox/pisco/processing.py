from datetime import datetime
import os
import numpy as np
import pandas as pd
from typing import Tuple, List, Union

import numpy as np

from .extraction import Extractor as ex


class Metadata:
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
    
    def _print_metadata(self):
        print(f"Header  : {self.header_size} bytes")
        print(f"Record  : {self.record_size} bytes")
        print(f"Spectrum: {self.number_of_channels} channels")
        print(f"Data    : {self.number_of_measurements} measurements")
        return

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
        self.number_of_measurements = 100 #(file_size - self.header_size - 8) // (self.record_size + 8)
        return
    
    def _read_record_size(self) -> Union[int, None]:
        self.f.seek(self.header_size + 8, 0)
        self.record_size = np.fromfile(self.f, dtype='uint32', count=1)[0]
        return
    
    def _verify_header(self) -> None:
        """
        Verifies the header size by comparing it with the header size at the end of the header.
        """
        # Reset file pointer to the beginning
        self.f.seek(0)

        # Skip first int32 value
        np.fromfile(self.f, dtype='uint32', count=1)

        # Skip header content
        np.fromfile(self.f, dtype='uint8', count=self.header_size)

        # Read header size at the end of the header
        header_size_check = np.fromfile(self.f, dtype='uint32', count=1)[0]

        # Check if header sizes match
        assert self.header_size == header_size_check, "Header size mismatch"

    def _read_iasi_common_header_metadata(self) -> Tuple[int, int]:
        """
        Reads the header of the binary file to obtain the header size and number of channels.
        """
        # Read header entries
        self.header_size = np.fromfile(self.f, dtype='uint32', count=1)[0]
        self.byte_order = np.fromfile(self.f, dtype='uint8', count=1)[0]
        self.format_version = np.fromfile(self.f, dtype='uint32', count=1)[0]
        self.satellite_identifier = np.fromfile(self.f, dtype='uint32', count=1)[0]
        self.record_header_size = np.fromfile(self.f, dtype='uint32', count=1)[0]
        self.brightness_temperature_brilliance = np.fromfile(self.f, dtype='bool', count=1)[0]
        self.number_of_channels = np.fromfile(self.f, dtype='uint32', count=1)[0]
        self.channel_IDs = np.fromfile(self.f, dtype='uint32', count=self.number_of_channels)
        self.AVHRR_brilliance = np.fromfile(self.f, dtype='bool', count=1)[0]
        self.number_of_L2_sections = np.fromfile(self.f, dtype='uint16', count=1)[0]
        # self.table_of_L2_sections = np.fromfile(self.f, dtype='uint32', count=self.number_of_L2_sections)[0]

        # Read header size at the end of the header, check for a match
        self._verify_header()       
        return
    
    def get_iasi_common_header(self):
        self._read_iasi_common_header_metadata()
        self._read_record_size()
        self._count_measurements()
        self._print_metadata()
        return


    def _get_iasi_common_record_fields(self) -> List[tuple]:
        # Format of fields in binary file (field_name, data_type, data_size, cumulative_data_size)
        common_fields = [
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
                        ('height_of_station', 'float32', 4, 50)]
        return common_fields
    
    def _get_iasi_l1c_record_fields(self) -> List[tuple]:
        # Format of L1C-specific fields in binary file (field_name, data_type, data_size, cumulative_data_size),
        # cumulative total continues from the fourth digit of the last tuple in common_fields.
        l1c_fields = [
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
        return l1c_fields
    
    def _get_iasi_l2_record_fields(self) -> List[tuple]:
        # Format of L2-specific fields in binary file (field_name, data_type, data_size, cumulative_data_size),
        # cumulative total continues from the fourth digit of the last tuple in common_fields.
        l2_fields = [
                    ('superadiabatic_indicator', 'uint8', 1, 51),
                    ('land_sea_qualifier', 'uint8', 1, 52),
                    ('day_nght_qualifier', 'uint8', 1, 53),
                    ('processing_technique', 'uint32', 4, 57),
                    ('sun_glint_indicator', 'uint8', 1, 58),
                    ('cloud_formation_and_height_assignment', 'uint32', 4, 62),
                    ('instrument_detecting_clouds', 'uint32', 4, 66),
                    ('validation_flag_for_IASI_L1_product', 'uint32', 4, 70),
                    ('quality_completeness_of_retrieval', 'uint32', 4, 74),
                    ('retrieval_choice_indicator', 'uint32', 4, 78),
                    ('satellite_manoeuvre_indicator', 'uint32', 4, 82)]
        return l2_fields
    
    def _get_ozo_record_fields(self):
        pass

    def _get_trg_record_fields(self):
        pass
    
    def _get_clp_record_fields(self):
        pass

    def _get_twt_record_fields(self):
        pass

    def _get_ems_record_fields(self):
        pass


class Preprocessor:
    """
    Processor for the intermediate binary file of IASI L2 products output by OBR script.

    Attributes:
    filepath (str): Path to the binary file.
    targets (List[str]): List of target variables to extract.
    """
    def __init__(self, filepath: str, data_level: str):
        self.filepath = filepath
        self.data_level = data_level
        self.f: object = None
        self.header: Metadata = None
        self.data_record_df = pd.DataFrame()

    def open_binary_file(self) -> None:
        # Open binary file
        print("")
        print("Loading intermediate L1C file:")
        self.f = open(self.filepath, 'rb')
        
        # Get structure of file header and data record
        self.header = Metadata(self.f)
        self.header.get_iasi_common_header()
        return

    def close_binary_file(self):
        self.f.close()
        return
    

    def read_record_fields(self, fields: List[tuple]) -> None:
        """
        Reads the data of each field from the binary file and store it in the field_df dictionary.

        This function only extracts the first 8 fields and the ones listed in the targets attribute.
        """
        # Iterate over each field
        for field, dtype, dtype_size, cumsize in fields:
            print(f"Extracting: {field}")

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
            self.data_record_df[field] = data
        return


    def read_spectral_radiance(self, fields: List[tuple]) -> None:
        """
        Extracts and stores the spectral radiance measurements from the binary file.
        """
        print("Extracting: radiance")

        # Determine the position of the anchor point for spectral radiance data in the binary file
        last_field_end = fields[-1][-1] # End of the surface_type field

        # Go to spectral radiance data (skip header and previous record data, "12"s are related to reading )
        start_read_position = self.header.header_size + 12 + last_field_end + (4 * self.header.number_of_channels) #12
        self.f.seek(start_read_position, 0)
        
        # Calculate the offset to skip to the next measurement
        byte_offset = self.header.record_size + 8 - (4 * self.header.number_of_channels)
        
        # Initialize an empty numpy array to store the spectral radiance data
        data = np.empty((self.header.number_of_channels, self.header.number_of_measurements))

        # Iterate over each measurement and extract the spectral radiance data
        for measurement in range(self.header.number_of_measurements):
            value = np.fromfile(self.f, dtype='float32', count=self.header.number_of_channels, sep='', offset=byte_offset)
            data[:, measurement] = np.nan if len(value) == 0 else value

        # Assign channel IDs and values to DataFrame
        for i, id in enumerate(self.header.channel_IDs):
            self.data_record_df[f'Channel {id}'] = data[i, :]
        return


    def _calculate_local_time(self) -> None:
        """
        Calculate the local time (in hours, UTC) that determines whether it is day or night at a specific longitude.

        Returns:
        np.ndarray: Local time (in hours, UTC) within a 24 hour range, used to determine day (6-18) or night (0-6, 18-23).
        """

        # Retrieve the necessary field data
        hour, minute, millisecond, longitude = self.data_record_df['hour'], self.data_record_df['minute'], self.data_record_df['millisecond'], self.data_record_df['longitude']

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

    def build_local_time(self) -> List:
        """
        Stores the local time Boolean indicating whether the current time is day or night.
        """
        # Calculate the local time
        local_time = self._calculate_local_time()

        # Store the Boolean indicating day (True) or night (False) in the DataFrame
        self.data_record_df['Local Time'] = (6 < local_time) & (local_time < 18)
        return


    def build_datetime(self) -> List:
        """
        Stores the datetime components to a single column and drops the elements.
        """
        # Create 'Datetime' column
        self.data_record_df['Datetime'] = self.data_record_df['year'].apply(lambda x: f'{int(x):04d}') + \
                                    self.data_record_df['month'].apply(lambda x: f'{int(x):02d}') + \
                                    self.data_record_df['day'].apply(lambda x: f'{int(x):02d}') + '.' + \
                                    self.data_record_df['hour'].apply(lambda x: f'{int(x):02d}') + \
                                    self.data_record_df['minute'].apply(lambda x: f'{int(x):02d}') + \
                                    self.data_record_df['millisecond'].apply(lambda x: f'{int(x/10000):02d}')

        # Drop original time element columns
        self.data_record_df = self.data_record_df.drop(columns=['year', 'month', 'day', 'hour', 'minute', 'millisecond'])
        return  
    

    def filter_bad_spectra(self) -> None:
        """
        Filters bad spectra based on IASI L1C data quality flags and date.
        """
        print("Filtering spectra:")
        if self.data_record_df['Datetime'] <= "201202208":
            # Treat data differently if before February 8 2012
            check_quality_flag = self.data_record_df['quality_flag'] == 0
            check_data = self.data_record_df.drop(['quality_flag', 'date_column'], axis=1).sum(axis=1) > 0
            good_flag = check_quality_flag & check_data
        else:
            check_quality_flags = (self.data_record_df['quality_flag_1'] == 0) & (self.data_record_df['quality_flag_2'] == 0) & (self.data_record_df['quality_flag_3'] == 0)
            good_flag = check_quality_flags

        # Throw away bad data, return the good
        good_df = self.data_record_df[good_flag]
        print(f"{np.round((len(good_df) / len(self.data_record_df)) * 100, 2)} % good data of {len(self.data_record_df)} spectra")
        return good_df
    

    def preprocess_files(self) -> None:
        
        # Open binary file and extract metadata
        self.open_binary_file()
        
        # Read common IASI record fields and store to pandas DataFrame
        print("\nCommon Record Fields:")
        self.read_record_fields(self.header._get_iasi_common_record_fields())
        if self.data_level == "l1c":
            # Read L1C-specific record fields and add to DataFrame
            print("\nL1C Record Fields:")
            self.read_record_fields(self.header._get_iasi_l1c_record_fields())
            self.read_spectral_radiance(self.header._get_iasi_l1c_record_fields())
        elif self.data_level == "l2":
            # Read L2-specific record fields and add to DataFrame
            print("\nL2 Record Fields:")
            self.read_record_fields(self.header._get_iasi_l2_record_fields())
        self.close_binary_file()

        # Construct Local Time column
        self.build_local_time()
        # Construct Datetime column and remove individual time elements
        self.build_datetime()
        # Remove observations (DataFrame rows) based on IASI quality flags
        self.filter_bad_spectra()

        # print the DataFrame
        print(self.data_record_df.head())