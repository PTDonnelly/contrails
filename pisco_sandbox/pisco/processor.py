from datetime import datetime
import os
import numpy as np
import pandas as pd
from typing import Tuple, List, Union

import numpy as np

from .extractor import Extractor as ex


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


class L1CProcessor:
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
        print("Loading intermediate L1C file:")
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
        fields = self.header.get_iasi_l1c_record_fields()

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