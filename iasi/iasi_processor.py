from datetime import datetime
import glob
import numpy as np
import pandas as pd
from typing import Tuple, List, Union

import numpy as np


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
        self.header_size, self.number_of_channels = self.read_header()
        self.record_size = self.read_record_size()
        self.number_of_measurements = 5#self.count_measurements()
        self.print_metadata()

        # Get fields information and prepare to store extracted data
        self.fields = self.get_fields()
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


    def print_metadata(self):
        print(f"Header  : {self.header_size} bytes")
        print(f"Record  : {self.record_size} bytes")
        print(f"Spectrum: {self.number_of_channels} channels")
        print(f"Data    : {self.number_of_measurements} measurements")
        return
    

    def read_header(self) -> Tuple[int, int]:
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

        # Verify the header size
        self.verify_header(header_size)
        return header_size, number_of_channels


    def verify_header(self, header_size: int) -> None:
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


    def read_record_size(self) -> Union[int, None]:
        self.f.seek(self.header_size + 8)
        record_size = np.fromfile(self.f, dtype='uint32', count=1)[0]
        return None if record_size == 0 else record_size


    def get_fields(self) -> List[tuple]:
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


    def read_field_data(self) -> None:
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
        self.field_data["datetime"] = [self.field_data['year'],
                                        self.field_data['month'],
                                        self.field_data['day'],
                                        self.field_data['hour'],
                                        self.field_data['minute'],
                                        self.field_data['millisecond']]
        

    def calculate_local_time(self) -> np.ndarray:
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
    

    def store_space_time_coordinates(self) -> List:
        """
        Stores the space-time coordinates and a Boolean indicating whether the current time is day or night.

        Returns:
            List: A list containing longitude, latitude and a Boolean indicating day or night. 
        """
        # Calculate the local time
        local_time = self.calculate_local_time()

        # Return the longitude, latitude, and a Boolean indicating day (True) or night (False)
        return [self.field_data['longitude'], self.field_data['latitude'], ((6 < local_time) & (local_time < 18))]


    def store_spectral_radiance(self) -> np.ndarray:
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


    def store_target_parameters(self) -> List:
        """
        Stores the target parameters from the field data.

        Returns:
            List: A list of the target parameters from the field data.
        """
        return [data for field, data in self.field_data.items() if (field in self.targets)]


    def store_datetime_components(self) -> List:
        """
        Stores the datetime components from the datetime field data.

        Returns:
            List: A list of the datetime components from the field data.
        """
        return [datetime for datetime in self.field_data['datetime']]


    def extract_data(self) -> Tuple[List[int], np.array]:
        """
        Extracts relevant data from the observation dataset, processes it, and consolidates it into a 2D array.

        Returns:
            Tuple[List[int], np.array]: A tuple containing a header (which represents lengths of various components of data)
            and a 2D numpy array of consolidated data.

        """
        # Extract and process binary data
        self.read_field_data()
        space_time_coordinates = self.store_space_time_coordinates()
        radiances = self.store_spectral_radiance()
        target_parameters = self.store_target_parameters()
        datetimes = self.store_datetime_components()

        # Concatenate processed observations into a single 2D array (number of parameters x number of measurements).
        data = np.concatenate((space_time_coordinates, radiances, target_parameters, datetimes), axis=0)

        # Construct a header that contains the lengths of each of the components of the concatenated data.
        # This can be used for easy separation of columns in the future.
        header = [len(space_time_coordinates), len(radiances), len(target_parameters), len(datetimes)]

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
    def save_observations(outpath: str, outfile: str, header: list, data: np.array) -> None:
        """
        Saves the observation data to a file.
        
        Args:
            outpath (str): The path to save the file.
            outfile (str): The name of the output file.
            header (list): A list of integers to be written as the first line.
            data (np.ndarray): A numpy array containing the observation data.
            
        Returns:
            None
        """
        # Open your file in write mode
        with open(f"{outpath}{outfile}.txt", 'w') as f:
        
            # Write the integers to the first line, separated by spaces
            f.write(' '.join(map(str, header)) + '\n')

            # Write the 2D numpy array to the file, line by line
            for row in np.transpose(data):
                f.write(' '.join(map(str, row)) + '\n')

        # # Transpose the data to match the previous structure
        # data = np.transpose(data)
        
        # # Create a DataFrame with the transposed data
        # df = pd.DataFrame(data, columns=header)
        
        # # Save the DataFrame to a file in HDF5 format
        # df.to_hdf(f"{outpath}{outfile}.h5", key='df', mode='w')
        return

    
    def extract_spectra(self, datapath_out: str, year: str, month: str, day: str):
        # Extract and process binary data
        header, all_data = self.extract_data()
        
        # Check observation quality and filter out bad observations
        good_data = self.filter_bad_observations(all_data, date=datetime(self.year, self.month, self.day))

        # Define the output filename
        outfile = f"IASI_L1C_{year}_{month}_{day}"
        
        # Save outputs to file
        self.save_observations(datapath_out, header, good_data, outfile)

class L2Processor:
    """
    Processor for the intermediate binary file of IASI L2 products output by OBR script.

    Attributes:
    filepath (str): Path to the binary file.
    """
    def __init__(self, filepath: str, lat_range: Tuple[int, int], lon_range: Tuple[int, int]):
        self.filepath = filepath
        self.lat_range = lat_range
        self.lon_range = lon_range
        self.filtered_data = None


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
        self.filtered_data.to_hdf(f"{self.filepath}.h5", key='df', mode='w')


    def _filter_data(self):
        data = []
        for _, row in self.df.iterrows():
            if (self.lat_range[0] <= row['Latitude'] <= self.lat_range[1]) and (self.lon_range[0] <= row['Longitude'] <= self.lon_range[1]):
                if (row['Cloud Phase'] == 2) and np.isfinite(row['Cloud-top Temperature']):
                    data.append([row['Latitude'], row['Longitude'], row['Datetime'], row['Cloud Fraction'], row['Cloud-top Temperature'], row['Cloud-top Pressure'], row['Cloud Phase']])
        self.filtered_data =  pd.DataFrame(data, columns=['Latitude', 'Longitude', 'Orbit', 'Datetime', 'Cloud Fraction', 'Cloud-top Temperature', 'Cloud-top Pressure', 'Cloud Phase'])

    
    def extract_ice_clouds(self):
        self._filter_data()
        self._save_data()
