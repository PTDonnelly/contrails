from datetime import datetime
import pandas as pd
import numpy as np
import subprocess
from typing import Tuple, List, Union
import snoop

class IASI_L1C:
    """
    Processing class to read and extract necessary data from binary file.

    Attributes:
    file_path (str): Path to the binary file.
    targets (List[str]): List of target variables to extract.
    """
    def __init__(self, file_path: str, targets: List[str]):
        """Initializes the IASI_L1C object with file_path and target variables."""
        self.file_path = file_path
        self.targets = targets


    def __enter__(self) -> 'IASI_L1C':
        """
        Opens the binary file and prepares the preprocessing.
        
        Returns:
        self: The IASI_L1C object itself.
        """
        # Open binary file
        self.f = open(self.file_path, 'rb')
        
        # Get structure of file header and data record
        self.header_size, self.number_of_channels = self.read_header()
        self.record_size = self.read_record_size()
        self.number_of_measurements = 10# self.count_measurements()
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

        # Generate datetime field if it's listed in the targets
        if "datetime" in self.targets:
            self.generate_datetime()


    def generate_datetime(self):
        """
        Generates the datetime field using the year, month, day, hour, minute and millisecond fields.
        """
        v_datetime = np.vectorize(self.vectorized_datetime)
        self.field_data["datetime"] = v_datetime(self.field_data['year'],
                                                self.field_data['month'],
                                                self.field_data['day'],
                                                self.field_data['hour'],
                                                self.field_data['minute'],
                                                self.field_data['millisecond'])

    @staticmethod
    def vectorized_datetime(year: int, month: int, day: int, hour: int, minute: int, millisecond: int) -> datetime:
        """
        Converts separate year, month, day, hour, minute and millisecond values into a datetime object.

        Args:
        year (int): The year component of the timestamp.
        month (int): The month component of the timestamp.
        day (int): The day component of the timestamp.
        hour (int): The hour component of the timestamp.
        minute (int): The minute component of the timestamp.
        millisecond (int): The millisecond component of the timestamp.

        Returns:
        datetime: The corresponding datetime object.
        """
        return datetime(int(year), int(month), int(day), int(hour), int(minute), int(millisecond // 1000))


    def calculate_adjusted_time(self) -> np.ndarray:
        """
        Calculate the adjusted time (in hours, UTC) that determines whether it is day or night at a specific longitude.

        Returns:
        np.ndarray: Adjusted time (in hours, UTC) within a 24 hour range, used to determine day (6-18) or night (0-6, 18-23).
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
        # Calculate the adjusted time
        adjusted_time = self.calculate_adjusted_time()

        # Return the longitude, latitude, and a Boolean indicating day (True) or night (False)
        return [self.field_data['longitude'], self.field_data['latitude'], ((6 < adjusted_time) & (adjusted_time < 18))]


    def store_spectral_radiance(self) -> np.ndarray:
        """
        Extracts and stores the spectral radiance measurements from the binary file.

        Returns:
            np.ndarray: An array of spectral radiance measurements for each channel.
        """
        print("Extracting: radiance")

        # Determine the position of the anchor point for spectral radiance data in the binary file
        last_field_end = self.fields[-1][-1] # End of the surface_type field

        # Calculate the starting point for reading spectral radiance data
        start_read_position = self.header_size + 12 + last_field_end + (4 * self.number_of_channels)
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
        return [data for field, data in self.field_data.items() if field in self.targets]

    @classmethod
    def filter_bad_observations(cls, data: np.array, date: object) -> np.array:
        if date <= datetime(2012, 2, 8):
            check_quality_flag = data[-3, :] == 0
            check_data = np.sum(data[2:-4, :], axis=1) > 0
            good_flag = np.logical_and(check_quality_flag, check_data)
        else:
            check_quality_flags = np.logical_and(data[-6, :] == 0, data[-5, :] == 0, data[-4, :] == 0)
            # check_data = np.sum(data[0:-6, :], axis=1) > 0
            good_flag = check_quality_flags#np.logical_and(check_quality_flags)#, check_data)
        
        good_data = data[:, good_flag]
        print(f"{np.round((data.shape[1] / good_data.shape[1]) * 100, 2)} % good data of {data.shape[1]} observations")
        return good_data

    @classmethod
    def save_observations(cls, filename: str, header: list, data: np.array) -> None:
        
        # Open your file in write mode
        with open(filename, 'w') as f:
        
            # Write the integers to the first line, separated by spaces
            f.write(' '.join(map(str, header)) + '\n')

            # Write the 2D numpy array to the file, line by line
            for row in np.transpose(data):
                f.write(' '.join(map(str, row)) + '\n')

    @classmethod
    def read_observations(cls, filename: str) -> List[np.ndarray]:
        """
        Reads observation data from a file and converts them into a list of numpy arrays.
        
        :param filename: The name of the file to read from.
        :return: A list of numpy arrays containing the observation data.
        """
        # Read the first line of the file to determine how to split the data.
        splits = cls.read_splits(filename)
        
        # Process the data lines based on the splits and return the resulting list of arrays.
        return cls.process_data_lines(filename, splits)

    @staticmethod
    def read_splits(filename: str) -> List[int]:
        """
        Reads the first line of a file and converts it into a list of integers.
        
        :param filename: The name of the file to read from.
        :return: A list of integers representing how to split the data in the file.
        """
        with open(filename, 'r') as f:
            first_line = f.readline()
        return list(map(int, first_line.strip().split()))

    @staticmethod
    def process_data_lines(filename: str, splits: List[int]) -> List[List[np.ndarray]]:
        """
        Processes the data lines in a file based on a list of splits.
        
        :param filename: The name of the file to read from.
        :param splits: A list of integers representing how to split the data.
        :return: A list of numpy arrays containing the split data.
        """
        with open(filename, 'r') as f:
            next(f)  # Skip the first line
            lines = f.readlines()

        # Remove the last split which is past the end of the array.
        split_indices = np.cumsum(splits)[:-1]

        list_of_arrays = []
        for line in lines:
            raw_data = np.array(line.strip().split(), dtype=object)
            
            # Split the raw array according to splits and convert to appropriate types
            arrays = np.hsplit(raw_data, split_indices)
            
            # Convert the first two sections to float
            arrays[0] = arrays[0].astype(float)
            arrays[1] = arrays[1].astype(float)
            
            # Leave the third section as strings
            list_of_arrays.append(arrays)
        return np.vstack(list_of_arrays)


def build_command(path_in: str, year: int, month: int, day: int, iasi_channels: list, filter: str, file_out: str):
    """Execute OBR command and produce intermediate binary files"""

    run_dir = f"/home/pdonnelly/data/obr_v4 "
    filepath = f"-d {path_in} "
    first_date = f"-fd {year:04d}-{month:02d}-{day:02d} "
    last_date = f"-ld {year:04d}-{month:02d}-{day:02d} "
    channels = f"-c {iasi_channels[0]}-{iasi_channels[-1]} "
    filter = "" #f"-mf {file_in}"
    output = f"of bin -out {file_out} "
    return f"{run_dir}{filepath}{first_date}{last_date}{channels}{filter}{output}"

def main():

    # Define inputs for OBR tool
    path_in = '/bdd/IASI/L1C/'
    years = [2020]
    months = [1] #, 2, 3]
    days = [1] #[day for day in range(28)] 
    iasi_channels = [1, 2, 3]
    filter = 'W_XX-EUMETSAT-Darmstadt,SOUNDING+SATELLITE,METOPA+IASI_C_EUMC_20200101000253_68496_eps_o_l1.bin'
    file_out = 'file_out.bin'

    # Specify location of IASI L1C products binary file
    file_path = "C:\\Users\\padra\\Documents\\Research\\data\\iasi\\test_file.bin"

    # Specify target IASI L1C products
    targets = ['satellite_zenith_angle','quality_flag_1','quality_flag_2','quality_flag_3','cloud_fraction','surface_type', 'datetime']

    # Loop through date-times
    for year in years:
        for month in months:
            for day in days:

                # Construct command-line executable
                command = build_command(path_in, year, month, day, iasi_channels, filter, file_out)

                # # Extract IASI spectra from raw binary files (create intermediate binary files)
                # subprocess.run(command, shell=True)

                # Process extracted IASI spectra from intermediate binary files
                with IASI_L1C(file_path, targets) as file:
                    file.read_field_data()
                    space_time_coordinates = file.store_space_time_coordinates()
                    radiance = file.store_spectral_radiance()
                    target_parameters = file.store_target_parameters()
                
                # Concatenate processed observations into a single 2-D array (number of parameters x number of measurements)
                all_data = np.concatenate((space_time_coordinates, radiance, target_parameters), axis=0)

                # Check observation quality and filter out bad observations
                good_data = IASI_L1C.filter_bad_observations(data=all_data, date=datetime(year, month, day))

                # Save outputs to file
                filename = f"iasi_L1C_{year}_{month}_{day}.txt"
                header = [len(space_time_coordinates), len(radiance), len(target_parameters)]
                IASI_L1C.save_observations(filename, header, good_data)

                # arrays = IASI_L1C.read_observations(filename)
                # location, spectra, target_parameters = arrays[:, 0], arrays[:, 1],arrays[:, 2]
                # for spectrum in spectra:
                #     print(spectrum)

if __name__ == "__main__":
    main()