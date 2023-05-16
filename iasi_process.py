import datetime
import numpy as np
import struct
from typing import List, Tuple, BinaryIO
import snoop


def read_uint(file: BinaryIO, size_format: str = 'I') -> int:
    """Read an unsigned integer from a binary file.
    
    Args:
        file (BinaryIO): The binary file to read from.
        size_format (str, optional): The format of the unsigned integer. Defaults to 'I'.
        
    Returns:
        int: The unsigned integer read from the file.
    """
    return struct.unpack(size_format, file.read(struct.calcsize(size_format)))[0]

def count_channels(f: object):
    f.read(4)
    f.read(1)
    f.read(1)
    number_of_channels = read_uint(f)
    f.seek(0)
    return number_of_channels

def read_header(f: object):
    header_size = read_uint(f)
    f.seek(header_size, 1)
    header_size_check = read_uint(f)
    assert header_size == header_size_check, "Header size mismatch"
    return header_size

def read_record(f):
    record_size = read_uint(f)
    if (record_size == 0):
        return None
    else:
        return record_size

def count_measurements(f: object, header_size: int, record_size: int):
    file_size = f.seek(0, 2)
    number_of_measurements = (file_size - header_size - 8) // (record_size + 8)
    return number_of_measurements

def read_field_data(f:object, fields: list, target: list, header_size: int, record_size: int, number_of_measurements: int) -> dict:
    """
    Read and store field data from a binary file.

    Args:
        f (file object): Binary file object to read the field data from.
        fields (List[Tuple[str, str, int]]): A list of tuples containing field names, data types, and size of data types.
        header_size (int): The size of the header in the binary file.
        record_size (int): The size of a record in the binary file.
        number_of_measurements (int): The total number of measurements in the binary file.
        target (List[str]): A list of field names to be read.

    Returns:
        dict: A dictionary containing the field data with field names as keys and data as values.
    """

    field_data = {}

    # Iterate over the fields list and read the field data
    for i, (field, dtype, dtype_size) in enumerate(fields):
        if (i < 8) or (field in target):
            
            # Move the file cursor to the correct position
            f.seek(header_size + 12 + dtype_size)
            
            # Read the field data
            field_data[field] = np.fromfile(f,
                                            dtype,
                                            number_of_measurements,
                                            f"{record_size + 8 - dtype_size}x").reshape(number_of_measurements,-1)

    # Create a timestamp for the observation
    if "ObsDateNum" in target:
        field_data["ObsDateNum"] = datetime(field_data['yyyy'],
                                            field_data['mm'],
                                            field_data['dd'],
                                            field_data['HH'],
                                            field_data['MM'],
                                            field_data['ms'] // 1000)
            
    return field_data

def calculate_day_or_night(field_data: dict):
    """
    Calculate adjusted time (in hours) that will be used to determine 
    whether it is day or night at a specific longitude.

    Args:
        hour (int): Hour component of the timestamp.
        minute (int): Minute component of the timestamp.
        millisecond (int): Millisecond component of the timestamp.
        longitude (int): Longitude coordinate for which day or night is to be determined.

    Returns:
        np.ndarray: Adjusted time (in hours) used to determine day or night.
    """

    # Define inputs
    hour, minute, millisecond, longitude = field_data['HH'], field_data['MM'], field_data['ms'], field_data['lon']

    # Calculate the total time in hours, minutes, and milliseconds
    total_time = hour * 1e4 + minute * 1e2 + millisecond / 1e3

    # Extract the hour component from total_time
    hour_component = np.floor(total_time / 10000)

    # Extract the minute component from total_time
    minute_component = np.mod((total_time - np.mod(total_time, 100)) / 100, 100)

    # Extract the second component from total_time and convert it to minutes
    second_component_in_minutes = np.mod(total_time, 100) / 60

    # Calculate the total time in hours
    total_time_in_hours = (hour_component + minute_component + second_component_in_minutes) / 60

    # Convert longitude to time in hours and add it to the total time
    total_time_with_longitude = total_time_in_hours + longitude / 15

    # Add 24 hours to the total time to ensure it is always positive
    total_time_positive = total_time_with_longitude + 24

    # Take the modulus of the total time by 24 to get the time in the range of 0 to 23 hours
    time_in_range = np.mod(total_time_positive, 24)

    # Subtract 6 hours from the total time, shifting the reference for day and night (so that 6 AM becomes 0)
    time_shifted = time_in_range - 6

    # Take the modulus again to ensure the time is within the 0 to 23 hours range
    return np.mod(time_shifted, 24)

@snoop
def read_bin_L1C(fields: list, file: str, target: list) -> Tuple[np.ndarray]:
    """Read L1C binary data from a given file.
    
    Args:
        fields (List[Tuple[str, str, int]]): A list of tuples containing field names, data types, and size of data types.
        file (str): The file path to the binary file.
        target (List[str]): A list of fields to be extracted from the binary file.
        
    Returns:
        Tuple[np.ndarray]: A tuple containing the extracted fields as numpy arrays.
    """

    with open(file, 'rb') as f:
        # Calculate number of channels   
        number_of_channels = count_channels(f)

        # Read and check header size
        header_size = read_header(f)
        
        # Read record size
        record_size = read_record(f)
        
        # Calculate number of measurements
        number_of_measurements = count_measurements(f, header_size, record_size)
        
        # Read and store field data from binary file
        field_data = read_field_data(f, fields, target, header_size, record_size, number_of_measurements)

        # Calculate adjusted time (in hours) to determine whether it is day or night
        day_or_night_time = calculate_day_or_night(field_data)

        # Initialize the output with longitude, latitude, and a boolean mask for day/night
        output = [field_data['lon'], field_data['lat'], day_or_night_time < 12]
        
        # Read the L (spectral radiance) matrix
        L = np.fromfile(f, dtype='float32', count=number_of_measurements * number_of_channels, sep='').reshape(number_of_measurements, number_of_channels)
        
        # Append the L matrix to the output
        output.append(L)
    
    # Return the output as a tuple of numpy arrays
    return tuple(output)

# Format of fields in binary file
fields = [
    ('yyyy', 'uint16', 2),
    ('mm', 'uint8', 1),
    ('dd', 'uint8', 1),
    ('HH', 'uint8', 1),
    ('MM', 'uint8', 1),
    ('ms', 'uint32', 4),
    ('lat', 'float32', 4),
    ('lon', 'float32', 4),
    ('sza', 'float32', 4),
    ('saazim', 'float32', 4),
    ('soza', 'float32', 4),
    ('soazim', 'float32', 4),
    ('ifov', 'uint32', 4),
    ('onum', 'uint32', 4),
    ('scanLN', 'uint32', 4),
    ('Hstat', 'float32', 4),
    ('dayver', 'uint16', 2),
    ('SC1', 'uint32', 4),
    ('EC1', 'uint32', 4),
    ('FQ1', 'uint32', 4),
    ('SC2', 'uint32', 4),
    ('EC2', 'uint32', 4),
    ('FQ2', 'uint32', 4),
    ('SC3', 'uint32', 4),
    ('EC3', 'uint32', 4),
    ('FQ3', 'uint32', 4),
    ('CLfrac', 'uint32', 4),
    ('SurfType', 'uint8', 1)
]

# Specify location of IASI L1C products binary file
test_file = "C:\\Users\\padra\\Documents\\Research\\data\\iasi\\test_file.bin"

# Specify target IASI L1C products
target = ['sza','FQ1','FQ2','FQ3','CLfrac','SurfType']

# Extract IASI L1C products from binary file
read_bin_L1C(fields, test_file, target)