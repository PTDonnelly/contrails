from datetime import datetime
import numpy as np
import struct
from typing import List, Tuple, Union, BinaryIO
import snoop
from sorcery import dict_of

def verify_header(f: object, header_size: int) -> None:
    f.seek(0)
    np.fromfile(f, dtype='uint32', count=1)
    np.fromfile(f, dtype='uint8', count=header_size)
    header_size_check = np.fromfile(f, dtype='uint32', count=1)[0]
    assert header_size == header_size_check, "Header size mismatch"

def read_header(f: object) -> Tuple[int, int, int]:
    """Read the header of the binary file and extract metadata.
    
    Args:
        f (object): A file object pointing to the binary file.
    
    Returns:
        header_size (int): The size of the header in bytes.
        number_of_channels (int): The number of channels in the binary file.
    """
    
    # Read the header size (1 uint32 value)
    header_size = np.fromfile(f, dtype='uint32', count=1)[0]
    
    # Skip
    np.fromfile(f, dtype='uint8', count=1)
    np.fromfile(f, dtype='uint32', count=3)
    np.fromfile(f, dtype='bool', count=1)

    # Read the number of channels (1 uint32 value)
    number_of_channels = np.fromfile(f, dtype='uint32', count=1)[0]

    # Reset pointer and verify second header_size
    verify_header(f, header_size)

    return header_size, number_of_channels

def read_record(f: object, header_size: int) -> Union[int, None]:
    """Read the size of a record in the binary file.
    
    Args:
        f (object): A file object pointing to the binary file.
    
    Returns:
        record_size (Union[int, None]): The size of the record in bytes if it exists, otherwise None.
    """

    # Reset the pointer
    f.seek(0)
    # Skip bytes until the record size
    np.fromfile(f, dtype='uint8', count=header_size+8)
    # Read the record size
    record_size = np.fromfile(f, dtype='uint32', count=1)[0]
    if (record_size == 0):
        return None
    else:
        return record_size

def count_measurements(f: BinaryIO, header_size: int, record_size: int) -> int:
    """Count the number of measurements in the binary file.

    Args:
        f (obect): The binary file object.
        header_size (int): The size of the header in bytes.
        record_size (int): The size of each record in bytes.
    
    Returns:
        number_of_measurements (int): The number of measurements in the file.
    """

    # Move the file cursor to the end of the file and get the file size in bytes
    file_size = f.seek(0, 2)
    return (file_size - header_size - 8) // (record_size + 8)

def vectorized_datetime(year: int, month: int, day: int, hour: int, minute: int, millisecond: int):
    return datetime(year, month, day, hour, minute, millisecond // 1000)

def read_record_data(f:object, fields: list, targets: list, header_size: int, record_size: int, number_of_measurements: int) -> dict:
    """
    Read and store field data from a binary file.

    Args:
        f (file object): Binary file object to read the field data from.
        fields (List[Tuple[str, str, int]]): A list of tuples containing field names, data types, and size of data types.
        header_size (int): The size of the header in the binary file.
        record_size (int): The size of a record in the binary file.
        number_of_measurements (int): The total number of measurements in the binary file.
        targets (List[str]): A list of optional field names to be read.

    Returns:
        field_data (dict): A dictionary containing the field data with field names as keys and data as values.
    """

    field_data = {}

    # Iterate over the fields list and read the field data
    for i, (field, dtype, dtype_size, cumsize) in enumerate(fields):
        if (i < 8) or (field in targets):
            
            print(i, field, dtype, dtype_size, cumsize)

            # Skip file header
            pointer = header_size + 12 + cumsize
            f.seek(pointer, 0)
            # print(f"POINTER: {pointer}")
            
            # Skip bytes up to the next measurement
            byte_offset = record_size + 8 - dtype_size
            print(f"OFFSET: {record_size} + 8 - {dtype_size} = {byte_offset}")
            
            # Read binary data, skip bytes, reshape into 2D array
            field_data[field] = np.fromfile(f, dtype=dtype, count=number_of_measurements, offset=byte_offset)#.reshape(number_of_measurements,-1)
            print(field_data[field])
            print('')
        # if i == 2:
        #     exit()

    # # Create a timestamp for the observation
    # if "datetime" in targets:       
    #     v_datetime = np.vectorize(vectorized_datetime)
    #     field_data["datetime"] = v_datetime(field_data['yyyy'],
    #                                         field_data['mm'],
    #                                         field_data['dd'],
    #                                         field_data['HH'],
    #                                         field_data['MM'],
    #                                         field_data['ms'])
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
        np.ndarray: Adjusted time (in hours) used to determine day (0-11) or night (12-23).
    """

    # Define inputs
    hour, minute, millisecond, longitude = field_data['HH'], field_data['MM'], field_data['ms'], field_data['lon']

    # Calculate the total time in hours, minutes, and milliseconds
    total_time = (hour * 1e4) + (minute * 1e2) + (millisecond / 1e3)

    # Extract the components from total_time, calculate the total time in hours
    hour_component = np.floor(total_time / 10000)
    minute_component = np.mod((total_time - np.mod(total_time, 100)) / 100, 100)
    second_component_in_minutes = np.mod(total_time, 100) / 60
    total_time_in_hours = (hour_component + minute_component + second_component_in_minutes) / 60

    # Convert longitude to time in hours and add it to the total time

    print(type(total_time_in_hours), type(longitude))
    total_time_with_longitude = total_time_in_hours + (longitude / 15)

    # Add 24 hours to the total time to ensure it is always positive
    total_time_positive = total_time_with_longitude + 24

    # Take the modulus of the total time by 24 to get the time in the range of 0 to 23 hours
    time_in_range = np.mod(total_time_positive, 24)

    # Subtract 6 hours from the total time, shifting the reference for day and night (so that 6 AM becomes 0)
    time_shifted = time_in_range - 6

    # Take the modulus again to ensure the time is within the 0 to 23 hours range
    return np.mod(time_shifted, 24)

def store_space_time_coordinates(field_data: dict, adjusted_time):
    return [field_data['lon'], field_data['lat'], adjusted_time < 12]

def store_spectral_radiance(f, number_of_measurements, number_of_channels, output):
    return np.fromfile(f, dtype='float32', count=number_of_measurements*number_of_channels).reshape(number_of_measurements, number_of_channels)

def store_target_parameters(field_data: dict):
    return [data for field, data in field_data.items() if field in targets]

# @snoop
def read_bin_L1C(fields: list, file: str, targets: list) -> Tuple[np.ndarray]:
    """Read L1C binary data from a given file.
    
    Args:
        fields (List[Tuple[str, str, int]]): A list of tuples containing field names, data types, and size of data types.
        file (str): The file path to the binary file.
        target (List[str]): A list of fields to be extracted from the binary file.
        
    Returns:
        Tuple[np.ndarray]: A tuple containing the extracted fields as numpy arrays (lon, lat, fractional_hour, radiance, target parameters).
    """

    with open(file, 'rb') as f:
               
        # Extract metadata from header   
        header_size, number_of_channels = read_header(f)
        
        # Extract metadata from record
        record_size = read_record(f, header_size)
        
        # Calculate number of measurements
        number_of_measurements = count_measurements(f, header_size, record_size)
        
        # Read and store field data from binary file
        field_data = read_record_data(f, fields, targets, header_size, record_size, number_of_measurements)

        # Calculate adjusted time (in hours) to determine whether it is day or night
        adjusted_time = calculate_day_or_night(field_data)

        # Initialize the output with longitude, latitude, and adjusted
        output = store_space_time_coordinates(field_data, adjusted_time)
        
        # Read the spectral radiance (L) matrix, append to the output
        output.append(store_spectral_radiance(f, number_of_measurements, number_of_channels, output))

        # Add target parameters to output
        output.extend(store_target_parameters(field_data))

    # Return the output as a tuple of numpy arrays
    return tuple(output)

# Format of fields in binary file (field_name, data_type, data_size, cumulative_data_size)
fields = [
    ('yyyy', 'uint16', 2, 2),
    ('mm', 'uint8', 1, 3),
    ('dd', 'uint8', 1, 4),
    ('HH', 'uint8', 1, 5),
    ('MM', 'uint8', 1, 6),
    ('ms', 'uint32', 4, 10),
    ('lat', 'float32', 4, 14),
    ('lon', 'float32', 4, 18),
    ('sza', 'float32', 4, 22),
    ('saazim', 'float32', 4, 26),
    ('soza', 'float32', 4, 30),
    ('soazim', 'float32', 4, 34),
    ('ifov', 'uint32', 4, 38),
    ('onum', 'uint32', 4, 42),
    ('scanLN', 'uint32', 4, 46),
    ('Hstat', 'float32', 4, 50),
    ('dayver', 'uint16', 2, 52),
    ('SC1', 'uint32', 4, 56),
    ('EC1', 'uint32', 4, 60),
    ('FQ1', 'uint32', 4, 64),
    ('SC2', 'uint32', 4, 68),
    ('EC2', 'uint32', 4, 72),
    ('FQ2', 'uint32', 4, 76),
    ('SC3', 'uint32', 4, 80),
    ('EC3', 'uint32', 4, 84),
    ('FQ3', 'uint32', 4, 88),
    ('CLfrac', 'uint32', 4, 92),
    ('SurfType', 'uint8', 1, 93)
]

# Specify location of IASI L1C products binary file
test_file = "C:\\Users\\padra\\Documents\\Research\\data\\iasi\\test_file.bin"

# Specify target IASI L1C products
targets = ['sza','FQ1','FQ2','FQ3','CLfrac','SurfType']#, 'datetime']

# Extract IASI L1C products from binary file
read_bin_L1C(fields, test_file, targets)