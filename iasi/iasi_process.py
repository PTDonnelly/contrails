from datetime import datetime
import numpy as np
import struct
from typing import List, Tuple, Union, BinaryIO
import snoop
from sorcery import dict_of

def count_channels(f: object) -> int:
    """Determine the number of channels in the binary file.
    
    Args:
        f (object): A file object pointing to the binary file.
    
    Returns:
        number_of_channels (int): The number of channels in the binary file.
    """
    
    # Read 4 uint32 values and discard them
    np.fromfile(f, dtype='uint32', count=4)
    # Read 1 uint8 value and discard it
    np.fromfile(f, dtype='uint8', count=1)
    # Read 1 bool value and discard it
    np.fromfile(f, dtype='bool', count=1)

    # Read the number of channels (1 uint32 value)
    number_of_channels = np.fromfile(f, dtype='uint32', count=1)[0]

    # Move pointer back to beginning of binary file
    f.seek(0, 0)

    return number_of_channels

def read_header(f: object) -> int:
    """Read the header of the binary file and verify its size.
    
    Args:
        f (object): A file object pointing to the binary file.
    
    Returns:
        header_size (int): The size of the header in bytes.
    
    Raises:
        AssertionError: If the header size values do not match.
    """
    
    # Read the header size (1 uint32 value)
    header_size = np.fromfile(f, dtype='uint32', count=1)[0]
    # Skip the header content by reading header_size uint8 values
    np.fromfile(f, dtype='uint8', count=header_size)
    # Read the header size again (1 uint32 value)
    header_size_check = np.fromfile(f, dtype='uint32', count=1)[0]
    # Verify that header is read properly
    assert header_size == header_size_check, "Header size mismatch"
    return header_size

def read_record(f: object) -> Union[int, None]:
    """Read the size of a record in the binary file.
    
    Args:
        f (object): A file object pointing to the binary file.
    
    Returns:
        record_size (Union[int, None]): The size of the record in bytes if it exists, otherwise None.
    """
    
    # Read the record size (1 uint32 value)
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

def read_field_data(f:object, fields: list, targets: list, header_size: int, record_size: int, number_of_measurements: int) -> dict:
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
    for i, (field, dtype, _, cumsize) in enumerate(fields):
        if (i < 8) or (field in targets):
           
            # Move the file cursor to the correct position
            f.seek(header_size + 12 + cumsize)
            
            # Read binary data, skip bytes, reshape into 2D array
            field_data[field] = np.fromfile(f, dtype=dtype, count=number_of_measurements).reshape(number_of_measurements,-1)

    # Create a timestamp for the observation
    def vectorized_datetime(year, month, day, hour, minute, millisecond):
        datetime_dict = dict_of(year, month, day, hour, minute, millisecond // 1000)
        print(datetime_dict)
        exit()
        return datetime(year, month, day, hour, minute, millisecond // 1000)
    
    if "datetime" in targets:
    #     field_data["datetime"] = datetime(field_data['yyyy'],
    #                                         field_data['mm'],
    #                                         field_data['dd'],
    #                                         field_data['HH'],
    #                                         field_data['MM'],
    #                                         field_data['ms'] // 1000)

        v_datetime = np.vectorize(vectorized_datetime)

        obs_date_num = v_datetime(field_data['yyyy'],
                                    field_data['mm'],
                                    field_data['dd'],
                                    field_data['HH'],
                                    field_data['MM'],
                                    field_data['ms'])

        field_data["datetime"] = obs_date_num
        
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

def store_space_time_coordinates(field_data, adjusted_time):
    return [field_data['lon'], field_data['lat'], adjusted_time < 12]

def store_spectral_radiance(f, number_of_measurements, number_of_channels):
    return np.fromfile(f, dtype='float32', count=number_of_measurements*number_of_channels).reshape(number_of_measurements, number_of_channels)

def store_target_parameters(field_data):
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
               
        # Calculate number of channels   
        number_of_channels = count_channels(f)

        # Read and check header size
        header_size = read_header(f)
        
        # Read record size
        record_size = read_record(f)
        
        # Calculate number of measurements
        number_of_measurements = count_measurements(f, header_size, record_size)
        
        # Read and store field data from binary file
        field_data = read_field_data(f, fields, targets, header_size, record_size, number_of_measurements)

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
targets = ['sza','FQ1','FQ2','FQ3','CLfrac','SurfType', 'datetime']

# Extract IASI L1C products from binary file
read_bin_L1C(fields, test_file, targets)