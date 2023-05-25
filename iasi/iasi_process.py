from datetime import datetime
import numpy as np
from typing import Tuple, Union, BinaryIO

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
    return datetime(int(year), int(month), int(day), int(hour), int(minute), int(millisecond // 1000))

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
            print(f"Extracting: {field}")

            # Skip file header
            pointer = header_size + 12 + cumsize
            f.seek(pointer, 0)

            # Skip bytes up to the next measurement
            byte_offset = record_size + 8 - dtype_size
    
            # Allocate memory, read binary data, skip empty values, save to dictionary
            data = np.empty(number_of_measurements)
            for measurement in range(number_of_measurements):
                value = np.fromfile(f, dtype=dtype, count=1, sep='', offset=byte_offset)
                if len(value) == 0:
                    data[measurement] = np.nan
                else:
                    data[measurement] = value
            field_data[field] = data

    # Create a timestamp for the observation
    if "datetime" in targets:       
        v_datetime = np.vectorize(vectorized_datetime)
        field_data["datetime"] = v_datetime(field_data['year'],
                                            field_data['month'],
                                            field_data['day'],
                                            field_data['hour'],
                                            field_data['minute'],
                                            field_data['millisecond'])
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
    hour, minute, millisecond, longitude = field_data['hour'], field_data['minute'], field_data['millisecond'], field_data['longitude']

    # Calculate the total time in hours, minutes, and milliseconds
    total_time = (hour * 1e4) + (minute * 1e2) + (millisecond / 1e3)

    # Extract the components from total_time, calculate the total time in hours
    hour_component = np.floor(total_time / 10000)
    minute_component = np.mod((total_time - np.mod(total_time, 100)) / 100, 100)
    second_component_in_minutes = np.mod(total_time, 100) / 60
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

def store_space_time_coordinates(field_data: dict, adjusted_time):
    return [field_data['longitude'], field_data['latitude'], adjusted_time < 12]

def store_spectral_radiance(f, header_size, record_size, number_of_measurements, number_of_channels):
    print(f"Extracting: radiance")

    # Skip file header and other record fields
    anchor = fields[-1][-1] # End of the surface_type field
    pointer = header_size + 12 + (anchor) + (4 * number_of_channels)
    f.seek(pointer, 0)
    
    # Skip bytes until the next measurement
    byte_offset = record_size + 8 - (4 * number_of_channels)

    # Allocate memory, read binary data, skip empty values
    data = np.empty((number_of_measurements, number_of_channels))
    for measurement in range(number_of_measurements):
        value = np.fromfile(f, dtype='float32', count=number_of_channels, sep='', offset=byte_offset)
        if len(value) == 0:
            data[measurement, :] = np.nan
        else:
            data[measurement, :] = value
    return data

def store_target_parameters(field_data: dict):
    return [data for field, data in field_data.items() if field in targets]

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
        number_of_measurements = 10

        # Read and store field data from binary file
        field_data = read_record_data(f, fields, targets, header_size, record_size, number_of_measurements)

        # Calculate adjusted time (in hours) to determine whether it is day or night
        adjusted_time = calculate_day_or_night(field_data)

        # Initialize the output with longitude, latitude, and adjusted
        output = store_space_time_coordinates(field_data, adjusted_time)
        
        # Read the spectral radiance matrix, append to the output
        output.append(store_spectral_radiance(f, header_size, record_size, number_of_measurements, number_of_channels))

        # Add target parameters to output
        output.extend(store_target_parameters(field_data))

    # Return the output as a tuple of numpy arrays 
    return output

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
    ('surface_type', 'uint8', 1, 93)
]

# Specify location of IASI L1C products binary file
test_file = "C:\\Users\\padra\\Documents\\Research\\data\\iasi\\test_file.bin"

# Specify target IASI L1C products
targets = ['surface_type']#'satellite_zenith_angle','quality_flag_1','quality_flag_2','quality_flag_3','cloud_fraction','surface_type', 'datetime']

# Extract IASI L1C products from binary file
longitude, latitude, time, radiance, target_parameters = read_bin_L1C(fields, test_file, targets)
