import glob
import os
from multiprocessing import Pool
from typing import Tuple, List, Union

import numpy as np

class L1CReader:
    def __init__(self, filepath: str, filename: str):
        self.filepath = filepath
        self.filename = filename
        self.locations = None
        self.local_times = None
        self.spectra = None
        self.target_parameters = None
        self.datetimes = None
   
    def read_file(self) -> List[np.ndarray]:
        """
        Read the file and split the data into separate arrays.
        
        Returns:
            A list of numpy arrays containing the split data.
        Raises:
            IOError: If there is an error reading the file.
        """
        try:
            # Read the first line to get the header and determine where to split the columns
            with open(self.filename, "r") as file:
                split_indices = [int(val) for val in file.readline().strip().split()]
            cumulative_split_indices = np.cumsum(split_indices)
            
            # Load the data from the file
            data = np.loadtxt(self.filename, skiprows=1)
            
            # Split the data array into columns (ignoring the last element)
            return np.hsplit(data, cumulative_split_indices[:-1])
        except IOError as e:
            raise IOError(f"Error reading file: {str(e)}")


    def get_data(self) -> None:
        """
        Process the data from the file and assign it to the corresponding attributes.
        
        Raises:
            ValueError: If the data arrays have unexpected shapes.
        """
        array1, array2, array3, array4 = self.read_file()
        self.locations = self.get_location(array1)
        self.local_times = self.get_local_time(array1)
        self.spectra = array2
        self.target_parameters = array3
        self.datetimes = array4

    @staticmethod
    def get_location(array: np.ndarray) -> List[np.ndarray]:
        """
        Extract the longitude [0] and latitude [1] data from the input array.
        
        Args:
            array: A numpy array containing the data.
        
        Returns:
            A numpy array containing the location data.
        Raises:
            ValueError: If the array shape is not as expected.
        """
        if array.shape[1] >= 1:
            return [tuple((lon, lat)) for lon, lat in zip(array[:, 0], array[:, 1])]
        raise ValueError("Unexpected array shape for location data.")

    @staticmethod
    def get_local_time(array: np.ndarray) -> np.ndarray:
        """
        Extract the local time data from the input array.
        
        Args:
            array: A numpy array containing the data.
        
        Returns:
            A numpy array containing the local time data.
        Raises:
            ValueError: If the array shape is not as expected.
        """
        if array.shape[1] != 1:
            return array[:, 2]
        raise ValueError("Unexpected array shape for local time data.")
    
    @staticmethod
    def build_datetime(datetime: np.array) -> object:
        year, month, dat, hour, minute, millisecond = [int(element) for element in datetime]
        return str(dt(year, month, dat, hour, minute, int(millisecond // 1000))).split()
    

def find_data_L1C(path_L1: str):
    
    for file in path_L1:
        filepath, filename = "path", "name"
        L1CReader(filepath, filename)
    
    return 

def get_data_L2(path_L2: str):
    data = np.loadtxt(path_L2, delimiter=',')
    locations = data[:, 0]
    times = data[:, 1]
    return locations, times

def process_file(filename):
    
    # Initialize an empty list to store the extracted spectra.
    extracted_spectra = []

    # Process each text file
    with open(filename, 'r') as file:
        # Read the spectra and corresponding location-time information from the file
        spectra = np.loadtxt(file, usecols=(1,))  # Assuming spectra are in the second column

        # Assuming location and time information are in the first and third columns respectively
        file_locations = np.loadtxt(file, usecols=(0,))
        file_times = np.loadtxt(file, usecols=(2,))

        # Find the indices of matching location-time points in dataset 2
        matching_indices = np.where(np.isin(locations, file_locations) & np.isin(times, file_times))

        # Extract the spectra for the matching location-time points
        extracted_spectra.extend(spectra[matching_indices])
    return

def search_data(path_L1C, path_L2, num_processes, chunk_size):

    # Get the list of text files in dataset 1 directory
    data_L1C = find_data_L1C(path_L1C)

    # Load dataset 2 (locations and times) into memory
    data_L2 = get_data_L2(path_L2)

    # Create a pool of worker processes
    with Pool(processes=num_processes) as pool:
        for file in data_L1C:
            filepath = os.path.join(path_L1C, file)

            # Split the text file into chunks
            chunks = split_file(filepath, chunk_size)

            # Process each chunk in parallel
            pool.map(process_file, chunks)

            # Check if any location-time point in dataset 2 matches the current chunk
            matching_points = [point for point in data_L2 if point_matches_chunk(point, chunks)]

            # Process the matching points further if needed
            process_matching_points(matching_points)


def main():

    path_L1C = "C:\\Users\\padra\\Documents\\Research\\github\\contrails\\iasi\\L1C_test.txt"
    path_L2 = "C:\\Users\\padra\\Documents\\Research\\github\\contrails\\iasi\\L2_test.txt"
    search_data(path_L1C, path_L2, num_processes=4, chunk_size=100000)

    return

if __name__ == "__main__":
    main()