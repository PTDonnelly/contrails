from datetime import datetime
import os
import subprocess
from typing import List, Union

import snoop

from iasi_config import Config
from iasi_processor import L1CProcessor as L1C
from iasi_processor import L2Processor as L2

class IASIExtractor:
    def __init__(self):
        """
        Initialize the IASI extractor class with given parameters.

        Args:
            year (int): The year for which data is to be processed.
            months (List[int]): List of months for which data is to be processed.
            days (List[int]): List of days for which data is to be processed.
            data_level (str): Type of data path. Accepts 'l1C' or 'l2'.
        """
        self.config = Config()
        self.config.set_parameters()
        self.data_level: str = self.config.data_level
        self.year: str = None
        self.month: str = None
        self.day: str = None
        self.datapath_in: str = None
        self.datapath_out: str = None
        self.datafile_in: str = None
        self.datafile_out: str = None
        self.l1c_targets: List[str] = None


    def _get_suffix(self):
        old_suffix=".bin"
        if self.data_level == 'l1C':
            new_suffix=".bin"
        elif self.data_level == 'l2':
            new_suffix=".out"
        else:
            raise ValueError("Invalid data path type. Accepts 'l1C' or 'l2'.")
        return old_suffix, new_suffix

    def rename_files(self):
        old_suffix, new_suffix = self._get_suffix()
        if os.path.isdir(self.datapath_out):
            for filename in os.scandir(self.datapath_out):
                if filename.name.endswith(old_suffix):
                    new_filename = f"{filename.name[:-len(old_suffix)]}{new_suffix}"
                    os.rename(filename.path, os.path.join(self.datapath_out, new_filename))


    def _delete_intermediate_file(self, intermediate_file: str):
        # Delete intermediate binary file (after extracting spectra and metadata)
        os.remove(intermediate_file)
        pass
    
    def _process_l2(self, intermediate_file: str):
        """
        Process level 2 IASI data.

        Extracts and processes IASI cloud products from intermediate csv files and
        stores all data points with Cloud Phase == 2 (ice).

        The result is a HDF5 file containing all locations of ice cloud from this intermediate file.
        """
        with L2(intermediate_file, self.config.latitude_range, self.config.longitude_range) as file:
            file.extract_ice_clouds()
        return

    def _process_l1c(self, intermediate_file: str) -> None:
        """
        Process level 1C IASI data.

        Extracts and processes IASI data from intermediate binary files,
        applies quality control and saves the output.

        The result is a HDF5 file containing all good spectra from this intermediate file.
        """
        # Process extracted IASI data from intermediate binary files
        with L1C(intermediate_file, self.config.targets) as file:
            file.extract_spectra(self.datapath_out, self.year, self.month, self.day)
        return

    def process(self) -> None:
        """
        Runs separate processors for the IASI data based on its level, because
        each intermediate file is different. 

        Raises:
            ValueError: If the data level is neither 'l1C' nor 'l2'.
        """
        # Point to intermediate binary file of IASI products (L1C: OBR, L2: BUFR)
        intermediate_file = f"{self.datapath_out}{self.datafile_out}"
        
        # Choose the processing function based on the data level
        if self.data_level == 'l1C':
            self._process_l1c(intermediate_file)
        elif self.data_level == 'l2':
            self._process_l2(intermediate_file)
        else:
            # If the data level is not 'l1C' or 'l2', raise an error
            raise ValueError("Invalid data path type. Accepts 'l1C' or 'l2'.")
        
        # Delete the intermediate file
        self._delete_intermediate_file(intermediate_file)


    def _build_parameters(self) -> str:
        """
        Builds the parameter string for the IASI data extraction command.
        
        Returns:
            str: Parameters for the command.
        """
        # Define the parameters for the command
        list_of_parameters = [
            f"-fd {self.year}-{self.month}-{self.day} -ld {self.year}-{self.month}-{self.day}",  # first and last day
            f"-c {self.config.channels[0]}-{self.config.channels[-1]}",  # spectral channels
            f"-of bin"  # output file format
        ]
        # Join the parameters into a single string and return
        return ' '.join(list_of_parameters)

    def _get_command(self) -> str:
        """
        Builds the command to extract IASI data based on the data level.

        Raises:
            ValueError: If the data level is neither 'l1C' nor 'l2'.

        Returns:
            str: Command to run for data extraction.
        """
        if self.data_level == 'l1C':
            # Define the path to the run executable
            runpath = f"./bin/obr_v4"
            # Get the command parameters
            parameters = self._build_parameters()
            # Return the complete command
            return f"{runpath} -d {self.datapath_in}{self.datafile_in} {parameters} -out {self.datapath_out}{self.datafile_out}"
        elif self.data_level == 'l2':
            # Define the path to the run executable
            runpath = "./bin/BUFR_iasi_clp_reader_from20190514"
            # Return the complete command
            return f"{runpath} {self.datapath_in}{self.datafile_in} {self.datapath_out}{self.datafile_out}"
        else:
            # If the data level is not 'l1C' or 'l2', raise an error
            raise ValueError("Invalid data path type. Accepts 'l1C' or 'l2'.")

    def _run_command(self) -> None:
        """
        Executes the command to extract IASI data.
        """
        # Get the command to run
        command = self._get_command()
        # Run the command in a bash shell
        subprocess.run(['bash', '-c', command], check=True)
        # Print the type of the command string
        print(type(command))

    def _create_run_directory(self) -> None:
        """
        Creates the directory to save the output files, based on the input file name and time.
        """
        # Get the output file name from the input file name
        self.datafile_out = self.datafile_in.split(",")[2]
        # Determine if the time is during the day or night
        hour = int(self.datafile_out[27:29])
        time = "day" if (6 <= hour <= 18) else "night"
        # Update the output data path
        self.datapath_out = f"{self.datapath_out}{time}/"
        # Create the output directory if it doesn't exist
        os.makedirs(self.datapath_out, exist_ok=True)

    def preprocess(self) -> None:
        """
        Preprocesses the IASI data.
        """
        # Create the output directory
        self._create_run_directory()
        # Run the command to extract the data
        self._run_command()

    @snoop
    def process_files(self) -> None:
        """
        Processes all IASI files in the input directory.
        """
        # Check if the input data path exists
        if os.path.isdir(self.datapath_in):
            # Process each file in the directory
            for datafile_in in os.scandir(self.datapath_in):
                # Set the current input file
                self.datafile_in = datafile_in.name
                # Preprocess the current file
                self.preprocess()
                # Process the current file
                self.process()

    

    def _get_datapath_out(self) -> str:
        """
        Gets the data path for the output based on the data level, year, month, and day.

        Raises:
            ValueError: If the data level is neither 'l1C' nor 'l2'.
            
        Returns:
            str: Output data path.
        """
        # Check if the data level is either 'l1C' or 'l2'
        if (self.data_level == 'l1C') or (self.data_level == 'l2'):
            # Format the output path string and return it
            return f"/data/pdonnelly/iasi/metopc/{self.data_level}/{self.year}/{self.month}/{self.day}/"
        else:
            # If the data level is not 'l1C' or 'l2', raise an error
            raise ValueError("Invalid data path type. Accepts 'l1C' or 'l2'.")

    def _get_datapath_in(self) -> str:
        """
        Gets the data path for the input based on the data level, year, month, and day.
        
        Raises:
            ValueError: If the data level is neither 'l1C' nor 'l2'.
            
        Returns:
            str: Input data path.
        """
        # Check if the data level is 'l1C'
        if self.data_level == 'l1C':
            # Format the input path string and return it
            return f"/bdd/metopc/{self.data_level}/iasi/{self.year}/{self.month}/{self.day}/"
        # Check if the data level is 'l2'
        elif self.data_level == 'l2':
            # Format the input path string with an additional 'clp/' at the end and return it
            return f"/bdd/metopc/{self.data_level}/iasi/{self.year}/{self.month}/{self.day}/clp/"
        else:
            # If the data level is not 'l1C' or 'l2', raise an error
            raise ValueError("Invalid data path type. Accepts 'l1C' or 'l2'.")

    def get_datapaths(self) -> None:
        """
        Retrieves the input and output data paths.
        """
        # Get the input data path
        self.datapath_in = self._get_datapath_in()
        # Get the output data path
        self.datapath_out = self._get_datapath_out()

    
