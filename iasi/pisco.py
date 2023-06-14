import os
import subprocess
from typing import Optional, Tuple

from configure import Config
from process import L1CProcessor, L2Processor
from correlate import Correlator

class Pisco:
    def __init__(self):
        """
        Initialize the Pisco class with given parameters.

        Args:
            year (int): The year for which data is to be processed.
            months (List[int]): List of months for which data is to be processed.
            days (List[int]): List of days for which data is to be processed.
            data_level (str): Type of data path. Accepts 'l1c' or 'l2'.
        """
        # Instantiate the Config class and set_parameters() for analysis
        self.config = Config()
        self.config.set_parameters()
        
        self.data_level: str = None
        self.year: str = None
        self.month: str = None
        self.day: str = None
        self.datapath_in: str = None
        self.datapath_out: str = None
        self.datafile_in: str = None
        self.datafile_out: str = None
        # self.datafile_l1c: str = None
        # self.datafile_l2: str = None

        

    def _get_datapath_out(self) -> str:
        """
        Gets the data path for the output based on the data level, year, month, and day.

        Raises:
            ValueError: If the data level is neither 'l1c' nor 'l2'.
            
        Returns:
            str: Output data path.
        """
        if (self.data_level == 'l1c') or (self.data_level == 'l2'):
            return f"{self.config.datapath_out}{self.data_level}/{self.year}/{self.month}/{self.day}/"
        else:
            # If the data level is not 'l1c' or 'l2', raise an error
            raise ValueError("Invalid data path type. Accepts 'l1c' or 'l2'.")

    def _get_datapath_in(self) -> str:
        """
        Gets the data path for the input based on the data level, year, month, and day.
        
        Raises:
            ValueError: If the data level is neither 'l1c' nor 'l2'.
            
        Returns:
            str: Input data path.
        """
        # Check the data level
        if self.data_level == 'l1c':
            # Format the input path string and return it
            return f"/bdd/metopc/{self.data_level}/iasi/"
        elif self.data_level == 'l2':
            # Format the input path string with an additional 'clp/' at the end and return it
            return f"/bdd/metopc/{self.data_level}/iasi/{self.year}/{self.month}/{self.day}/clp/"
        else:
            # If the data level is not 'l1c' or 'l2', raise an error
            raise ValueError("Invalid data path type. Accepts 'l1c' or 'l2'.")

    def get_datapaths(self) -> None:
        """
        Retrieves the input and output data paths.
        """
        # Get the input data path
        self.datapath_in = self._get_datapath_in()
        # Get the output data path
        self.datapath_out = self._get_datapath_out()


    def _check_preprocessed_files(self, result: object) -> bool:
        if "No L1C data files found" in result.stdout:
            print(result.stdout)
            return False
        elif "No L2 data files found" in result.stdout:
            print(result.stdout)
            return False
        else:
            return True
    
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
            ValueError: If the data level is neither 'l1c' nor 'l2'.

        Returns:
            str: Command to run for data extraction.
        """
        if self.data_level == 'l1c':
            # Define the path to the run executable
            runpath = f"./bin/obr_v4"
            # Get the command parameters
            parameters = self._build_parameters()
            # Return the complete command
            return f"{runpath} -d {self.datapath_in} {parameters} -out {self.datapath_out}{self.datafile_out}"
        elif self.data_level == 'l2':
            # Define the path to the run executable
            runpath = "./bin/BUFR_iasi_clp_reader_from20190514"
            # Return the complete command
            return f"{runpath} {self.datapath_in}{self.datafile_in} {self.datapath_out}{self.datafile_out}"
        else:
            # If the data level is not 'l1c' or 'l2', raise an error
            raise ValueError("Invalid data path type. Accepts 'l1c' or 'l2'.")

    def _run_command(self) -> Optional[bool]:
        """
        Executes and monitors the command to extract IASI data.
        """
        # Build the command string to execute the binary script
        command = self._get_command()
        print(command)
        try:
            # Run the command in a bash shell and capture the output
            result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            # The subprocess module will raise a CalledProcessError if the process returns a non-zero exit status
            # The standard error of the command is available in e.stderr
            raise RuntimeError(f"{str(e)}, stderr: {e.stderr.decode('utf-8')}")
        except Exception as e:
            # Catch any other exceptions
            raise RuntimeError(f"An unexpected error occurred while running the command '{command}': {str(e)}")
        return result
    
    def _create_run_directory(self) -> None:
        """
        Creates the directory to save the output files, based on the input file name and time.
        
        Returns:
            intermediate_file (str): the full path to the intermediate file produced by IASI extraction script.Z
        """
        if self.data_level == 'l1c':
            # Get the output file name from the input file name
            self.datafile_out = "intermediate.bin"
        elif self.data_level == 'l2':
            self.datafile_out = self.datafile_in.split(",")[2]
            # Determine if the time is during the day or night
            hour = int(self.datafile_out[27:29])
            time = "day" if (6 <= hour <= 18) else "night"
            
            # Trim day/night subdirectory from any previous iterations
            if ("day" in self.datapath_out) or ("night" in self.datapath_out):
                self.datapath_out = f"{os.path.dirname(os.path.dirname(self.datapath_out))}/"
            # Update the output data path
            self.datapath_out = f"{self.datapath_out}{time}/"
        else:
            # If the data level is not 'l1c' or 'l2', raise an error
            raise ValueError("Invalid data path type. Accepts 'l1c' or 'l2'.")
        
        # Create the output directory if it doesn't exist
        os.makedirs(self.datapath_out, exist_ok=True)
        return f"{self.datapath_out}{self.datafile_out}"

    def preprocess(self) -> Tuple[bool, str]:
        """
        Preprocesses the IASI data.
        """
        # Create the output directory and point to intermediate file (L1C: OBR, L2: BUFR)
        intermediate_file = self._create_run_directory()
        # Run the command to extract the data
        result = self._run_command()
        # Check if files are produced. If not, skip processing
        check = self._check_preprocessed_files(result)
        return intermediate_file, check

    def process_files(self) -> None:
        """
        Processes all IASI files in the input directory.
        """
        if self.data_level == 'l1c':
            # Preprocess the current input file. If no IASI data files are found, skip processing (empty file still created, delete after)
            intermediate_file, check = self.preprocess()
            if check:
                # Process the current file
                self.process(intermediate_file)
            else:
                # Delete the intermediate file (intermediate file will only be a few bytes, so there is not much I/O overhead)
                self._delete_intermediate_reduction_data(intermediate_file)
        elif self.data_level == 'l2':
            # Check if the input data path exists
            if os.path.isdir(self.datapath_in):
                # Process each file in the directory
                for datafile_in in os.scandir(self.datapath_in):
                    # Check that entry is a file
                    if datafile_in.is_file():
                        # Set the current input file
                        self.datafile_in = datafile_in.name
                        # Preprocess the current input file. If no IASI data files are found, skip processing (empty file still created, delete after)
                        intermediate_file, check = self.preprocess()
                        if check:
                            # Process the current file
                            self.process(intermediate_file)
                        else:
                            # Delete the intermediate file (intermediate file will only be a few bytes, so there is not much I/O overhead)
                            self._delete_intermediate_reduction_data(intermediate_file)


    def _delete_intermediate_reduction_data(self, intermediate_file: str):
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
        with L2Processor(intermediate_file, self.config.latitude_range, self.config.longitude_range, self.config.cloud_phase) as file:
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
        with L1CProcessor(intermediate_file, self.config.targets) as file:
            file.extract_spectra(self.datapath_out, self.datafile_out, self.year, self.month, self.day)
        return

    def process(self, intermediate_file: str) -> None:
        """
        Runs separate processors for the IASI data based on its level, because
        each intermediate file is different. 

        Raises:
            ValueError: If the data level is neither 'l1c' nor 'l2'.
        """
        # Choose the processing function based on the data level
        if self.data_level == 'l1c':
            self._process_l1c(intermediate_file)
        elif self.data_level == 'l2':
            self._process_l2(intermediate_file)
        else:
            # If the data level is not 'l1c' or 'l2', raise an error
            raise ValueError("Invalid data path type. Accepts 'l1c' or 'l2'.")


    def _get_suffix(self):
        old_suffix=".bin"
        if self.data_level == 'l1c':
            new_suffix=".bin"
        elif self.data_level == 'l2':
            new_suffix=".out"
        else:
            raise ValueError("Invalid data path type. Accepts 'l1c' or 'l2'.")
        return old_suffix, new_suffix

    def rename_files(self):
        old_suffix, new_suffix = self._get_suffix()
        if os.path.isdir(self.datapath_out):
            for filename in os.scandir(self.datapath_out):
                if filename.name.endswith(old_suffix):
                    new_filename = f"{filename.name[:-len(old_suffix)]}{new_suffix}"
                    os.rename(filename.path, os.path.join(self.datapath_out, new_filename))


    def correlate_l1c_l2(self):
        with Correlator(self.datapath_out, self.datafile_out, self.config.cloud_phase) as file:
            file.filter_spectra()
        return
