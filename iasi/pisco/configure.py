from typing import List
import commentjson

class Config:
    def __init__(self, path_to_config_file: str):
        self.data_level: str = ""
        # Initialise the Config class with your JSON configuration file
        with open(path_to_config_file, 'r') as file:
            # Access the parameters directly as attributes of the class. 
            self.__dict__ = commentjson.load(file)
            
        # Perform any necessary post-processing before executing
        self.channels: List[int] = self.set_channels()
    
    def set_channels(self):
        # Set the list of IASI channel indices
        n = 8461
        self.channels: List[int] = [(i + 1) for i in range(n)]

    def check_mode_and_data_level(self):
        # Check execution mode data level inputs agree before execution
        if self.mode == "Process":
            if (self.L1C and self.L2) or (not self.L1C and not self.L2):
                raise ValueError("Invalid data path type. Process mode requires either 'l1C' or 'l2'.")