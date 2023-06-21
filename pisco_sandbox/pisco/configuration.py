from typing import List
import commentjson

class Configurer:
    def __init__(self, path_to_config_file: str):
        self.data_level: str = ""
        
        # Initialise the Config class with your JSON configuration file
        with open(path_to_config_file, 'r') as file:
            # Access the parameters directly as attributes of the class. 
            self.__dict__ = commentjson.load(file)
            
        # Perform any necessary post-processing before executing
        self.channels: List[int] = self.set_channels(n=10)
    
    @staticmethod
    def set_channels(n=8461):
        # Set the list of IASI channel indices (defaults to maximum of 8461)
        return [(i + 1) for i in range(n)]