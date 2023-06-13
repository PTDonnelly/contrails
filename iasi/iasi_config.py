from typing import List, Tuple, Optional

class Config:
    def __init__(self):
        self.year_list: List[int] = []
        self.month_list: List[int] = []
        self.day_list: Optional[List[int]] = []
        self.days_in_months: List[int] = []
        self.data_level: str = ""
        self.mode: str = None
        self.datapath_out: str = None
        self.targets: List[str] = []
        self.channels: List[int] = []
        self.latitude_range: Tuple[float, float] = ()
        self.longitude_range: Tuple[float, float] = ()
        self.cloud_phase: int = None

    def set_parameters(self):
        
        # Specify date range for zeroth-level extraction
        self.year_list = [2020]
        self.month_list = [3]
        self.day_list = [1] # List[int] (for specific days) or None (to scan days in month)
        self.days_in_months = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
  
        # Sets the data path for the processed output files (defined by user)
        self.datapath_out = f"/data/pdonnelly/iasi/metopc/"

        # Specify the processing mode ("Process" | "Correlate")
        self.mode = "Process"
        
        # Set data level to analyse (if self.mode == "Correlate", will default to ["l2", "l1c"])
        self.data_level = ["l1c"]

        # Verify data level and mode (based on self.mode)
        self.set_data_level()
        
        # Set Level 1C parameters
        self.set_l1c()

        # Set Level 2 parameters
        self.set_l2()


    def set_data_level(self):
        # Specify level of IASI data for zeroth-level extraction ("L1C" | "L2")
        if self.mode == "Process":
            self.data_level = self.data_level
        elif self.mode == "Correlate":
            self.data_level = ["l2", "l1c"]
        else:
            raise ValueError("Invalid analysis mode. Accepts 'Process' or 'Correlate'.")
        # Check mode data level input agrees before execution
        self.check_mode()


    def set_l1c(self):
        # Specify target IASI L1C products
        self.targets = ['satellite_zenith_angle', 'quality_flag_1', 'quality_flag_2',
                        'quality_flag_3', 'cloud_fraction', 'surface_type']
        
        # Create a list of all IASI channel indices
        self.channels = [(i + 1) for i in range(8461)]


    def set_l2(self):
        # Set spatial range of binning
        self.latitude_range = (-90, 90)
        self.longitude_range = (-180, 180)
        
        # Set the cloud phase desired from the L2 products
        self.cloud_phase = 2


    def check_mode(self):
        if (self.mode == "Process") & (len(self.data_level) > 1):
            # If the data level is not 'l1C' or 'l2', raise an error
            raise ValueError("Invalid data path type. Accepts 'l1C' or 'l2'.")
        elif (self.mode == "Correlate") & (self.data_level != ["l2", "l1c"]):
            # If the data level does not contain 'l1C' and 'l2', raise an error
            raise ValueError("Invalid data path type. Must be ['l2', 'l1c'].")