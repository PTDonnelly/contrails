from typing import List, Tuple, Optional

class Config:
    def __init__(self):
        self.year_list: List[int] = []
        self.month_list: List[int] = []
        self.day_list: Optional[List[int]] = []
        self.days_in_months: List[int] = []
        self.data_level: List[str] = []
        self.mode: str = None
        self.datapath_out: str = None
        self.targets: List[str] = []
        self.channels: List[int] = []
        self.latitude_range: Tuple[float, float] = ()
        self.longitude_range: Tuple[float, float] = ()
        self.cloud_phase: int = None

    def set_parameters(self):
        self.set_filepath_parameters()
        self.set_spatial_parameters()
        self.set_temporal_parameters()
        self.set_level_1C_parameters()
        self.set_level_2_parameters()
        self.set_processing_mode_and_data_level()

    def set_filepath_parameters(self):
        # Sets the data path for the processed output files (defined by user)
        self.datapath_out = f"/data/pdonnelly/iasi/metopc/"
    

    def set_spatial_parameters(self):
        # Set spatial range of binning
        self.latitude_range = (-90, 90)
        self.longitude_range = (-180, 180)


    def set_temporal_parameters(self):
         # Specify date range for zeroth-level extraction
        self.year_list = [2022]
        self.month_list = [3]
        self.day_list = [24] # List[int] (for specific days) or None (to scan days in month)
        self.days_in_months = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]


    def set_level_1C_parameters(self):
        # Specify target IASI L1C products
        self.targets = ['satellite_zenith_angle', 'quality_flag_1', 'quality_flag_2',
                        'quality_flag_3', 'cloud_fraction', 'surface_type']
        
        # Create a list of all IASI channel indices
        self.channels = [(i + 1) for i in range(8461)]


    def set_level_2_parameters(self):
        # Set the cloud phase desired from the L2 products
        self.cloud_phase = 2


    def set_processing_mode_and_data_level(self):
        # Specify the processing mode ("Process" | "Correlate")
        self.mode = "Process"
        # Specify the IASI product for extraction
        L1C, L2 = True, True
        
        # Specify level of IASI data for zeroth-level extraction ("L1C" | "L2")
        if self.mode == "Process":
            if L1C:
                self.data_level = ["l1c"]
            elif L2:
                self.data_level = ["l2"]
        elif self.mode == "Correlate":
            self.data_level = ["l2", "l1c"]
        else:
            raise ValueError("Invalid analysis mode. Accepts 'Process' or 'Correlate'.")

        # Check mode data level input agrees before execution
        if self.mode == "Process" and (L1C and L2) or (not L1C and not L2):
            raise ValueError("Invalid data path type. Process mode requires either 'l1C' or 'l2'.")
        elif self.mode == "Correlate" and self.data_level != ["l2", "l1c"]:
            raise ValueError("Invalid data path type. Correlate mode requires ['l2', 'l1c'].")