from typing import List, Tuple

class Config:
    def __init__(self):
        self.year_list: List[int] = []
        self.month_list: List[int] = []
        self.days_in_months: List[int] = []
        self.data_level: str = ""
        self.datapath_in: str = None
        self.datapath_out: str = None
        self.targets: List[str] = []
        self.channels: List[int] = []
        self.latitude_range: Tuple[float, float] = ()
        self.longitude_range: Tuple[float, float] = ()
        self.cloud_phase: int = None

    def set_parameters(self):
        
        # Specify date range for zeroth-level extraction
        self.year_list = [2020]
        self.month_list = [1]
        self.days_in_months = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

        # Specify level of IASI data for zeroth-level extraction ("L1C" | "L2")
        self.data_level = "l2"

        # Sets the data path for the binary input files and processed output files based on the data level
        self.datapath_in = f"/bdd/metopc/{self.data_level}/iasi/"
        self.datapath_out = f"/data/pdonnelly/iasi/metopc/{self.data_level}/"
        
        ### Level 1C
        # Specify target IASI L1C products
        self.targets = ['satellite_zenith_angle', 'quality_flag_1', 'quality_flag_2',
                        'quality_flag_3', 'cloud_fraction', 'surface_type']
        
        # Create a list of all IASI channel indices
        self.channels = [(i + 1) for i in range(8461)]

        ### Level 2
        # Set spatial range of binning
        self.latitude_range = (-90, 90)
        self.longitude_range = (-180, 180)
        
        # Set the cloud phase desired from the L2 products
        self.cloud_phase = 2
