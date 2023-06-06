from datetime import date, timedelta
import os
import subprocess
from typing import List, Union

class IASIExtractor:
    def __init__(self, data_level: str):
        """
        Initialize IASI_L2 class with given parameters.

        Args:
            year (int): The year for which data is to be processed.
            months (List[int]): List of months for which data is to be processed.
            days (List[int]): List of days for which data is to be processed.
            data_level (str): Type of data path. Accepts 'l1C' or 'l2'.
        """
        self.data_level: str = data_level
        self.year: str = None
        self.month: str = None
        self.day: str = None
        self.datapath_in: str = None
        self.datapath_out: str = None
        self.datafile_in: str = None
        self.datafile_out: str = None


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


    def _delete_intermediate_file(self):
        pass
    
    def _process_l1c():
        pass
    
    def _process_l2():
        pass

    def process(self):
        if self.data_level == 'l1C':
            self._process_l1c()
        elif self.data_level == 'l2':
            self._process_l2()
        else:
            raise ValueError("Invalid data path type. Accepts 'l1C' or 'l2'.")
        self._delete_intermediate_file()
        return


    def _build_parameters(self):
        iasi_channels = [(i + 1) for i in range(8461)]
        list_of_parameters = [f"-fd {self.year}-{self.month}-{self.day} -ld {self.year}-{self.month}-{self.day}", # first and last day
                                f"-c {iasi_channels[0]}-{iasi_channels[-1]}", # channels
                                f"-of bin" # outputfile format
                            ]
        return ' '.join(list_of_parameters)
   
    def _get_command(self):
        if self.data_level == 'l1C':
            runpath = f"./bin/obr_v4"            
            parameters = self._build_parameters()
            return f"{runpath} -d {self.datapath_in}{self.datafile_in} {parameters} -out {self.datapath_out}{self.datafile_out}"
        elif self.data_level == 'l2':
            runpath = "./bin/BUFR_iasi_clp_reader_from20190514"
            return f"{runpath} {self.datapath_in}{self.datafile_in} {self.datapath_out}{self.datafile_out}"
        else:
            raise ValueError("Invalid data path type. Accepts 'l1C' or 'l2'.")

    def _run_command(self):
        command = self._get_command()
        subprocess.run(['bash', '-c', command], check=True)
        print(type(command))

    def _create_run_directory(self):
        self.datafile_out = self.datafile_in.split(",")[2]
        hour = int(self.datafile_out[27:29])
        time = "day" if (hour >= 6) and (hour <= 18) else "night"
        self.datapath_out = f"{self.datapath_out}{time}/"
        os.makedirs(self.datapath_out, exist_ok=True)
        return
    
    def preprocess(self):
        self._create_run_directory()
        self._run_command()
    
    def extract_data(self):
        if os.path.isdir(self.datapath_in):
            for datafile_in in os.scandir(self.datapath_in):
                self.datafile_in = datafile_in.name
                self.preprocess()
                self.process()
    

    def _get_datapath_out(self):
        if (self.data_level == 'l1C') or (self.data_level == 'l2'):
            return f"/data/pdonnelly/iasi/metopc/{self.data_level}/{self.year}/{self.month}/{self.day}/"
        else:
            raise ValueError("Invalid data path type. Accepts 'l1C' or 'l2'.")

    def _get_datapath_in(self):
        if self.data_level == 'l1C':
            return f"/bdd/metopc/{self.data_level}/iasi/{self.year}/{self.month}/{self.day}/"
        elif self.data_level == 'l2':
            return f"/bdd/metopc/{self.data_level}/iasi/{self.year}/{self.month}/{self.day}/clp/"
        else:
            raise ValueError("Invalid data path type. Accepts 'l1C' or 'l2'.")
       
    def get_datapaths(self):
        self.datapath_in = self._get_datapath_in()
        self.datapath_out = self._get_datapath_out()
        return


def main():
    
    years = [2020]
    months = [1]
    days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    data_level="l2"

    # Instantiate an Extractor to get data from raw binary files
    extractor = IASIExtractor(data_level)
    
    for year in years:
        extractor.year = f"{year:04d}"
        
        for im, month in enumerate(months):
            extractor.month = f"{month:02d}"
            
            for day in range(1, days[im-1] + 1):
                extractor.day = f"{day:02d}"
                
                extractor.get_datapaths()
                extractor.extract_data()
                extractor.rename_files()


if __name__ == "__main__":
    main()
