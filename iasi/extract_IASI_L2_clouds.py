import os
import subprocess
from typing import List, Union

class IASIExtractor:
    def __init__(self, year: int, months: List[int], days: List[int], data_level: str):
        """
        Initialize IASI_L2 class with given parameters.

        Args:
            year (int): The year for which data is to be processed.
            months (List[int]): List of months for which data is to be processed.
            days (List[int]): List of days for which data is to be processed.
            data_level (str): Type of data path. Accepts 'L1C' or 'L2'.
        
        Methods:
        _create_output_directory():
            Creates the output directory if it does not exist and returns its path.
        _process_files_for_date(month: str, day: str):
            Processes all files for a specific date.
        _process_file(datafile_in: str):
            Processes a specific file.
        _run_data_processing(datafile_in: str, datafile_out: str):
            Runs the data processing command for a specific file.
        _rename_files_with_suffix(old_suffix: str, new_suffix: str):
            Renames all files in the output directory, replacing the old suffix with the new suffix.
        process_files():
            Processes all files for all dates in the given year.
        """
        self.year = year
        self.months = months
        self.days = days
        self.data_level = data_level
        self.datapath_out = self._create_output_directory()

    def _create_output_directory(self):
        datapath_out = f"/data/pdonnelly/IASI/metopc/{self.year}"
        os.makedirs(datapath_out, exist_ok=True)
        return datapath_out

    def _get_command(self, datafile_in: str, datafile_out: str):
        if self.data_level == 'L1C':
            iasi_channels = [(i + 1) for i in range(8461)]
            executable = f"/home/pdonnelly/data/obr_v4"
            filepath = f"-d {datafile_in}"
            first_date = f"-fd {self.year:04d}-{self.month:02d}-{self.day:02d}"
            last_date = f"-ld {self.year:04d}-{self.month:02d}-{self.day:02d}"
            channels = f"-c {iasi_channels[0]}-{iasi_channels[-1]}"
            filter = "" #f"-mf {file_in}"
            output = f"-of bin -out {datafile_out}"
            
            return f"{executable} {filepath} {first_date} {last_date} {channels} {filter} {output}"
        elif self.data_level == 'L2':
            executable = "/BUFR_iasi_clp_reader_from20190514 "
            return f"{executable} {datafile_in} {self.datapath_out}{datafile_out}"
        else:
            raise ValueError("Invalid data path type. Accepts 'L1C' or 'L2'.")

    def _run_data_processing(self, datafile_in: str, datafile_out: str):
        command = self._get_command(datafile_in, datafile_out)
        subprocess.run(command, shell=True)
        print(command)

    def _process_file(self, datafile_in: str):
        datafile_out = datafile_in.split(",")[2]
        hour = datafile_out[27:29]
        if int(hour) <= 6 or int(hour) >= 18:
            self._run_data_processing(datafile_in, datafile_out)

    def _get_datapath_in(self, month: str, day: str):
        if self.data_level == 'L1C':
            return f"/bdd/metopc/l1c/iasi/{self.year}/{month}/{day}/"
        elif self.data_level == 'L2':
            return f"/bdd/metopc/l2/iasi/{self.year}/{month}/{day}/clp"
        else:
            raise ValueError("Invalid data path type. Accepts 'L1C' or 'L2'.")
        
    def _process_files_for_date(self, month: str, day: str):
        datapath_in = self._get_datapath_in(month, day)
        if os.path.isdir(datapath_in):
            for datafile_in in os.scandir(datapath_in):
                self._process_file(datafile_in.name)

    def _rename_files_with_suffix(self, old_suffix: str, new_suffix: str):
        if os.path.isdir(self.datapath_out):
            for filename in os.scandir(self.datapath_out):
                if filename.name.endswith(old_suffix):
                    new_filename = f"{filename.name[:-len(old_suffix)]}{new_suffix}"
                    os.rename(filename.path, os.path.join(self.datapath_out, new_filename))

    def process_files(self):
        for month in self.months:
            for day in range(1, self.days[month-1] + 1):
                self.month = month
                self.day = day
                self._process_files_for_date(f"{month:02d}", f"{day:02d}")

def main():
    year = 2020
    months = [1]
    days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    
    data = IASIExtractor(year, months, days, data_level="L2")
    data.process_files()
    data._rename_files_with_suffix(old_suffix=".bin", new_suffix=".out")


if __name__ == "__main__":
    main()
