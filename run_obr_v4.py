import subprocess
from sorcery import dict_of

def build_command(path_in: str, year: int, month: int, day: int, iasi_channels: list, file_in: str, file_out: str):
    run_dir = f"/home/pdonnelly/data/obr_v4"
    filepath = f"-d {path_in}"
    first_date = f"-fd {year:04d}-{month:02d}-{day:02d}"
    last_date = f"-ld {year:04d}-{month:02d}-{day:02d}"
    channels = f"c {iasi_channels[0]}-{iasi_channels[-1]}"
    mf = f"-mf {file_in}"
    suffix = f"of bin -out {file_out}"
    return f"{run_dir} {filepath} {first_date} {last_date} {channels} {mf} {suffix}"

# Define inputs for OBR tool
path_in = '/bdd/IASI/L1C/'
years = [2020]
months = [1] #, 2, 3]
days = [1] #[day for day in range(28)] 
iasi_channels = [1, 2, 3]
file_in = 'W_XX-EUMETSAT-Darmstadt,SOUNDING+SATELLITE,METOPA+IASI_C_EUMC_20200101000253_68496_eps_o_l1.bin'
file_out = 'file_out.bin'

# Loop through date-times
for year in years:
    for month in months:
        for day in days:

            # Construct command-line executable
            command = build_command(path_in, year, month, day, iasi_channels, file_in, file_out)
            
            print(command)
            
            # Execute OBR command
            subprocess.run(command, shell=True)