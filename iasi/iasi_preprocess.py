import subprocess

def build_command(path_in: str, year: int, month: int, day: int, iasi_channels: list, filter: str, file_out: str):
    run_dir = f"/home/pdonnelly/data/obr_v4 "
    filepath = f"-d {path_in} "
    first_date = f"-fd {year:04d}-{month:02d}-{day:02d} "
    last_date = f"-ld {year:04d}-{month:02d}-{day:02d} "
    channels = f"-c {iasi_channels[0]}-{iasi_channels[-1]} "
    filter = "" #f"-mf {file_in}"
    output = f"of bin -out {file_out} "
    return f"{run_dir}{filepath}{first_date}{last_date}{channels}{filter}{output}"

# Define inputs for OBR tool
path_in = '/bdd/IASI/L1C/'
years = [2020]
months = [1] #, 2, 3]
days = [1] #[day for day in range(28)] 
iasi_channels = [1, 2, 3]
filter = 'W_XX-EUMETSAT-Darmstadt,SOUNDING+SATELLITE,METOPA+IASI_C_EUMC_20200101000253_68496_eps_o_l1.bin'
file_out = 'file_out.bin'

# Loop through date-times
for year in years:
    for month in months:
        for day in days:

            # Construct command-line executable
            command = build_command(path_in, year, month, day, iasi_channels, filter, file_out)

            # Extract IASI spectra (execute OBR command and produce intermediate binary files)
            subprocess.run(command, shell=True)

            # Filter IASI spectra (read and filter intermediate binary files)

