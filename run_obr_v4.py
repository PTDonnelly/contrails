import subprocess
from sorcery import dict_of

def build_command(inputs: dict, year: int, month: int, day: int):
    run_dir = f"/home/pdonnelly/ancillary_data/obr_v4"
    filepath = f"-d {inputs['path_in']}"
    first_date = f"-fd {year:04d}-{month:02d}-{day:02d}"
    last_date = f"-ld {year:04d}-{month:02d}-{day:02d}"
    channels = f"c {inputs['iasi_channels'][0]}-{inputs['iasi_channels'][-1]}"
    mf = f"-mf {inputs['file_in']}"
    suffix = f"of bin -out {inputs['file_out']}"
    return f"{run_dir} {filepath} {first_date} {last_date} {channels} {mf} {suffix}"

def set_inputs():
    """Define inputs for OBR tool"""
    path_in = '/path/to/data'
    years = [2020]
    months = [1] #, 2, 3]
    days = [1] #[day for day in range(28)] 
    iasi_channels = [1, 2, 3]
    file_in = 'file_in.bin'#'W_XX-EUMETSAT-Darmstadt,SOUNDING+SATELLITE,METOPA+IASI_C_EUMC_20200101000253_68496_eps_o_l1.bin'
    file_out = 'file_out.bin'
    return dict_of(path_in, years, months, days, iasi_channels, file_in, file_out)

# Create dictionary containing OBR inputs
inputs = set_inputs()

# Loop through date-times
for year in inputs['years']:
    for month in inputs['months']:
        for day in inputs['days']:

            # Construct command-line executable
            command = build_command(inputs, year, month, day)

            print(command)
            # # Execute OBR command
            # subprocess.run(command, shell=True)