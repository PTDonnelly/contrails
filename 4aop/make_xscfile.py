import numpy as np
import pandas as pd
import os

def get_iwabuchi(optical_dir: str, filename: str):
    data = np.loadtxt(f"{optical_dir}{filename}.txt", comments='#', delimiter=None, unpack=True)
    return data[0, :], data[1:13, :], data[13:, :]


def create_aer_file(xsc_file_path, aer_template_path, file_path):
    
    # Function to get column formats
    def get_column_formats():
        return {
            0: "{:0>7.2f}".format,  # Leading zeros, two decimal places
            1: "{:0>7.2f}".format,  # Leading zeros, two decimal places
            2: "{:.2E}".format,     # Scientific notation with 2 decimal places
            3: "{}".format,         # String, as is
            4: "{:0>4.2f}".format,  # Leading zeros, two decimal places
            5: "{:0>4.2f}".format,  # Leading zeros, two decimal places
            6: "{:0>4.2f}".format,  # Leading zeros, two decimal places
            7: "{}".format          # String, as is
        }

    # Function to format dataframe
    def format_dataframe(df, column_formats):
        formatted_df = df.copy()
        for col, fmt in column_formats.items():
            formatted_df[col] = df[col].apply(fmt)
        return formatted_df

    # Read the template .aer file into a DataFrame
    df = pd.read_csv(aer_template_path, delim_whitespace=True, header=None, skiprows=1)

    # Extract header from the template .aer file
    with open(aer_template_path, 'r') as file:
        header = file.readline().strip()

    # Get the scattering file name from the xsc_file_path
    scatterer = os.path.basename(xsc_file_path).split('_')[-1].split('.')[0]

    # Modify the scattering scheme column
    df.iloc[:, 3] = scatterer

    # Apply column formats
    column_formats = get_column_formats()
    formatted_df = format_dataframe(df, column_formats)

    # Write to new aer file
    new_aerfile = f"{file_path}aer4atest_{scatterer}.dsf"
    with open(new_aerfile, 'w') as f:
        f.write(header + '\n')
    formatted_df.to_csv(new_aerfile, mode='a', sep=' ', index=False, header=False)


optical_dir = "C:\\Users\\padra\\Documents\\Research\\projects\\contrails\\scattering_calculations\\optical_data\\"
filename = 'iwabuchi_optical_properties'
wavelengths, real, imaginary = get_iwabuchi(optical_dir, filename)


# Example usage:
file_path = 'C:\\Users\\padra\\Documents\\Research\\projects\\contrails\\4aop\\datatm\\'
aer_template_path = os.path.join(file_path, 'aer4atest_baum.dsf')

# When a new xsc file is created:
xsc_file_path = 'path_to_new_xsc_file.dat'
create_aer_file(xsc_file_path, aer_template_path, file_path)