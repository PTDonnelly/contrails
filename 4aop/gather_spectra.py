import os
import numpy as np
import pandas as pd

def read_plt_file(filename):
    """Load data from filename, handling possible errors."""
    try:
        return np.loadtxt(filename)
    except (IOError, ValueError) as e:
        print(f"Error reading {filename}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    return None

data_dir = "C:\\Users\\padra\\Documents\\Research\\projects\\contrails\\4aop\\outputs\\"
input_files = [
    "spc4asub0001_inv_co2_CAMS_test_iasil1c_clear.plt",
    # "spc4asub0001_inv_co2_CAMS_test_iasil1c_baum.plt",
    # "spc4asub0001_inv_co2_CAMS_test_iasil1c_cir100.plt",
    # "spc4asub0001_inv_co2_CAMS_test_iasil1c_cir200.plt",
    # "spc4asub0001_inv_co2_CAMS_test_iasil1c_cir300.plt",
    # "spc4asub0001_inv_co2_CAMS_test_iasil1c_con1601.plt",
    # "spc4asub0001_inv_co2_CAMS_test_iasil1c_con1602.plt",
    # "spc4asub0001_inv_co2_CAMS_test_iasil1c_con1605.plt",
    # "spc4asub0001_inv_co2_CAMS_test_iasil1c_con1610.plt",
    # "spc4asub0001_inv_co2_CAMS_test_iasil1c_con2301.plt",
    # "spc4asub0001_inv_co2_CAMS_test_iasil1c_con2302.plt",
    # "spc4asub0001_inv_co2_CAMS_test_iasil1c_con2305.plt",
    # "spc4asub0001_inv_co2_CAMS_test_iasil1c_con2310.plt"
    "spc4asub0001_inv_co2_CAMS_test_iasil1c_con2305f1.plt",
    # "spc4asub0001_inv_co2_CAMS_test_iasil1c_con2305f2.plt",
    # "spc4asub0001_inv_co2_CAMS_test_iasil1c_con2305f3.plt",
    # "spc4asub0001_inv_co2_CAMS_test_iasil1c_con2305f4.plt",
    # "spc4asub0001_inv_co2_CAMS_test_iasil1c_con2305f5.plt",
    # "spc4asub0001_inv_co2_CAMS_test_iasil1c_con2305f6.plt",
    # "spc4asub0001_inv_co2_CAMS_test_iasil1c_con2305f7.plt",
    # "spc4asub0001_inv_co2_CAMS_test_iasil1c_con2305f8.plt",
    "spc4asub0001_inv_co2_CAMS_test_iasil1c_con2305f9.plt"
  ]

def main():
    """Main execution function."""
    files = [os.path.join(data_dir, file) for file in input_files]
    
    # Initialize empty DataFrames for radiance and brightness temperature
    df_radiance = pd.DataFrame()
    df_temperature = pd.DataFrame()

    for file in files:
        data = read_plt_file(file)
        
        if data is not None:
            wavenumbers = data[:, 0]
            radiance = data[:, 1]
            brightness_temperature = data[:, 2]

            # Extract the last 5 characters of the filename (excluding the extension)
            header = [part.split('.') for part in file.split('_')].pop()[0]

            # Populate DataFrames
            df_radiance[header] = radiance
            df_temperature[header] = brightness_temperature

    # Set the first column as wavenumbers in both DataFrames
    df_radiance.insert(0, 'Wavenumbers', wavenumbers)
    df_temperature.insert(0, 'Wavenumbers', wavenumbers)

    # Save DataFrames to CSV
    df_radiance.to_csv(f"{data_dir}spectra_radiance.csv", index=False, sep='\t')
    df_temperature.to_csv(f"{data_dir}spectra_brightness_temperature.csv", index=False, sep='\t')
    print("Data saved to CSV files.")

if __name__ == "__main__":
    main()
