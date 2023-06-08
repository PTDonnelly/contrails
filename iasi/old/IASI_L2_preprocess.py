import glob
import numpy as np
import pandas as pd

# Set the directory path and input file pattern
datapath_in = "C:\\Users\\padra\\Documents\\Research\\projects\\contrails\\iasi\\anni_L2_bash_idl\\"
datafiles_in = "clp_20220324000252.out"

# Search for matching files
datafile_list = glob.glob(f"{datapath_in}{datafiles_in}")

# # Initialize iasidata list for storing data points
# iasidata = []

# Define latitude and longitude range
latitude_min = -90
latitude_max = 90
longitude_min = -180
longitude_max = 180

# Loop through each file
for datafile in datafile_list:
    
    # Read the file into a DataFrame
    df = pd.read_csv(datafile, sep='\s+', header=None)

    # Assign column names to the DataFrame
    df.columns = ['Latitude', 'Longitude', 'Datetime', 'Orbit Number','Scanline Number', 'Pixel Number',
                  'Cloud Fraction', 'Cloud-top Temperature', 'Cloud-top Pressure', 'Cloud Phase',
                  'Column11', 'Column12', 'Column13', 'Column14', 'Column15', 'Column16', 'Column17', 'Column18']
    
    # Filter data points based on conditions and append to iasidata list
    data = []
    for index, row in df.iterrows():

        if (latitude_min <=  row['Latitude'] <= latitude_max) and (longitude_min <= row['Longitude'] <= longitude_max):
            
            if (row['Cloud Phase'] == 2) and np.isfinite(row['Cloud-top Temperature']):
                
                print([row['Latitude'], row['Longitude'], row['Datetime'], row['Cloud Fraction'], row['Cloud-top Temperature'], row['Cloud-top Pressure'], row['Cloud Phase']])
                data.append([row['Latitude'], row['Longitude'], row['Datetime'], row['Cloud Fraction'], row['Cloud-top Temperature'], row['Cloud-top Pressure'], row['Cloud Phase']])

    # Convert data list to a new DataFrame
    filtered_data_df = pd.DataFrame(data, columns=['Latitude', 'Longitude', 'Orbit', 'Datetime', 'Cloud Fraction', 'Cloud-top Temperature', 'Cloud-top Pressure', 'Cloud Phase'])

    # Save the data to a file
    filtered_data_df.to_hdf(f"{datafile}.h5", key='df', mode='w')
