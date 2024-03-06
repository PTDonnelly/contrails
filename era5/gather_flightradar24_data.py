import os
import pandas as pd
import numpy as np
import logging

# Configure logging at the module level
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to sort lon-lat pairs within a row
def sort_lon_lat_pairs(row):
    # Extracting lon-lat pairs, ignoring NaN values
    pairs = [(row[f'Lon{i}'], row[f'Lat{i}']) for i in range(1, (len(row) - 1) // 2 + 1) if pd.notna(row[f'Lon{i}'])]
    # Sorting the pairs based on longitude first, then latitude
    sorted_pairs = sorted(pairs, key=lambda x: (x[0], x[1]))
    # Flattening the sorted pairs back into a list
    sorted_values = [val for pair in sorted_pairs for val in pair]
    # Filling the row with sorted values and NaN for any missing values
    new_row = [row['Flight Number']] + sorted_values + [pd.NA] * (len(row) - 1 - len(sorted_values))
    return new_row

def interpolate_to_grid(lon, lat, grid_size=0.1):
    """
    Interpolate a series of longitude and latitude points onto a 0.1 x 0.1 degree grid.
    Returns a list of tuples (lon, lat) representing the interpolated points.
    """
    interpolated_points = []
    
    # Ensure there are at least two points to interpolate between
    if len(lon) < 2 or len(lat) < 2:
        return list(zip(lon, lat))
    
    for i in range(len(lon) - 1):
        current_point = (lon[i], lat[i])
        next_point = (lon[i + 1], lat[i + 1])
        
        interpolated_points.append(current_point)
        
        # Calculate the number of grid points between the current and next point
        num_interpolated_points_lon = int(abs(next_point[0] - current_point[0]) / grid_size)
        num_interpolated_points_lat = int(abs(next_point[1] - current_point[1]) / grid_size)
        num_interpolated_points = max(num_interpolated_points_lon, num_interpolated_points_lat)
        
        if num_interpolated_points > 0:
            lon_step = (next_point[0] - current_point[0]) / (num_interpolated_points + 1)
            lat_step = (next_point[1] - current_point[1]) / (num_interpolated_points + 1)
            
            for step in range(1, num_interpolated_points + 1):
                interpolated_lon = current_point[0] + lon_step * step
                interpolated_lat = current_point[1] + lat_step * step
                interpolated_points.append((interpolated_lon, interpolated_lat))
    
    interpolated_points.append(next_point)  # Add the last point
    
    return interpolated_points

def apply_interpolation(row):
    lon_lat_pairs = [(row[f'Lon{i}'], row[f'Lat{i}']) for i in range(1, (len(row) + 1) // 2) if not pd.isna(row[f'Lon{i}'])]
    lon, lat = zip(*lon_lat_pairs) if lon_lat_pairs else ([], [])
    interpolated = interpolate_to_grid(list(lon), list(lat))
    
    # Flatten the list of tuples and pad with NaNs as necessary
    interpolated_flat = [val for pair in interpolated for val in pair]
    return pd.Series(interpolated_flat + [np.nan] * (len(row) - 1 - len(interpolated_flat)))

# Define the directory containing your CSV files
directory_path = file_path = 'G:\\My Drive\\Research\\Postdoc_2_CNRS_LATMOS\\data\\FlightRadar24'

# Define the column names based on the inferred structure
column_names = [
    'Hex Code', 'Altitude', 'Flight Number', 'Date', 'Departure Airport ICAO Code',
    'Departure Airport IATA Code', 'Aircraft Type', 'Flight Event/Status',
    'Flight IATA Number/Additional Identifier', 'Unknown/Additional Identifier',
    'Latitude', 'Longitude', 'Airline Code', 'Arrival Airport ICAO Code',
    'Arrival Airport IATA Code', 'Aircraft Registration', 'Time'
]

aggregated_dfs = []

# Initialize an empty DataFrame to store all flight paths
all_flight_paths_df = pd.DataFrame(columns=['Flight Number', 'Latitudes', 'Longitudes'])

# Iterate over each CSV file in the directory
for i, filename in enumerate(os.listdir(directory_path)):
    if filename.endswith(".csv"):
        file_path = os.path.join(directory_path, filename)
        
        # Load the CSV file into a DataFrame
        logging.info(f"Reading: {file_path}")
        df = pd.read_csv(file_path, names=column_names, dtype={'Flight Number': str})

        # Remove rows where 'Flight Number' is NaN or an empty string
        df = df[df['Flight Number'].notna() & (df['Flight Number'] != '')]
        df['Flight Number'] = df['Flight Number'].astype(str)

        df = df.loc[0]

        print(df.head())


        # Group the DataFrame by 'Flight Number' and aggregate the Latitude and Longitude into lists
        logging.info("Scanning flights")
        # def custom_agg(x):
        #     logging.debug(f"Aggregating: {x.name}")  # x.name should be 'Latitude' or 'Longitude'
        #     return list(x)

        # aggregated = df.groupby('Flight Number', as_index=False).agg({
        #     'Latitude': custom_agg,
        #     'Longitude': custom_agg
        # })

        
        aggregated = df.groupby(['Flight Number']).agg({
            'Latitude': lambda x: list(x),
            'Longitude': lambda x: list(x)
        }).reset_index()

        print(aggregated.head())
        print(aggregated.dtypes)

        exit()
        # Rename columns for clarity and consistency
        aggregated.columns = ['Flight Number', 'Latitudes', 'Longitudes']

        # Append the aggregated data for this file to the master DataFrame
        aggregated_dfs.append(aggregated)
    
    if i == 0:
        break

all_flight_paths_df = pd.concat(aggregated_dfs, ignore_index=True)

print(all_flight_paths_df[['Flight Number']].head())

# print(all_flight_paths_df.head())

exit()
# In case of  duplicate entries for the same 'Flight Number', further aggregate by concatenating lists for the same flight number
logging.info("Final aggregation check")
aggregated_flight_paths_df = all_flight_paths_df.groupby('Flight Number').agg({
    'Latitudes': "sum",  # Use "sum" instead of sum
    'Longitudes': "sum"  # Use "sum" instead of sum
}).reset_index()

print(aggregated_flight_paths_df.head())

exit()
# Expand the Latitudes and Longitudes lists into separate rows for each coordinate pair
logging.info("Reformatting latitudes and longitude")
lat_long_expanded = aggregated_flight_paths_df.apply(
    lambda x: pd.Series(zip(x['Latitudes'], x['Longitudes'])), axis=1
).stack().reset_index(level=1, drop=True).apply(pd.Series)

lat_long_expanded.columns = ['Latitude', 'Longitude']

# Add the flight number back to the expanded DataFrame
lat_long_expanded['Flight Number'] = lat_long_expanded.index

print(lat_long_expanded.head())
exit()

# Convert the flight paths to a format suitable for a DataFrame
data_for_df = []

logging.info("Converting dictionary to DataFrame")
for flight_number, coordinates in flight_paths.items():
    row = [flight_number]
    for lat, lon in coordinates:
        row.extend([lon, lat])  # Append longitude first, then latitude
    data_for_df.append(row)

# Find the maximum number of columns for any flight
max_columns = max(len(row) for row in data_for_df)

# Adjust the column names generation to reflect 'lon' and 'lat' separately
logging.info("Renaming columns")
num_pairs = (max_columns - 1) // 2
column_names = ["Flight Number"] + [f"{coord}{i}" for i in range(1, num_pairs + 1) for coord in ("Lon", "Lat")]

# Create the DataFrame with the corrected column names
df_flight_paths = pd.DataFrame(data_for_df, columns=column_names)

# Applying the sorting function to each row
logging.info("Sorting latitude-longitude-pairs")
sorted_rows = df_flight_paths.apply(sort_lon_lat_pairs, axis=1, result_type='expand')

# Assigning the original column names to the sorted DataFrame
sorted_df_flight_paths = pd.DataFrame(sorted_rows.to_list(), columns=df_flight_paths.columns)

# Apply interpolation to each row and update the DataFrame with interpolated values
logging.info("Interpolating onto regular 0.1x0.1 grid")
interpolated_rows = sorted_df_flight_paths.apply(apply_interpolation, axis=1)

# Update column names for the interpolated DataFrame
num_cols = max(len(row.dropna()) for _, row in interpolated_rows.iterrows())
column_names = ["Flight Number"] + [f"Lon{i}" if i % 2 != 0 else f"Lat{i//2}" for i in range(1, num_cols)]
interpolated_df = pd.DataFrame(interpolated_rows.to_list(), columns=column_names)

# Fixing the DataFrame's structure if necessary
interpolated_df = interpolated_df.reindex(columns=sorted_df_flight_paths.columns, fill_value=np.nan)

# Save the corrected DataFrame to a CSV file
csv_output_path_corrected = os.path.join(directory_path, 'flight_paths_corrected.csv')
logging.info(f"Saving: {csv_output_path_corrected}")
interpolated_df.to_csv(csv_output_path_corrected, index=False)