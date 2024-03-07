# Let's first read the uploaded file to understand its structure and content better.
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

# Load the dataset
file_path = 'G:\\My Drive\\Research\\Postdoc_2_CNRS_LATMOS\\data\\FlightRadar24\\2019-01-01.csv'

# Define the column names based on the inferred structure
column_names = [
    'Hex Code', 'Altitude', 'Flight Number', 'Date', 'Departure Airport ICAO Code',
    'Departure Airport IATA Code', 'Aircraft Type', 'Flight Event/Status',
    'Flight IATA Number/Additional Identifier', 'Unknown/Additional Identifier',
    'Latitude', 'Longitude', 'Airline Code', 'Arrival Airport ICAO Code',
    'Arrival Airport IATA Code', 'Aircraft Registration', 'Time'
]

# Reload the dataset with the correct headers
df = pd.read_csv(file_path, names=column_names, skiprows=1)


# Identify Departure Points
# Assuming the first record of each flight on a given day as its departure point
departure_points = df.drop_duplicates(subset=['Flight Number', 'Date'], keep='first')

# Identify Arrival Points
# Filtering for 'gate_arrival' events to find arrivals
arrival_points = df[df['Flight Event/Status'] == 'gate_arrival']

# Merge Departure and Arrival Points based on Flight Number and Date to correlate them
correlated_flights = pd.merge(
    departure_points,
    arrival_points,
    on=['Flight Number', 'Date'],
    suffixes=('_dep', '_arr'),
    how='inner'
)

# Selecting relevant columns for clarity
correlated_flights_df = correlated_flights[['Flight Number', 'Date', 'Latitude_dep', 'Longitude_dep', 'Latitude_arr', 'Longitude_arr']]

# Display the first few rows to verify correlation
print(correlated_flights_df.head())

# Assuming 'df' is your DataFrame and it contains 'Latitude' and 'Longitude' columns for plotting
latitude = correlated_flights_df['Latitude_dep']
longitude = correlated_flights_df['Longitude_dep']

# Create a plot with the PlateCarree projection
fig = plt.figure(figsize=(10, 5))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines()  # Add coastlines to the map

# Add the scatter plot on the map
# You might need to adjust 'latitude' and 'longitude' if your DataFrame structure is different
ax.scatter(longitude, latitude, color='blue', marker='o', s=0.1, alpha=0.1, transform=ccrs.Geodetic())

# Add gridlines and labels for better readability
ax.gridlines(draw_labels=True)

plt.title('Flight Locations')
plt.show()