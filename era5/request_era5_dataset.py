import cdsapi
from datetime import datetime, timedelta

def daterange(start_date, end_date):
    """Generate a range of dates from start_date to end_date."""
    for n in range(int((end_date - start_date).days) + 1):
        yield start_date + timedelta(n)

variables = [
    {"name": "Specific rain water content", "short_name": "crwc", "paramID": 75},
    {"name": "Specific snow water content", "short_name": "cswc", "paramID": 76},
    {"name": "Eta-coordinate vertical velocity", "short_name": "etadot", "paramID": 77},
    {"name": "Geopotential", "short_name": "z", "paramID": 129},
    {"name": "Temperature", "short_name": "t", "paramID": 130},
    {"name": "U component of wind", "short_name": "u", "paramID": 131},
    {"name": "V component of wind", "short_name": "v", "paramID": 132},
    {"name": "Specific humidity", "short_name": "q", "paramID": 133},
    {"name": "Vertical velocity", "short_name": "w", "paramID": 135},
    {"name": "Vorticity (relative)", "short_name": "vo", "paramID": 138},
    {"name": "Logarithm of surface pressure", "short_name": "lnsp", "paramID": 152},
    {"name": "Divergence", "short_name": "d", "paramID": 155},
    {"name": "Ozone mass mixing ratio", "short_name": "o3", "paramID": 203},
    {"name": "Specific cloud liquid water content", "short_name": "clwc", "paramID": 246},
    {"name": "Specific cloud ice water content", "short_name": "ciwc", "paramID": 247},
    {"name": "Fraction of cloud cover", "short_name": "cc", "paramID": 248},
]


# Initialize the CDS API client
c = cdsapi.Client()

# Define the start and end dates
start_date = datetime(2013, 3, 1)
end_date = datetime(2013, 5, 31)

# Define the geographic locations (North, West, South, East)
locations = [
    [60, -60, 30, 0],  # Atlantic
    [40, -125, 30, -115],  # Specific region in North America
    [-10, 110, -40, 150],  # Specific region in the Southern Hemisphere
]

# Iterate over each date and location
for single_date in daterange(start_date, end_date):
    for area in locations:
        output_file = f'C:\\Users\\donnelly\\Documents\\projects\\data\\era5\\{single_date.strftime("%Y%m%d")}_{area[0]}_{area[1]}_{area[2]}_{area[3]}.nc'
        
        data_retrieval = {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'variable': ['t', 'u', 'v', 'q', 'clwc', 'ciwc', 'cc'],
            'pressure_level': ['200', '250', '300'],
            'year': single_date.strftime("%Y"),
            'month': single_date.strftime("%m"),
            'day': single_date.strftime("%d"),
            'time': '12:00',
            'area': area,
            'grid': [1, 1],
        }
        
        # Send the API request
        c.retrieve(
            'reanalysis-era5-pressure-levels',
            data_retrieval,
            output_file
        )
