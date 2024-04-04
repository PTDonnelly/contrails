from datetime import datetime
import cdsapi
import calendar

# Initialize the CDS API client
c = cdsapi.Client()

# Define the start and end dates
start_year, end_year = 2014, 2017
start_month, end_month = 3, 5

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

# Define the geographic regions
regions = {
    # "GMCS": {"lat": [0, 45], "lon": [-60, -100]}, # Gulf of Mexico / Carribean Sea
    # "NAO": {"lat": [30, 60], "lon": [0, -60]}, # North Atlantic Ocean
    "NPO": {"lat": [30, 60], "lon": [-180, -120]} # North Pacific Ocean
    # "SCS": {"lat": [0, 30], "lon": [90, 150]} # South Schina Sea
}

# Define atmospheric variables to extract
variables = ['clwc', 'ciwc', 'cc', 'd', 'w', 'vo']

# Iterate over years and months
for year in range(start_year, end_year + 1):
    for month in range(start_month, end_month + 1 if year < end_year else end_month + 1):
        days_in_month = calendar.monthrange(year, month)[1]
        days = [str(day).zfill(2) for day in range(1, days_in_month + 1)]

        for region, coordinates in regions.items():
            for variable in variables:
                output_file = f"/data/pdonnelly/era5/requested_data/{variable}.{year}{str(month).zfill(2)}_{region}.nc"
                
                # Format region co-ordinates for API (North, West, South, East)
                west, east = min(coordinates["lon"]), max(coordinates["lon"])
                south, north = min(coordinates["lat"]), max(coordinates["lat"])

                data_retrieval = {
                    'product_type': 'reanalysis',
                    'format': 'netcdf',
                    'variable': variable,
                    'pressure_level': ['200', '300'],
                    'year': str(year),
                    'month': str(month).zfill(2),
                    'day': days,
                    'time': ['06:00', '07:00', '08:00', '09:00', '10:00', '11:00', '12:00', '13:00', '14:00', '15:00', '16:00', '17:00', '18:00'],
                    'area': [north, west, south, east],
                    'grid': [1, 1]
                }
                
                # Send the API request
                c.retrieve(
                    'reanalysis-era5-pressure-levels',
                    data_retrieval,
                    output_file
                )
