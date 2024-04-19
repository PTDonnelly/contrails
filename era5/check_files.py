import os
import datetime

# Define the directory to scan
directory = "/data/pdonnelly/era5/processed_files"

# Define the range of dates
start_year = 2018
end_year = 2023
start_month = 3
end_month = 5

# Define the variables and their target levels
variables_dict = {
    "cloud cover": {"short_name": "cc", "target_level": [200, 500, 750, 950]},
    "temperature": {"short_name": "ta", "target_level": [200, 500, 750, 950]},
    "specific humidity": {"short_name": "q", "target_level": [200, 500, 750, 950]},
    "relative humidity": {"short_name": "r", "target_level": [200, 500, 750, 950]},
    "geopotential": {"short_name": "geopt", "target_level": [200, 500, 750, 950]},
    "eastward wind": {"short_name": "u", "target_level": [200, 500, 750, 950]},
    "northward wind": {"short_name": "v", "target_level": [200, 500, 750, 950]},
    "ozone mass mixing ratio": {"short_name": "o3", "target_level": [200, 500, 750, 950]},
}

# Generate date range
def generate_date_range(start_year, end_year, start_month, end_month):
    start_date = datetime.date(start_year, start_month, 1)
    end_date = datetime.date(end_year, end_month, 1)
    while start_date <= end_date:
        day = datetime.date(start_date.year, start_date.month, 1)
        while day.month == start_date.month:
            yield day
            day += datetime.timedelta(days=1)
        start_date += datetime.timedelta(days=32)
        start_date = start_date.replace(day=1)

# Check for missing files
missing_files = []
for variable, info in variables_dict.items():
    for level in info["target_level"]:
        for date in generate_date_range(start_year, end_year, start_month, end_month):
            formatted_date = date.strftime("%Y-%m-%d")
            filename = f"{info['short_name']}_{level}_daily_1x1_{formatted_date}.csv"
            file_path = os.path.join(directory, filename)
            if not os.path.exists(file_path):
                missing_files.append(filename)

# Print missing files
for missing_file in missing_files:
    print(missing_file)

print(len(missing_files))
