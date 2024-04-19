import os
import datetime

# Define the directory to scan
directory = "/data/pdonnelly/era5/processed_files"

# Define the range of dates
start_year = 2018
end_year = 2023
months_of_interest = [3, 4, 5]  # March, April, May

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

# Function to generate a date range for specific months each year
def generate_dates(start_year, end_year, months):
    for year in range(start_year, end_year + 1):
        for month in months:
            start_date = datetime.date(year, month, 1)
            end_date = datetime.date(year, month, 28) + datetime.timedelta(days=4)  # Get to the last day of the month safely
            end_date = end_date - datetime.timedelta(days=end_date.day)
            while start_date <= end_date:
                yield start_date
                start_date += datetime.timedelta(days=1)

# Check for missing files
missing_files = []
for variable, info in variables_dict.items():
    for level in info["target_level"]:
        for date in generate_dates(start_year, end_year, months_of_interest):
            formatted_date = date.strftime("%Y-%m-%d")
            filename = f"{info['short_name']}_{level}_daily_1x1_{formatted_date}.csv"
            file_path = os.path.join(directory, filename)
            if not os.path.exists(file_path):
                missing_files.append(filename)

# Print missing files
for missing_file in missing_files:
    print(missing_file)
print(len(missing_files))