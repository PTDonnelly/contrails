import gzip
import shutil
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature


# Define the directory containing your CSV files
directory_path = 'G:\\My Drive\\Research\\Postdoc_2_CNRS_LATMOS\\data\\Eurocontrol'

def unzip_files(directory_path, delete_gzip_files=True):
    # Iterate over the gzipped files
    for file_name in os.listdir(directory_path):
        if file_name.endswith(".gz"):
            
            # Define the full file path without the ".gz" extension
            file_path = os.path.join(directory_path, file_name)
            file_root, _ = os.path.splitext(file_path)
            
            # Open gzipped file in read mode
            with gzip.open(f"{file_root}.gz", 'rb') as f_in:
                # Open new file in write mode
                with open(f"{file_root}", 'wb') as f_out:
                    # Copy contents and save
                    shutil.copyfileobj(f_in, f_out)

            if delete_gzip_files:
                # Delete the gzipped file
               os.remove(f"{file_root}.gz")


def main():

    # Check for compressed files and unzip
    unzip_files(directory_path)

    # Inspect the CSVs
    for file_name in os.listdir(directory_path):
        if ("Flights" in file_name) and (file_name.endswith(".csv")):
            print(file_name)
            # Define the full file path without the ".gz" extension
            file_path = os.path.join(directory_path, file_name)

            df = pd.read_csv(file_path)

            # Sub-sample the dataset
            rng = np.random.RandomState(42)
            indices = rng.choice(df.shape[0], size=1000, replace=False)
            subset_df = df.iloc[indices]


            # Extract departure and arrival points
            departure_points = subset_df[['ADEP Latitude', 'ADEP Longitude']]
            arrival_points = subset_df[['ADES Latitude', 'ADES Longitude']]

            # Initialize the plot
            fig = plt.figure(figsize=(20, 10))
            ax = plt.axes(projection=ccrs.PlateCarree())
            ax.add_feature(cfeature.COASTLINE)
            ax.add_feature(cfeature.BORDERS, linestyle=':')

            # Plot each flight's path
            for departure, arrival in zip(departure_points.itertuples(index=False), arrival_points.itertuples(index=False)):
                ax.plot([departure[1], arrival[1]], [departure[0], arrival[0]],
                        color='orange', linewidth=0.5, alpha=0.5,
                        transform=ccrs.Geodetic())

            # Add gridlines and set extent
            ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
            ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())

            plt.show()
            exit()





if __name__ == "__main__":
    main()