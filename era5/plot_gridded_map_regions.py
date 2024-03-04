import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import os

def plot_earth_with_windowed_regions(regions, output_directory):
    """
    Plot the Earth map and highlight a windowed region specified by
    latitude and longitude coordinates.
    
    Parameters:
    - regions (dict): A collection of geographic regions and their bounding co-ordinates
    """
    # Create a figure with PlateCarree projection
    fig = plt.figure(figsize=(8, 2), dpi=300)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    
    # Add the map features
    ax.coastlines()
    # Set the extent to the Northern Hemisphere
    ax.set_extent([-180, 180, 0, 90], crs=ccrs.PlateCarree())

    for region, coordinates in regions.items():
        lon_min, lon_max = min(coordinates["lon"]), max(coordinates["lon"])
        lat_min, lat_max = min(coordinates["lat"]), max(coordinates["lat"])

        # Add a rectangle to highlight the windowed region
        ax.add_patch(plt.Rectangle((lon_min, lat_min), lon_max - lon_min, lat_max - lat_min,
                                edgecolor='red', facecolor='none', linewidth=1, transform=ccrs.PlateCarree()))
        
        # Draw custom gridlines within the windowed region
        # Generate the gridline coordinates
        lons = np.arange(lon_min, lon_max + 1, 1)
        lats = np.arange(lat_min, lat_max + 1, 1)
        
        # Draw longitude lines
        for lon in lons:
            ax.plot([lon, lon], [lat_min, lat_max], color='black', linewidth=0.25, transform=ccrs.PlateCarree())
        
        # Draw latitude lines
        for lat in lats:
            ax.plot([lon_min, lon_max], [lat, lat], color='black', linewidth=0.25, transform=ccrs.PlateCarree())
    
    # Add a title
    plt.title("Dense Flight Corridors: Common Grid for IASI and ERA5", fontsize=10)
    # Set x-axis and y-axis labels
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    # Define the range of ticks to be shown
    ax.set_xticks(np.arange(-180, 181, 60), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(0, 91, 30), crs=ccrs.PlateCarree())
    # Optional: Customize tick labels, for example, to increase font size or change style
    ax.xaxis.set_tick_params(labelsize=10)
    ax.yaxis.set_tick_params(labelsize=10)

    plt.savefig(os.path.join(output_directory, f'geographic_regions.png'), bbox_inches='tight', dpi=300)
    plt.close()

output_directory = "C:\\Users\\donnelly\\Documents\\projects\\era5"


# Define the geographic regions (South, North, West, East)
regions = {
    "North Atlantic": {"lat": [30, 60], "lon": [0, -60]},  # Note the conversion of 15° W to -15 for consistency with the -180 to 180 degree convention
    "South China Sea": {"lat": [0, 30], "lon": [90, 150]},
    "North Pacific": {"lat": [30, 60], "lon": [-180, -120]}  # Note the conversion of longitudes to negative for W
}

plot_earth_with_windowed_regions(regions, output_directory)
