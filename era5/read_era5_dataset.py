import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr
import os
import pandas as pd
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from sklearn.preprocessing import StandardScaler
import seaborn as sns

def get_region_from_data_file(data_file):
    # Extract the base name of the file (without the full path)
    base_name = os.path.basename(data_file)
    
    # Split the base name into name and extension
    name_without_extension, _ = os.path.splitext(base_name)
    
    # Split by the first underscore assuming the first part is always the date
    parts = name_without_extension.split('_', 1)
    
    # Return the region part. If the format is unexpected, return an empty string or a meaningful default value.
    return parts[1] if len(parts) > 1 else ""

def get_land_mask(mean_ds):
    # Extract spatial grid
    lat = mean_ds.latitude
    lon = mean_ds.longitude

    # Get bounds from dataset attributes
    west = lon.min().item()
    east = lon.max().item()
    south = lat.min().item()
    north = lat.max().item()
    # The shape of your target grid
    height = len(lat)
    width = len(lon)

    # Get land boundaries
    land = cfeature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='black', facecolor='none')
    land_shapely = [geom for geom in land.geometries()]
    land_gdf = gpd.GeoDataFrame(geometry=land_shapely)

    # Define the transform that aligns your grid with the raster's spatial reference
    transform = rasterio.transform.from_bounds(west, south, east, north, width, height)

    # Rasterize the land polygons ()'out_shape' should match the dimensions of the target grid)
    land_mask = rasterize(land_gdf.geometry, out_shape=(height, width), transform=transform, all_touched=True)
    return land_mask

def remove_land_values(ds, land_mask):
    # Loop through each variable in the dataset
    for variable, attribute in ds.items():
        
        # Check if the DataArray has the same spatial dimensions as the land_mask
        if set(['latitude', 'longitude']).issubset(attribute.dims):
            
            # Apply the mask: Set values to NaN where land_mask is True
            ds[variable] = attribute.where(land_mask==0, np.nan)
    return ds

def plot_masked_data_from_df(combined_df, iasi_df, output_directory, variable='t'):
    # Prepare DataFrame: separate regions and pressure levels
    regions = combined_df['region'].unique()
    reference_pressure_level = 250 # mbar

    # Create a figure with subplots in a 3x1 layout
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(10, 5), subplot_kw={'projection': ccrs.PlateCarree()})
    plt.suptitle("ERA5 Temperatures at 250 mbar", fontsize=12, y=0.62)

    # Ensure axs is an array, even with a single subplot
    axs = axs.flatten()

    for ax, region in zip(axs, regions):
        region_df = combined_df[(combined_df['region'] == region) & (combined_df['level'] == reference_pressure_level)]

        # Dynamically determine the unique counts of latitude and longitude
        unique_lats = region_df['latitude'].unique()
        unique_lons = region_df['longitude'].unique()
        lat_count = len(unique_lats)
        lon_count = len(unique_lons)

        # Ensure data is sorted by latitude and longitude
        region_df_sorted = region_df.sort_values(by=['latitude', 'longitude'])

        # Extract lat, lon, and data arrays
        lat = unique_lats
        lon = unique_lons
        data = region_df_sorted[variable].to_numpy()

        # Reshape the data according to the actual dimensions
        try:
            data_reshaped = data.reshape(lat_count, lon_count)
            data_reshaped = np.flipud(data_reshaped) # Sort latitudes by ascending order
        except ValueError as e:
            print(f"Error reshaping data for region {region}: {e}")
            print(f"Expected shape: ({lat_count}, {lon_count}), Got: {data.size}")
            continue  # Skip this iteration

        # Plotting the data
        mesh = ax.pcolormesh(lon, lat, data_reshaped, transform=ccrs.PlateCarree(), cmap='cividis')

        # Add the land feature boundaries
        land = cfeature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='black', facecolor='none')
        ax.add_feature(land)

        # Add locations of IASI measurements
        ax.scatter(iasi_df['Longitude'], iasi_df['Latitude'], s=2, color='red', transform=ccrs.PlateCarree())

        # Add a colorbar
        cbar = fig.colorbar(mesh, ax=ax, orientation='horizontal', pad=0.1, aspect=30)
        cbar.set_label(r"$T_{250 mbar} (K)$")

        # Setting title and labels
        ax.set_title(f'{region}', fontsize=10)
        ax.set_xlabel('Longitude', fontsize=10)
        ax.set_ylabel('Latitude', fontsize=10)
        
        # Assuming you can calculate or have min and max values for lon and lat per region
        ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()], crs=ccrs.PlateCarree())

        # Setting gridlines and labels
        gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, linewidth=0.25, linestyle='--', color='black')
        gl.top_labels = False
        gl.right_labels = False
        gl.bottom_labels = True
        gl.left_labels = True
        gl.xlabel_style = {'size': 10, 'color': 'black'}
        gl.ylabel_style = {'size': 10, 'color': 'black'}

    # Adjust layout
    plt.savefig(os.path.join(output_directory, f"masked_data_{variable}.png"), bbox_inches='tight', dpi=300)
    plt.close()

def plot_statistics_from_df(data, output_directory, n_samples=1000, columns_drop=None, state_seed=42):   
    # Sub-sample the dataset
    rng = np.random.RandomState(state_seed)
    indices = rng.choice(data.shape[0], size=n_samples, replace=False)

    # Drop horizontal and vertical grids, and the liquid cloud parameter
    columns_drop = ['latitude', 'longitude', 'level', 'clwc', ]
    # Drop the unwanted columns if specified
    if columns_drop is not None:
        subset = data.iloc[indices].drop(columns=columns_drop)
    else:
        subset = data.iloc[indices]

    # Initialize the StandardScaler
    scaler = StandardScaler()
    standardised_dataset = scaler.fit_transform(subset)
    standardised_df = pd.DataFrame(standardised_dataset, columns=['t', 'u', 'v', 'q', 'ciwc', 'cc'])

    # Create a pair plot
    pairplot = sns.pairplot(data=standardised_df,
                            plot_kws={"s": 10},
                            kind='scatter',
                            diag_kind='hist',
                            diag_kws = {'bins':10},
                            corner=True)

    plt.suptitle('Pair Plot of Sampled Data', y=1.02)  # Adjust title position

    # Finish and save the plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_directory, f'dataset_pairplot.png'), bbox_inches='tight')
    plt.close()

def main():
    # Open CSV file containing IASI OLR measurements
    iasi_data_path = "G:\\My Drive\\Research\\Postdoc_2_CNRS_LATMOS\\data\\iasi"
    iasi_data_file = "spectra_and_cloud_products.csv"
    iasi_df = pd.read_csv(os.path.join(iasi_data_path, iasi_data_file), sep='\t')

    # Open the NetCDF file and calculate the average across the time dimension
    era5_data_path = "G:\\My Drive\\Research\\Postdoc_2_CNRS_LATMOS\\data\\era5"
    data_files = ["20130301_North Pacific.nc", "20130301_North Atlantic.nc", "20130301_South China Sea.nc"]
    
    # Specify the location to save reduced data and plots
    # output_directory = "C:\\Users\\donnelly\\Documents\\projects\\era5"
    output_directory = "C:\\Users\\padra\\Documents\\Research\\projects\\contrails\\era5"

    # Initialise a list to store the mean datasets for each region
    era5_dfs = []

    for data_file in data_files:
        # Open NetCDF data and compute daily average
        era5_ds = xr.open_dataset(os.path.join(era5_data_path, data_file))
        mean_era5_ds = era5_ds.mean(dim='time')

        # Extract geographic region from the file name
        region = get_region_from_data_file(data_file)

        # Create land mask, interpolate onto dataset grid
        land_mask = get_land_mask(mean_era5_ds)
        
        # Remove overland values from dataset
        masked_mean_era5_ds = remove_land_values(mean_era5_ds, land_mask)

        # Store the masked mean dataset in to a DataFrame
        era5_df = masked_mean_era5_ds.to_dataframe().reset_index()

        # Add a 'region' column to the DataFrame
        era5_df['region'] = region

        # Append the DataFrame to the list
        era5_dfs.append(era5_df)

    # # Combine all regional DataFrames into a single DataFrame and save to a CSV file
    combined_era5_df = pd.concat(era5_dfs, ignore_index=True)
    combined_era5_df.to_csv(os.path.join(output_directory, "combined_data.csv"), sep='\t', index=False)

    # Plot data
    # plot_statistics_from_df(combined_era5_df, output_directory)
    plot_masked_data_from_df(combined_era5_df, iasi_df, output_directory)



if __name__ == "__main__":
    main()