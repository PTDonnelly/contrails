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

def plot_masked_data(mean_datasets_by_region, land, output_directory, variable='t'):
    # # First, determine the global min and max across all datasets for consistent color scaling
    # global_min = min(mean_ds[variable].min().item() for mean_ds in mean_datasets_by_region.values())
    # global_max = max(mean_ds[variable].max().item() for mean_ds in mean_datasets_by_region.values())

    # Create a figure with subplots in a 3x1 layout
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(10, 5), subplot_kw={'projection': ccrs.PlateCarree()})
    plt.suptitle("ERA5 Temperatures at 250 mbar", fontsize=12, y=0.62)

    # Ensure axs is an array, even with a single subplot
    axs = axs.flatten()

    # Iterate over each region and its dataset
    for ax, (region, mean_ds) in zip(axs, mean_datasets_by_region.items()):
        # Extract spatial grid and data
        lat = mean_ds.latitude
        lon = mean_ds.longitude
        data = mean_ds[variable].sel(level=250)

        # Plotting the data
        mesh = ax.pcolormesh(lon, lat, data, transform=ccrs.PlateCarree(), cmap='cividis')

        # Add the land feature boundaries
        land = cfeature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='black', facecolor='none')
        ax.add_feature(land)

        # Add a colorbar
        cbar = fig.colorbar(mesh, ax=ax, orientation='horizontal', pad=0.1, aspect=30)
        cbar.set_label(r"$T_{250 mbar} (K)$")

        # Setting title and labels
        ax.set_title(f'{region}', fontsize=10)
        ax.set_xlabel('Longitude', fontsize=10)
        ax.set_ylabel('Latitude', fontsize=10)
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
    plt.savefig(os.path.join(output_directory, 'masked_data_subplot.png'), bbox_inches='tight', dpi=300)
    plt.close()

def plot_statistics(data, output_directory, n_samples=1000, columns_drop=None, state_seed=42):   
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
    # Specify the location to save reduced data and plots
    output_directory = "C:\\Users\\donnelly\\Documents\\projects\\era5"

    # Open the NetCDF file and calculate the average across the time dimension
    data_path = "C:\\Users\\donnelly\\Documents\\projects\\data"
    data_files = ["20130301_North Pacific.nc", "20130301_North Atlantic.nc", "20130301_South China Sea.nc"]
    
    # Initialize a dictionary to store the mean datasets for each region
    mean_datasets_by_region = {}
    dfs = []

    for data_file in data_files:
        ds = xr.open_dataset(os.path.join(data_path, data_file))
        mean_ds = ds.mean(dim='time')

        # Extract geographic region from the file name
        region = get_region_from_data_file(data_file)

        # Create land mask, interpolate onto dataset grid
        land_mask = get_land_mask(mean_ds)
        
        # Remove overland values from dataset
        masked_mean_ds = remove_land_values(mean_ds, land_mask)

        # Store the masked mean dataset in the dictionary using the region as the key
        mean_datasets_by_region[region] = masked_mean_ds

        # Flatten the dataset to a DataFrame
        df = masked_mean_ds.to_dataframe().reset_index()

        # Append the DataFrame to the list
        dfs.append(df)

    # Combine all regional DataFrames into a single DataFrame and save to a CSV file
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df.to_csv(os.path.join(output_directory, "combined_data.csv"), sep='\t', index=False)
    plot_statistics(combined_df, output_directory)

    # # Plot data
    # plot_masked_data(mean_datasets_by_region, output_directory)



if __name__ == "__main__":
    main()