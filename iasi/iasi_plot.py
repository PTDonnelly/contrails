from datetime import datetime as dt
from typing import Tuple, List, Union

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cartopy.crs as ccrs
from cartopy.feature import NaturalEarthFeature

from iasi_process import L1CReader

class PlotParameters:
    def __init__(self, data: object, parameters: dict):
        self.data = data
        self.parameters = parameters
        self.title = self.set_title()
        self.xlabel = self.set_xlabel()
        self.ylabel = self.set_ylabel()
        self.xlim = self.set_xlim()
        self.ylim = self.set_ylim()
        self.xscale = self.set_xscale()
        self.yscale = self.set_yscale()

    def set_title(self, title='Plot'):
        return self.parameters['title'] if 'title' in self.parameters else title

    def set_xlabel(self, xlabel='x'):
        return self.parameters['xlabel'] if 'xlabel' in self.parameters else xlabel

    def set_ylabel(self, ylabel='y'):
        return self.parameters['ylabel'] if 'ylabel' in self.parameters else ylabel

    def set_xlim(self):
        return np.min(self.data), np.max(self.data)

    def set_ylim(self):
        return np.min(self.data), np.max(self.data)

    def set_xscale(self, xscale='linear'):
        return self.parameters['xscale'] if 'xscale' in self.parameters else xscale

    def set_yscale(self, yscale='linear'):
        return self.parameters['yscale'] if 'yscale' in self.parameters else yscale


class L1CPlotter:
    def __init__(self, filepath, filename, parameters):
        self.data = L1CReader(filepath, filename)
        self.params = PlotParameters(self.data, parameters)


    def plot_spectrum(self) -> None:
        
        # Load data
        self.data.get_data()
        for i, (spectrum, datetime) in enumerate(zip(self.data.spectra, self.data.datetimes)):
            
            x = range(len(spectrum))
            y = spectrum

            _, ax = plt.subplots(dpi=300)
            ax.plot(x, y, lw=0.5, color='k')
            ax.set_title(self.params.title)
            ax.set_xlabel(self.params.xlabel)
            ax.set_ylabel(self.params.ylabel)
            ax.set_yscale(self.params.yscale)

            # Build output filename and save figure
            date, time = self.data.build_datetime(datetime)
            figname = f"IASI_L1C_{date}_{time}_{i}.png".replace(":", ".")
            plt.savefig(f"{self.data.filepath}{figname}", dpi=150, bbox_inches='tight')
        return


    def plot_observation_points(self) -> None:
        
        # Load data
        self.data.get_data()

        # Create a figure and axes
        _, ax = plt.subplots(figsize=(10, 6), subplot_kw={'projection': ccrs.PlateCarree()})

        # Add land outlines
        land = NaturalEarthFeature(category='physical', name='land', scale='50m', edgecolor='k', facecolor='lightgray')
        ax.add_feature(land)

        # Set up the colormap
        colors = cm.jet(np.linspace(0, 1, self.data.datetimes.shape[0]))
        
        # Plot points chronologically
        for (lon, lat), datetime, color in zip(self.data.locations, self.data.datetimes, colors):
            ax.plot(lon, lat, marker='.', color=color, markersize=2)
            # ax.text(lon, lat, timestamp.strftime('%Y-%m-%d'), va='bottom', ha='right', fontsize=8, color='blue')

        # Set plot title and labels
        ax.set_title('Latitude-Longitude Points')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')

        # Set plot extent
        ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())

        # Add gridlines
        ax.gridlines(linestyle='--', linewidth=0.5, color='gray', alpha=0.5)

        # Show the plot
        # Build output filename and save figure
        figname = f"IASI_L1C_points.png".replace(":", ".")
        plt.savefig(f"{self.data.filepath}{figname}", dpi=150, bbox_inches='tight')


def main():

    # Specify location of IASI L1C outputs txt files
    # filepath = "E:\\data\\iasi\\"
    filepath = "C:\\Users\\padra\\Documents\\Research\\github\\contrails\\iasi\\"

    filenames = ["iasi_L1C_2020_1_1.txt"]
    
    # Loop through date-times
    for filename in filenames:
        
        plot_parameters = {'title': 'IASI Spectrum', 'xlabel': 'Channels', 'ylabel': 'Radiance (...)'}
        plot = L1CPlotter(filepath, filename, plot_parameters)
        # plot.plot_spectrum()
        plot.plot_observation_points()

if __name__ == "__main__":
    main()