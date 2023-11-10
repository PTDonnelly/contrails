import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import logging
import json

# Constants
CONFIG_FILE = "aerosol_config.json"

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_config():
    """Load configuration from a JSON file."""
    with open(CONFIG_FILE, 'r') as f:
        return json.load(f)

def read_reference_profile(config):

    # Read the template .aer file into a DataFrame
    reference_aerfile = os.path.join(config.get('aerosol_profile_directory'), 'aer4atest_baum.dsf')
    df_columns = ['layer_top_pressure',
                'layer_base_pressure',
                'reference_opacity',
                'scattering_model',
                'single_scattering_albedo',
                'asymmetry_parameter',
                'model_exponent',
                'phase_function'
    ]
    
    # Extract header from the reference file
    with open(reference_aerfile, 'r') as file:
        header = file.readline().strip()

    return header, pd.read_csv(reference_aerfile, delim_whitespace=True, names=df_columns, header=None, skiprows=1)

def main():
    config = get_config()
    
    header, df = read_reference_profile(config)

    # Set up the plot
    plt.figure(figsize=(4, 6))

    # Plot each layer
    opacity = df['reference_opacity']
    pressure = df[['layer_top_pressure', 'layer_base_pressure']].mean(axis=1)
    
    # Plot data
    plt.plot(opacity, pressure, lw=0.5, ls='--', color='r')
    plt.plot(opacity.replace(0, np.nan), pressure, lw=1.5, color='r', label='Baum et al. (2014)')

    # Set the axes to logarithmic scale
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim((4e-4, 1e-1))
    plt.ylim((1e3, 10))

    # Adding labels and title
    plt.xlabel('Aerosol Optical Depth')
    plt.ylabel('Pressure (hPa)')
    plt.title('Vertical Profile of Natural Cirrus')

    # Add a legend
    plt.legend()

   
    # Save figure
    plt.savefig(f"{config.get('aerosol_profile_directory')}aer_comparison.png", dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    main()