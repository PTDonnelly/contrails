import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import os

# Constants
DATA_DIR = "C:\\Users\\padra\\Documents\\Research\\projects\\contrails\\4aop\\outputs\\"
INPUT_FILES = [
    "spc4asub0001_inv_co2_CAMS_test_iasil1c_icehr20iasi1c.plt",
    "spc4asub0001_inv_co2_CAMS_test_iasil1c_ice_test_icehr20iasi1c.plt"
]
OUTPUT_FILE = os.path.join(DATA_DIR, "ice_vs_noice.png")
RADIANCE_COLUMN = 1
LABELS = ['Cloud-free', 'Cirrus (Baum et al., 2014)']
YLABELS = ['Radiance (W $m^{-2} sr^{-1} (cm^{-1})^{-1}$)', 'Residuals (%)']
SCALE = 'linear'
FIG_SIZE = (8, 6)
FONT_SIZE = 10
COLOURS = ['black', 'red']

def read_plt_file(filename):
    """Load data from filename, handling possible errors."""
    try:
        return np.loadtxt(filename)
    except (IOError, ValueError) as e:
        print(f"Error reading {filename}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    return None

def plot_radiance(ax, x_data, y_data, label, colour='black'):
    """Plot a column of data on a given axis."""
    ax.plot(x_data, y_data, label=label, color=colour, lw=0.75)

def plot_residuals(ax, x_data, y_data, colour='black'):
    """Plot a column of data on a given axis."""
    ax.fill_between(x_data, 0, y_data, color=colour, alpha=0.5)


def plot_data(files, output_file):
    """Generate plots for the provided data."""
    fig = plt.figure(figsize=FIG_SIZE)
    gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1]) 
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1], sharex=ax1)

    data1 = read_plt_file(files[0])
    data2 = read_plt_file(files[1])
    
    if data1 is not None and data2 is not None:
        wavenumbers = data1[:, 0]
        radiance1 = data1[:, RADIANCE_COLUMN]
        radiance2 = data2[:, RADIANCE_COLUMN]
        difference = ((radiance2 - radiance1)/radiance1) * 100
        
        plot_radiance(ax1, wavenumbers, radiance1, LABELS[0], COLOURS[0])
        plot_radiance(ax1, wavenumbers, radiance2, LABELS[1], COLOURS[1])
        
        plot_residuals(ax2, wavenumbers, difference)
        
        for iax, ax in enumerate([ax1, ax2]):
            ax.set_ylabel(YLABELS[iax], fontsize=FONT_SIZE)
            ax.set_yscale(SCALE)
            ax.tick_params(axis='y', labelsize=FONT_SIZE)
            ax.set_xlim(wavenumbers[-1], wavenumbers[0])  # This will ensure descending x-axis
            ax.grid(ls='--', lw=0.5, color='grey')
        ax2.set_ylim(-6, 6)
        # ax2.plot([max(wavenumbers), min(wavenumbers)], [0, 0], ls='--', lw=1, color='black')


    ax1.legend(fontsize=FONT_SIZE)
    ax2.set_xlabel(r'Wavenumbers ($cm^{-1}$)', size=FONT_SIZE)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)

def main():
    """Main execution function."""
    files = [os.path.join(DATA_DIR, file) for file in INPUT_FILES]
    plot_data(files, OUTPUT_FILE)

if __name__ == "__main__":
    main()
