import numpy as np
import matplotlib.pyplot as plt
import os

# Constants
DATA_DIR = "C:\\Users\\padra\\Documents\\Research\\projects\\contrails\\4aop\\outputs\\"
INPUT_FILES = [
    "spc4asub0001_inv_co2_CAMS_test_iasil1c_ice_clearhr20iasi1c.plt",
    "spc4asub0001_inv_co2_CAMS_test_iasil1c_ice_test_ice_baumhr20iasi1c.plt",
    "spc4asub0001_inv_co2_CAMS_test_iasil1c_ice_con1601hr20iasi1c.plt",
    "spc4asub0001_inv_co2_CAMS_test_iasil1c_ice_con2301hr20iasi1c.plt"
]
OUTPUT_FILE = f"{DATA_DIR}ice_vs_noice.png"  # Add your output directory here

COLUMNS = [
    (1, 'Radiance (W $m^{-2} sr^{-1} (cm^{-1})^{-1}$)', 'log', 'black'),
    (2, 'Brightness Temperature (K)', 'linear', 'red'),
]
FIG_SIZE = (12, 6)
FONT_SIZE = 10

def read_plt_file(filename):
    """Load data from filename, handling possible errors."""
    try:
        return np.loadtxt(filename)
    except (IOError, ValueError) as e:
        print(f"Error reading {filename}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    return None

def plot_column(ax, x_data, y_data, y_label, yscale, color):
    """Plot a column of data on a given axis."""
    ax.plot(x_data, y_data, color=color, lw=0.75)
    ax.set_ylabel(y_label, color=color, fontsize=FONT_SIZE)
    ax.set_yscale(yscale)
    ax.tick_params(axis='y', color=color, labelcolor=color, size=FONT_SIZE, labelsize=FONT_SIZE)
    ax.set_xlim(x_data[-1], x_data[0])  # This will ensure descending x-axis

def plot_data(files, output_file):
    """Generate plots for the provided data."""
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=FIG_SIZE)

    for file in files:
        data = read_plt_file(file)
        if data is not None:
            wavenumbers = data[:, 0]
            plot_column(ax1, wavenumbers, data[:, COLUMNS[0][0]], COLUMNS[0][1], COLUMNS[0][2], COLUMNS[0][3])
            plot_column(ax2, wavenumbers, data[:, COLUMNS[1][0]], COLUMNS[1][1], COLUMNS[1][2], COLUMNS[1][3])

    ax1.set_xlabel(r'Wavenumbers ($cm^{-1}$)', size=FONT_SIZE)
    ax2.set_xlabel(r'Wavenumbers ($cm^{-1}$)', size=FONT_SIZE)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)

def main():
    """Main execution function."""
    files = [os.path.join(DATA_DIR, file) for file in INPUT_FILES]
    plot_data(files, OUTPUT_FILE)

if __name__ == "__main__":
    main()
