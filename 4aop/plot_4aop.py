import numpy as np
import matplotlib.pyplot as plt

# Constants
DATA_DIR = "C:\\Users\\padra\\Documents\\Research\\projects\\contrails\\4aop\\outputs\\"
INPUT_FILE = f"{DATA_DIR}spc4asub0001test_test_iasil1c_trophr20iasi1c.plt"
OUTPUT_FILE = INPUT_FILE.split(".")[0]

COLUMNS = [
    (1, 'Radiance (W $m^{-2} sr^{-1} cm^{-1}$)', 'black'),
    (2, 'Brightness Temperature (K)', 'red')
]
FIG_SIZE = (6, 3)
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

def plot_column(ax, x_data, y_data, y_label, color):
    """Plot a column of data on a given axis."""
    ax.plot(x_data, y_data, color=color, lw=0.75)
    ax.set_ylabel(y_label, color=color, fontsize=FONT_SIZE)
    ax.tick_params(axis='y', color=color, labelcolor=color, size=FONT_SIZE, labelsize=FONT_SIZE)
    ax.set_xlim(x_data[-1], x_data[0])  # This will ensure descending x-axis

def plot_data(data):
    """Generate plots for the provided data."""
    wavenumbers = data[:, 0]
    fig, ax1 = plt.subplots(figsize=FIG_SIZE)
    plot_column(ax1, wavenumbers, data[:, COLUMNS[0][0]], COLUMNS[0][1], COLUMNS[0][2])
    
    ax2 = ax1.twinx()
    plot_column(ax2, wavenumbers, data[:, COLUMNS[1][0]], COLUMNS[1][1], COLUMNS[1][2])
    
    ax1.set_xlabel(r'Wavenumbers ($cm^{-1}$)', size=FONT_SIZE)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_FILE}.png", dpi=300)

def main():
    """Main execution function."""
    data = read_plt_file(INPUT_FILE)
    if data is not None:
        plot_data(data)

if __name__ == "__main__":
    main()
