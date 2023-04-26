import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

def get_iwabuchi(optical_dir: str, filename: str):
    data = np.loadtxt(f"{optical_dir}{filename}.txt", comments='#', delimiter=None, unpack=True)
    return data[0, :], data[1:13, :], data[13:, :]

def get_ybounds(yarray: npt.ArrayLike, xarray: npt.ArrayLike, xmin: float, xmax: float):
    xkeep = np.where((xarray >= xmin) & (xarray <= xmax))
    return np.min(yarray[:, xkeep]), np.max(yarray[:, xkeep])

def format_plot(ax, ylabel: str, ylim: tuple):
    ax.grid(axis='both', markevery=1, color='gray', lw=0.5, ls='--')
    ax.set_ylabel(ylabel)
    ax.set_ylim(ylim)
    ax.tick_params(axis='both', size=6, pad=2)

def plot_iwabuchi():
    optical_dir = 'optical_data/'
    filename = 'iwabuchi_optical_properties'
    wavelengths, real, imaginary = get_iwabuchi(optical_dir, filename)
    xmin, xmax = 7.0, 15.0
    colors = cm.seismic(np.linspace(0, 1, real.shape[0]))

    _, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 4), dpi=300)

    # Plot real and imaginary parts
    for part, ax, ylabel in zip((real, imaginary), (ax1, ax2), ('n', 'k')):
        for i, color in enumerate(colors):
            ax.plot(wavelengths, part[i, :], color=color, label=f'T = {160 + i*10} K')
        format_plot(ax, ylabel, get_ybounds(part, wavelengths, xmin, xmax))

    # Clean up plot
    ax1.set_title('Refractive Index')
    ax2.set_xlabel('Wavelength (um)')
    plt.xlim(xmin, xmax)

    # Add a legend
    plt.legend(loc='center right', fontsize=5.5, bbox_to_anchor=(1.15, 1))
    
    # Save figure
    plt.savefig(f"{optical_dir}{filename}.png", dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    plot_iwabuchi()
