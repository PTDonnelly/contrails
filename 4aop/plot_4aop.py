import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import os
import pandas as pd

def read_plt_file(filename):
    """Load data from filename, handling possible errors."""
    try:
        return np.loadtxt(filename)
    except (IOError, ValueError) as e:
        print(f"Error reading {filename}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    return None

def plot_spectrum(ax, x_data, y_data, label, colour='black', linestyle='-'):
    """Plot a column of data on a given axis and return the line."""
    line = ax.plot(x_data, y_data, label=label, color=colour, linestyle=linestyle, lw=0.5)
    return line[0]  # Return the line object for the legend

def plot_residuals(ax, x_data, y_data, colour='black', linestyle='-'):
    """Plot a column of data on a given axis."""
    ax.plot(x_data, y_data, color=colour, linestyle=linestyle, lw=0.5)

def plot_histogram(ax, data, nbins, label, colour, linestyle):
    """Plot a histogram of data on a given axis."""
    ax.hist(data, bins=50, color=colour, linestyle=linestyle, label=label, histtype='step', range=(max_residual, 0))

# Inputs
data_dir = "C:\\Users\\padra\\Documents\\Research\\projects\\contrails\\4aop\\outputs\\"
# data_file = "spectra_radiance"
data_file = "spectra_brightness_temperature"
output_file = os.path.join(data_dir, f"comparison_{data_file}_contrail.png")
fig_size = (8, 6)
font_size = 10
if "radiance" in data_file:
    ylabels = ['Radiance (W $m^{-2} sr^{-1} (cm^{-1})^{-1}$)', 'Residuals (%)']
    max_residual = -10
    nbins = 5
elif "brightness_temperature" in data_file:
    ylabels = [r'Brightness Temperature, $T_{B}$ (K)', r'$\Delta T_{B}$ (K)']
    max_residual = -6
    nbins = 6

# Custom labels and colors/linestyles
plot_settings = {
    "clear": ("Clear-sky", "black", "-"),
    # "baum": ("Baum+2014", "grey", "-"),
    # "cir100": (r"Hess+2008a, $T$ = 248K, $r$ = 92 $\mu$m", "grey", "-"),
    # "cir200": (r"Hess+2008b, $T$ = 223K, $r$ = 57 $\mu$m", "grey", "-"),
    # "cir300": (r"Hess+2008c, $T$ = 223K, $r$ = 34 $\mu$m", "grey", "-"),
    # "con1601": (r"$T$ = 160 K, $r$ = 1 $\mu$m", "blue", ":"),
    # "con1602": (r"$T$ = 160 K, $r$ = 2 $\mu$m", "green", ":"),
    # "con1605": (r"$T$ = 160 K, $r$ = 5 $\mu$m", "orange", ":"),
    # "con1610": (r"$T$ = 160 K, $r$ = 10 $\mu$m", "red", ":"),
    # "con2301": (r"$T$ = 230 K, $r$ = 1 $\mu$m", "blue", "-"),
    # "con2302": (r"$T$ = 230 K, $r$ = 2 $\mu$m", "green", "-"),
    # "con2305": (r"$T$ = 230 K, $r$ = 5 $\mu$m", "orange", "-"),
    # "con2310": (r"$T$ = 230 K, $r$ = 10 $\mu$m", "red", "-")
    "con2305f1": (r"$T$ = 230 K, $f$ = 0.1", "blue", "-"),
    # "con2305f2": (r"$T$ = 230 K, $f$ = 0.2", "k", "-"),
    # "con2305f3": (r"$T$ = 230 K, $f$ = 0.3", "k", "-"),
    # "con2305f4": (r"$T$ = 230 K, $f$ = 0.4", "k", "-"),
    # "con2305f5": (r"$T$ = 230 K, $f$ = 0.5", "k", "-"),
    # "con2305f6": (r"$T$ = 230 K, $f$ = 0.6", "k", "-"),
    # "con2305f7": (r"$T$ = 230 K, $f$ = 0.7", "k", "-"),
    # "con2305f8": (r"$T$ = 230 K, $f$ = 0.8", "k", "-"),
    "con2305f9": (r"$T$ = 230 K, $f$ = 0.9", "orange", "-")
    # "con2305": (r"$T$ = 230 K, $f$ = 1.0", "orange", "-")
}

def main():
    """Main execution function."""
    data_path = os.path.join(data_dir, f"{data_file}.csv")
    
    df = pd.read_csv(data_path, sep='\t')
    wavenumbers = df['Wavenumbers']

    fig = plt.figure(figsize=fig_size, dpi=300)
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[3, 1])
    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[1, 0], sharex=ax1)
    ax3 = plt.subplot(gs[1, 1])
    ax_legend = plt.subplot(gs[0, 1])

    base_spectrum = df.iloc[:, 1]
    line_handles = []  # To store line handles for the legend

    for column in df.columns[1:]:
        spectrum = df[column]
        if "radiance" in data_file:
            difference = ((spectrum - base_spectrum) / base_spectrum) * 100
        elif "brightness_temperature" in data_file:
            difference = (spectrum - base_spectrum)
        label, colour, linestyle = plot_settings.get(column, ("Unknown", "grey", "-"))
        line = plot_spectrum(ax1, wavenumbers, spectrum, label, colour, linestyle)
        line_handles.append(line)
        plot_residuals(ax2, wavenumbers, difference, colour, linestyle)
        if column != 'clear':
            plot_histogram(ax3, difference.values, nbins, label, colour, linestyle)

    for iax, ax in enumerate([ax1, ax2]):
        ax.set_ylabel(ylabels[iax], fontsize=font_size)
        ax.tick_params(axis='y', labelsize=font_size)
        ax.set_xlim(wavenumbers.iloc[-1], wavenumbers.iloc[0])  # Descending x-axis
        ax.grid(ls='--', lw=0.5, color='grey')

    ax_legend.legend(handles=line_handles, labels=[setting[0] for setting in plot_settings.values()], loc='center', fontsize=font_size-3)
    ax_legend.axis('off')  # Turn off the axis for the legend

    ax3.set_xlabel(ylabels[1])
    ax3.set_ylabel('Frequency')
    ax3.yaxis.set_label_position("right")
    ax3.yaxis.set_ticks_position("right")
    ax3.set_xlim(max_residual, 0)
    ax3.grid(ls='--', lw=0.5, color='grey')

    ax2.set_ylim(max_residual, 0)
    ax2.set_xlabel(r'Wavenumbers ($cm^{-1}$)', size=font_size)
    plt.subplots_adjust(hspace=0.15, wspace=0.15)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    main()