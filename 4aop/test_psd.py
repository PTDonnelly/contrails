import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import lognorm
from scipy import interpolate
import json

def build_psd(sigma, D_median, D_max, num_points):
    s = np.log(sigma)
    scale = np.exp(np.log(D_median) - s**2 / 2)
    
    # Create a lognormal distribution object
    distribution = lognorm(s=s, scale=scale)

    # Evaluate the PSD
    D = np.linspace(D_max/num_points, D_max, num_points)
    psd_values = distribution.pdf(D)

    # Integrate the PSD over the diameter range
    total_concentration = np.trapz(psd_values, D)

    # Normalize the PSD so that it integrates to 1 particle per cm³
    normalized_psd_values = psd_values / total_concentration

    # Check the normalization
    assert np.isclose(np.trapz(normalized_psd_values, D), 1.0), "PSD not properly normalized"

    return normalized_psd_values, D

def get_config():
    """Load configuration from a JSON file."""
    CONFIG_FILE = "aerosol_config.json"

    with open(CONFIG_FILE, 'r') as f:
        return json.load(f)
config = get_config()

radii = [1, 2, 5, 10]
resolutions = np.arange(15, 22, 1)

# Plotting setup
fig = plt.figure(figsize=(6, 9))
nplots = len(radii)
gs = gridspec.GridSpec(nplots, 2, height_ratios=[1, 1, 1, 1], width_ratios=[1, 1])
fig.suptitle('Scattering Properties of\nWater-Ice Mie Particles', ha='center')
colors = [plt.cm.jet(x) for x in np.linspace(0, 1, len(resolutions))]

for i, radius in enumerate(radii):
    HR_psd, HR_D = build_psd(1.5, 2*radius, 6*radius, 1024)
    idx = np.where(HR_psd == np.max(HR_psd))
    mean = HR_D[idx]
    scale = np.exp(np.log(mean) - 1.5**2 / 2)
    mean, scale = mean[0], scale[0]

    for (res, color) in zip(resolutions, colors):
        psd, D = build_psd(1.5, 2*radius, 6*radius, res)

        # Plot
        ax1 = plt.subplot(gs[i, 0])
        ax1.plot(HR_D, HR_psd, color='k', ls=':')
        
        ax1.axvline(x=mean, color='k', lw=0.5, ls='--')
        ax1.axvline(x=mean+scale, color='k', lw=0.5, ls='--')
        ax1.axvline(x=mean-scale, color='k', lw=0.5, ls='--')

        ax1.plot(D, psd, color=color)
        ax1.set_xlabel(r'Particle Diameter ($\mu$m)', labelpad=1)
        ax1.set_xscale('log')
        ax1.set_xlim((mean-scale, mean+scale))

        ax2 = plt.subplot(gs[i, 1])
        ax2.plot(HR_D, HR_psd/HR_psd, color='k', ls=':')
        ax2.axvline(x=mean, color='k', lw=0.5, ls='--')
        ax2.axvline(x=mean+scale, color='k', lw=0.5, ls='--')
        ax2.axvline(x=mean-scale, color='k', lw=0.5, ls='--')

        f = interpolate.interp1d(D, psd, kind='cubic', fill_value='extrapolate')
        psd_new = f(HR_D)
        ax2.plot(HR_D, psd_new/HR_psd, color=color)
        ax2.set_xscale('log')
        ax2.set_xlim((mean-scale, mean+scale))
        ax2.set_ylim((0.99, 1.01))


plt.subplots_adjust(hspace=0.15)

# Save figure
plt.savefig(f"{config.get('aerosol_scattering_directory')}psd_comparison.png", dpi=300, bbox_inches='tight')