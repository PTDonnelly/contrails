import os
import pandas as pd
import numpy as np
from scipy.integrate import quad
from scipy.stats import lognorm

def get_xsc(xsc_file):
    with open(xsc_file, 'r') as file:
        lines = file.readlines()
        data_lines = lines[17:34]  # Assuming these are the lines you need

        # Preallocate a NumPy array of the correct shape
        num_cols = len(data_lines[0].split()) - 2
        data = np.empty((len(data_lines), num_cols))

        # Fill in the array with data
        for i, line in enumerate(data_lines):
            values = line.split()[2:]  # Get all values starting from the 3rd
            data[i] = np.array(values, dtype=float)  # Convert values to float and assign to row i
    return pd.DataFrame(data)

def particle_size_distribution(D_median):
    # Constants
    sigma = 1.5
    D_max = 3 * D_median
    num_points = 20

    # Distribution parameters
    s = np.log(sigma)
    scale = np.exp(np.log(D_median) - s**2 / 2)
    
    # Create a lognormal distribution object
    distribution = lognorm(s=s, scale=scale)

    # Evaluate the PSD
    D_range = np.linspace(D_max/num_points, D_max, num_points)
    psd_values = distribution.pdf(D_range)

    # Integrate the PSD over the diameter range
    total_concentration = np.trapz(psd_values, D_range)

    # Normalize the PSD so that it integrates to 1 particle per cm³
    normalized_psd_values = psd_values / total_concentration

    # Check the normalization
    assert np.isclose(np.trapz(normalized_psd_values, D_range), 1.0), "PSD not properly normalized"

    return normalized_psd_values, D_range

def extinction_coefficient(D_median, N_p, sigma_D):
    N_D = particle_size_distribution(D_median)
    N_D_p = quad(ext_coeff, D1, D2)
    return N_D_p * sigma_D



def optical_thickness_gradient(D_median, N_p, sigma_D, D1, D2):
    ext_coeff = extinction_coefficient(D_median, N_p, sigma_D)
    integral, _ = quad(ext_coeff, D1, D2)
    return integral

def ice_water_content(D, p):
    density_ice = 917  # kg/m^3
    volume = 4/3 * np.pi * (D/2)**3 # Volume of a sphere with diameter D
    mD = density_ice * volume
    return particle_size_distribution(D, p) * mD

def ice_water_path(p1, p2, D1, D2):
    # Integrate ice water content over the pressure range and particle sizes
    def integral_function(p):
        integral, _ = quad(ice_water_content, D1, D2, args=(p,))
        return integral
    IWP, _ = quad(integral_function, p1, p2)
    return IWP


# Inputs
data_dir = "C:\\Users\\padra\\Documents\\Research\\projects\\contrails\\4aop\\"
aer_file = "\\datatm\\aer4atest_baum.dsf"
xsc_file = "\\datscat\\aerosols_con1610.dat"
# output_file = os.path.join(data_dir, f".png")

aer_df = pd.read_csv(f"{data_dir}{aer_file}", sep='\t', skiprows=1)
xsc_df = get_xsc(f"{data_dir}{xsc_file}")

# Calculate optical thickness gradient with pressure
D_median = 5 # in microns
D1, D2 = 1, 60
p_top, p_base = aer_df.columns[0, 1]
optical_depth = aer_df.columns[2]
sigma_D = xsc_df.columns[1]
dtau_dz = optical_thickness_gradient(D_median, N_p, sigma_D, D1, D2)

# print(f"Optical thickness gradient at {z_sample} m: {dtau_dz}")

# # Calculate Ice Water Path
# iwp = ice_water_path(z1, z2, D1, D2)
# print(f"Ice Water Path: {iwp}")