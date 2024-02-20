import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chisquare

def reconstruct_spectrum(half_res_spectrum, covariance_matrix):
    num_channels = covariance_matrix.shape[0]
    reconstructed_spectrum = np.zeros(num_channels)
    
    # Fill in the known values
    reconstructed_spectrum[::2] = half_res_spectrum
    
    # Indices of the missing values
    missing_indices = np.arange(1, num_channels, 2)
    
    # Ensure we don't go out of bounds on the last index
    valid_indices = missing_indices[missing_indices + 1 < num_channels]
    
    # Calculate weights
    a = covariance_matrix[valid_indices, valid_indices - 1] / \
        (covariance_matrix[valid_indices, valid_indices - 1] + covariance_matrix[valid_indices, valid_indices + 1])
    b = covariance_matrix[valid_indices, valid_indices + 1] / \
        (covariance_matrix[valid_indices, valid_indices - 1] + covariance_matrix[valid_indices, valid_indices + 1])
    
    # Reconstruct missing values
    reconstructed_spectrum[valid_indices] = a * reconstructed_spectrum[valid_indices - 1] + b * reconstructed_spectrum[valid_indices + 1]
    
    return reconstructed_spectrum


def calculate_means_and_residuals(original_spectrum, half_res_spectrum, reconstructed_spectrum):
    # Calculate means
    original_mean = np.mean(original_spectrum)
    half_res_mean = np.mean(half_res_spectrum)
    reconstructed_mean = np.mean(reconstructed_spectrum)
    
    print("Mean of Original Spectrum:", original_mean)
    print("Mean of Half-Resolution Spectrum:", half_res_mean)
    print("Mean of Reconstructed Spectrum:", reconstructed_mean)
    
    # Calculate residuals
    original_residuals = original_spectrum - original_mean
    half_res_residuals = half_res_spectrum - original_mean
    reconstructed_residuals = reconstructed_spectrum - original_mean
    
    # Print out the means of the residuals for verification
    print("Mean Residual for Original Spectrum:", np.mean(original_residuals))
    print("Mean Residual for Half-Resolution Spectrum:", np.mean(half_res_residuals))
    print("Mean Residual for Reconstructed Spectrum:", np.mean(reconstructed_residuals))

    # Perform chi-square test
    # For half-resolution spectrum
    chi2_half_res, p_half_res = chisquare(f_obs=half_res_spectrum, f_exp=original_spectrum[::2])
    print(f"Chi-Square Test for Half-Resolution Spectrum: Chi2 = {chi2_half_res}, p-value = {p_half_res}")

    # For reconstructed spectrum
    chi2_reconstructed, p_reconstructed = chisquare(f_obs=reconstructed_spectrum, f_exp=original_spectrum)
    print(f"Chi-Square Test for Reconstructed Spectrum: Chi2 = {chi2_reconstructed}, p-value = {p_reconstructed}")

    
    # Generate histograms
    plt.figure(figsize=(12, 8))
    
    plt.hist(original_residuals, bins=50, alpha=0.5, label='Original Spectrum Residuals')
    plt.hist(half_res_residuals, bins=50, alpha=0.5, label='Half-Resolution Spectrum Residuals')
    plt.hist(reconstructed_residuals, bins=50, alpha=0.5, label='Reconstructed Spectrum Residuals')
    
    plt.xlabel('Residual Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of Residuals')
    plt.legend()
    
    plt.show()


# Example usage with mock data
if __name__ == "__main__":
       
    # original_spectrum = 1
    
    # half_res_spectrum = original_spectrum[::2]
    
    # # Reconstruct original spectrum using weighted average
    # covariance_matrix = 1
    # reconstructed_spectrum = reconstruct_spectrum(half_res_spectrum, covariance_matrix)

    original_spectrum = np.random.normal(loc=0, scale=1, size=1000)  # Mock original spectrum
    half_res_spectrum = original_spectrum[::2]  # Mock half-resolution spectrum (for demonstration)
    reconstructed_spectrum = np.random.normal(loc=0, scale=1, size=1000)  # Mock reconstructed spectrum (for demonstration)
    
    calculate_means_and_residuals(original_spectrum, half_res_spectrum, reconstructed_spectrum)


