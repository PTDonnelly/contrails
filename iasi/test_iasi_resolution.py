import gzip
from turtle import right
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chisquare

def reconstruct_spectrum(original_spectrum, half_res_spectrum, covariance_matrix):
    # Initialize the reconstructed spectrum with original values, missing values can be NaN or some placeholder
    reconstructed_spectrum = original_spectrum.copy()

    # Fill in known values from the half-resolution spectrum (every other value in this case)
    reconstructed_spectrum.iloc[::2] = half_res_spectrum

    # Find missing indices in the reconstructed spectrum
    missing_indices = reconstructed_spectrum[reconstructed_spectrum.isna()].index

    for idx in missing_indices:
        # Get weights from the covariance matrix for adjacent channels
        if idx > 0 and idx < len(original_spectrum) - 1:
            weight_left = covariance_matrix.loc[idx, idx - 1]
            weight_right = covariance_matrix.loc[idx, idx + 1]

            # Get adjacent values in the spectrum
            value_left = reconstructed_spectrum.iloc[idx - 1]
            value_right = reconstructed_spectrum.iloc[idx + 1]

            # Calculate the weighted average for the missing value
            reconstructed_value = (weight_left * value_left + weight_right * value_right) / (weight_left + weight_right)
            reconstructed_spectrum.iloc[idx] = reconstructed_value

    return reconstructed_spectrum

def quantile_binning(series, num_bins):
    quantiles = np.linspace(0, 1, num_bins + 1)
    bin_edges = series.quantile(quantiles).unique()
    return np.histogram(series, bins=bin_edges)

def normalize_frequencies(observed, expected):
    """
    Normalize observed and expected frequencies to have the same total sum.
    
    Parameters:
    observed (np.array): Observed frequencies.
    expected (np.array): Expected frequencies.
    
    Returns:
    np.array: Normalized observed frequencies.
    np.array: Normalized expected frequencies.
    """
    total_observed = observed.sum()
    total_expected = expected.sum()
    
    # Calculate the combined total and the desired sum for each set of frequencies
    combined_total = total_observed + total_expected
    desired_sum = combined_total / 2
    
    # Normalize the frequencies
    observed_normalized = (observed / total_observed) * desired_sum
    expected_normalized = (expected / total_expected) * desired_sum
    
    return observed_normalized, expected_normalized

def calculate_means_and_residuals(original_spectrum, half_res_spectrum, reconstructed_spectrum):
    # Calculate means
    original_mean = np.mean(original_spectrum)
    half_res_mean = np.mean(half_res_spectrum)
    reconstructed_mean = np.mean(reconstructed_spectrum)
    print("Mean of Original Spectrum:", original_mean)
    print("Mean of Half-Resolution Spectrum:", half_res_mean)
    print("Mean of Reconstructed Spectrum:", reconstructed_mean)
    
    # Calculate residuals
    original_residuals = np.subtract(original_spectrum, original_mean)
    half_res_residuals = np.subtract(half_res_spectrum, half_res_mean)
    reconstructed_residuals = np.subtract(reconstructed_spectrum, reconstructed_mean)
    # Print out the means of the residuals for verification
    print("Mean Residual for Original Spectrum:", np.mean(original_residuals))
    print("Mean Residual for Half-Resolution Spectrum:", np.mean(half_res_residuals))
    print("Mean Residual for Reconstructed Spectrum:", np.mean(reconstructed_residuals))

    # Define the range of num_bins to test
    num_bins_range = range(5, 101, 5)  # From 5 to 100 in steps of 5

    # Initialize lists to store the results
    chi2_half_res_list = []
    p_half_res_list = []
    chi2_reconstructed_list = []
    p_reconstructed_list = []

    for num_bins in num_bins_range:
        # Calculate histograms using quantile binning
        hist_original, bin_edges = np.histogram(original_spectrum, bins=num_bins) #quantile_binning(original_spectrum, num_bins)
        hist_half_res, _ = np.histogram(half_res_spectrum, bins=bin_edges)
        hist_reconstructed, _ = np.histogram(reconstructed_spectrum, bins=bin_edges)

        # Normalize frequencies and perform Chi-square tests using the histograms
        hist_half_res_normalized, hist_original_normalized = normalize_frequencies(hist_half_res, hist_original)
        chi2_half_res, p_half_res = chisquare(f_obs=hist_half_res_normalized, f_exp=hist_original_normalized)
        hist_reconstructed_normalized, hist_original_normalized = normalize_frequencies(hist_reconstructed, hist_original)
        chi2_reconstructed, p_reconstructed = chisquare(f_obs=hist_reconstructed_normalized, f_exp=hist_original_normalized)

        # Output the results
        # print(f"Chi-Square Test for Half-Resolution Spectrum: Chi2 = {chi2_half_res}, p-value = {p_half_res}")
        # print(f"Chi-Square Test for Reconstructed Spectrum: Chi2 = {chi2_reconstructed}, p-value = {p_reconstructed}")

        # Store the results
        chi2_half_res_list.append(chi2_half_res)
        p_half_res_list.append(p_half_res)
        chi2_reconstructed_list.append(chi2_reconstructed)
        p_reconstructed_list.append(p_reconstructed)

    # Plotting
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))

    # Panel 1: Chi-square statistics
    ax[0].plot(num_bins_range, chi2_half_res_list, label='Half-Resolution', marker='o')
    ax[0].plot(num_bins_range, chi2_reconstructed_list, label='Reconstructed', marker='s')
    ax[0].set_xlabel('Number of Bins')
    ax[0].set_ylabel('Chi-square Statistic')
    ax[0].set_title('Chi-square Statistic vs. Number of Bins')
    ax[0].legend()

    # Panel 2: P-values
    ax[1].plot(num_bins_range, p_half_res_list, label='Half-Resolution', marker='o')
    ax[1].plot(num_bins_range, p_reconstructed_list, label='Reconstructed', marker='s')
    ax[1].set_xlabel('Number of Bins')
    ax[1].set_ylabel('P-value')
    ax[1].set_title('P-value vs. Number of Bins')
    ax[1].legend()

    plt.tight_layout()
    plt.show()


# Example usage with mock data
if __name__ == "__main__":
    
    # Read in the original spectrum
    data_file = "c:\\Users\\donnelly\\Documents\\projects\\iasi\\spectra_and_cloud_products.pkl.gz"
    with gzip.open(data_file, 'rb') as f:
        df = pickle.load(f)
        spectrum_df = df[[col for col in df.columns if 'Spectrum' in col]]

    # Construct original and half-resolution spectra
    original_spectrum = spectrum_df.iloc[0]
    half_res_spectrum = original_spectrum[::2]

    # Read in IASI covariance matrix
    covariance_file = "c:\\Users\\donnelly\\Documents\\projects\\iasi\\covariance_matrix.csv"
    covariance_matrix_df = pd.read_csv(covariance_file, sep="\t")

    # Reconstruct original spectrum using weighted average
    reconstructed_spectrum = reconstruct_spectrum(original_spectrum, half_res_spectrum, covariance_matrix_df)
    
    calculate_means_and_residuals(original_spectrum, half_res_spectrum, reconstructed_spectrum)


