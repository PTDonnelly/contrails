import gzip
from turtle import right
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chisquare

from tqdm import tqdm

def reconstruct_spectrum(original_spectrum, half_res_spectrum, weights_left, weights_right):
    # Initialize the reconstructed spectrum with original values, missing values can be NaN or some placeholder
    reconstructed_spectrum = original_spectrum.copy()

    # Fill in known values from the half-resolution spectrum (every other value in this case)
    reconstructed_spectrum.iloc[::2] = half_res_spectrum

    # Find missing indices in the reconstructed spectrum
    missing_indices = reconstructed_spectrum[reconstructed_spectrum.isna()].index.to_numpy()

    # Pre-extract values for efficiency
    reconstructed_spectrum_values = reconstructed_spectrum.values

    for idx in missing_indices:
        # Get weights from the covariance matrix for adjacent channels
        if idx > 0 and idx < len(original_spectrum) - 1:
            # Get adjacent values in the spectrum
            value_left = reconstructed_spectrum_values[idx - 1]
            value_right = reconstructed_spectrum_values[idx + 1]

            # Use pre-computed weights
            weight_left = weights_left[idx - 1]
            weight_right = weights_right[idx + 1]

            # Calculate the weighted average for the missing value
            reconstructed_value = (weight_left * value_left + weight_right * value_right) / (weight_left + weight_right)
            reconstructed_spectrum_values[idx] = reconstructed_value

    return reconstructed_spectrum

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

def process_spectra_over_bins(df, weights_left, weights_right, number_of_bins_range):
    # Initialize dictionaries to store the results for each number_of_bins
    chi2_results_half_res = {bins: [] for bins in number_of_bins_range}
    chi2_results_reconstructed = {bins: [] for bins in number_of_bins_range}
    p_values_half_res = {bins: [] for bins in number_of_bins_range}
    p_values_reconstructed = {bins: [] for bins in number_of_bins_range}

    df = df.reset_index(drop=True)

    for row in tqdm(df.itertuples(index=False), total=df.shape[0]):
        original_spectrum = pd.Series(row)

        # Loop over the range of number_of_bins
        for number_of_bins in number_of_bins_range:
            half_res_spectrum = original_spectrum[::2]
            reconstructed_spectrum = reconstruct_spectrum(original_spectrum, half_res_spectrum, weights_left, weights_right)

            # Calculate histograms for the original, half-resolution, and reconstructed spectra
            hist_original, bin_edges = np.histogram(original_spectrum, bins=number_of_bins)
            hist_half_res, _ = np.histogram(half_res_spectrum, bins=bin_edges)
            hist_reconstructed, _ = np.histogram(reconstructed_spectrum, bins=bin_edges)

            # Normalize frequencies and perform Chi-square tests
            hist_half_res_normalized, hist_original_normalized = normalize_frequencies(hist_half_res, hist_original)
            chi2_half_res, p_half_res = chisquare(f_obs=hist_half_res_normalized, f_exp=hist_original_normalized)
            hist_reconstructed_normalized, hist_original_normalized = normalize_frequencies(hist_reconstructed, hist_original)
            chi2_reconstructed, p_reconstructed = chisquare(f_obs=hist_reconstructed_normalized, f_exp=hist_original_normalized)

            # Store the results for this number_of_bins
            reduced_chi2_half_res = chi2_half_res / number_of_bins
            reduced_chi2_reconstructed = chi2_reconstructed / number_of_bins
            chi2_results_half_res[number_of_bins].append(reduced_chi2_half_res)
            chi2_results_reconstructed[number_of_bins].append(reduced_chi2_reconstructed)
            p_values_half_res[number_of_bins].append(p_half_res)
            p_values_reconstructed[number_of_bins].append(p_reconstructed)

    # Plotting
    fig, axs = plt.subplots(2, 1, figsize=(12, 10))

    mean_chi2_half_res_list = []
    mean_chi2_reconstructed_list = []
    mean_p_half_res_list = []
    mean_p_reconstructed_list = []

    # Plot for each number_of_bins
    for number_of_bins in number_of_bins_range:
        axs[0].scatter([number_of_bins] * len(chi2_results_half_res[number_of_bins]), chi2_results_half_res[number_of_bins],
                            marker='o', s=5, color='orange', alpha=0.5)
        axs[0].scatter([number_of_bins] * len(chi2_results_reconstructed[number_of_bins]), chi2_results_reconstructed[number_of_bins],
                            marker='o', s=5, color='blue', alpha=0.5)
        axs[1].scatter([number_of_bins] * len(p_values_half_res[number_of_bins]), p_values_half_res[number_of_bins],
                            marker='o', s=5, color='orange', alpha=0.5)
        axs[1].scatter([number_of_bins] * len(p_values_reconstructed[number_of_bins]), p_values_reconstructed[number_of_bins],
                            marker='o', s=5, color='blue', alpha=0.5)

        # Calculate means
        mean_chi2_half_res_list.append(np.nanmean(chi2_results_half_res[number_of_bins]))
        mean_chi2_reconstructed_list.append(np.nanmean(chi2_results_reconstructed[number_of_bins]))
        mean_p_half_res_list.append(np.nanmean(p_values_half_res[number_of_bins]))
        mean_p_reconstructed_list.append(np.nanmean(p_values_reconstructed[number_of_bins]))

    # Line plots for averages
    axs[0].plot(number_of_bins_range, mean_chi2_half_res_list, ls='-', lw=2, color='orange')
    axs[0].plot(number_of_bins_range, mean_chi2_reconstructed_list, ls='-', lw=2, color='blue')
    axs[1].plot(number_of_bins_range, mean_p_half_res_list, ls='-', lw=2, color='orange')
    axs[1].plot(number_of_bins_range, mean_p_reconstructed_list, ls='-', lw=2, color='blue')

    # Customizing the legend to avoid duplicate labels
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='orange', marker='.', linestyle='', markersize=10),
                    Line2D([0], [0], color='blue', marker='.', linestyle='', markersize=10),
                    Line2D([0], [0], color='orange', linestyle='-', markersize=10),
                    Line2D([0], [0], color='blue', linestyle='-', markersize=10)]
    axs[0].legend(custom_lines, ['Half-Resolution', 'Reconstructed', 'Average Half-Res', 'Average Reconstructed'])
    axs[1].legend(custom_lines, ['Half-Resolution', 'Reconstructed', 'Average Half-Res', 'Average Reconstructed'])

    axs[0].set_xlabel('Number of Bins', fontsize=20)
    axs[0].set_ylabel('Chi-square Statistic', fontsize=20)
    axs[0].set_title('Chi-square Statistic vs. Number of Bins', fontsize=20)
    axs[0].set_ylim([0, number_of_bins_range[-1]])
    axs[0].set_ylim([0, 70])
    axs[1].set_xlabel('Number of Bins', fontsize=20)
    axs[1].set_ylabel('P-value', fontsize=20)
    axs[1].set_title('P-value vs. Number of Bins', fontsize=20)
    axs[1].set_ylim([0, number_of_bins_range[-1]])
    axs[1].set_ylim([0.7, 1])
    plt.tick_params(axis='both', labelsize=16)
    plt.tight_layout()
    plt.savefig("c:\\Users\\donnelly\\Documents\\projects\\data\\chisquare_test.png")

def main():
    # Read in the original spectrum
    data_file = "c:\\Users\\donnelly\\Documents\\projects\\data\\spectra_and_cloud_products.pkl.gz"
    with gzip.open(data_file, 'rb') as f:
        df = pickle.load(f)
        spectrum_df = df[[col for col in df.columns if 'Spectrum' in col]]

    # Read in IASI covariance matrix
    covariance_file = "c:\\Users\\donnelly\\Documents\\projects\\iasi\\covariance_matrix.csv"
    covariance_matrix_df = pd.read_csv(covariance_file, sep="\t")

    # Example: Pre-computing weights for adjacent indices
    weights_left = covariance_matrix_df.values[np.arange(spectrum_df.shape[1] - 1), np.arange(1, spectrum_df.shape[1])]
    weights_right = covariance_matrix_df.values[np.arange(1, spectrum_df.shape[1]), np.arange(spectrum_df.shape[1] - 1)]
    
    # Define the range of number_of_bins to test
    number_of_bins_range = range(5, 101, 1)
    process_spectra_over_bins(spectrum_df, weights_left, weights_right, number_of_bins_range)


if __name__ == "__main__":
    main()

