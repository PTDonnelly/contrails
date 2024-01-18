import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load your data
data_dir = "C:\\Users\\padra\\Documents\\Research\\projects\\contrails\\4aop\\outputs\\"
df = pd.read_csv(f"{data_dir}spectra_radiance.csv", skiprows=1, sep='\t')

# Transpose the data so that each spectrum (cloud configuration) is a row
# and skip the first column (wavenumber) by starting from the second column
df_transposed = df.T

# Remove the first row which contains the wavelength information
df_transposed = df_transposed.iloc[1:]

# Convert the data to numeric, as it might be read as string
df_transposed = df_transposed.apply(pd.to_numeric, errors='coerce')

# Calculate the correlation matrix
corr_matrix = df_transposed.corr(numeric_only=True)

# Display the correlation matrix
# print(corr_matrix)

# Plotting the correlation matrix
plt.figure(figsize=(8, 6), dpi=300)
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm')
plt.title('Correlation Matrix of Spectral Data under Different Cloud Configurations')
plt.savefig(f"{data_dir}correlation.png", dpi=300)
