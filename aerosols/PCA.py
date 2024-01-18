import os
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Inputs
data_dir = "C:\\Users\\padra\\Documents\\Research\\projects\\contrails\\4aop\\outputs\\"
data_file = "spectra_radiance"
output_file = os.path.join(data_dir, f"PCA_{data_file}.png")

N = 4

# Load your data
data_path = os.path.join(data_dir, f"{data_file}.csv")
df = pd.read_csv(data_path, sep='\t')

# Standardizing the data so that each feature contributes equally to the variance
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[['con2301', 'con2302', 'con2305', 'con2310']].values)

# Performing PCA for N components
pca = PCA(n_components=N)  # Replace N with the number of components you want
principal_components = pca.fit_transform(scaled_data)

# Creating a DataFrame for the principal components
pc_df = pd.DataFrame(data=principal_components, columns=[f'PC{i+1}' for i in range(N)])
pc_df['Wavenumbers'] = df['Wavenumbers']

# Visualizing the principal components
plt.figure(figsize=(8, 6))

# Creating a pair plot
sns.pairplot(pc_df, hue='Wavenumbers', palette='cividis')
# Create a PairGrid instance with the lower triangle option
grid = sns.PairGrid(pc_df, corner=True)  # corner=True will create only the lower triangle

# Map the plots to the grid
# For scatter plots
grid.map_lower(sns.scatterplot, hue=pc_df['Wavenumbers'], palette='cividis')

# For histograms on the diagonal
grid.map_diag(sns.histplot, color='grey')

# Add legend and adjust plot
plt.legend(title='Wavenumbers', bbox_to_anchor=(1.05, 1), loc=2)
plt.suptitle('Triangle Plot of Principal Components', y=1.02)  # Adjust title positioning
plt.savefig(output_file, dpi=300, bbox_inches='tight')

# Display the variance ratio of each component
print('Explained variance ratio:', pca.explained_variance_ratio_)
