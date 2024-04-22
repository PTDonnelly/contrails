import pandas as pd
import os
from datetime import datetime, timedelta
import seaborn as sns
from statsmodels.graphics.gofplots import qqplot
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import cross_val_predict, cross_val_score, cross_validate
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer, PowerTransformer, FunctionTransformer
from sklearn.model_selection import RepeatedKFold
import numpy as np

class Dataset:
    def __init__(self, base_path, start_date, end_date, scaling_method):
        self.base_path: str = base_path
        self.start_date: datetime = start_date
        self.end_date: datetime = end_date
        self.scaling_method: str = scaling_method
        self.input: pd.DataFrame = None
        self.target: pd.DataFrame = None
        self.feature_names: list = None
        self.combined_df = None
        self._load_dataset()

    @staticmethod
    def _convert_longitude(lon) -> pd.Series:
        """
        Convert longitude from -180 to 180 range to 0 to 360 range.
        """
        return lon % 360
    
    def read_data(self) -> None:
        data_path = os.path.join(self.base_path, 'era5_iasi_combined.csv')
        self.combined_df = pd.read_csv(data_path, sep='\t')
        return
    
    def get_era5_fields(self):
        # Extract ERA5 fields excluding certain columns
        columns = ['Date', 'Latitude', 'Longitude', 'OLR_mean', 'OLR_icy', 'OLR_clear']#, 'cc_200', 'cc_300', 'o3_300']
        self.feature_names = [col for col in self.combined_df.columns if col not in columns]
        return

    def split_data(self, target_feature):
        self.input = self.combined_df[self.feature_names]
        self.target = self.combined_df[target_feature]
        return

    def categorise_data(self, categorical_feature):
        # Calculate quintiles for the target feature and categorize
        categorical_values = pd.qcut(self.target, 5, labels=False)
        self.input.loc[:, categorical_feature] = categorical_values
        return

    def standardise_data(self):
        if self.scaling_method == "standard":
            scaler = StandardScaler()
        elif self.scaling_method == "robust":
            scaler = RobustScaler()
        elif self.scaling_method == "quantile_uniform":
            scaler = QuantileTransformer(output_distribution='uniform')
        elif self.scaling_method == "quantile_normal":
            scaler = QuantileTransformer(output_distribution='normal')
        elif self.scaling_method == "yeo":
            scaler = PowerTransformer(method='yeo-johnson')
        elif self.scaling_method == "log":
            scaler = FunctionTransformer(np.log1p, validate=False)  # Apply log transformation (log1p for handling zero by adding 1)
        
        # Define a pipeline that imputes missing values, then scales the data
        pipeline = make_pipeline(
            SimpleImputer(strategy='mean'),
            scaler
        )
        # Standardize the input features
        self.input = pd.DataFrame(
            pipeline.fit_transform(self.input),
            index=self.input.index,
            columns=self.input.columns
        )
        return

    def _load_dataset(self):
        self.read_data()
        self.get_era5_fields()
        self.split_data(target_feature='OLR_mean')
        self.categorise_data(categorical_feature='OLR_quintile')
        if not self.scaling_method == 'raw':
            self.standardise_data()

class DataAnalysis:
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def plot_era5_fields_by_olr_quintiles(self):
        # Assuming 'OLR_quintile' is already computed and added to self.dataset.input
        _, axes = plt.subplots(8, 5, figsize=(10, 10), dpi=300)
        axes = axes.flatten()
        features = self.dataset.feature_names
        
        for i, feature in enumerate(features):
            ax = axes[i]
            sns.kdeplot(data=self.dataset.input, x=feature, hue='OLR_quintile', fill=False, linewidth=1, ax=ax, palette='coolwarm', legend=False)
            ax.set_title(feature, fontsize=10)
            ax.set_xlabel("")
            ax.set_ylabel("")
            if i % 4 == 3:
                ax.set_xlabel("Z-score", fontsize=8)
            if i // 4 == 0:
                ax.set_ylabel("Density", fontsize=8)

        plt.tight_layout()
        plt.savefig(os.path.join(self.dataset.base_path, f"olr_era5_correlation_kde_{self.dataset.scaling_method}.png"), bbox_inches='tight')
        plt.close()
        return

    def plot_qq(self):
        _, axes = plt.subplots(8, 5, figsize=(10, 10), dpi=300)
        axes = axes.flatten()
        features = self.dataset.feature_names
        
        for i, feature in enumerate(features):
            ax = axes[i]
            qqplot(self.dataset.input[feature], line='s', ax=ax)
            ax.set_title(feature, fontsize=10)
            ax.set_xlabel("")
            ax.set_ylabel("")
            if i % 4 == 3:
                ax.set_xlabel("Theoretical Quantities", fontsize=8)
            if i // 4 == 0:
                ax.set_ylabel("Sample Quantities", fontsize=8)

        plt.tight_layout()
        plt.savefig(os.path.join(self.dataset.base_path, f"olr_era5_correlation_qq_{self.dataset.scaling_method}.png"), bbox_inches='tight')
        plt.close()
        return
    
    def preprocess_and_fit(self):
        # Define a pipeline that includes RidgeCV
        pipeline = make_pipeline(RidgeCV(alphas=np.logspace(-6, 6, 13)))

        # Define cross-validation strategy
        cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)

        # Perform cross-validation and fit the model
        scores = cross_validate(pipeline, self.dataset.input[self.dataset.feature_names], self.dataset.target, scoring='r2', cv=cv, return_estimator=True)

        # Extract coefficients from each estimator in the cross-validation
        coefs = pd.DataFrame([est[-1].coef_ for est in scores['estimator']], columns=self.dataset.feature_names)
        # Calculate the median value of coefficients for each feature
        medians = coefs.median().sort_values()
        # Reorder the dataframe columns based on the sorted median values
        coefs = coefs[medians.index]

        # Create the box plot with reordered features
        color = {"whiskers": "black", "medians": "black", "caps": "black"}
        coefs.plot.box(vert=False, color=color, figsize=(10, 8))
        plt.axvline(x=0, color="black", linestyle="--")
        plt.title("Coefficients of Ridge Models via Cross-Validation")
        plt.savefig(os.path.join(self.dataset.base_path, f"ridge_coefficients_{self.dataset.scaling_method}.png"), bbox_inches='tight')
        plt.close()
        return

    def plot_ridge(self):
        alphas = np.logspace(-6, 6, 61)
        coefs = []
        
        # Set up and fit RidgeCV to find the optimal alpha
        ridge_cv = RidgeCV(alphas=alphas, store_cv_values=True).fit(self.dataset.input[self.dataset.feature_names], self.dataset.target)
        optimal_alpha = ridge_cv.alpha_  # This is the optimal alpha found by RidgeCV

        # Fit Ridge model for each alpha to track coefficient paths
        for alpha in alphas:
            ridge = Ridge(alpha=alpha).fit(self.dataset.input[self.dataset.feature_names], self.dataset.target)
            coefs.append(ridge.coef_)

        # Plotting the coefficient paths
        coefs_df = pd.DataFrame(coefs, index=alphas, columns=self.dataset.feature_names)
        plt.figure(figsize=(10, 8), dpi=300)
        for column in coefs_df.columns:
            plt.plot(coefs_df.index, coefs_df[column], label=column)

        plt.axvline(x=optimal_alpha, linewidth=0.75, color='k')
        plt.text(0.5*optimal_alpha, 0.75*plt.ylim()[1], f'{optimal_alpha:.3}', rotation=90, verticalalignment='baseline')
        plt.xscale('log')
        plt.xlabel('Alpha')
        plt.ylabel('Coefficient')
        plt.title('Coefficient Paths')
        plt.legend()
        plt.savefig(os.path.join(self.dataset.base_path, f"ridge_paths_{self.dataset.scaling_method}.png"), bbox_inches='tight')
        plt.close()
        return

    def plot_correlation_matrix(self):
        # Concatenate the input features and target variable to form a new DataFrame
        data_for_corr = pd.concat([self.dataset.input[self.dataset.feature_names], self.dataset.target.rename('Target')], axis=1)
        
        # Calculate the correlation matrix
        corr_matrix = data_for_corr.corr()

        # Create a mask to display only the lower triangle of the matrix, as it is symmetrical
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

        # Set up the matplotlib figure
        plt.figure(figsize=(12, 10))

        # Draw the heatmap with the mask and correct aspect ratio
        sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', vmax=.3, center=0,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True, fmt=".2f")

        # Fix the issue with the top and bottom cutting of the heatmap
        plt.ylim(len(corr_matrix), 0)
        plt.xlim(0, len(corr_matrix))

        # Adjust the layout to prevent overlap
        plt.tight_layout()

        # Save the figure
        plt.savefig(os.path.join(self.dataset.base_path, f"correlation_matrix_{self.dataset.scaling_method}.png"), bbox_inches='tight', dpi=300)
        plt.close()

def main():
    # Usage
    base_path = 'G:/My Drive/Research/Postdoc_2_CNRS_LATMOS/data/machine_learning/'
    start_date = datetime(2018, 3, 1)
    end_date = datetime(2023, 5, 31)

    # Initialise the Dataset class with your parameters
    data = Dataset(base_path, start_date, end_date, scaling_method='yeo')

    # Initialise the DataAnalysis class with the Dataset instance
    analysis = DataAnalysis(data)

    # Use the methods of the DataAnalysis class to generate plots and perform other analysis
    analysis.plot_era5_fields_by_olr_quintiles()
    analysis.plot_qq()
    analysis.preprocess_and_fit()
    analysis.plot_ridge()
    analysis.plot_correlation_matrix()

if __name__ == "__main__":
    main()