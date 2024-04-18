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
        self.input[categorical_feature] = pd.qcut(self.target, 5, labels=False)
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
        self.standardise_data()

def remove_outliers(X, y, columns):
    # # Define a function that replaces values more than 5 stds from the mean with the mean
    # def replace_outliers_with_mean(series):
    #     mean = np.nan#series.mean()
    #     std = series.std()
    #     cutoff = std * 2
    #     lower, upper = mean - cutoff, mean + cutoff
    #     series = np.where(np.abs(series - mean) > cutoff, mean, series)
    #     return pd.Series(series, index=X.index)

    # # Apply the function to the specified columns
    # X[columns] = X[columns].apply(replace_outliers_with_mean)

    # Initialize a filter for rows to keep
    mask = pd.Series(True, index=X.index)
    
    for column in columns:
        # Compute the mean and standard deviation for the column
        mean = X[column].mean()
        std = X[column].std()
        cutoff = std * 1

        # Update the mask to keep only data within +- cutoff
        mask &= X[column].between(mean - cutoff, mean + cutoff)
    
    # Apply the mask to X and y to filter out outlier rows
    X_filtered = X[mask]
    y_filtered = y[mask]

    return X_filtered, y_filtered

def plot_era5_fields_by_olr_quintiles(X, y, base_path, era5_fields, scaling_method):
    # Add quintile category to ERA5 fields
    data_fields = era5_fields.copy()
    data_fields.append('OLR_quintile')
    
    # Draw and flatten the axes
    _, axes = plt.subplots(4, 4, figsize=(10, 10), dpi=300)
    axes = axes.flatten()
    fontsize = 7

    # Loop through the features and axes to plot each feature's KDE on its subplot
    features = X[era5_fields].columns
    for i, feature in enumerate(features):
        sns.kdeplot(data=X,
                    x=feature,
                    hue='OLR_quintile',
                    fill=False,
                    linewidth=1, 
                    ax=axes[i],
                    palette='coolwarm',
                    legend=False)

        axes[i].set_title(feature)
        axes[i].set_xlabel("")
        if i // 4 == 3:
            axes[i].set_xlabel("Z-score")

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.savefig(os.path.join(base_path, f"olr_era5_correlation_kde_{scaling_method}.png"), bbox_inches='tight', dpi=300)

def plot_qq(X, y, base_path, era5_fields, scaling_method):
    # Add quintile category to ERA5 fields
    data_fields = era5_fields.copy()
    data_fields.append('OLR_quintile')
    
    # Draw and flatten the axes
    _, axes = plt.subplots(4, 4, figsize=(10, 10), dpi=300)
    axes = axes.flatten()
    fontsize = 7

    # Loop through the features and axes to plot each feature's KDE on its subplot
    features = X[era5_fields].columns
    for i, feature in enumerate(features):        
        qqplot(X[feature],
               line='s',
               ax=axes[i])
        
        axes[i].set_title(feature)
        axes[i].set_xlabel("")
        if i // 4 == 3:
            axes[i].set_xlabel("Theroetical Quantities")

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.savefig(os.path.join(base_path, f"olr_era5_correlation_qq_{scaling_method}.png"), bbox_inches='tight', dpi=300)

def preprocess_and_fit(X, y, base_path, era5_fields, scaling_method):
    # Define a pipeline that first imputes missing values, then scales the data, and finally fits RidgeCV
    pipeline = make_pipeline(
        RidgeCV(alphas=np.logspace(-6, 6, 13))
    )

    # Define cross-validation strategy
    cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)

    # Perform cross-validation and fit the model
    scores = cross_validate(pipeline, X[era5_fields], y, scoring='r2', cv=cv, return_estimator=True)

    # # If you need to access the RidgeCV models or their coefficients after fitting:
    # for est in scores['estimator']:
    #     print(est[-1].alpha_)  # Access the chosen alpha for RidgeCV
    #     print(est[-1].coef_)   # Access the coefficients

    # mean_r2 = np.mean(scores['test_score'])
    # std_r2 = np.std(scores['test_score'])
    # print(f"Average R2 score across all folds: {mean_r2:.3f}")
    # print(f"Standard deviation of R2 scores across all folds: {std_r2:.3f}")

    # Extract coefficients from each estimator in the cross-validation
    coefs = pd.DataFrame(
        [est[-1].coef_ for est in scores['estimator']],
        columns=era5_fields,
    )

    # Calculate the median value of coefficients for each feature
    medians = coefs.median().sort_values()

    # Reorder the dataframe columns based on the sorted median values
    coefs = coefs[medians.index]

    # Define color settings for the box plot
    color = {"whiskers": "black", "medians": "black", "caps": "black"}

    # Create the box plot with reordered features
    coefs.plot.box(vert=False, color=color, figsize=(10, 8))

    # Add a vertical line at x=0 for reference
    plt.axvline(x=0, color="black", linestyle="--")

    # Set the title and show the plot
    plt.title("Coefficients of Ridge Models via Cross-Validation")
    plt.savefig(os.path.join(base_path, f"ridge_coefficients_{scaling_method}.png"), bbox_inches='tight', dpi=300)

    return

def plot_ridge(X, y, base_path, era5_fields, scaling_method):
    alphas = np.logspace(-6, 6, 61)
    coefs = []

    for alpha in alphas:
        ridge = Ridge(alpha=alpha).fit(X[era5_fields], y)
        coefs.append(ridge.coef_)

    coefs = pd.DataFrame(coefs, index=alphas, columns=era5_fields)
    plt.figure(figsize=(10, 8), dpi=300)
    for column in coefs.columns:
        plt.plot(coefs.index, coefs[column], label=column)

    plt.xscale('log')
    plt.xlabel('Alpha')
    plt.ylabel('Coefficient')
    plt.title('Coefficient Paths')
    plt.legend()
    plt.savefig(os.path.join(base_path, f"ridge_testing_{scaling_method}.png"), bbox_inches='tight', dpi=300)

def main():
    # Usage
    base_path = 'G:/My Drive/Research/Postdoc_2_CNRS_LATMOS/data/machine_learning/'
    start_date = datetime(2018, 3, 1)
    end_date = datetime(2023, 5, 31)
    
    # Testing
    scaling_method = "standard"

    # Initialise the Dataset class with your parameters
    data = Dataset(base_path, start_date, end_date, scaling_method)

    # Access the input features and target variable
    X = data.input
    y = data.target

    columns = data.feature_names#['cc_200', 'cc_300']
    X_clean, y_clean = X, y #remove_outliers(X, y, columns)
    
    plot_era5_fields_by_olr_quintiles(X_clean, y_clean, base_path, data.feature_names, data.scaling_method)
    plot_qq(X, y, base_path, data.feature_names, data.scaling_method)
    preprocess_and_fit(X_clean, y_clean, base_path, data.feature_names, data.scaling_method)
    plot_ridge(X_clean, y_clean, base_path, data.feature_names, data.scaling_method)

if __name__ == "__main__":
    main()