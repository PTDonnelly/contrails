import pandas as pd
import os
from datetime import datetime, timedelta
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import cross_val_predict, cross_val_score, cross_validate
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedKFold
import numpy as np

def convert_longitude(lon):
    """
    Convert longitude from -180 to 180 range to 0 to 360 range.
    """
    return lon % 360

def read_and_combine_daily_data(iasi_base_path, era5_base_path, start_date, end_date):
    combined_data = []

    # Generate date range
    current_date = start_date
    while current_date <= end_date:
        year = current_date.strftime('%Y')
        month = current_date.strftime('%m')
        day = current_date.strftime('%d')
        date_str = current_date.strftime('%Y-%m-%d')

        # Construct file paths
        iasi_file_path = os.path.join(iasi_base_path, year, month, day, 'spectra_and_cloud_products_binned.csv')
        era5_file_path = os.path.join(era5_base_path, f'daily_1x1_{date_str}.csv')

        if os.path.exists(iasi_file_path) and os.path.exists(era5_file_path):
            # Read data
            iasi_df = pd.read_csv(iasi_file_path, sep='\t')
            era5_df = pd.read_csv(era5_file_path, sep='\t')

            # Convert IASI longitudes to 0-360 range
            iasi_df['Longitude'] = iasi_df['Longitude'].apply(convert_longitude)

            # Combine data based on latitude and longitude
            combined_df = pd.merge(era5_df, iasi_df, on=['Date', 'Latitude', 'Longitude'])
            combined_data.append(combined_df)

        current_date += timedelta(days=1)

    combined_df = pd.concat(combined_data, ignore_index=True) if combined_data else pd.DataFrame()
    
    # Define training and testing sets
    era5_fields = [col for col in combined_df.columns if col not in ['Date', 'Latitude', 'Longitude', 'OLR_mean', 'OLR_icy', 'OLR_clear']]
    X = combined_df[era5_fields]
    y = combined_df['OLR_mean']

    return X, y, era5_fields

def standardise_data(X, y):
    # Calculate quintiles for 'OLR_mean'
    X = X.copy()
    X['OLR_quintile'] = pd.qcut(y, 5, labels=False)

    # Define a pipeline that first imputes missing values, then scales the data, and finally fits RidgeCV
    pipeline = make_pipeline(
        SimpleImputer(strategy='mean'),  # Fill NaN values with the mean of each column
        StandardScaler()
    )
    # Transform, convert back to DataFrame
    X_scaled = pipeline.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)
    return X_scaled_df

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
        cutoff = std * 0.5

        # print(column, mean, std)

        # Update the mask to keep only data within +- cutoff
        mask &= X[column].between(mean - cutoff, mean + cutoff)
    
    # Apply the mask to X and y to filter out outlier rows
    X_filtered = X[mask]
    y_filtered = y[mask]

    return X_filtered, y_filtered

def plot_era5_fields_by_olr_quintiles(X, y, base_path, era5_fields):
    # Add quintile category to ERA5 fields
    data_fields = era5_fields.copy()
    data_fields.append('OLR_quintile')

    # # Randomly sample the data set
    # rng = np.random.RandomState(42)
    # indices = rng.choice(X.shape[0], size=500, replace=False)
    # subset = X.iloc[indices]

    # # Create pair plot
    # pair_plot = sns.pairplot(subset,
    #                          vars=data_fields,
    #                          hue='OLR_quintile',
    #                          palette='coolwarm')
    # plt.show()
    
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
        if i // 4 == 3:
            axes[i].set_xlabel("Z-score")

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.savefig(f"{base_path}/iasi/olr_era5_correlation_kde.png", bbox_inches='tight', dpi=300)

def preprocess_and_fit(X, y, base_path, era5_fields):
    # Define a pipeline that first imputes missing values, then scales the data, and finally fits RidgeCV
    pipeline = make_pipeline(
        SimpleImputer(strategy='mean'),  # Fill NaN values with the mean of each column
        StandardScaler(),
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
    plt.savefig(f"{base_path}/iasi/ridge_coefficients.png", bbox_inches='tight', dpi=300)

    return

def plot_ridge(X, y, base_path, era5_fields):
    alphas = np.logspace(-6, 6, 61)
    coefs = []

    # Impute and scale X before the loop since these steps do not depend on alpha
    imputer = SimpleImputer(strategy='mean').fit(X[era5_fields])
    X_imputed = imputer.transform(X[era5_fields])
    scaler = StandardScaler().fit(X_imputed)
    X_scaled = scaler.transform(X_imputed)

    for alpha in alphas:
        ridge = Ridge(alpha=alpha).fit(X_scaled, y)
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
    plt.savefig(f"{base_path}/iasi/ridge_testing.png", bbox_inches='tight', dpi=300)

def main():
    base_path = 'G:/My Drive/Research/Postdoc_2_CNRS_LATMOS/data/'
    start_date = datetime(2018, 3, 1)
    end_date = datetime(2023, 5, 31)
    
    # Define training and testing sets
    X, y, era5_fields = read_and_combine_daily_data(f"{base_path}iasi/binned_olr/", f"{base_path}era5/daily_combined/", start_date, end_date)
    
    # Convert true data values to Z-score values
    X_scaled = standardise_data(X, y)

    # Impute outliers down to mean
    columns = ['cc_200', 'cc_300']
    X_clean, y_clean = X_scaled, y#remove_outliers(X_scaled, y, columns)

    plot_era5_fields_by_olr_quintiles(X_clean, y_clean, base_path, era5_fields)
    preprocess_and_fit(X_clean, y_clean, base_path, era5_fields)
    plot_ridge(X_clean, y_clean, base_path, era5_fields)

if __name__ == "__main__":
    main()