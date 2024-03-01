import commentjson
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline

from tune_network import HyperModelTuner

def get_config(file_path):
    with open(file_path, 'r') as file:
        # Access the parameters directly as attributes of the class. 
        config = commentjson.load(file)
    return config

def plot_categorical_hyperparameters(df, categorical_hyperparameters):
    # Determine the number of rows/columns needed for the subplot grid
    n = len(categorical_hyperparameters)
    cols = 3  # Number of columns in the grid
    rows = n // cols + (n % cols > 0)  # Calculate rows needed

    fig, axs = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
    axs = axs.flatten()  # Flatten to iterate easily if it's a 2D array

    for i, param in enumerate(categorical_hyperparameters):
        if param in df.columns:
            sns.countplot(data=df, x=param, ax=axs[i])
            axs[i].set_title(f'Frequency of {param} choices')
            axs[i].set_xlabel(param)
            axs[i].set_ylabel('Count')
            axs[i].tick_params(axis='x', rotation=45)  # Rotate labels if they overlap
        else:
            axs[i].set_visible(False)  # Hide unused subplots

    plt.tight_layout()
    plt.show()

def plot_pairplot_for_continuous_hyperparameters(dataset: pd.DataFrame, continuous_hyperparameters: list, file_path: str):
    """
    Generates and saves a seaborn pair plot for a randomly sampled subset of the provided DataFrame.
    
    Saves a PNG image of the pair plot to the specified file path.
    """
    # Filter data to contain only columns in continuous_parameters
    dataset = dataset[continuous_hyperparameters]
    
    # Initialize the StandardScaler
    scaler = StandardScaler()

    # Standardize the continuous variables
    standardized_dataset = scaler.fit_transform(dataset)
    standardized_df = pd.DataFrame(standardized_dataset, columns=continuous_hyperparameters)

    # Create a pair plot of the standardized data
    pairplot = sns.pairplot(data=standardized_df,
                            kind='kde',
                            diag_kind='kde')
    
    # Finish and save the plot
    plt.tight_layout()
    plt.savefig(os.path.join(file_path, f'trial_configurations_pairplot.png'), bbox_inches='tight')
    plt.close()

def plot_cross_validation(dataset: pd.DataFrame, continuous_hyperparameters: list, file_path: str, alphas: np.array =np.logspace(-3, 1, num=30), n_jobs: int=2):
    """
    Performs cross-validation for Ridge regression, calculates the R2 score, and saves a boxplot of the coefficients to the specified file path.
    """
    # Filter data to contain only columns in continuous_parameters
    dataset = dataset[continuous_hyperparameters].dropna()

    # Separate independent (data) and dependent (target) variables
    data = dataset.drop(columns=['Score'])
    target = dataset['Score']

    # Create a Ridge regression model with standard scaling and cross-validation
    model = make_pipeline(StandardScaler(), RidgeCV(alphas=alphas))
    
    # Perform cross-validation
    cv_results = cross_validate(
        model,
        data,
        target,
        return_estimator=True,
        n_jobs=n_jobs
    )
    
    # Calculate and print R2 score
    score = cv_results["test_score"]
    print(f"R2 score: {score.mean():.3f} ± {score.std():.3f}")
    
    # Extract coefficients from Ridge models
    coefs = pd.DataFrame(
        [est[-1].coef_ for est in cv_results["estimator"]],
        columns=data.columns,
    )
    
    # Plot boxplot for coefficients
    color = {"whiskers": "black", "medians": "black", "caps": "black"}
    coefs.plot.box(vert=False, color=color)
    plt.axvline(x=0, color="black", linestyle="--")
    plt.title("Coefficients of Ridge models\n via cross-validation")
    
    # Finish and save the plot
    plt.savefig(os.path.join(file_path, f'trial_configurations_cross_validation.png'), bbox_inches='tight')
    plt.close()


def plot_trial_configurations(hyper_model):
    # Assume hyper_model.output_directory and hyper_model.number_of_best_trials are defined
    # Read in best trial configurations
    trials_df = pd.read_csv(os.path.join(hyper_model.output_directory, f"trial_configurations_top_{hyper_model.number_of_best_trials}.csv"), sep='\t')
    
    # continuous_hyperparameters = ['Score', 'loss', 'mae', 'mse', 'val_loss', 'val_mae', 'val_mse', 'dropout_0', 'learning_rate', 'dropout_rate_0', 'momentum', 'dropout_1', 'dropout_rate_1', 'dropout_2', 'dropout_rate_2', 'dropout_3', 'dropout_rate_3']
    continuous_hyperparameters = ['Score', 'learning_rate', 'dropout_rate_0', 'momentum', 'dropout_rate_1', 'dropout_rate_2', 'dropout_rate_3']
    categorical_hyperparameters = ['n_layers', 'units_0', 'activation_0', 'l1_0', 'l2_0', 'batch_norm_0', 'optimizer_type', 'units_1', 'activation_1', 'l1_1', 'l2_1', 'batch_norm_1', 'units_2', 'activation_2', 'l1_2', 'l2_2', 'batch_norm_2', 'units_3', 'activation_3', 'l1_3', 'l2_3', 'batch_norm_3', 'tuner/epochs', 'tuner/initial_epoch', 'tuner/bracket', 'tuner/round', 'tuner/trial_id']

    # plot_pairplot_for_continuous_hyperparameters(trials_df, continuous_hyperparameters, hyper_model.output_directory)
    plot_cross_validation(trials_df, continuous_hyperparameters, hyper_model.output_directory)
    # plot_categorical_hyperparameters(trials_df, categorical_hyperparameters)





# Read global parameters from JSON configuration file
config = get_config("config.jsonc")

# Instantiate a hyper model for tuning
hyper_model = HyperModelTuner(config)

# Preprocess data: organise into DataFrames and normalise
hyper_model.get_training_and_test_data()
hyper_model.scale_input_features()



plot_trial_configurations(hyper_model)