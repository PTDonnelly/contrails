import commentjson
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression

from tune_network import HyperModelTuner

def get_config(file_path):
    with open(file_path, 'r') as file:
        # Access the parameters directly as attributes of the class. 
        config = commentjson.load(file)
    return config

def plot_best_results(hyper_model: HyperModelTuner) -> None:
    # Read in best trial results
    history_df = pd.read_csv(os.path.join(hyper_model.output_directory, f"performance_metrics_per_epoch_top_{hyper_model.number_of_best_trials}.csv"), sep='\t')
    predictions_df = pd.read_csv(os.path.join(hyper_model.output_directory, f"predicted_data_per_model_top_{hyper_model.number_of_best_trials}.csv"), sep='\t')
    
    # Loop through each rank
    for rank in range(1, hyper_model.number_of_best_trials + 1):
        current_history_df = history_df[history_df['Rank'] == rank]
        current_predictions_df = predictions_df[predictions_df['Rank'] == rank]

        # Extract predicted and true values
        predictions = current_predictions_df['Predicted value'].values
        true_values = current_predictions_df['True value'].values
        # Fit a linear regression model to the predictions vs. true values
        lin_reg = LinearRegression().fit(predictions.reshape(-1, 1), true_values)
        line_x = np.linspace(predictions.min(), predictions.max(), 100)
        line_y = lin_reg.predict(line_x.reshape(-1, 1))

        # Create plots similar to the above with current_history_df and predictions
        fig, axs = plt.subplots(2, 2, figsize=(8, 8))

        # Plot training & validation loss values
        axs[0, 0].plot(current_history_df['Epoch'], current_history_df['loss'], label='Training')
        axs[0, 0].plot(current_history_df['Epoch'], current_history_df['val_loss'], label='Validation')
        axs[0, 0].set_title('Model Loss over Test Epochs')
        axs[0, 0].set_ylabel('Loss')
        axs[0, 0].set_xlabel('Epoch')
        axs[0, 0].legend(loc='upper right')

        # Leave the bottom right plot (1,0) empty
        axs[1, 1].axis('off')

        # Plot training & validation MSE
        axs[1, 0].plot(current_history_df['Epoch'], current_history_df['mse'], label='Train MSE')
        axs[1, 0].plot(current_history_df['Epoch'], current_history_df['val_mse'], label='Validation MSE')
        axs[1, 0].set_title('Model MSE over Test Epochs')
        axs[1, 0].set_ylabel('MSE')
        axs[1, 0].set_xlabel('Epoch')
        axs[1, 0].legend(loc='upper right')

        # Scatter plot of predicted vs true values in the top right plot
        axs[0, 1].scatter(true_values, predictions, s=2, marker='.', alpha=0.3)
        axs[0, 1].plot([true_values.min(), true_values.max()], [true_values.min(), true_values.max()], 'k--', lw=2)  # x=y line
        axs[0, 1].plot(line_x, line_y, color='red', label='Linear Fit')  # Linear fit
        axs[0, 1].set_title('Predicted vs True Values')
        axs[0, 1].set_xlabel('True Values')
        axs[0, 1].set_ylabel('Predicted Values')
        axs[0, 1].axis('square')  # Force square aspect ratio

        plt.tight_layout()
        plt.savefig(os.path.join(hyper_model.output_directory, f'performance_and_predictions_rank_{rank}.png'))
        plt.close()  # Close the plot to free memory

def plot_continuous_hyperparameters(df, continuous_hyperparameters):
    for param in continuous_hyperparameters:
        if param in df.columns:
            sns.histplot(data=df, x=param, kde=True)
            plt.title(f'Distribution of {param}')
            plt.xlabel(param)
            plt.ylabel('Frequency')
            plt.show()

def plot_categorical_hyperparameters(df, categorical_hyperparameters):
    for param in categorical_hyperparameters:
        if param in df.columns:
            sns.countplot(data=df, x=param)
            plt.title(f'Frequency of {param} choices')
            plt.xlabel(param)
            plt.ylabel('Count')
            plt.xticks(rotation=45)  # Rotate labels if they overlap
            plt.show()

def plot_hyperparameters(output_directory, number_of_best_trials):
    # Read in best trial configurations
    trials_df = pd.read_csv(os.path.join(output_directory, f"trial_configurations_top_{number_of_best_trials}.csv"), sep='\t')
    
    continuous_hyperparameters = ['val_loss', 'val_mae', 'val_mse', 'loss', 'mae', 'mse', 'dropout_0', 'lr_adam', 'dropout_rate_0', 'dropout_1', 'dropout_rate_1', 'dropout_rate_2', 'lr_sgd', 'momentum_sgd', 'lr_rmsprop', 'dropout_3']
    categorical_hyperparameters = ['n_layers', 'units_0', 'activation_0', 'batch_norm_0', 'optimizer', 'units_1', 'activation_1', 'batch_norm_1', 'units_2', 'activation_2', 'batch_norm_2', 'units_3', 'activation_3', 'batch_norm_3']

    plot_continuous_hyperparameters(trials_df, continuous_hyperparameters)
    plot_categorical_hyperparameters(trials_df, categorical_hyperparameters)

# Read global parameters from JSON configuration file
config = get_config("config.jsonc")

# Instantiate a hyper model for tuning
hyper_model = HyperModelTuner(config)

# Preprocess data: organise into DataFrames and normalise
hyper_model.get_training_and_test_data()
hyper_model.scale_input_features()

# Plot epoch-wise performance and predictions for the best trials
plot_best_results(hyper_model)

# plot_hyperparameters(output_directory, number_of_best_trials)