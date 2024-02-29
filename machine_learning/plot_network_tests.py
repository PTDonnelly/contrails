import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.linear_model import LinearRegression

def plot_best_model(output_directory, number_of_trials):
    # Concatenate all history DataFrames together
    history_df = pd.read_csv(os.path.join(output_directory, f"performance_metrics_per_epoch_top_{number_of_trials}.csv"), sep='\t')

    # Concatenate all predictions DataFrames together
    predictions_df = pd.read_csv(os.path.join(output_directory, f"predicted_data_per_model_top_{number_of_trials}.csv"), sep='\t')
    
    print(history_df.head())

    print(predictions_df.head())

    
    # Assume `X_test` and `y_test` are defined
    predictions = predictions_df['Predicted value'].values
    true_values = predictions_df['True value'].values
    # Fit a linear regression model to the predictions vs. true values
    lin_reg = LinearRegression().fit(predictions.reshape(-1, 1), true_values)
    # Use the fitted model to get predicted values for a line
    line_x = np.linspace(predictions.min(), predictions.max(), 100)
    line_y = lin_reg.predict(line_x.reshape(-1, 1))
    
    # Create a figure and a set of subplots
    fig, axs = plt.subplots(2, 2, figsize=(8, 8))

    # Plot training & validation loss values
    axs[0, 0].plot(history_df['epoch'], history_df['loss'], label='Training')
    axs[0, 0].plot(history_df['epoch'], history_df['val_loss'], label='Validation')
    axs[0, 0].set_title('Model Loss over Test Epochs')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].legend(loc='upper right')

    # Leave the bottom right plot (1,0) empty
    axs[1, 1].axis('off')

    # Plot training & validation MSE
    axs[1, 0].plot(history_df['epoch'], history_df['mse'], label='Train MSE')
    axs[1, 0].plot(history_df['epoch'], history_df['val_mse'], label='Validation MSE')
    axs[1, 0].set_title('Model MSE over Test Epochs')
    axs[1, 0].set_ylabel('MSE')
    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].legend(loc='upper right')

    # Scatter plot of predicted vs true values in the top right plot
    axs[0, 1].scatter(true_values, predictions, s=2, marker='.', alpha=0.3)
    axs[0, 1].plot([true_values.min(), true_values.max()], [true_values.min(), true_values.max()], 'k--', lw=2)  # x=y line
    axs[0, 1].plot(line_x, line_y, color='red', label='Linear Fit') # Linear fit
    axs[0, 1].set_title('Predicted vs True Values')
    axs[0, 1].set_xlabel('True Values')
    axs[0, 1].set_ylabel('Predicted Values')
    axs[0, 1].axis('square')  # Force square aspect ratio

    plt.tight_layout()
    plt.savefig(os.path.join(output_directory, 'performance.png'))

    return


# Output directory
output_directory = 'C:\\Users\\donnelly\\Documents\\projects\\machine_learning\\models\\tuning\\housing_prices'

# Train on the best hyper parameters and plot epoch-wise performance
plot_best_model(number_of_trials=1)

plot