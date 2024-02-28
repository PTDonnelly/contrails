from kerastuner import HyperModel
from kerastuner.tuners import Hyperband
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, BatchNormalization
from keras.optimizers import Adam, SGD, RMSprop
from keras.regularizers import l1_l2
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from pathlib import Path
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import tensorflow as tf

def get_training_and_test_data(validation_split=0.2, state_seed=42):
    # Load the dataset
    housing = fetch_california_housing(as_frame=True)
    X = housing.data
    y = housing.target

    X_shuffled, y_shuffled = shuffle(X, y, random_state=state_seed)

    # Split the dataset
    return train_test_split(X_shuffled, y_shuffled, test_size=validation_split, random_state=state_seed)

def scale_input_features(X_train, X_test):
    # Scale the features
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

def save_best_trials(tuner, X_train, X_test, y_train, y_test, number_of_trials=1, epochs=10):
    # List to hold DataFrames for each trial's history and the predicted
    all_histories = []
    all_predictions = []

    # Get the best hyperparameters for the specified number of trials
    best_hyperparameters = tuner.get_best_hyperparameters(number_of_trials=number_of_trials)
    
    for rank, hps in enumerate(best_hyperparameters, start=1):
        # Build the model with the current set of hyperparameters
        model = tuner.hypermodel.build(hps)
        
        # Train the model
        history = model.fit(X_train, y_train, epochs=epochs, validation_split=0.2, verbose=2)

        # Make predictions and add to DataFrame
        predictions = model.predict(X_test)
        predictions_df = pd.DataFrame(predictions, columns=['Predicted value'])
        predictions_df['True value'] = y_test.values
        predictions_df['Rank'] = rank
        
        # Convert the training history to a DataFrame
        history_df = pd.DataFrame(history.history)
        history_df['Epoch'] = history.epoch
        # Add a column for the trial's rank
        history_df['Rank'] = rank

        # Append this trial's history and predictions DataFrames to the list
        all_histories.append(history_df)
        all_predictions.append(predictions_df)

    # Concatenate all history DataFrames together
    combined_history_df = pd.concat(all_histories, axis=0).reset_index(drop=True)
    combined_history_df.to_csv(os.path.join(tuner.project_dir, f"performance_metrics_per_epoch_top_{number_of_trials}.csv"), sep='\t')

    # Concatenate all predictions DataFrames together
    combined_predictions_df = pd.concat(all_predictions, axis=0).reset_index(drop=True)
    combined_predictions_df.to_csv(os.path.join(tuner.project_dir, f"predicted_data_per_model_top_{number_of_trials}.csv"), sep='\t')

    return
    
def save_trial_configurations(tuner):
    # Collect trial information
    trial_details = []

    for trial in tuner.oracle.get_best_trials(num_trials=10):
        # Retrieve histories for each metric
        val_loss_history = trial.metrics.get_history('val_loss')
        val_mae_history = trial.metrics.get_history('val_mae')
        val_mse_history = trial.metrics.get_history('val_mse')
        
        # Extracting values from the metrics' histories
        val_loss_values = [entry['value'] for entry in val_loss_history]
        val_mae_values = [entry['value'] for entry in val_mae_history]
        val_mse_values = [entry['value'] for entry in val_mse_history]

        trial_info = {
            'trial_id': trial.trial_id,
            'hyperparameters': trial.hyperparameters.values,
            'score': trial.score,  # Ensure this score aligns with your primary objective
            'val_loss_history': val_loss_values,
            'val_mae_history': val_mae_values,
            'val_mse_history': val_mse_values,
        }
        trial_details.append(trial_info)

    # Convert to DataFrame for easier manipulation and visualization
    trials_df = pd.DataFrame(trial_details)
    trials_df.to_csv(os.path.join(tuner.project_dir, f"tuning_trials.csv"), sep='\t')
    print(trials_df.head())

    return

def plot_best_model(output_directory, X_train, X_test, y_train, y_test, number_of_trials):
    # Concatenate all history DataFrames together
    history_df = pd.read_csv(os.path.join(output_directory, f"performance_metrics_per_epoch_top_{number_of_trials}.csv"), sep='\t')

    # Concatenate all predictions DataFrames together
    predictions_df = pd.read_csv(os.path.join(output_directory, f"predicted_data_per_model_top_{number_of_trials}.csv"), sep='\t')
    
    print(history_df.head())

    print(predictions_df.head())

    
    # Assume `X_test` and `y_test` are defined
    predictions = predictions_df['Precitions']
    true_values = y_test.values
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
    axs[0, 1].scatter(y_test, predictions, s=2, marker='.', alpha=0.3)
    axs[0, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)  # x=y line
    axs[0, 1].plot(line_x, line_y, color='red', label='Linear Fit') # Linear fit
    axs[0, 1].set_title('Predicted vs True Values')
    axs[0, 1].set_xlabel('True Values')
    axs[0, 1].set_ylabel('Predicted Values')
    axs[0, 1].axis('square')  # Force square aspect ratio

    plt.tight_layout()
    plt.savefig(os.path.join(output_directory, 'performance.png'))

    return


class MyHyperModel(HyperModel):
    def __init__(self, input_shape):
        self.input_shape = input_shape
    
    def build(self, hp):
        model = Sequential()
        
        # Define the input layer
        model.add(Input(shape=(self.input_shape,)))
        
        # Dynamic addition of layers
        for i in range(hp.Int('n_layers', 1, 4)):
            model.add(Dense(units=hp.Int(f'units_{i}', min_value=32, max_value=512, step=32),
                            activation=hp.Choice(f'activation_{i}', ['relu', 'tanh', 'sigmoid']),
                            kernel_regularizer=l1_l2(l1=hp.Float(f'l1_{i}', 0, 0.01, step=0.005),
                                                      l2=hp.Float(f'l2_{i}', 0, 0.01, step=0.005))))
            if hp.Boolean(f'batch_norm_{i}'):
                model.add(BatchNormalization())
            if hp.Float(f'dropout_{i}', 0, 0.5):
                model.add(Dropout(rate=hp.Float(f'dropout_rate_{i}', min_value=0.0, max_value=0.5, step=0.1)))
        
        # Output layer
        model.add(Dense(1, activation='linear'))
        
        # Optimizer selection
        optimizer_type = hp.Choice('optimizer', ['adam', 'sgd', 'rmsprop'])
        if optimizer_type == 'adam':
            optimizer = Adam(learning_rate=hp.Choice('lr_adam', [1e-2, 1e-3, 1e-4]))
        elif optimizer_type == 'sgd':
            optimizer = SGD(
                learning_rate=hp.Choice('lr_sgd', [1e-2, 1e-3, 1e-4]),
                momentum=hp.Float('momentum_sgd', 0.0, 0.9))
        elif optimizer_type == 'rmsprop':
            optimizer = RMSprop(learning_rate=hp.Choice('lr_rmsprop', [1e-2, 1e-3, 1e-4]))

        # Compile model
        model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae', 'mse'])
        
        return model

# Output directory
output_directory = 'C:\\Users\\donnelly\\Documents\\projects\\machine_learning\\models\\tuning\\housing_prices'
# Use 20% of the training data for validation
validation_split = 0.2
# Number of ranks to take from the best fits
number_of_trials = 1

# Get data and normalise
X_train, X_test, y_train, y_test = get_training_and_test_data()
X_train_scaled, X_test_scaled = scale_input_features(X_train, X_test)

# Construct the keras tuner for sampling hyperparameter grid
tuner = Hyperband(
    MyHyperModel(input_shape=X_train.shape[1]),
    objective='val_mse',
    max_epochs=10,
    directory=output_directory,
    project_name='hypermodel_test',
    overwrite=True
)

# Set up early stopping to stop training if the validation performance doesn't improve for a specified number of epochs.
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
tuner.search(X_train, y_train, epochs=10, validation_split=validation_split, callbacks=[stop_early])

# Assuming tuner, X_train, and y_train are already defined
save_best_trials(tuner, X_train_scaled, X_test_scaled, y_train, y_test)

# # Gather results into pd.DataFrame
# save_trial_configurations(tuner)

# Train on the best hyper parameters and plot epoch-wise performance
plot_best_model(tuner, X_train, X_test, y_train, y_test, number_of_trials)