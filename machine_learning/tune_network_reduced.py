# Standard library imports
import os
from typing import Dict, List
import commentjson
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression

# Related third-party imports
from keras.layers import BatchNormalization, Dense, Dropout, Input
from keras.models import Sequential
from keras.optimizers import Adam, RMSprop, SGD
from keras.regularizers import l1_l2
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils import shuffle
import tensorflow as tf
import keras_tuner as kt
from keras_tuner import HyperModel
from keras_tuner.tuners import Hyperband


class HyperModelTuner(HyperModel):
    def __init__(self, config: dict):
        self.performance_metrics = self.define_performance_metrics()
        self.validation_split = config["validation_split"]
        self.max_epochs = config["max_epochs"]
        self.number_of_best_trials = config["number_of_best_trials"]
        self.project_directory: str = config["project_directory"]
        self.test_directory: str = f"{config['test_directory']}_{self.max_epochs}"
        self.output_directory: str = self.set_output_directory()
        self.X_train: pd.DataFrame = None
        self.X_test: pd.DataFrame = None
        self.y_train: pd.DataFrame = None
        self.y_test: pd.DataFrame = None
        self.X_train_scaled: pd.DataFrame = None
        self.X_test_scaled: pd.DataFrame = None
    
    def set_output_directory(self) -> None:
        return os.path.join(self.project_directory, self.test_directory)
    
    def get_training_and_test_data(self, state_seed: int=42) -> None:
        # Load the dataset
        housing = fetch_california_housing(as_frame=True)

        X = housing.data#.drop(columns=["Longitude", "Latitude"])
        y = housing.target

        # Combine X and y into a single DataFrame for easier filtering
        data = housing.frame

        # Filter out rows where the target (y) is 5 or more
        filtered_data = data[data['MedHouseVal'] < 5]

        # Separate the filtered data back into X and y
        X_filtered = filtered_data.drop(columns=['MedHouseVal'])
        y_filtered = filtered_data['MedHouseVal']

        X_shuffled, y_shuffled = shuffle(X_filtered, y_filtered, random_state=state_seed)

        # Split the dataset
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X_shuffled, y_shuffled, test_size=self.validation_split, random_state=state_seed)
        return
    
    def scale_input_features(self):
        # # Apply normalistion to input features such that they vary in the range [0, 1]
        # scaler = MinMaxScaler()
        # self.X_train_scaled = scaler.fit_transform(self.X_train)
        # self.X_test_scaled = scaler.transform(self.X_test)

        # Initialize the StandardScaler
        scaler = StandardScaler()
        self.X_train_scaled = scaler.fit_transform(self.X_train)
        self.X_test_scaled = scaler.transform(self.X_test)
        return

    def define_performance_metrics(self):
        return [
            {"name": "loss", "direction": "min"},
            {"name": "mae", "direction": "min"},
            {"name": "mse", "direction": "min"},
            {"name": "val_loss", "direction": "min"},
            {"name": "val_mae", "direction": "min"},
            {"name": "val_mse", "direction": "min"}
        ]
    
    def build(self, hp):
        model = Sequential()
        
        # Define the input layer to have the same number of node as input features
        model.add(Input(shape=(self.X_train.shape[1],)))
        
        # Dynamic addition of layers
        for i in range(hp.Int('n_layers', 1, 3)):
            model.add(Dense(units=hp.Int(f'units_{i}', min_value=16, max_value=32, step=8),
                            activation='relu'))

            # model.add(BatchNormalization())
            # model.add(Dropout(rate=hp.Float(f'dropout_rate_{i}', **self.hyperparameters['dropout_rate'])))
    
        # Output layer
        model.add(Dense(1, activation='linear'))
        
        # Optimizer selection
        # optimizer = Adam(learning_rate=hp.Choice('lr_adam', [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]))
        optimizer = Adam(learning_rate=1e-4)

        # Extract metric names for model compilation, validation metrics tracked automatically by keras
        metrics_list = [
            metric["name"] for metric in self.performance_metrics
            if not metric["name"].startswith("val_") and metric["name"] != "loss"
        ]
        # Compile model
        model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=metrics_list)
        
        return model

    def get_tuner_objectives(self):
        objectives = [
            kt.Objective(metric["name"], direction=metric["direction"]) for metric in self.performance_metrics
        ]
        return objectives

def get_config(file_path):
    with open(file_path, 'r') as file:
        # Access the parameters directly as attributes of the class. 
        config = commentjson.load(file)
    return config

def run_tuner(hyper_model: HyperModelTuner) -> object:
    # Construct the keras tuner for sampling hyperparameter grid
    tuner = Hyperband(
        hyper_model,
        objective=hyper_model.get_tuner_objectives(),
        max_epochs=hyper_model.max_epochs,
        directory=hyper_model.project_directory,
        project_name=hyper_model.test_directory,
        overwrite=True
    )
    
    # Set up early stopping to stop training if the validation performance doesn't improve for a specified number of epochs.
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    tuner.search(hyper_model.X_train, hyper_model.y_train, validation_split=hyper_model.validation_split, callbacks=[stop_early])
    return tuner

def gather_model_details(model: object, rank: int, X_test: pd.DataFrame, y_test: pd.DataFrame, history: object) -> List[pd.DataFrame]:
    predictions = model.predict(X_test)
    predictions_df = pd.DataFrame(predictions, columns=['Predicted value'])
    predictions_df['True value'] = y_test.reset_index(drop=True)
    predictions_df['Rank'] = rank
    
    history_df = pd.DataFrame(history.history)
    history_df['Epoch'] = history.epoch
    history_df['Rank'] = rank

    return predictions_df, history_df

def gather_trial(rank: int, trial_id: str, hps: object, history: object, metrics: Dict[str, str]) -> dict:
    # Extract basic trial information
    trial = {
        'Rank': rank,
        'Trial ID': trial_id,  # Directly use the passed trial_id
        'Score': history.history['val_loss'][-1],  # Example default metric
    }

    # Add performance metrics
    for metric in metrics:
        metric_name = metric["name"]
        if metric_name in history.history:
            # Determine if the metric should be minimised or maximised
            if metric["direction"] == "min":
                trial[metric_name] = min(history.history[metric_name])
            elif metric["direction"] == "max":
                trial[metric_name] = max(history.history[metric_name])
        else:
            trial[metric_name] = None
    
    trial.update(hps.values)
    
    return trial

def save_aggregated_metrics(all_histories: list, all_predictions: list, all_trials: list, output_directory: str, number_of_best_trials: int) -> None:
    combined_history_df = pd.concat(all_histories, ignore_index=True)
    combined_predictions_df = pd.concat(all_predictions, ignore_index=True)
    trials_df = pd.DataFrame(all_trials)
    
    combined_history_df.to_csv(os.path.join(output_directory, f"performance_metrics_per_epoch_top_{number_of_best_trials}.csv"), sep='\t', index=False)
    combined_predictions_df.to_csv(os.path.join(output_directory, f"predicted_data_per_model_top_{number_of_best_trials}.csv"), sep='\t', index=False)
    trials_df.to_csv(os.path.join(output_directory, f"trial_configurations_top_{number_of_best_trials}.csv"), sep='\t', index=False)
    return

def save_best_trials(hyper_model: HyperModelTuner, tuner: Hyperband) -> None:
    all_histories = []
    all_predictions = []
    all_trials = []

    best_trials = tuner.oracle.get_best_trials(hyper_model.number_of_best_trials)

    for rank, trial in enumerate(best_trials, start=1):
        # Get trial number and hyperparameters of each trial
        trial_id = trial.trial_id
        hps = trial.hyperparameters

        # Train a new model on these hyperparameters
        model = tuner.hypermodel.build(hps)
        history = model.fit(hyper_model.X_train, hyper_model.y_train,
                            epochs=hyper_model.max_epochs,
                            validation_split=hyper_model.validation_split,
                            verbose=2)
        
        # Gather predictions and epoch-wise performance metrics for best trials
        predictions_df, history_df = gather_model_details(model, rank, hyper_model.X_test, hyper_model.y_test, history)
        
        # Gather trial configurations for best trials
        trial = gather_trial(rank, trial_id, hps, history, hyper_model.performance_metrics)

        # Aggreate results for writing to CSV
        all_histories.append(history_df)
        all_predictions.append(predictions_df)
        all_trials.append(trial)

    # Saving detailed metrics and predictions
    save_aggregated_metrics(all_histories, all_predictions, all_trials, hyper_model.output_directory, hyper_model.number_of_best_trials)

    return

def plot_best_results(hyper_model: HyperModelTuner) -> None:
    # Read in best trial results
    history_df = pd.read_csv(os.path.join(hyper_model.output_directory, f"performance_metrics_per_epoch_top_{hyper_model.number_of_best_trials}.csv"), sep='\t')
    predictions_df = pd.read_csv(os.path.join(hyper_model.output_directory, f"predicted_data_per_model_top_{hyper_model.number_of_best_trials}.csv"), sep='\t')
    
    # Loop through each rank
    for rank in range(1, hyper_model.number_of_best_trials + 1):
        current_history_df = history_df[history_df['Rank'] == rank]
        current_predictions_df = predictions_df[predictions_df['Rank'] == rank]

        if not current_predictions_df.empty and len(current_predictions_df) > 1:

            # Extract predicted and true values
            predictions = current_predictions_df['Predicted value'].values
            true_values = current_predictions_df['True value'].values
            # Fit a linear regression model to the predictions vs. true values
            lin_reg = LinearRegression().fit(predictions.reshape(-1, 1), true_values)
            r_squared = lin_reg.score(predictions.reshape(-1, 1), true_values)
            line_x = np.linspace(predictions.min(), predictions.max(), 100)
            line_y = lin_reg.predict(line_x.reshape(-1, 1))

            # Create plots similar to the above with current_history_df and predictions
            fig, axs = plt.subplots(2, 2, figsize=(8, 8))

            # Plot training & validation loss values
            axs[0, 0].plot(current_history_df['Epoch'], current_history_df['loss'], label='Training')
            axs[0, 0].plot(current_history_df['Epoch'], current_history_df['val_loss'], label='Validation')
            axs[0, 0].set_title('Model Loss over Test Epochs')
            axs[0, 0].set_ylabel('Loss')
            axs[0, 0].set_yscale('log')
            axs[0, 0].set_xlabel('Epoch')
            axs[0, 0].legend(loc='upper right')

            # Leave the bottom right plot (1,0) empty
            axs[1, 1].axis('off')

            # Plot training & validation MSE
            axs[1, 0].plot(current_history_df['Epoch'], current_history_df['mse'], label='Train MSE')
            axs[1, 0].plot(current_history_df['Epoch'], current_history_df['val_mse'], label='Validation MSE')
            axs[1, 0].set_title('Model MSE over Test Epochs')
            axs[1, 0].set_ylabel('MSE')
            axs[1, 0].set_yscale('log')
            axs[1, 0].set_xlabel('Epoch')
            axs[1, 0].legend(loc='upper right')

            # Scatter plot of predicted vs true values in the top right plot
            axs[0, 1].scatter(true_values, predictions, s=2, marker='.', alpha=0.3)
            axs[0, 1].plot([true_values.min(), true_values.max()], [true_values.min(), true_values.max()], 'k--', lw=2)  # x=y line
            axs[0, 1].plot(line_x, line_y, color='red', label='Linear Fit')  # Linear fit
            axs[0, 1].text(0.95, 0.95, f'$R^2 = {r_squared:.2f}$', fontsize=12, va='top', ha='right', transform=axs[0, 1].transAxes)
            axs[0, 1].set_title('Predicted vs True Values')
            axs[0, 1].set_xlabel('True Values')
            axs[0, 1].set_ylabel('Predicted Values')
            axs[0, 1].axis('square')  # Force square aspect ratio

            plt.tight_layout()
            plt.savefig(os.path.join(hyper_model.output_directory, f'performance_and_predictions_rank_{rank}.png'))
            plt.close()  # Close the plot to free memory
        else:
            print(f"Not enough data for plotting results. Number of samples: {len(predictions_df)}")
    return

def main():
    # Read global parameters from JSON configuration file
    config = get_config("config.jsonc")

    # Instantiate a hyper model for tuning
    hyper_model = HyperModelTuner(config)

    # Preprocess data: organise into DataFrames and normalise
    hyper_model.get_training_and_test_data()
    hyper_model.scale_input_features()

    # Build and run the hyper model tuner
    tuner = run_tuner(hyper_model)

    # Extract general view of all best trials as well as per-trial predictions and performance
    save_best_trials(hyper_model, tuner)

    # Plot epoch-wise performance and predictions for the best trials
    plot_best_results(hyper_model)

if __name__ == "__main__":
    main()