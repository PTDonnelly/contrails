# Standard library imports
import os
from typing import Dict, List

# Related third-party imports
from keras.layers import BatchNormalization, Dense, Dropout, Input
from keras.models import Sequential
from keras.optimizers import Adam, RMSprop, SGD
from keras.regularizers import l1_l2
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
import tensorflow as tf
import keras_tuner as kt
from keras_tuner import HyperModel
from keras_tuner.tuners import Hyperband


class HyperModelTuner(HyperModel):
    def __init__(self, performance_metrics: Dict[str, str], validation_split: float, number_of_best_trials: int, max_epochs: int):
        self.performance_metrics = performance_metrics
        self.validation_split = validation_split
        self.max_epochs = max_epochs
        self.number_of_best_trials = number_of_best_trials
        self.ouput_directory: str = None
        self.project_name: str = None
        self.X_train: pd.DataFrame = None
        self.X_test: pd.DataFrame = None
        self.y_train: pd.DataFrame = None
        self.y_test: pd.DataFrame = None
        self.X_train_scaled: pd.DataFrame = None
        self.X_test_scaled: pd.DataFrame = None
    
    def get_training_and_test_data(self, validation_split: float=0.2, state_seed: int=42) -> None:
        # Load the dataset
        housing = fetch_california_housing(as_frame=True)
        X = housing.data
        y = housing.target

        X_shuffled, y_shuffled = shuffle(X, y, random_state=state_seed)

        # Split the dataset
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X_shuffled, y_shuffled, test_size=validation_split, random_state=state_seed)
        return
    
    def scale_input_features(self):
        # Apply normalistion to input features such that they vary in the range [0, 1]
        scaler = MinMaxScaler()
        self.X_train_scaled = scaler.fit_transform(self.X_train)
        self.X_test_scaled = scaler.transform(self.X_test)
        return

    def build(self, hp):
        model = Sequential()
        
        # Define the input layer to have the same number of node as input features
        model.add(Input(shape=(self.X_train.shape[1],)))
        
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


def run_tuner(hyper_model: HyperModelTuner) -> object:
    # Construct the keras tuner for sampling hyperparameter grid
    tuner = Hyperband(
        hyper_model,
        objective=hyper_model.get_tuner_objectives(),
        max_epochs=hyper_model.max_epochs,
        directory=hyper_model.ouput_directory,
        project_name=hyper_model.project_name,
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

        print(type(rank))

        # Train a new model on these hyperparameters
        model = tuner.hypermodel.build(hps)
        history = model.fit(hyper_model.X_train, hyper_model.y_train, epochs=hyper_model.max_epochs, validation_split=hyper_model.validation_split, verbose=2)
        
        # Gather predictions and epoch-wise performance metrics for best trials
        predictions_df, history_df = gather_model_details(model, rank, hyper_model.X_test, hyper_model.y_test, history)
        
        # Gather trial configurations for best trials
        trial = gather_trial(rank, trial_id, hps, history, hyper_model.performance_metrics)

        # Aggreate results for writing to CSV
        all_histories.append(history_df)
        all_predictions.append(predictions_df)
        all_trials.append(trial)

    # Saving detailed metrics and predictions
    save_aggregated_metrics(all_histories, all_predictions, all_trials, hyper_model.ouput_directory, hyper_model.number_of_best_trials)

    return


def main():
    # Fraction of data kept aside for testing, and the fraction of training data kept aside for validation during training
    validation_split = 0.2
    # Number of ranks to take from the best fits (e.g. 10 = Top 10 best fits)
    number_of_best_trials = 10
    # Maximum number of epochs for an individual model
    max_epochs = 10
    # Define performance metrics for the modelling and tuning process
    performance_metrics = [
        {"name": "val_loss", "direction": "min"},
        {"name": "val_mae", "direction": "min"},
        {"name": "val_mse", "direction": "min"},
        {"name": "loss", "direction": "min"},
        {"name": "mae", "direction": "min"},
        {"name": "mse", "direction": "min"}
    ]

    # Instantiate a hyper model for tuning
    hyper_model = HyperModelTuner(performance_metrics, validation_split, max_epochs, number_of_best_trials)
    hyper_model.ouput_directory = 'C:\\Users\\donnelly\\Documents\\projects\\machine_learning\\models\\tuning\\housing_prices'
    hyper_model.project_name = "hypermodel_test"

    # Preprocess data: organise into DataFrames and normalise
    hyper_model.get_training_and_test_data()
    hyper_model.scale_input_features()

    # Build and run the hyper model tuner
    tuner = run_tuner(hyper_model)

    # Extract general view of all best trials as well as per-trial predictions and performance
    save_best_trials(hyper_model, tuner)

if __name__ == "__main__":
    main()