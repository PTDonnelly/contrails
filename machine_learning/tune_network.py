# Standard library imports
import os
from typing import Dict, List, Optional
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
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.pipeline import make_pipeline
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
        self.hyperparameters: dict = self.define_hyperparameters()
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
        
        # Separate independent (X) and dependent (y) variables
        X = housing.data
        y = housing.target

        # Plot statistics overview
        plot_pairplot(housing.frame, target_variable='MedHouseVal', file_path=self.output_directory)
        plot_ridge_regression_cross_validation(X, y, file_path=self.output_directory)

        X_shuffled, y_shuffled = shuffle(X, y, random_state=state_seed)

        # Split the dataset
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X_shuffled, y_shuffled, test_size=self.validation_split, random_state=state_seed)
        return
    
    def scale_input_features(self):
        # # Apply normalistion to input features such that they vary in the range [0, 1]
        # scaler = MinMaxScaler()
        # self.X_train_scaled = scaler.fit_transform(self.X_train)
        # self.X_test_scaled = scaler.transform(self.X_test)
        # # Apply standardisation to input features such that they have a mean of zero and standard deviation of 1
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
    
    def define_hyperparameters(self):
        """Define the hyperparameter space."""
        return {
            'number_of_layers': {'min_value': 1, 'max_value': 4, 'step': 1},
            'nodes_per_layer': {'min_value': 32, 'max_value': 512, 'step': 32},
            'activation': ['relu', 'tanh', 'sigmoid'],
            'l1_regularisation': {'min_value': 0, 'max_value': 0.01, 'step': 0.005},
            'l2_regularisation': {'min_value': 0, 'max_value': 0.01, 'step': 0.005},
            'dropout_rate': {'min_value': 0.0, 'max_value': 0.5, 'step': 0.1},
            'optimizer': ['adam', 'sgd', 'rmsprop'],
            'learning_rate': [1e-2, 1e-3, 1e-4],
            'momentum': {'min_value': 0.0, 'max_value': 0.9},
        }
    
    def build(self, hp):
        """Build the model."""
        model = Sequential()
        
        # Add input layer
        self.add_input_layer(model)

        # Dynamically add hidden layers
        self.add_hidden_layers(model, hp)

        # Add output layer
        self.add_output_layer(model)
        
        # Choose and set the optimizer
        optimizer = self.choose_optimizer(hp)

        # Compile the model
        self.compile_model(model, optimizer, hp)
        
        return model
        
    def add_input_layer(self, model: object) -> None:
        """Add the input layer to the model."""
        model.add(Input(shape=(self.X_train.shape[1],)))
        return
    
    def add_hidden_layers(self, model: object, hp: object) -> None:
        """
        Add hidden layers to the model based on hyperparameters.
        
        Args:
            model: The neural network model to which the layers will be added.
            hp: Hyperparameter object for hyperparameter tuning.
        """
        for i in range(hp.Int('n_layers', **self.hyperparameters['n_layers'])):
            model.add(Dense(units=hp.Int(f'units_{i}', **self.hyperparameters['units']),
                            activation=hp.Choice(f'activation_{i}', self.hyperparameters['activation']),
                            kernel_regularizer=l1_l2(l1=hp.Float(f'l1_{i}', **self.hyperparameters['l1']),
                                                        l2=hp.Float(f'l2_{i}', **self.hyperparameters['l2']))))
            
            if hp.Boolean(f'batch_norm_{i}'):
                model.add(BatchNormalization())
            
            if hp.Float(f'dropout_{i}', **self.hyperparameters['dropout_rate']):
                model.add(Dropout(rate=hp.Float(f'dropout_rate_{i}', **self.hyperparameters['dropout_rate'])))
        return
        
    def add_output_layer(self, model: object) -> None:
        """
        Add the output layer to the model with linear activation.
        
        Args:
            model: The neural network model to which the output layer will be added.
        """
        model.add(Dense(1, activation='linear'))
        return

    def choose_optimizer(self, hp: object) -> object:
        """
        Choose the optimizer based on hyperparameters.
        
        Args:
            hp: Hyperparameter object for hyperparameter tuning.
        
        Returns:
            Optimizer object configured with the selected hyperparameters.
        """
        # Default optimizer choices
        optimizer_choices = self.hyperparameters.get('optimizer', ['adam', 'sgd', 'rmsprop'])
        optimizer_type = hp.Choice('optimizer', optimizer_choices)
        
        # Default learning rates
        learning_rate = self.hyperparameters.get('learning_rate', [1e-2, 1e-3, 1e-4])
        
        if optimizer_type == 'adam':
            return Adam(learning_rate=hp.Choice('lr', learning_rate))
        elif optimizer_type == 'sgd':
            momentum = self.hyperparameters.get('momentum', {'min_value': 0.0, 'max_value': 0.9, 'step': 0.1})
            return SGD(learning_rate=hp.Choice('lr', learning_rate), momentum=hp.Float('momentum', **momentum))
        else:  # Default case for rmsprop
            return RMSprop(learning_rate=hp.Choice('lr', learning_rate))

    def compile_model(self, model: object, optimizer: object, hp: object) -> None:
        """Compile the model."""
        metrics_list = [metric for metric in ['mae', 'mse'] if metric in hp.values]
        model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=metrics_list)
        return

        # model = Sequential()
        
        # # Define the input layer to have the same number of node as input features
        # model.add(Input(shape=(self.X_train.shape[1],)))
        
        # # Dynamic addition of layers
        # for i in range(hp.Int('n_layers', 1, 4)):
        #     model.add(Dense(units=hp.Int(f'units_{i}', min_value=32, max_value=512, step=32),
        #                     activation=hp.Choice(f'activation_{i}', ['relu', 'tanh', 'sigmoid']),
        #                     kernel_regularizer=l1_l2(l1=hp.Float(f'l1_{i}', 0, 0.01, step=0.005),
        #                                               l2=hp.Float(f'l2_{i}', 0, 0.01, step=0.005))))
        #     if hp.Boolean(f'batch_norm_{i}'):
        #         model.add(BatchNormalization())
        #     if hp.Float(f'dropout_{i}', 0, 0.5):
        #         model.add(Dropout(rate=hp.Float(f'dropout_rate_{i}', min_value=0.0, max_value=0.5, step=0.1)))
        
        # # Output layer
        # model.add(Dense(1, activation='linear'))
        
        # # Optimizer selection
        # optimizer_type = hp.Choice('optimizer', ['adam', 'sgd', 'rmsprop'])
        # if optimizer_type == 'adam':
        #     optimizer = Adam(learning_rate=hp.Choice('lr_adam', [1e-2, 1e-3, 1e-4]))
        # elif optimizer_type == 'sgd':
        #     optimizer = SGD(
        #         learning_rate=hp.Choice('lr_sgd', [1e-2, 1e-3, 1e-4]),
        #         momentum=hp.Float('momentum_sgd', 0.0, 0.9))
        # elif optimizer_type == 'rmsprop':
        #     optimizer = RMSprop(learning_rate=hp.Choice('lr_rmsprop', [1e-2, 1e-3, 1e-4]))

        # # Extract metric names for model compilation, validation metrics tracked automatically by keras
        # metrics_list = [
        #     metric["name"] for metric in self.performance_metrics
        #     if not metric["name"].startswith("val_") and metric["name"] != "loss"
        # ]
        # # Compile model
        # model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=metrics_list)
        
        # return model

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


def plot_pairplot(data: pd.DataFrame, target_variable: str, file_path: str, n_samples: int=500, columns_drop=None, n_quantiles=5, state_seed: int=42):
    """
    Generates and saves a seaborn pair plot for a randomly sampled subset of the provided DataFrame.

    Parameters:
    - data (pd.DataFrame): The DataFrame containing the data to plot.
    - target_variable (str): The name of the column in 'data' to use as the target variable for coloring.
    - file_path (str): The directory path where the plot image will be saved.
    - n_samples (int, optional): The number of samples to randomly select from 'data' for plotting. Default is 500.
    - columns_drop (list, optional): A list of column names to be excluded from the plot. Default is None.
    - n_quantiles (int, optional): The number of quantiles to use for discretizing the target variable. Default is 5.
    - state_seed (int, optional): A seed for the random number generator for reproducibility. Default is 42.

    Saves a PNG image of the pair plot to the specified file path.
    """
    rng = np.random.RandomState(state_seed)
    indices = rng.choice(data.shape[0], size=n_samples, replace=False)

    # Drop the unwanted columns if specified
    if columns_drop is not None:
        subset = data.iloc[indices].drop(columns=columns_drop)
    else:
        subset = data.iloc[indices]

    # Quantize the target variable and keep the midpoint for each interval
    subset[target_variable] = pd.qcut(subset[target_variable], n_quantiles, retbins=False)
    subset[target_variable] = subset[target_variable].apply(lambda x: x.mid)

    # Create a pair plot
    pairplot = sns.pairplot(data=subset,
                            hue=target_variable,
                            kind='scatter',
                            diag_kind='kde',
                            palette="viridis")

    plt.suptitle('Pair Plot of Sampled Data', y=1.02)  # Adjust title position

    # Finish and save the plot
    plt.tight_layout()
    plt.savefig(os.path.join(file_path, f'dataset_pariplot.png'))
    plt.close()

def plot_ridge_regression_cross_validation(data: pd.DataFrame, target: pd.DataFrame, file_path: str, alphas: np.array =np.logspace(-3, 1, num=30), n_jobs: int=2):
    """
    Performs Ridge regression with cross-validation on the provided dataset and plots the distribution of the estimated coefficients.

    Parameters:
    - data (pd.DataFrame): The DataFrame containing the predictor variables.
    - target (pd.DataFrame): The DataFrame containing the target variable.
    - file_path (str): The directory path where the plot image will be saved.
    - alphas (np.array, optional): Array of alpha values to test in Ridge regression. Default is logspace(-3, 1, num=30).
    - n_jobs (int, optional): The number of CPUs to use to do the computation. -1 means using all processors. Default is 2.

    Performs cross-validation for Ridge regression, calculates the R2 score, and saves a boxplot of the coefficients to the specified file path.
    """
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
    plt.savefig(os.path.join(file_path, f'dataset_ridge_regression_cross_validation.png'))
    plt.close()

def plot_best_results(hyper_model: HyperModelTuner) -> None:
    """
    Generates and saves plots for the best trial results based on rank. Each plot contains
    training and validation loss, training and validation MSE, and a scatter plot of
    predicted vs true values along with a linear fit for each rank of trial.

    Parameters:
    - hyper_model (HyperModelTuner): An instance of HyperModelTuner containing attributes
      such as output_directory and number_of_best_trials that dictate where to save the
      plots and how many best trial results to plot.

    This function reads the performance metrics and predicted data from CSV files stored
    in the output directory of the hyper_model, generates plots for each specified rank of
    trial, and saves each plot as a PNG file in the output directory. The plots include a
    line plot for model loss over epochs, a line plot for model MSE over epochs, and a
    scatter plot for predicted vs true values with a linear regression fit.
    """
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

            # Calculate fraction of predictions that are within one standard deviation of the truth
            differences = true_values - predictions
            sigma = np.std(differences)
            within_1_sigma = np.sum(np.abs(differences) <= sigma) / len(differences)

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
            # Add text for R^2 and fraction within 1-sigma to the plot
            axs[0, 1].text(0.95, 0.85, f'$R^2 = {r_squared:.2f}$\n$1\\sigma$ fraction = {within_1_sigma:.2f}',
                           fontsize=12, va='top', ha='right', transform=axs[0, 1].transAxes)
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

def plot_trial_configurations():
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

    # Plot distribution of hyper parameters across best trials
    plot_trial_configurations()
if __name__ == "__main__":
    main()