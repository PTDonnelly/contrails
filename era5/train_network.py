import numpy as np
import tensorflow as tf
print(tf.__version__)
import keras
print(keras.__version__)
exit()
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import MinMaxScaler

def build_model(input_shape):
    """"
    An input layer that matches the shape of your input data (input_shape should be the number of features in your dataset).
    Two hidden layers, each with 64 neurons and ReLU activation functions.
    An output layer with a single neuron (since this is a regression problem; adjust according to your specific task).
    The model uses Mean Squared Error (MSE) as the loss function and Mean Absolute Error (MAE) and MSE as metrics for evaluation.
    The RMSprop optimizer is used with a learning rate of 0.001.
    """

    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(input_shape,)))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))

    # Using default parameters
    model.compile(optimiser=Adam(), 
                loss='mean_squared_error',  # Use an appropriate loss function for your problem
                metrics=['mae', 'mse'])  # Mean Absolute Error as a metric; can be adjusted
    
    return model

housing = fetch_california_housing()
print(housing.data.shape, housing.target.shape)
print(housing.feature_names)
exit()

# Assuming data is a 2D numpy array where rows are samples and columns are features
normalizer = MinMaxScaler()
data_normalized = normalizer.fit_transform(data)

# Build the model
input_shape = X_train.shape[1]  # Assuming X_train is a 2D numpy array where rows are samples
model = build_model(input_shape)

# Show summary
model.summary()

# Train the model
history = model.fit(
    X_train, Y_train,
    epochs=100,  # Number of epochs (iterations over the entire dataset)
    validation_split=0.2,  # Use 20% of the training data for validation
    verbose=1,  # Show progress
)

# Evaluate the model
test_loss, test_mae, test_mse = model.evaluate(X_test, Y_test, verbose=2)
print(f"Test set Mean Absolute Error: {test_mae}")