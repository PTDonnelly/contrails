from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

def build_model(input_shape, learning_rate=0.0001):
    """"
    An input layer that matches the shape of your input data (input_shape should be the number of features in your dataset).
    Two hidden layers, each with 64 neurons and ReLU activation functions.
    An output layer with a single neuron (since this is a regression problem; adjust according to your specific task).
    The model uses Mean Squared Error (MSE) as the loss function and Mean Absolute Error (MAE) and MSE as metrics for evaluation.
    The RMSprop optimizer is used with a learning rate of 0.001.
    """

    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(input_shape,)))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(1, activation='linear'))

    # Using default parameters
    model.compile(optimizer=Adam(learning_rate=learning_rate), 
                loss='mean_squared_error',  # Use an appropriate loss function for your problem
                metrics=['mae', 'mse'])
    
    return model