from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

def build_model(
        input_shape:int,
        learning_rate: float=0.0001,
        number_of_layers: int=3,
        nodes_first_layer: int=16,
        nodes_second_layer: int=32,
        nodes_third_layer: int=8
        ):
    model = Sequential()
    model.add(Dense(nodes_first_layer, activation='relu', input_shape=(input_shape,)))
    if number_of_layers >= 2:
        model.add(Dense(nodes_second_layer, activation='relu'))
    if number_of_layers >= 3:
        model.add(Dense(nodes_third_layer, activation='relu'))
    model.add(Dense(1, activation='linear'))

    # Using default parameters
    model.compile(optimizer=Adam(learning_rate=learning_rate), 
                loss='mean_squared_error',  # Use an appropriate loss function for your problem
                metrics=['mae', 'mse'])
    
    return model