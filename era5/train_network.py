import matplotlib.pyplot as plt
from keras import metrics
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

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

# Use 20% of the training data for validation
validation_split = 0.2

# Load the dataset
housing = fetch_california_housing(as_frame=True)
X = housing.data
y = housing.target

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=validation_split)

# Scale the features
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build the model
input_shape = X_test_scaled.shape[1]  # Assuming X_train is a 2D numpy array where rows are samples
model = build_model(input_shape)

# Show summary
model.summary()

# Train the model
history = model.fit(
    X_train_scaled, y_train,
    epochs=10,
    validation_split=validation_split,
    verbose=2
)

# Evaluate the model
test_loss, test_mse, test_mae = model.evaluate(X_test_scaled, y_test, verbose=2)

# Make predictions
predictions = model.predict(X_test_scaled)

# Printing the first 5 predicted and real targets
for i, (truth, prediction) in enumerate(zip(y_test.values, predictions)):
    print(truth, prediction[0])
    if i == 10:
        break

# Printing the MSE
print("Mean Squared Error:", test_mse)
print("Mean Absolute Error:", test_mae)


# Create a figure and a set of subplots
fig, axs = plt.subplots(2, 2, figsize=(8, 8))

# Plot training & validation loss values
axs[0, 0].plot(history.history['loss'], label='Train')
axs[0, 0].plot(history.history['val_loss'], label='Validation')
axs[0, 0].set_title('Model Loss')
axs[0, 0].set_ylabel('Loss')
axs[0, 0].set_xlabel('Epoch')
axs[0, 0].legend(loc='upper right')

# Leave the bottom right plot (1,0) empty
axs[1, 1].axis('off')

# Plot training & validation MSE
axs[1, 0].plot(history.history['mse'], label='Train MSE')
axs[1, 0].plot(history.history['val_mse'], label='Validation MSE')
axs[1, 0].set_title('Model MSE')
axs[1, 0].set_ylabel('MSE')
axs[1, 0].set_xlabel('Epoch')
axs[1, 0].legend(loc='upper right')

# Scatter plot of predicted vs true values in the top right plot
axs[0, 1].scatter(y_test, predictions, s=2, marker='.', alpha=0.3)
axs[0, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)  # Diagonal line
axs[0, 1].set_title('Predicted vs True Values')
axs[0, 1].set_xlabel('True Values')
axs[0, 1].set_ylabel('Predicted Values')
axs[0, 1].axis('square')  # Force square aspect ratio

plt.tight_layout()
plt.show()