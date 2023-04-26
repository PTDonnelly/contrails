import numpy as np
from atmos import Retrieval

# Create a retrieval object
fouraop_config = {
    # Add your 4A/OP configuration here
}
retrieval = Retrieval(fouraop_config)

# Define input parameters for the retrieval
x_a = np.array([1.0, 1.0])  # A priori state vector
y_obs = np.array([2.0, 3.0])  # Observed measurements
s_a = np.diag([0.1, 0.1])  # A priori covariance matrix
s_y = np.diag([0.5, 0.5])  # Measurement covariance matrix

# Run the retrieval
result = retrieval.run_retrieval(x_a, y_obs, s_a, s_y)

print("Retrieved state vector:", result.x)
print("A posteriori covariance matrix:", result.s)
