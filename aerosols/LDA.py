import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import confusion_matrix, accuracy_score

# Load your data
data_dir = "C:\\Users\\padra\\Documents\\Research\\projects\\contrails\\4aop\\outputs\\"
# The last row is assumed to be the cloud category
data = pd.read_csv(f"{data_dir}spectra_radiance.csv").T

# The first row (now column after transpose) contains wavelength information and can be dropped for the analysis
# The last column is the cloud category
X = data.iloc[1:, :-1]  # All columns except the first and last
y = data.iloc[1:, -1]   # All columns, only the last row (now column)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Perform LDA
lda = LDA()
X_train_lda = lda.fit_transform(X_train, y_train)
X_test_lda = lda.transform(X_test)

# Evaluate the model
y_pred = lda.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Calculate the covariance matrix
cov_matrix = np.cov(X_train_lda.T)
print("Covariance Matrix:\n", cov_matrix)
