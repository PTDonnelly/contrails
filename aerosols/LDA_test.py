import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

# Fit the LDA model
lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X, y)

# Plot the original dataset and LDA results side-by-side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot the original dataset
colors = ['red', 'green', 'blue']
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    ax1.scatter(X[y == i, 0], X[y == i, 1], color=color, alpha=0.8, label=target_name)
ax1.set_title('Original Dataset')
ax1.set_xlabel('Sepal length')
ax1.set_ylabel('Sepal width')
ax1.legend()

# Plot the LDA results
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    ax2.scatter(X_lda[y == i, 0], X_lda[y == i, 1], color=color, alpha=0.8, label=target_name)
ax2.set_title('LDA Results')
ax2.set_xlabel('LDA Component 1')
ax2.set_ylabel('LDA Component 2')
ax2.legend()

plt.show()
