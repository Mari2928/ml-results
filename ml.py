import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Generate synthetic regression data
X, y = make_regression(n_samples=100, n_features=1, noise=20, random_state=42)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Add a bias term (intercept) to the features
X_train_bias = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
X_test_bias = np.hstack([np.ones((X_test.shape[0], 1)), X_test])

# Optimize weights (manual optimization)
# weights = [bias, coefficient]
manual_weights = [0, 0]  # You can adjust these values to experiment

def predict(X, weights):
    return X.dot(weights)

# Calculate predictions using the custom weights
y_pred_manual = predict(X_test_bias, manual_weights)

# Calculate Mean Squared Error for the manually set weights
mse_manual = mean_squared_error(y_test, y_pred_manual)

