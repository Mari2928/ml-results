import os
import ml
from pathlib import Path
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

def predict(X, weights):
    return X.dot(weights)

# Calculate predictions using the custom weights
y_pred_manual = predict(X_test_bias, ml.manual_weights)

# Calculate Mean Squared Error for the manually set weights
mse_manual = mean_squared_error(y_test, y_pred_manual)

output = Path("/home/matthew.t/ml_results.txt")
output_all = Path("/home/matthew.t/all_results.npy")

# log output to be grabbed by Artemis
with open(output, 'a') as f:
    f.write(f"Weights: {ml.manual_weights}; MSE: {mse_manual}\n")
full_output = output.read_text()
print(full_output)

# save all results
if os.path.exists(output_all):
    all_results= np.load(output_all, allow_pickle=True).item()
else:
    all_results = {}
    all_results["weights"] = []
    all_results["mses"]= []
all_results["weights"].append(ml.manual_weights)
all_results["mses"].append(mse_manual)
with open(output_all, "wb") as f:
    np.save(f, all_results)
