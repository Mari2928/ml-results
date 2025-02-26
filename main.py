import ml
from pathlib import Path
import numpy as np
import os

result = ml.mse_manual
weights = ml.manual_weights

output = Path("/home/matthew.t/ml_results.txt")
output_all = Path("/home/matthew.t/all_results.npy")

# log output to be grabbed by Artemis
with open(output, 'a') as f:
    f.write(f"Weights: {weights}; MSE: {result}\n")
full_output = output.read_text()
print(full_output)

# save all results
if os.path.exists(output_all):
    all_results= np.load(output_all, allow_pickle=True).item()
else:
    all_results = {}
    all_results["weights"] = []
    all_results["mses"]= []
all_results["weights"].append(weights)
all_results["mses"].append(result)
with open(output_all, "wb") as f:
    np.save(f, all_results)
