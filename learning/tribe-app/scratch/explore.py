import numpy as np

preds = np.load("preds.npy")

for t in range(preds.shape[0]):
    print(t, round(float(preds[t].mean()), 4))