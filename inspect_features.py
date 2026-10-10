import numpy as np
d = np.load("features.npz", allow_pickle=True)
for k in d.files:
    print(k, d[k].shape)