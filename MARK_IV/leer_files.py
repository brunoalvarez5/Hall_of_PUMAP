import pandas as pd
import numpy as np

data_1 = np.load("/home/bruno/ESCI/TFG/parametric_UMAP_code/MARK_IV/allen-celltypes+human-cortex+various-cortical-areas-encodings.npy", allow_pickle=True)
print("this is encodings:")
print(data_1)



################################################

#load binary assuming int32, but you should confirm dtype)
labels = np.fromfile('/home/bruno/ESCI/TFG/parametric_UMAP_code/MARK_IV/allen-celltypes+human-cortex+various-cortical-areas-labels.bin', dtype=np.int32)

#it's pairs of (ground truth, prediction) so reshape into (n_samples, 2)
labels = labels.reshape(-1, 2)

#split into groundtrueth and predicted
ground_truth = labels[:, 0]
predictions = labels[:, 1]

print("this is ground thruth from labels:")
print(ground_truth[:5])
print("this is predictions from labels:")
print(predictions[:5])

################################################


data_3 = np.load("/home/bruno/ESCI/TFG/parametric_UMAP_code/MARK_IV/allen-celltypes+human-cortex+various-cortical-areas-predictions.npy", allow_pickle=True)
print("this is predictions:")
print(data_3)