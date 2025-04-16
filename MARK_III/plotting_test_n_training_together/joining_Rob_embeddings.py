import onnxruntime as ort
import numpy as np
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans


cortex_data = np.load("/home/bruno/ESCI/TFG/parametric_UMAP_code/MARK_III/plotting_test_n_training_together/allen-celltypes+human-cortex+various-cortical-areas-encodings.npy").astype(np.float32)
cortexm1_data = np.load("/home/bruno/ESCI/TFG/parametric_UMAP_code/MARK_III/plotting_test_n_training_together/allen-celltypes+human-cortex+m1-encodings.npy").astype(np.float32)

print(cortex_data)
print(cortexm1_data)
print(cortex_data.shape)
print(cortexm1_data.shape)
print(cortex_data[0])

joined_data = np.concatenate((cortex_data, cortexm1_data))

print(joined_data.shape)
print(joined_data[0])

#loading the onnx model
path_model = "/home/bruno/ESCI/TFG/parametric_UMAP_code/MARK_III/plotting_test_n_training_together/Mark_III_model.onnx"
model = ort.InferenceSession(path_model)
input_name = model.get_inputs()[0].name


embedding = []
for row in joined_data:
    #run inference
    row = row.reshape(1, -1)  # Reshape to match the model input
    result = model.run(None, {input_name: row})[0]
    embedding.append(result)

print(embedding[0])
print(embedding)

with open("/home/bruno/ESCI/TFG/parametric_UMAP_code/MARK_III/plotting_test_n_training_together/cortex_n_m1_processed_together_embeddings.csv", "w", newline="") as f:    
    writer = csv.writer(f, delimiter=',')
    writer.writerow(["Dim1", "Dim2"])  #correct header
    for row in embedding:
        writer.writerow([float(row[0][0]), float(row[0][1])])  #assuming each row is [x, y]