import onnxruntime as ort
import numpy as np
import csv

#load ONNX model
onnx_model_path = "/home/bruno/ESCI/TFG/parametric_UMAP_code/MARK_III/Mark_III_model.onnx"  # Change this to your actual ONNX file
session = ort.InferenceSession(onnx_model_path)

#load datasets
cortex_data = np.load("/home/bruno/ESCI/TFG/parametric_UMAP_code/MARK_III/allen-celltypes+human-cortex+various-cortical-areas-encodings.npy").astype(np.float32)
cortexm1_data = np.load("/home/bruno/ESCI/TFG/parametric_UMAP_code/MARK_III/allen-celltypes+human-cortex+m1-encodings.npy").astype(np.float32)


#get name of the input layer
input_name = session.get_inputs()[0].name


embedding_cortex = []
for row in cortex_data:
    #run inference
    row = row.reshape(1, -1)  # Reshape to match the model input
    result = session.run(None, {input_name: row})[0]
    embedding_cortex.append(result)

embedding_cortexm1 = []
for row in cortexm1_data:
    #run inference
    row = row.reshape(1, -1)  # Reshape to match the model input
    result = session.run(None, {input_name: row})[0]
    embedding_cortexm1.append(result)

with open("/home/bruno/ESCI/TFG/parametric_UMAP_code/MARK_III/training_set_cortex_embeddings.csv", "w", newline="") as f:    
    writer = csv.writer(f)
    writer.writerow(["Dim1", "Dim2"])  #correct header
    for row in embedding_cortex:
        writer.writerow([float(row[0][0]), float(row[0][1])])  #assuming each row is [x, y]

with open("/home/bruno/ESCI/TFG/parametric_UMAP_code/MARK_III/test_set_cortex+m1_embeddings.csv", "w", newline="") as f:    
    writer = csv.writer(f, delimiter=',')
    writer.writerow(["Dim1", "Dim2"])  #correct header
    for row in embedding_cortexm1:
        writer.writerow([float(row[0][0]), float(row[0][1])])  #assuming each row is [x, y]
