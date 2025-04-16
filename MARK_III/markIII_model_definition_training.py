from parametric_umap import *
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
from sklearn.preprocessing import StandardScaler


#to get a 32D dataset, I will use some samples that my friend and neighbour Rob had stored
#lets load it 
import numpy as np
# Load the .npy file
data = np.load("/home/bruno/ESCI/TFG/parametric_UMAP_code/MARK_III/allen-celltypes+human-cortex+various-cortical-areas-encodings.npy") 

# Now 'data' is a NumPy array, similar to loading a CSV
print(data)

df = pd.DataFrame(data)
print(df)


tensors = df.to_numpy()
print(tensors)

#define the class model
embedder_X = ParametricUMAP(n_neighbors=15, min_dist=0.09, parametric_reconstruction = False, random_state = 42)


#train the model
embedding_X = embedder_X.fit_transform(tensors)

#check the structure of the model
for layer in embedder_X.encoder.layers:
    print(layer.name, layer.__class__.__name__, layer.get_config())


#applying the model to the data
embedding_data_1 = embedder_X.transform(tensors)

#Now embedding_new has the processed values, so we will save it in a .csv file
#print(embedding_new)
df_result = pd.DataFrame(embedding_new)
df_result.to_csv("/home/bruno/ESCI/TFG/parametric_UMAP_code/MARK_III/Mark_III_training_results.csv", index = False)


#now we are going to save the project in ONNX
path = "/home/bruno/ESCI/TFG/parametric_UMAP_code/MARK_II/Mark_III_model.onnx"

embedder_X.to_ONNX(path)
