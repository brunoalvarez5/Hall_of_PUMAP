from parametric_umap import *
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
from sklearn.preprocessing import StandardScaler


#load data
#df_joint = pd.read_csv("Hippie_joint_embeddings.csv")

#remove index column and extract features + labels
#X_joint = df_joint.iloc[:, 1:-1].values #numerical embeddings

#labels_joint = df_joint["label"].values


#to get a 32D dataset
#lets load it 
df = pd.read_csv("./data_to_replicate_tests.csv")

tensors = df.to_numpy()
#print(tensors)

#define the class model
embedder_X = ParametricUMAP(n_neighbors=15, min_dist=0.09, parametric_reconstruction = False, random_state = 42)


#train the model
embedding_X = embedder_X.fit_transform(tensors)

#check the structure of the model
for layer in embedder_X.encoder.layers:
    print(layer.name, layer.__class__.__name__, layer.get_config())
##



#applying the model to the data
embedding_new = embedder_X.transform(tensors)

#Now embedding_new has the processed values, so we will save it in a .csv file
#print(embedding_new)
df_result = pd.DataFrame(embedding_new)
df_result.to_csv("/home/bruno/ESCI/TFG/parametric_UMAP_code/tests_comarison/Using_source_code_from_library/result_pytorch_and_TF_natively.csv", index = False)


#now we are going to save the project in ONNX
path = "/home/bruno/ESCI/TFG/parametric_UMAP_code/tests_comarison/Using_source_code_from_library/pumap_model_tester.onnx"

embedder_X.to_ONNX(path)
