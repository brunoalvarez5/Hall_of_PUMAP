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


#to get a 32D dataset, I will use some samples that my friend and neighbour Rob had stored
#lets load it 
import numpy as np
# Load the .npy file
data = np.load("/home/bruno/ESCI/TFG/parametric_UMAP_code/MARK_II/sample-encodings.npy")

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
embedding_new = embedder_X.transform(tensors)

#Now embedding_new has the processed values, so we will save it in a .csv file
#print(embedding_new)
df_result = pd.DataFrame(embedding_new)
df_result.to_csv("/home/bruno/ESCI/TFG/parametric_UMAP_code/MARK_II/Mark_II_results.csv", index = False)

import tensorflow as tf
import tf2onnx

#get the encoder of the model?
encoder_model = embedder_X.encoder

# Build model with flexible batch size
encoder_model.build(input_shape=(None, tensors.shape[1]))


from tensorflow.keras import Model, Input

# Define new input layer
inputs = Input(shape=(tensors.shape[1],), name="input")
# Pass input through the sequential model
outputs = encoder_model(inputs)
# Build a functional model
functional_encoder = Model(inputs, outputs)



#convert keras model to ONNX
spec = (tf.TensorSpec((None, tensors.shape[1]), tf.float32, name="input"),)

output_path = "/home/bruno/ESCI/TFG/parametric_UMAP_code/MARK_II/Mark_II_model.onnx"

model_proto, _ = tf2onnx.convert.from_keras(
    functional_encoder, 
    input_signature=spec, 
    output_path=output_path, 
    opset=13,
)

