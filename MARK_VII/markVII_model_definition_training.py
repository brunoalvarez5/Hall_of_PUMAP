from parametric_umap import *
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
from sklearn.preprocessing import StandardScaler

#to get a 32D dataset, I will use some samples that my friend and neighbour Rob had stored
#lets load it 
import numpy as np
# Load the .npy file
data_1 = np.load("/home/bruno/ESCI/TFG/parametric_UMAP_code/MARK_VII/embeddings_1.npy")
data_2 = np.load("/home/bruno/ESCI/TFG/parametric_UMAP_code/MARK_VII/embeddings_from_collided_csv.npy") 
print('####################################################################################################################################')
print(data_2.shape)
print('####################################################################################################################################')
data = np.vstack([data_1, data_2])
# Now 'data' is a NumPy array, similar to loading a CSV
print(data)

df = pd.DataFrame(data)
print(df)


tensors = df.to_numpy()
print(tensors)

#define the class model
embedder_X = ParametricUMAP(n_neighbors=15)


#train the model
embedding_X = embedder_X.fit_transform(tensors)

#check the structure of the model
for layer in embedder_X.encoder.layers:
    print(layer.name, layer.__class__.__name__, layer.get_config())


#applying the model to the data
embedding_data_1 = embedder_X.transform(tensors)

#Now embedding_new has the processed values, so we will save it in a .csv file
#print(embedding_new)
df_result = pd.DataFrame(embedding_data_1)
df_result.to_csv("/home/bruno/ESCI/TFG/parametric_UMAP_code/MARK_VII/Mark_VII_training_results.csv", index = False)


import tensorflow as tf
import tf2onnx

#get the encoder of the model?
encoder_model = embedder_X.encoder

#make model with flexible batch size
encoder_model.build(input_shape=(None, tensors.shape[1]))


from tensorflow.keras import Model, Input

#defdine new input layer
inputs = Input(shape=(tensors.shape[1],), name="input")
#pass input through the sequential model
outputs = encoder_model(inputs)
#build a functional model
functional_encoder = Model(inputs, outputs)



#convert keras model to ONNX
spec = (tf.TensorSpec((None, tensors.shape[1]), tf.float32, name="input"),)

output_path = "/home/bruno/ESCI/TFG/parametric_UMAP_code/MARK_VII/Mark_VII_model.onnx"

model_proto, _ = tf2onnx.convert.from_keras(
    functional_encoder, 
    input_signature=spec, 
    output_path=output_path, 
    opset=13,
)

