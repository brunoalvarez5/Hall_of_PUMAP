import onnxruntime as ort
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the ONNX model
path = "/home/bruno/ESCI/TFG/parametric_UMAP_code/tests_comarison/Using_source_code_from_library/pumap_model_tester.onnx"
session = ort.InferenceSession(path)

# Example data preprocessing (assuming you need to scale new data)
scaler = StandardScaler()
#X = np.random.rand(10, 32)  # Example new data

#we save X to replicate in other cases with the same vectors
import pandas as pd 
#df = pd.DataFrame(X)
#df.to_csv("data_to_replicate_tests.csv", index = False)

#now lets load it 
X_df = pd.read_csv("data_to_replicate_tests.csv")

XX = X_df.to_numpy()
#XX = XX[0]
#scaled_XX = scaler.fit_transform(XX)

scaled_XX = XX

# Run inference
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

y = session.run([output_name], {input_name: scaled_XX[0:1].astype(np.float32)})[0]
#print(scaled_XX[0])
print(y)
output_embeddings = []
i = 0
for x in range(1, len(scaled_XX)+1):
    print(x)
    
    y = session.run([output_name], {input_name: scaled_XX[i:x].astype(np.float32)})[0]
    #breakpoint()
    print(type(y))
    #output_embeddings.append(session.run([output_name], {input_name: x.astype(np.float32)})[0])
    output_embeddings.append(y[0])
    i+=1

print(len(scaled_XX))

# Print results
print("Embedding shape:", output_embeddings)
print("First few embeddings:\n", output_embeddings[:5])

df_result = pd.DataFrame(output_embeddings)
df_result.to_csv("/home/bruno/ESCI/TFG/parametric_UMAP_code/tests_comarison/Using_source_code_from_library/result_pytorch_ONNX.csv", index = False)

