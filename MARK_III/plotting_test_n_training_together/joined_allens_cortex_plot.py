import onnxruntime as ort
import numpy as np
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

#loading the onnx model
path_model = "/home/bruno/ESCI/TFG/parametric_UMAP_code/MARK_III/plotting_test_n_training_together/Mark_III_model.onnx"

#load datasets
cortex_data = np.loadtxt("/home/bruno/ESCI/TFG/parametric_UMAP_code/MARK_III/plotting_test_n_training_together/training_set_cortex_embeddings.csv", delimiter=",", skiprows=1).astype(np.float32)
cortexm1_data = np.loadtxt("/home/bruno/ESCI/TFG/parametric_UMAP_code/MARK_III/plotting_test_n_training_together/test_set_cortex+m1_embeddings.csv", delimiter=",", skiprows=1).astype(np.float32)


print(cortex_data.shape)
print(cortexm1_data.shape)

joined_data = np.concatenate((cortex_data, cortexm1_data))
print(joined_data.shape)

plt.figure(figsize=(8,6))
sns.scatterplot(x=joined_data[:,0], y=joined_data[:,1], s=10)
plt.title("cortex and m1 embeddings joined")
plt.show()

# Initialize the KMeans model
kmeans = KMeans(n_clusters=5, random_state=42)

# Fit the KMeans model
kmeans.fit(joined_data)

# Retrieve the cluster labels
cluster_labels = kmeans.labels_

# Visualize the clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(x=joined_data[:, 0], y=joined_data[:, 1], hue=cluster_labels, palette="viridis", s=10)
plt.title("KMeans Clustering of Cortex and M1 Embeddings")
plt.legend(title="Cluster")
plt.show()



#NOW FOR THE FILE WITH ALL TOGETHER
#"/home/bruno/ESCI/TFG/parametric_UMAP_code/MARK_III/plotting_test_n_training_together/cortex_n_m1_processed_together_embeddings.csv"

data = np.loadtxt("/home/bruno/ESCI/TFG/parametric_UMAP_code/MARK_III/plotting_test_n_training_together/cortex_n_m1_processed_together_embeddings.csv", delimiter=",", skiprows=1).astype(np.float32)

# Initialize the KMeans model
kmeans = KMeans(n_clusters=5, random_state=42)

# Fit the KMeans model
kmeans.fit(data)

# Retrieve the cluster labels
cluster_labels = kmeans.labels_

# Visualize the clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=cluster_labels, palette="viridis", s=10)
plt.title("KMeans Clustering of Cortex and M1 Embeddings processed at the same time")
plt.legend(title="Cluster")
plt.show()
