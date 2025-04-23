import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import sys
import onnxruntime as ort
import numpy as np
import csv


cortex_data = pd.read_csv("/home/bruno/ESCI/TFG/parametric_UMAP_code/MARK_III/training_set_cortex_embeddings.csv")
cortexm1_data = pd.read_csv("/home/bruno/ESCI/TFG/parametric_UMAP_code/MARK_III/test_set_cortex+m1_embeddings.csv")
#allen-celltypes+human-cortex+various-cortical-areas-encodings.npy
#allen-celltypes+human-cortex+various-cortical-areas-predictions.npy
#allen-celltypes+human-cortex+m1-encodings.npy
#allen-celltypes+human-cortex+m1-predictions.npy


cortex_data_pred = np.load("/home/bruno/ESCI/TFG/parametric_UMAP_code/MARK_III/allen-celltypes+human-cortex+various-cortical-areas-predictions.npy").astype(np.float32)
cortexm1_data_pred = np.load("/home/bruno/ESCI/TFG/parametric_UMAP_code/MARK_III/allen-celltypes+human-cortex+m1-predictions.npy").astype(np.float32)

print(cortex_data_pred)
print('#######################################################')
print(cortexm1_data_pred)
print('#######################################################')
print(cortex_data_pred.shape)
print('#######################################################')
print(cortexm1_data_pred.shape)
print('#######################################################')
#label encoder de cell type a numero, los archivos terminados en "predictions" contienen el cell type, pero esta en formato de numero porque lo ha procesado un modelo para que otro modelo pueda leerlo ya que solo leen numeros no letras
#from sklearn.preprocessing import LabelEncoder ---- esto es un ejemplo

cortex_data['cell_type'] = cortex_data_pred
cortexm1_data['cell_type'] = cortexm1_data_pred

print(cortex_data.shape)
print(cortexm1_data.shape)

#convertir a categoría y luego usar los códigos
cortex_data['cell_type'] = cortex_data['cell_type']

#visualizar cortex data
plt.figure(figsize=(8, 6))
sns.scatterplot(data=cortex_data, x='Dim1', y='Dim2', hue='cell_type', palette="tab10", s=10)
plt.title("Cortex umap with cell types")
plt.legend(title='Cell Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


#visualizar cortexm1 data
plt.figure(figsize=(8, 6))
sns.scatterplot(data=cortexm1_data, x='Dim1', y='Dim2', hue='cell_type', palette='tab10', s=10)
plt.title("Cortex+m1 umap with cell types")
plt.legend(title='Cell type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()