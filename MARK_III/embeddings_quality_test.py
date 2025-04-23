import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import sys

#coger el path al archivo y guardarlo en variable path
print("Enter path to the file you wanna use to test the model: ")
path = sys.stdin.readline().strip()

#guardando el nombre del archivo en name, 
name = path.split('/')[-1]
print(f'You entered: {name}')

# Cargar los embeddings del conjunto de entrenamiento desde el CSV
train_embeddings = pd.read_csv(path)
#/home/bruno/ESCI/TFG/parametric_UMAP_code/MARK_III/training_set_cortex_embeddings.csv
#/home/bruno/ESCI/TFG/parametric_UMAP_code/MARK_III/test_set_cortex+m1_embeddings.csv




#visualizar los embeddings (opcional)
plt.figure(figsize=(8,6))
sns.scatterplot(x='Dim1', y='Dim2', data=train_embeddings, s=10)
plt.title(f'Embeddings 2D - {name}')
plt.xlabel("Dim1")
plt.ylabel("Dim2")
plt.show()


#aplicar clustering con K-Means
num_clusters = 5  # Ajusta este número según lo que consideres adecuado
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
train_embeddings['Cluster'] = kmeans.fit_predict(train_embeddings[['Dim1', 'Dim2']])


# Visualizar los clusters
plt.figure(figsize=(8,6))
sns.scatterplot(x='Dim1', y='Dim2', hue='Cluster', palette='viridis', data=train_embeddings, s=10)
plt.title(f'Clustering K-Means - {name}')
plt.xlabel("Dim1")
plt.ylabel("Dim2")
plt.show()


from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# Considera solo los puntos que fueron asignados a un clúster válido (por ejemplo, en DBSCAN algunos puntos pueden ser ruido con etiqueta -1)
# Usamos K-Means para este ejemplo:
X = train_embeddings[['Dim1', 'Dim2']].values
labels = train_embeddings['Cluster'].values
silhouette = silhouette_score(X, labels)
calinski = calinski_harabasz_score(X, labels)
davies = davies_bouldin_score(X, labels)


print(f'for the file {path} the metrics are:')
print("Silhouette Score:", silhouette)
print("Calinski-Harabasz Score:", calinski)
print("Davies-Bouldin Score:", davies)