Making use of the PUMAP github repo, Parametric UMAP provides a scalable solution with a neural network approach, it embeds high dimensional data into meaningful low dimensional spaces. The pipeline is adaptable to various setups with different improvements in the pipeline from mark I to mark III.

Pipeline description
All Mark models use a Parametric UMAP neural network that has been built with TensorFlow/Keras. The architecture follows the following structure:

Input Layer: Flattens the input tensors

Hidden Layers: Three Dense layers with 100 units each and ReLU activation

Output Layer: Embedding layer typically set to two dimensions


Exporting Models

Models are exported to ONNX format to enable efficient inference across diverse computational environments.

Embeddings Generation

Embeddings are generated and saved in cvs format through inference scripts (get_embeddings.py). With the resulting embeddings. Afterwards this embeddings are used to asses the quality of the model, by clustering either with the ground thruth or KMeans and using metrics such as Silhouette Score, Calinski-Harabasz Index, Davies-Bouldin Score to determine the quality.
