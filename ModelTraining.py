import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import SpectralClustering
from sklearn import metrics

class ModelTraining:
    def __init__(self):
        pass
    
    def get_cosine_sim(self, term_matrix):
        A = term_matrix
        similarity = np.dot(A, A.T)
        square_mag = np.diag(similarity)
        inv_square_mag = 1 / square_mag
        inv_square_mag[np.isinf(inv_square_mag)] = 0
        inv_mag = np.sqrt(inv_square_mag)
        cosine = similarity * inv_mag
        cosine = cosine.T * inv_mag
        cosine_similarities_df = pd.DataFrame(cosine)
        cosine_similarities_df.style.set_caption('Cosine Similarity Matrix')
        return cosine_similarities_df
    
    def get_euclidean_distance_sim(self, term_matrix):
        m, n = term_matrix.shape
        distance_matrix = np.zeros((m, m))
        for i in range(m):
            for j in range(i+1, m):
                distance = np.sqrt(np.sum((term_matrix[i] - term_matrix[j])**2))
                distance_matrix[i, j] = distance_matrix[j, i] = distance
        return distance_matrix
        

    def fit_pca(self, data, n_components=None):
        if n_components is None:
            pca = PCA(n_components=n_components)
            pca = pca.fit_transform(data)
        else:
            pca = PCA(n_components=n_components)
            pca = pca.fit_transform(data)
        return pca

    
    def fit_kmeans_cosine(self, cosine_similarities, n_clusters=3):
        kmeans = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', assign_labels='discretize', random_state=0)
        kmeans.fit_predict(cosine_similarities)
        return kmeans
    
    def fit_kmeans_euclidean(self, euclidean_distances, n_clusters=3):
        kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit_predict(euclidean_distances)
        return kmeans


    def fit_kmeans(self, data, n_clusters=3):
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(data)
        return kmeans
    
 
    # create a function that fits a kmeans ++ algorithm 
    def fit_kmeans_pp(self, data, n_clusters=3):
        kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(data)
        return kmeans
        
        
    def get_model_scores(self, pred_labels, actual_labels):
        accuracy = metrics.accuracy_score(actual_labels, pred_labels)
        precision = metrics.precision_score(actual_labels, pred_labels, average='weighted')
        recall = metrics.recall_score(actual_labels, pred_labels, average='weighted')
        f1_score = metrics.f1_score(actual_labels, pred_labels, average='weighted')
        # round scores to 2 decimal places
        
        print(f'Accuracy: {accuracy:.3f} Precision: {precision:.3f} Recall: {recall:.3f} F1-Score: {f1_score:.3f}')
        
    def fix_labels(self, labels):
        for i in labels:
            first = [x for x in labels if x == 0]
            second = [x for x in labels if x == 1]
            third = [x for x in labels if x == 2]
        return first+second+third