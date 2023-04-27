import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class Visualizations:
    def __init__(self):
        pass

    def find_optimal_clusters(self, data):
        # Elbow method for finding the optimal number of clusters
        Sum_of_squared_distances = []
        K = range(1,10)
        for k in K:
            km = KMeans(n_clusters=k)
            km = km.fit(data)
            Sum_of_squared_distances.append(km.inertia_)
        sns.lineplot(x=K, y=Sum_of_squared_distances)
        plt.xlabel('k')
        plt.ylabel('Sum_of_squared_distances')
        plt.title('Elbow Method For Optimal k')
        plt.show()

    def plot_confusion_matrix(self, predicted_labels, actual_labels, labels,  title, subplot):
        # Confusion matrix subplot
        cf_matrix = confusion_matrix(predicted_labels, predicted_labels)
        plt.rcParams.update({'font.size': 14})
        disp = ConfusionMatrixDisplay(cf_matrix, display_labels=['C1', 'C4', 'C7'])
        disp.plot(ax=subplot, xticks_rotation=45,  cmap="YlGnBu", colorbar=True, values_format='g')
        disp.ax_.set_title(title, fontsize=20)
        disp.ax_.tick_params(axis='both', which='major', labelsize=15)
        disp.ax_.set_ylabel('Actual', fontsize=14)
        disp.ax_.set_xlabel('Predicted', fontsize=14) 
        disp.im_.colorbar.remove()

    def plot_2d_pca(self, data, labels, centers, title, subplot):
        # 2D PCA subplot
        pca = PCA(n_components=2)
        pca.fit(data)
        transformed = pca.transform(data)
        colors = ['#DF2020', '#81DF20', '#2095DF']
        subplot.scatter(transformed[:,0], transformed[:,1], c=labels, cmap='Set1', s=100)       
        subplot.set_title(title, fontsize=20)
        subplot.set_xlabel('PC1', fontsize=18, labelpad=7)
        subplot.set_ylabel('PC2', fontsize=18)

    def plot_3d_pca(self, data, labels, title, subplot):
        # 3D PCA subplot
        pca = PCA(n_components=3)
        pca.fit(data)
        transformed = pca.transform(data)
        subplot.scatter(transformed[:,0], transformed[:,1], transformed[:,2], c=labels, cmap='Set1', s=100)
        subplot.set_title(title, fontsize=20)
        subplot.tick_params(axis='both', which='major', labelsize=15, pad=10)

    def plot_2d_tsne(self, tdf, title):
        # 2D t-SNE plot
        arr = tdf.to_numpy()
        pca = PCA(n_components=2)
        pca.fit(arr)
        transformed = pca.transform(arr)
        sns.scatterplot(x=transformed[:,0], y=transformed[:,1], hue= [2] * 8 + [0] * 8 + [1] * 8)
        plt.title(title)
        plt.legend()
        plt.xlabel('PC1', fontsize=18)
        plt.ylabel('PC2', fontsize=18)
        plt.tight_layout()
        plt.show()

    def plot_explained_var(self, data, title, subplot, n_components=8):
        # Explained variance plot
        pca = PCA(n_components=len(data)-2)
        pca.fit(data)
        subplot.plot(pca.explained_variance_ratio_)
        subplot.set_title(title, fontsize=20, pad=10)
        subplot.set_xlabel('PC', fontsize=15)
        subplot.set_ylabel('Square Distance', fontsize=15)
        subplot.axvline(x=2, color='r', linestyle='-')
        subplot.set_xticks(np.arange(0, len(data)-2, 4))
        subplot.set_ylim([0, .175])
        subplot.set_yticks(np.arange(0, .175, 0.025))
        subplot.axhline(y=0.5, color='r', linestyle='-')
        subplot.tick_params(axis='both', which='major', labelsize=20)
        subplot.title.set_size(20)
        subplot.margins(0.02)

    def plot_eigenvectors(self, data, title, subplot, n_components=20):
        # Eigenvectors plot
        pca = PCA(n_components)
        pca.fit(data)
        subplot.plot(pca.singular_values_)
        subplot.set_title(title, fontsize=20, pad=10)
        subplot.set_xlabel('PC', fontsize=15, labelpad=10)
        subplot.set_ylabel('Eigenvalues', fontsize=15)
        subplot.set_xticks(np.arange(0, 24, 4))
        subplot.tick_params(axis='both', which='major', labelsize=20)
        subplot.set_ylim([0.8, 2.1])
        subplot.margins(0.2)
        subplot.axhline(y=1, color='r', linestyle='-')

    def plot_most_bigrams(self, top_bigrams, top_bigrams_frequencies):
        # Top bigrams plot
        plt.figure(figsize=(20,10))
        plt.bar(range(25), top_bigrams_frequencies[:25])
        plt.xticks(range(25), top_bigrams[:25], rotation=45)
        plt.title('Top 25 Bigrams')
        plt.show()

    def plot_kmeans_centnoids(self, data, kmeans, title='K-Means Clustering',xlabel='PC 1', ylabel='PC 2'):
        # K-Means centroids plot
        pca = PCA(n_components=2)
        data = pca.fit_transform(data)
        kmeans_labels = kmeans.labels_
        centers = kmeans.cluster_centers_
        pca = PCA(n_components=2)
        centers = pca.fit_transform(centers.reshape(3, -1))
        plt.figure(figsize=(20,10))
        plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
        plt.scatter(data[:,0], data[:,1], c=kmeans_labels, cmap='viridis')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()

    def plot_kmeans_centnoids_subplot(self, data, kmeans, subplot, title='K-Means Clustering',xlabel='PC 1', ylabel='PC 2'):
        # K-Means centroids subplot
        pca = PCA(n_components=2)
        data = pca.fit_transform(data)
        kmeans_labels = kmeans.labels_
        centers = kmeans.cluster_centers_
        pca = PCA(n_components=2)
        centers = pca.fit_transform(centers.reshape(3, -1))
        subplot.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
        subplot.scatter(data[:,0], data[:,1], c=kmeans_labels, cmap='viridis')
        subplot.set_title(title, fontsize=20)
        subplot.set_xlabel(xlabel, fontsize=18)
        subplot.set_ylabel(ylabel, fontsize=18)
        subplot.tick_params(axis='both', which='major', labelsize=15)
        subplot.margins(0.02)

    def plot_kmeans_centnoids(self, data, kmeans, title='K-Means Clustering',xlabel='PC 1', ylabel='PC 2'):
        
        pca = PCA(n_components=2)
        data = pca.fit_transform(data)
        
        kmeans_labels = kmeans.labels_
        centers = kmeans.cluster_centers_
        pca = PCA(n_components=2)
        centers = pca.fit_transform(centers.reshape(3, -1))
        # plot the centers
        # make plot size 20x10
        plt.figure(figsize=(20,10))
        plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
        plt.scatter(data[:,0], data[:,1], c=kmeans_labels, cmap='viridis')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()
        
    # create plot_kmeans_centnoids function that takes a subplot as a parameter
    def plot_kmeans_centnoids_subplot(self, data, kmeans, subplot, title='K-Means Clustering',xlabel='PC 1', ylabel='PC 2'):
        
        pca = PCA(n_components=2)
        data = pca.fit_transform(data)
        
        kmeans_labels = kmeans.labels_
        centers = kmeans.cluster_centers_
        pca = PCA(n_components=2)
        centers = pca.fit_transform(centers.reshape(3, -1))
        # plot the centers
        subplot.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
        subplot.scatter(data[:,0], data[:,1], c=kmeans_labels, cmap='viridis')
        subplot.set_title(title, fontsize=20)
        subplot.set_xlabel(xlabel, fontsize=18)
        subplot.set_ylabel(ylabel, fontsize=18)
        subplot.tick_params(axis='both', which='major', labelsize=15)
        subplot.margins(0.02)
    
    
        
            
        
        