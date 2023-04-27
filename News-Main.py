#%%
import importlib
import DataImporter
importlib.reload(DataImporter)
from DataImporter import DataImporter
from TextPreprocessing import TextPreprocessing
from FeatureExtraction import FeatureExtraction
from ModelTraining import ModelTraining
from Visualizations import Visualizations
from Tf_Idf_Machine import Tf_Idf_Machine
import matplotlib.pyplot as plt
import pandas as pd
import sys
import warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.pyplot.show()
import os
import glob
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import SpectralClustering
from sklearn import metrics
print("Successfully Started Program")
print("Importing Data...")
#%%
preprocessor = TextPreprocessing()
featureExtractor = FeatureExtraction()
modelTraining = ModelTraining()
visualizations = Visualizations()


files = glob.glob('**/*.csv',recursive=True)
documents = pd.DataFrame()
for file in files:
    df = pd.read_csv(file)
    df = df.dropna()
    df = df.drop_duplicates()
    df = df.reset_index(drop=True)
    documents = documents.append(df)
keep = [
        'CNN',
        'Associated Press',
        'Newsweek',
        'Fox News']

political_orientation = {
                         'Associated Press': 2,
                         'CNN': 1,
                         'Fox News': 4,
                         'Newsweek': 3,
                         }
# if keep in column source, keep it
documents_unfiltered = documents

print("Data Preprocessing initialized. Will take a few minutes... Please Wait...")
documents = documents[documents['source'].isin(keep)]
documents = documents.reset_index(drop=True)


#%%
documents = documents.drop_duplicates(subset=['title'])
documents = documents.reset_index(drop=True)
documents = documents.sort_values(by=['source'])
documents = documents.reset_index(drop=True)
source_dict = dict(zip(documents['source'].unique(), range(len(documents['source'].unique()))))
source_dict
for i in range(len(documents)):
    documents.loc[i,'source_num'] = source_dict[documents.loc[i,'source']]
documents['source_num'] = documents['source_num'].astype(int)
documents['source_num'].value_counts()
documents = documents.groupby('source_num').head(200)
documents = documents.reset_index(drop=True)
documents['source_num'].value_counts()

#%%Tokenize and clean text

docList = documents['title'].tolist()
cleaned_tokens_all = []
cleaned_docs = []
corpus = []
for doc in docList:
    preprocessed = preprocessor.preprocess_text(doc)
    ner_results = preprocessor.combind_ner(preprocessed)
    merge_desired_bigrams = preprocessor.get_desired_bigrams(ner_results, [])
    cleaned_tokens_all.extend(merge_desired_bigrams)
    cleaned_docs.append(merge_desired_bigrams)

#%% Generate TF-IDF Matrix
corpus = []
for doc in cleaned_docs:
    corpus.append(' '.join(doc))
tf_idf_all = featureExtractor.gen_term_matrix(corpus)
tf_idf = featureExtractor.gen_term_matrix(corpus)
cosine_similarities = modelTraining.get_cosine_sim(tf_idf)



# %%f

print("News Title classification on 4 news sources keep ['CNN', 'Associated Press', 'Newsweek', 'Fox News']")

# Elbow Method for K means
from yellowbrick.cluster import KElbowVisualizer
model = KMeans()
visualizer = KElbowVisualizer(model, k=(2,30), timings= True)
visualizer.fit(cosine_similarities)  
visualizer.show() 

#%% apply PCA to reduce dimensionality and plot scree plot
pca = PCA(10)
pca.fit(cosine_similarities)

plt.plot(pca.explained_variance_)
plt.plot(np.cumsum(pca.explained_variance_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.title('Scree Plot of PCA n=10 on 4 News Sources')
plt.show()
plt.scatter(pca.components_[0], pca.components_[1])
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Scree Plot of PCA n=10 on 4 News Sources')
plt.scatter(pca.components_[0], pca.components_[1], c=documents['source_num'],cmap='rainbow')
plt.legend(documents['source'].unique())
plt.show()
#%%
# plot the first two principle components again but with using plotly this time
import plotly.express as px
fig = px.scatter(pca.components_.T, x=0, y=1, color=documents['source'], hover_data=[documents['source'], documents['title']])
# add a legned and change the colors to match the news sources
fig.update_layout(
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01
    ),
    coloraxis_showscale=False
)
fig.update_traces(marker=dict(size=8,))
fig.update_layout(
    title="PCA of 4 News Sources",
    xaxis_title="PC1",
    yaxis_title="PC2",
)
# make the plot larger
fig.update_layout(
    autosize=False,
    width=1000,
    height=1000,
)


fig.show()



# plot the third principle component in 3D with plotly using 'scatter_3d' and 'source' as color labels
import plotly.graph_objects as go
fig = go.Figure(data=[go.Scatter3d(
    x=pca.components_[0],
    y=pca.components_[1],
    z=pca.components_[2],
    mode='markers',
    marker=dict(
        size=4,
        color=documents['source_num'],                # set color to an array/list of desired values
        colorscale='Viridis',   # choose a colorscale
        opacity=0.6,  
    )
)])
# add a legned and change the colors to match the news sources
fig.update_layout(
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01,
        
    ),
    coloraxis_showscale=False
)
fig.update_layout(
    title="PCA of 3 News Sources",
    xaxis_title="PC1",
    yaxis_title="PC2",
    
)

fig.show()


#%% Run Kmeans and plot confusion matrix

kmeans = KMeans(n_clusters=4, random_state=0).fit(cosine_similarities)
kmeans.labels_
documents['kmeans'] = kmeans.labels_
documents['kmeans'].value_counts()
resultsDf = documents[['source','source_num','kmeans','title', 'sentimentScore']]

confusion_matrix(resultsDf['source_num'], resultsDf['kmeans'])
print('Confusion Matrix of Kmeans on 4 News Sources and 4 Clusters (cluster per political orientation)')
print(classification_report(resultsDf['source_num'], resultsDf['kmeans']))
sns.heatmap(confusion_matrix(resultsDf['source_num'], resultsDf['kmeans']), annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix of Kmeans on 4 News Sources and 4 Clusters (cluster per political orientation)')
plt.show()

# print a dataframe of titles with 'kmeans' values above 1
#%%
extremes = resultsDf[resultsDf['kmeans'] >= 1][['source', 'source_num', 'title','kmeans']]

print("Removed Non-Outliers. Preprocessing data again with only outliers... Please wait... ")
docList_extremes = extremes['title'].tolist()
cleaned_tokens_all = []
cleaned_docs_extremes = []
corpus_extremes = []
for doc in docList_extremes:
    preprocessed = preprocessor.preprocess_text(doc)
    ner_results = preprocessor.combind_ner(preprocessed)
    merge_desired_bigrams = preprocessor.get_desired_bigrams(ner_results, [])
    cleaned_tokens_all.extend(merge_desired_bigrams)
    cleaned_docs.append(merge_desired_bigrams)
    corpus_extremes.append(' '.join(merge_desired_bigrams))


tf_idf_extremes = featureExtractor.gen_term_matrix(corpus_extremes)
cosine_similarities_extremes = modelTraining.get_cosine_sim(tf_idf_extremes)
#%%
pca = PCA(n_components=10)
pca.fit(cosine_similarities_extremes)
# plot pca
plt.plot(np.cumsum(pca.explained_variance_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.show()
plt.scatter(pca.components_[0], pca.components_[1])
# 
plt.scatter(pca.components_[0], pca.components_[1], c=extremes['source_num'],cmap='rainbow')

plt.legend(documents['source'].unique())
# move the legend down 
plt.legend(documents['source'].unique(),bbox_to_anchor=(1, .8), loc='upper left', borderaxespad=0.)

    
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Scree Plot of PCA n=10 on Extreme values from previous itteration of Kmeans')
# add a legned and change the colors to match the news sources


kmeans = KMeans(n_clusters=4, random_state=0).fit(cosine_similarities_extremes)


documents['source'].unique()

#%% Run Kmeans and plot confusion matrix


kmeans = KMeans(n_clusters=3, random_state=0).fit(cosine_similarities_extremes)
kmeans.labels_

extremes['kmeans'] = kmeans.labels_

confusion_matrix(extremes['source_num'], extremes['kmeans'])
print('Confusion Matrix of Kmeans on 4 News Sources and 3 Clusters (cluster per political orientation)')
print(classification_report(extremes['source_num'], extremes['kmeans']))
sns.heatmap(confusion_matrix(extremes['source_num'], extremes['kmeans']), annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix of Kmeans on 4 News Sources and 4 Clusters with extremes removed')
plt.show()

#%% Run Spectral Clustering
from yellowbrick.cluster import KElbowVisualizer
model = SpectralClustering()
visualizer = KElbowVisualizer(model, k=(2,30), timings= True)
visualizer.fit(cosine_similarities_extremes)        # Fit data to visualizer
visualizer.show() 

#%%




# %% Run classification on only CNN and Fox News Cleaning
print('\n\n\n===================== Attempting Classification on CNN and Fox News Only================')
print('Confusion Matrix of Kmeans on CNN and Fox News. n=2 cluster')
files = glob.glob('**/*.csv',recursive=True)
documents = pd.DataFrame()
for file in files:
    df = pd.read_csv(file)
    df = df.dropna()
    df = df.drop_duplicates()
    df = df.reset_index(drop=True)
    documents = documents.append(df)
keep = [
        'CNN',
        'Fox News',
   ]
# if keep in column source, keep it

documents = documents[documents['source'].isin(keep)]
documents = documents.drop_duplicates(subset=['title'])
documents = documents.reset_index(drop=True)

# sort documents by 'source'
documents = documents.sort_values(by=['source'])
documents = documents.reset_index(drop=True)
# create a dictionary assigning each unique source a numrical value
source_dict = dict(zip(documents['source'].unique(), range(len(documents['source'].unique()))))
source_dict
# itterate through documents and create a new column with the numerical value of the source
for i in range(len(documents)):
    documents.loc[i,'source_num'] = source_dict[documents.loc[i,'source']]
documents['source_num'] = documents['source_num'].astype(int)
documents['source_num'].value_counts()

# keep 450 documents from each source
documents = documents.groupby('source_num').head(450)
documents = documents.reset_index(drop=True)
documents['source_num'].value_counts()

docList = documents['title'].tolist()
cleaned_tokens_all = []
cleaned_docs = []
corpus = []
for doc in docList:
    preprocessed = preprocessor.preprocess_text(doc)
    ner_results = preprocessor.combind_ner(preprocessed)
    merge_desired_bigrams = preprocessor.get_desired_bigrams(ner_results, [])
    cleaned_tokens_all.extend(merge_desired_bigrams)
    cleaned_docs.append(merge_desired_bigrams)
    corpus.append(' '.join(merge_desired_bigrams))
#%% Create TF-IDF matrix and cosine similarity matrix

tf_idf_all = featureExtractor.gen_term_matrix(corpus)
tf_idf = featureExtractor.gen_term_matrix(corpus)
cosine_similarities = modelTraining.get_cosine_sim(tf_idf)
#%%

from sklearn.feature_extraction.text import TfidfVectorizer

model = TfidfVectorizer()


X = model.fit_transform(corpus)

words = model.get_feature_names_out()

word2idx = dict(zip(words, range(len(words))))



# train a k-means model
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
kmeans.labels_

resultsDf = documents[['source','source_num','sentimentScore']]
resultsDf['kmeans'] = kmeans.labels_
resultsDf['title'] = documents['title']

# create a confusion matrix and classification report
confusion_matrix(resultsDf['source_num'], resultsDf['kmeans'])
print(classification_report(resultsDf['source_num'], resultsDf['kmeans']))
# plot the confusion matrix
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(confusion_matrix(resultsDf['source_num'], resultsDf['kmeans']), annot=True, fmt='d', ax=ax, cmap=plt.cm.Blues)
# make the color scale go from white to red



plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('CNN and Fox News K-Means Cosine Similarity')
plt.show()

#%% Run Spectral Clustering

kmeans = SpectralClustering(n_clusters=2, affinity='nearest_neighbors',
                            assign_labels='kmeans')
labels = kmeans.fit_predict(X)

resultsDf['spectral'] = labels


# apply pca to reduce dimensionality
#%%

from sklearn.decomposition import PCA
pca = PCA(2)
pca.fit(np.asarray(X.todense()))
X_pca = pca.transform(np.asarray(X.todense()))
# plot the clusters
#%%
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20,10))
fig.suptitle("CNN and Fox News K-Means Cosine Similarity PC2", size=25)
ax[0].scatter(X_pca[:, 0], X_pca[:, 1], c=resultsDf['source_num'], s=50, cmap='viridis')
ax[0].set_title('Actual', size=20)
# swap 0 and 1 in labels
labels = [1 if x == 0 else 0 for x in labels]
ax[1].scatter(X_pca[:, 0], X_pca[:, 1], c=labels, s=50, cmap='viridis')
ax[1].set_title('Predicted', size=20)
# add CNN and Fox News labels
ax[0].text(-0.1, 0.1, 'CNN', size=20)
ax[0].text(0.1, -0.1, 'Fox News', size=20)
ax[1].text(-0.1, 0.1, 'CNN', size=20)
ax[1].text(0.1, -0.1, 'Fox News', size=20)

plt.show()

# now plot the same as above side by side but using plotly
#%%
import plotly.graph_objects as go
fig = go.Figure()
fig.add_trace(go.Scatter(x=X_pca[:, 0], y=X_pca[:, 1], mode='markers', name='Actual',
                            marker=dict(color=resultsDf['source_num'], colorscale='Viridis')))
fig.add_trace(go.Scatter(x=X_pca[:, 0], y=X_pca[:, 1], mode='markers', name='Predicted',
                            marker=dict(color=labels, colorscale='Viridis')))
fig.update_layout(title="CNN and Fox News K-Means Cosine Similarity PC2", title_x=0.5, width=1000, height=500)
fig.show()

#%%
import plotly.graph_objs as go
import pandas as pd


from plotly.subplots import make_subplots


fig = make_subplots(rows=1, cols=2,
                    subplot_titles=("Actual", "Predicted"))
 
fig.add_trace(go.Scatter(x=X_pca[:, 0], y=X_pca[:, 1], mode='markers', name='Actual',
                            marker=dict(color=resultsDf['source_num'], colorscale='Viridis')), row=1, col=1)
fig.add_trace(go.Scatter(x=X_pca[:, 0], y=X_pca[:, 1], mode='markers', name='Predicted',
                            marker=dict(color=labels, colorscale='Viridis')), row=1, col=2)
fig.update_layout(title="CNN and Fox News K-Means Cosine Similarity PC2", title_x=0.5, width=1000, height=500)
fig.show()

# plot the confusion matrix
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(confusion_matrix(resultsDf['source_num'], resultsDf['spectral']), annot=True, fmt='d', ax=ax, cmap=plt.cm.Blues)
# make the color scale go from white to red
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('CNN and Fox News Spectral Clustering Cosine Similarity')
plt.show()

print("Program Finished! Peace out!")

# %%
