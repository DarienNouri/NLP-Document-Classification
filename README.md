# NLP-Document-Classification

This repository contains Python implementations of various text preprocessing techniques, clustering, and visualizations. The classes included are:

1. TextPreprocessing
2. TfidfVectorizer
3. ModelTraining
4. Visualizations

## TextPreprocessing

The `TextPreprocessing` class includes methods for preprocessing and tokenizing text documents. It utilizes libraries such as NLTK, spaCy, and regular expressions to perform tasks such as:

- Decontracting words (e.g., "won't" to "will not")
- Removing punctuation
- Tokenizing words
- Lemmatizing words
- Removing stop words
- Named entity recognition (NER)
- Extracting bigrams
- Combining NER and bigrams

## TfidfVectorizer

The `TfidfVectorizer` class is a custom implementation of the TF-IDF (Term Frequency-Inverse Document Frequency) vectorization technique. This class calculates the TF-IDF scores for a given set of documents and returns the top `n` terms with the highest TF-IDF scores for each document. The class also includes a method to create a term frequency matrix for the given documents.

## ModelTraining

The `ModelTraining` class includes methods for training clustering models using various similarity/distance metrics, such as:

- Cosine similarity
- Euclidean distance

The class also includes methods for fitting clustering models, such as:

- PCA (Principal Component Analysis)
- K-Means clustering
- K-Means++ clustering
- Spectral clustering

Additionally, the class provides methods for calculating evaluation metrics, such as accuracy, precision, recall, and F1-score, and for reordering labels for proper grouping.

## Visualizations

The `Visualizations` class includes methods for visualizing various aspects of the data and clustering models, such as:

- Elbow method for finding the optimal number of clusters
- Confusion matrix
- 2D and 3D PCA plots
- 2D t-SNE plot
- Explained variance plot
- Eigenvectors plot
- Top bigrams plot
- K-Means centroids plot

# Dependencies

To use these classes, you will need to install the following Python libraries:

- numpy
- pandas
- nltk
- spacy
- scikit-learn
- seaborn
- matplotlib

# Usage

To use the classes in this repository, simply import the classes and create instances of them. Then, you can call the class methods to preprocess text, vectorize documents, train clustering models, and visualize the results.

For example:

```python
from text_preprocessing import TextPreprocessing
from tfidf_vectorizer import TfidfVectorizer
from model_training import ModelTraining
from visualizations import Visualizations

# Preprocess the text documents
preprocessor = TextPreprocessing()
preprocessed_docs = preprocessor.preprocessing_pipeline(documents)

# Vectorize the documents using TF-IDF
tfidf_vectorizer = TfidfVectorizer(preprocessed_docs)
tfidf_vectors = tfidf_vectorizer.create_tf_idf_vectors()

# Train a K-Means clustering model
model_trainer = ModelTraining()
kmeans_model = model_trainer.fit_kmeans(tfidf_vectors, n_clusters=3)

# Visualize the results
visualizer = Visualizations()
visualizer.plot_kmeans_centnoids(tfidf_vectors, kmeans_model)
```

Please refer to the class methods and their docstrings for more information on their usage and expected input/output types.
