class TfIdfMachine:
    def __init__(self, documents_cleaned, all_words):
        """Initialize TfIdfMachine. Takes cleaned_docs (list of tokenized documents) and all_words (Set of unique words) as
        input.
            documents_cleaned: List of cleaned documents (each document is a list of words)
            all_words: Set of unique words
        """
        self.cleaned_docs = documents_cleaned
        self.all_words = all_words
        self.word_count = self._count_dict()
        self.total_documents = len(documents_cleaned)
        self.index_dict = self._create_index_dict()

    def _create_index_dict(self):
        """Create and return a dictionary that maps each word to its index."""
        return {word: i for i, word in enumerate(self.all_words)}

    def _count_dict(self):
        """Create and return a dictionary with the count of each word in the cleaned_docs."""
        word_count = {}
        for word in self.all_words:
            word_count[word] = sum([1 for sent in self.cleaned_docs if word in sent])
        return word_count

    def _term_frequency(self, document, word):
        """Calculate and return the term frequency of 'word' in 'document'."""
        N = len(document)
        occurance = document.count(word)
        return occurance / N

    def _inverse_doc_freq(self, word):
        """Calculate and return the inverse document frequency of 'word'."""
        word_occurance = self.word_count.get(word, 0) + 1
        return np.log(self.total_documents / word_occurance)

    def _tf_idf(self, sentence):
        """Calculate and return the TF-IDF vector for a given sentence."""
        tf_idf_vec = np.zeros(len(self.all_words))

        for word in sentence:
            index = self.index_dict[word]
            tf_idf_vec[index] = self._term_frequency(sentence, word) * self._inverse_doc_freq(word)

        return tf_idf_vec

    def _create_tf_matrix(self, document):
        """Create and return a Term Frequency matrix for a given document."""
        tf_matrix = np.zeros(len(self.all_words))

        for word in document:
            index = self.index_dict[word]
            tf_matrix[index] = document.count(word) * self._inverse_doc_freq(word)

        return tf_matrix

    def create_tf_idf_vectors(self):
        """Compute and return the TF-IDF vectors for all documents in cleaned_docs."""
        return [self._tf_idf(sent) for sent in self.cleaned_docs]

    def get_top_words(self, n=20):
        """Return the top 'n' terms with the highest TF-IDF scores for each document."""
        tf_matrix_list = [self._create_tf_matrix(doc) for doc in self.cleaned_docs]
        tf_df = pd.DataFrame(tf_matrix_list, columns=self.all_words)

        tf_df['top_words'] = tf_df.apply(lambda row: row.nlargest(n).index.tolist(), axis=1)
        top_words_per_doc = tf_df['top_words'].tolist()

        return top_words_per_doc
