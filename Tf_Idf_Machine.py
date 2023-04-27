import numpy as np
import pandas as pd
class Tf_Idf_Machine:
    def __init__(self, documents_cleaned, all_words):
        
        self.cleaned_docs = documents_cleaned
        self.all_words = all_words
        self.word_count = self.count_dict()
        self.total_documents = len(documents_cleaned)
     
     
    def add_index_per_word(self):
        index_dict = {} 
        i = 0
        for word in self.all_words:
            index_dict[word] = i
            i += 1
        return index_dict
        
    #Create a count dictionary
    
    def count_dict(self):
        word_count = {}
        for word in self.all_words:
            word_count[word] = 0
            for sent in self.cleaned_docs:
                if word in sent:
                    word_count[word] += 1
        return word_count
    
    


    def termfreq(self, document, word):
        N = len(document)
        occurance = len([token for token in document if token == word])
        return occurance/N
    def inverse_doc_freq(self, word):

        try:
            word_occurance = self.word_count[word] + 1
        except:
            word_occurance = 1
        return np.log(self.total_documents/word_occurance)
    def tf_idf(self, sentence):
        tf_idf_vec = np.zeros((len(self.all_words),))
        
        for word in sentence:
            tf = self.termfreq(sentence,word)
            idf = self.inverse_doc_freq(word)
            
            value = tf*idf
            #print(word)
            index_dict = self.add_index_per_word()
            tf_idf_vec[index_dict[word]] = value 
        return tf_idf_vec

    def tf_idf_words(self, document):
        tf_matrix = np.zeros((len(self.all_words),))
        
        for word in document:
            tf = self.termfreq(document,word)
            occurance = len([token for token in document if token == word])
            idf = self.inverse_doc_freq(word)
            
            value = tf*idf
            #print(word)
            index_dict = self.add_index_per_word()
            tf_matrix[index_dict[word]] = occurance
        return tf_matrix
        
        
    def create_TfIdf(self):
        vectors = []
        for sent in self.cleaned_docs:
            vec = self.tf_idf(sent)
            vectors.append(vec)
        return vectors
    
    def get_top_words(self, cleaned_docs, cleaned_tokens_all):
        tf_matrix = []
        for doc in self.cleaned_docs:
            tf = self.tf_idf_words(doc)
            tf_matrix.append(tf)
        tf_df = pd.DataFrame(tf_matrix, columns=cleaned_tokens_all)
        tf_df['top_words'] = tf_df.apply(lambda row: tf_df.columns[np.argsort(row.values)[-20:]].tolist(), axis=1)
        tf_df['top_words'] = tf_df['top_words'].apply(lambda x: ' '.join(x))
        tf_df['top_words']
        top_words_per_doc = tf_df['top_words'].tolist()
        #split the top words into a list of words
        top_words_per_doc = [doc.split() for doc in top_words_per_doc]
        return top_words_per_doc