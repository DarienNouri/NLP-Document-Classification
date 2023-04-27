from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np


class FeatureExtraction:
    def __init__(self):
        pass
    
    def gen_term_matrix(self, corpus):
        tfVec = TfidfVectorizer()
        try: tdf = tfVec.fit_transform([corpus])
        except: tdf = tfVec.fit_transform(corpus)
        bow = pd.DataFrame(tdf.toarray(), columns = tfVec.get_feature_names_out())
        return bow 
    
    def get_tf_idf(self, corpus):
        tfVec = TfidfVectorizer()
        try: tdf = tfVec.fit_transform([corpus])
        except: tdf = tfVec.fit_transform(corpus)
        bow = pd.DataFrame(tdf.toarray(), columns = tfVec.get_feature_names_out())
        return bow
    
    def extract_top_keywords(self, dtm_all, processed_tokens, folder_label):
        dtm_dict_all = dtm_all.to_dict(orient='records')[0]
        dtm_dict_1 = {}
        key_list = [*dtm_dict_all]
        for token in processed_tokens:
            if token in key_list:
                dtm_dict_1[token] = round(dtm_dict_all[token],4)
                
        folder_keywords_sorted = sorted(dtm_dict_1.items(), key=lambda x: x[1], reverse=True)
        # only keep keywords where tf-idf > .01
        folder_keywords_sorted = [tup for tup in folder_keywords_sorted if tup[1] > .01]
        folder_keywords_sorted = [[folder_label]+list(tup) for tup in folder_keywords_sorted]
        folder_keywords_sorted = list(map(list, folder_keywords_sorted))
        return folder_keywords_sorted