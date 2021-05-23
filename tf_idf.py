import numpy as np
import pandas as pd

from math import log
from numpy.core.numeric import count_nonzero


class Tf_Idf:
    """
    In this class, only consider using English
    """
    def __init__(self):
        params = None

    def _make_bow(self, docs):   
        docs_token = [doc.split() for doc in docs]
        token_list = \
            [item for sublist in docs_token for item in sublist]
        # Make Word To Index
        self.word2index = {}
        # Make Word To Index
        for token in token_list:
            if token not in self.word2index:
                self.word2index[token] = len(self.word2index)
        # Make Bag of Words
        bow = []
        for doc in docs_token:
            tmp_bow = [0] * len(self.word2index)
            for voca in doc:
                voca_index = self.word2index.get(voca)
                tmp_bow[voca_index] += 1
            bow.append(tmp_bow)
        self.bow = np.array(bow)
    
    def run(self):
        self.tf = self.bow
        self.idf = \
            [log(len(term) / np.count_nonzero(term)) for term in np.transpose(self.tf)]
        self.tf_idf = np.array(self.tf) * np.array(self.idf)


if __name__ == '__main__':
    sentence = input("Write Sentence : ").split(",")
    tfidf = Tf_Idf()
    tfidf._make_bow(sentence)
    tfidf.run()
    word_ls = list(tfidf.word2index.keys())
    
    for i in range(len(sentence)):
        print("=========== doc {} ===========".format(i+1))
        df = pd.concat([pd.DataFrame(tfidf.tf[i], index=word_ls, columns=['TF']),
                        pd.DataFrame(tfidf.idf, index=word_ls, columns=['idf']),
                        pd.DataFrame(tfidf.tf_idf[i],  index=word_ls, columns=['TF-IDF'])],
                    axis=1)
        print(df)

