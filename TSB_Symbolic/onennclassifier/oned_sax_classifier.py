from ..symbolic.sax.oned_sax import OneDSAX
from ..util.distance import mindist,matching_distance,hist_euclidean_dist,pairwise_distance,pairwise_histogram_distance
from ..util.distance_vectorized import symbol_vectorized, hamming_vectorized

import sys
import numpy as np

class OneDSAXDictionaryClassifier():
    def __init__(self,
        word_length=8,
        alphabet_size=4,
        window_size=12,
        remove_repeat_words=False,
        save_words=False,
        metric = 'symbolic_l1'):
        
        self.word_length = word_length
        self.alphabet_size = alphabet_size
        self.window_size = window_size
        self.remove_repeat_words = remove_repeat_words
        self.save_words = save_words
        self.metric = metric

    def fit(self,X,y=None):

        self.word_length = min(self.word_length, X.shape[-1])
        
        self.sax = OneDSAX(
            word_length=self.word_length,
            alphabet_size=self.alphabet_size,
            window_size=self.window_size,
            remove_repeat_words=self.remove_repeat_words,
            save_words=self.save_words
        )

        self.train_bags = self.sax.transform(X)

        self.train_hist = self.sax.histogram
        self.train_words_bps = self.sax.bp_words

        self.series_length = X.shape[1]
        self.breakpoints = self.sax.breakpoints
        self.train_words_bps = self.sax.bp_words
        # self.train_dist_mat = self.pairwise_distance(self.train_words_bps)
        self._y = y
        print(self.train_words_bps.shape)

        return self 
    def predict(self,X):
        self.predict_bags = self.sax.transform(X)
        self.predict_hist = self.sax.histogram
        self.predict_words_bps = self.sax.bp_words

        print(self.predict_words_bps.shape)

    

        if self.metric == 'symbolic_l1':
            pred_X = np.squeeze(self.predict_words_bps,axis=1)
            train_X = np.squeeze(self.train_words_bps,axis=1)

            dist_mat = symbol_vectorized(pred_X,train_X)

        elif self.metric == '1dsax_mindist':

            train_X = np.squeeze(self.train_words_bps, axis=1).reshape(self.train_words_bps.shape[0], -1, 2)
            pred_X = np.squeeze(self.predict_words_bps, axis=1).reshape(self.predict_words_bps.shape[0], -1, 2)
            
            dist_mat = np.zeros((pred_X.shape[0], train_X.shape[0]))

            for i in range(pred_X.shape[0]):
                for j in range(train_X.shape[0]):

                    dist_mat[i][j] = self.sax.sax_model.distance_1d_sax(pred_X[i], train_X[j])
        else:
            raise Exception(f"Sorry, the {self.metric} is not currently supported.")


        self.pred_dist_mat = dist_mat
        ind = np.argmin(dist_mat,axis=1)

        ind = ind.T
        pred = self._y[ind]

        return pred


    def transform(self,X):
        return self.sax.transform(X)

    def fit_transform(self,X,y=None):
        self.fit(X,y)
        X_transform = self.transform(X)

        return X_transform
    
  