from ..symbolic.sax.tfsax import TFSAX
from ..util.distance import mindist,matching_distance,hist_euclidean_dist,pairwise_distance,pairwise_histogram_distance
from ..util.distance_vectorized import symbol_vectorized, hamming_vectorized, euclidean_vectorized

import sys
import numpy as np
import scipy.stats

class TFSAXDictionaryClassifier():
    def __init__(self,
        word_length=8,
        alphabet_size=4,
        window_size=12,
        remove_repeat_words=False,
        save_words=False,
        metric = 'symboic_l1'
        ):
        
        self.word_length = word_length
        self.alphabet_size = alphabet_size
        self.window_size = window_size
        self.remove_repeat_words = remove_repeat_words
        self.save_words = save_words
        self.metric = metric

    def fit(self,X,y=None):

        self.train_data = X

        self.word_length = min(self.word_length, X.shape[-1], X.shape[0])

        self.sax = TFSAX(
            word_length=self.word_length,
            alphabet_size=self.alphabet_size,
            window_size=self.window_size,
            remove_repeat_words=self.remove_repeat_words,
            save_words=self.save_words
        )

        self.train_data = X
        self.train_bags = self.sax.transform(X)

        self.train_hist = self.sax.histogram
        self.train_words_bps = self.sax.bp_words

        self.series_length = X.shape[1]
        self.breakpoints = self.sax.breakpoints
        self.train_words_bps = self.sax.bp_words
        # self.train_dist_mat = self.pairwise_distance(self.train_words_bps)
        self._y = y

        return self 
    def predict(self,X):

        self.test_data = X
        self.predict_bags = self.sax.transform(X)
        self.predict_hist = self.sax.histogram
        self.predict_words_bps = self.sax.bp_words


        """DISTANCE MATRIX
        """
        if self.metric in ['symbolic_l1']:
            pred_X = np.squeeze(self.predict_words_bps,axis=1)
            train_X = np.squeeze(self.train_words_bps,axis=1)

            dist_mat = symbol_vectorized(pred_X,train_X)


        elif self.metric in ['tfsax_mindist']:

            dist_mat = self.sax.distance(self.train_words_bps, self.predict_words_bps, int(self.word_length/2), X.shape[1])

            self.dist_mat = dist_mat
        else:
            raise Exception(f"Sorry, the {self.metric} is not currently supported.")

        """OUTPUT DISTANCE MATRIX
        """
       
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
    