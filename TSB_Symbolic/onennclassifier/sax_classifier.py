from ..symbolic.sax.sax import SAX
from ..util.distance import mindist,matching_distance,hist_euclidean_dist,pairwise_distance,pairwise_histogram_distance
from ..util.distance_vectorized import hamming_vectorized,symbol_vectorized,sax_mindist,euclidean_vectorized,boss_vectorized,cosine_similarity_vectorized,kl_divergence

import sys
import numpy as np
import scipy

class SAXDictionaryClassifier():
    def __init__(self,
        word_length=4,
        alphabet_size=4,
        window_size=0,
        remove_repeat_words=False,
        save_words=False,
        metric = 'mindist',
        store_words=None,
        build_histogram=True):
        
        self.word_length = word_length
        self.alphabet_size = alphabet_size
        self.window_size = window_size
        self.remove_repeat_words = remove_repeat_words
        self.save_words = save_words
        self.metric = metric
        self.store_words = store_words
        self.build_histogram = build_histogram

    def fit(self,X,y=None):

        self.train_data = X
        self.ts_len = self.train_data.shape[-1]
        self.word_length = min(self.word_length, X.shape[-1])
        self.window_size = min(self.window_size, X.shape[-1])

        # load SAX approximation method
        self.sax = SAX(
            word_length=self.word_length,
            alphabet_size=self.alphabet_size,
            window_size=self.window_size,
            remove_repeat_words=self.remove_repeat_words,
            save_words=self.save_words,
            build_histogram = self.window_size > 0 and self.build_histogram
        )

        # transform 
        self.train_bags = self.sax.transform(X)

        self.train_hist = self.sax.histogram
        self.train_words_bps = self.sax.bp_words

        self.series_length = X.shape[1]
        self.breakpoints = self.sax.breakpoints
        self.train_words_bps = self.sax.bp_words
        if self.store_words is not None:
            pass
        self._y = y

        return self 

    def predict(self,X):

        self.test_data = X
        self.predict_bags = self.sax.transform(X)
        self.predict_hist = self.sax.histogram
        self.predict_words_bps = self.sax.bp_words

        """Calculate the dist mat for 1NN
        """
        
        if self.metric in ['hist_euclidean']:
            dist_mat = euclidean_vectorized(self.predict_hist,self.train_hist)
        
        elif self.metric in ['symbolic_l1']:
            pred_X = np.squeeze(self.predict_words_bps,axis=1)
            train_X = np.squeeze(self.train_words_bps,axis=1)

            dist_mat = symbol_vectorized(pred_X,train_X)
        
        elif self.metric in ['sax_mindist']:
            pred_X = np.squeeze(self.predict_words_bps,axis=1)
            train_X = np.squeeze(self.train_words_bps,axis=1)

            breakpoints = self.breakpoints
            breakpoints = [sys.float_info.min] + breakpoints
            breakpoints = np.array(breakpoints)
            breakpoints = np.tile(breakpoints,(self.word_length,1))

            dist_mat = sax_mindist(pred_X,train_X,breakpoints,self.ts_len)

        else:
            raise Exception(f"Sorry, the {self.metric} is not currently supported.")

        self.dist_mat = dist_mat
        self.pred_histogram = self.predict_hist.astype(float)
        # print("hist shape: ", self.pred_histogram.shape)

        """1NN and Predict results
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
    
    