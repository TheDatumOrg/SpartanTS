from TSB_Symbolic.symbolic.spartan import SPARTAN
from ..util.distance import pairwise_distance,pairwise_histogram_distance
from ..util.distance_vectorized import hamming_vectorized,symbol_vectorized,symbol_weighted,hamming_weighted,mindist_vectorized,sax_mindist,mindist_minmax,spartan_pca_mindist,euclidean_vectorized,boss_vectorized,cosine_similarity_vectorized,kl_divergence
import numpy as np

class SPARTANClassifier:
    def __init__(self,
                 alphabet_size=[8,8,8,4,4,2,2,2],
                 window_size=0,
                 word_length=8,
                 bit_budget = 16,
                 binning_method='equi-depth',
                 assignment_policy = 'DAA',
                 remove_repeat_words=False,
                 build_histogram = True,
                 metric = 'symbolic_l1',
                 lamda=0.5,
                 downsample = 1.0,
                 pca_solver = 'auto'
                 ):
        self.alphabet_size = alphabet_size
        self.window_size = window_size
        self.word_length = word_length
        self.binning_method = binning_method
        self.assignment_policy = assignment_policy
        self.remove_repeat_words = remove_repeat_words
        self.bit_budget = bit_budget
        self.metric = metric
        self.lamda = lamda
        self.build_histogram = build_histogram
        self.downsample = downsample
        self.pca_solver = pca_solver

        self.spartan = SPARTAN(
            alphabet_size=alphabet_size,
            window_size = window_size,
            word_length = word_length,
            binning_method = binning_method,
            assignment_policy = assignment_policy,
            remove_repeat_words=remove_repeat_words,
            lamda = lamda,
            bit_budget=self.bit_budget,
            build_histogram = window_size > 0 and self.build_histogram,
            downsample = self.downsample,
            pca_solver = self.pca_solver
        )
    def fit(self,X,y=None):
        self._y = y

        self._mean = np.mean(X)
        self._std = np.std(X)

        self._X = X
        train_X = (X - self._mean) / self._std


        # word_length = min(self.word_length, X.shape[-1], X.shape[0])
        word_length = min(self.word_length, X.shape[-1])
        
        if word_length < self.word_length:
            self.word_length = word_length
            self.spartan.word_length = self.word_length

            if isinstance(self.alphabet_size, list): # direct allocation
                alpha_size = np.mean(np.log2(self.alphabet_size))
                alpha_size = int(2**alpha_size)
                self.spartan.alphabet_size = [alpha_size for i in range(self.word_length)]
                print("shrink the alphabet size: ", self.spartan.alphabet_size)


        X_transform = self.spartan.fit_transform(train_X)
        
        self.pca_repr = self.spartan.pca_repr
        if self.window_size == 0:
            self.train_words = np.expand_dims(X_transform,axis=1)
        else:
            self.train_words = X_transform
            self.train_histogram = self.spartan.pred_histogram
        self.evcr = self.spartan.pca.explained_variance_ratio_
        return self

    def predict(self,X):
        
        pred_X = (X - self._mean) / self._std
        X_transform = self.spartan.transform(pred_X)
        if self.window_size > 0:
            self.pred_histogram = self.spartan.pred_histogram
            # print("hist shape: ", self.pred_histogram.shape)
        if self.window_size ==0 or self.window_size == X.shape[1]:
            self.pred_words = np.expand_dims(X_transform,axis=1)
        else:
            self.pred_words = X_transform
        # self.pred_words = self.pred_words.astype(np.uint32)

        self.predict_words_bps = self.pred_words
        
        if self.metric in ['hist_euclidean']:
            dist_mat =  euclidean_vectorized(self.pred_histogram,self.train_histogram)
        elif self.metric in ['symbolic_l1']:
            pred_X = np.squeeze(self.pred_words,axis=1)
            train_X = np.squeeze(self.train_words,axis=1)
            dist_mat = symbol_vectorized(pred_X,train_X)
        elif self.metric in ['pca_mindist']:
            pred_X = np.squeeze(self.pred_words,axis=1)
            train_X = np.squeeze(self.train_words,axis=1)
            breakpoints = self.spartan.mindist_breakpoints
            dist_mat = spartan_pca_mindist(pred_X,train_X,breakpoints)
        else:
            raise Exception(f"Sorry, the {self.metric} is not currently supported.")


        self.dist_mat = dist_mat
        ind = np.argmin(dist_mat,axis=1)
        ind = ind.T
        pred = self._y[ind]

        return pred