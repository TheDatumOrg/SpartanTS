import sys
import math
import time
import numpy as np
import multiprocessing

from numba import prange,njit

from ..util.distance import mindist,matching_distance,hist_euclidean_dist,pairwise_distance,pairwise_histogram_distance
from ..util.distance_vectorized import symbol_vectorized,hamming_vectorized,sax_mindist,mindist_minmax,euclidean_vectorized,boss_vectorized,cosine_similarity_vectorized,kl_divergence

from TSB_Symbolic.symbolic.sfa.sfa_fast import SFAFast

class SFADictionaryClassifier:
    def __init__(
        self,
        word_length=8,
        alphabet_size=4,
        window_size=0,
        norm=False,
        binning_method="equi-depth",
        anova=False,
        variance=False,
        bigrams=False,
        skip_grams=False,
        remove_repeat_words=False,
        lower_bounding=True,
        save_words=False,
        feature_selection="none",
        max_feature_count=256,
        p_threshold=0.05,
        random_state=None,
        return_sparse=True,
        return_pandas_data_series=False,
        n_jobs=-1,
        metric = 'sfa_mindist',
        build_histogram=True
    ):
        self.words = []
        self.breakpoints = []

        # we cannot select more than window_size many letters in a word
        self.word_length = word_length

        self.alphabet_size = alphabet_size
        self.window_size = window_size
        self.norm = norm
        self.lower_bounding = lower_bounding
        

        self.remove_repeat_words = remove_repeat_words

        self.save_words = save_words

        self.binning_method = binning_method
        self.anova = anova
        self.variance = variance

        self.bigrams = bigrams
        self.skip_grams = skip_grams
        self.n_jobs = n_jobs

        self.n_instances = 0
        self.series_length = 0
        self.letter_bits = 0

        # Feature selection part
        self.feature_selection = feature_selection
        self.max_feature_count = max_feature_count
        self.feature_count = 0
        self.relevant_features = None

        # feature selection is applied based on the chi-squared test.
        self.p_threshold = p_threshold

        self.return_sparse = return_sparse
        self.return_pandas_data_series = return_pandas_data_series

        self.random_state = random_state

        self.build_histogram = build_histogram

        if self.n_jobs < 1 or self.n_jobs > multiprocessing.cpu_count():
            self.n_jobs = min(multiprocessing.cpu_count(), 5)
        else:
            self.n_jobs = min(self.n_jobs, 1)

        # print(self.n_jobs)

        self.metric = metric

        from numba import set_num_threads

        set_num_threads(min(self.n_jobs, 1))

    def fit(self,X,y=None):
        self._y = y

        # remember class labels
        self.classes_ = np.unique(y)
        self.n_classes_ = self.classes_.shape[0]
        self._class_dictionary = {}

        if self.window_size ==0:
            window_size = X.shape[1]
        else:
            window_size = self.window_size
        self.inverse_sqrt_win_size = (
            1.0 / math.sqrt(window_size) if not self.lower_bounding else 1.0
        )

        for index, class_val in enumerate(self.classes_):
            self._class_dictionary[class_val] = index

        # self.word_length = min(self.word_length, X.shape[-1], X.shape[0])
        self.word_length = min(self.word_length, X.shape[-1]-1)
        self.window_size = min(self.window_size, X.shape[-1])
        
        self.sfa = SFAFast(
            window_size=self.window_size,
            word_length=self.word_length,
            alphabet_size=self.alphabet_size,
            norm=self.norm,
            binning_method=self.binning_method,
            anova=self.anova,
            variance=self.variance,
            bigrams=self.bigrams,
            skip_grams=self.skip_grams,
            remove_repeat_words=self.remove_repeat_words,
            lower_bounding=self.lower_bounding,
            save_words=self.save_words,
            feature_selection=self.feature_selection,
            max_feature_count=self.max_feature_count,
            p_threshold=self.p_threshold,
            random_state=self.random_state,
            return_sparse=self.return_sparse,
            return_pandas_data_series=self.return_pandas_data_series,
            build_histogram = self.build_histogram,
            n_jobs=self.n_jobs
        )

        X_transform = self.sfa.fit_transform(X,y)

        self.breakpoints = self.sfa.breakpoints
        X_words = self.sfa.words

        X_words_indices = self.words_to_indices(X_words)


        self.train_words = X_words
        self.train_word_indices = X_words_indices

        if self.window_size > 0 and self.build_histogram:
            self.train_hist = X_transform.toarray()
        # print(self.train_hist[1])

        if self.n_jobs < 1 or self.n_jobs > multiprocessing.cpu_count():
            n_jobs = multiprocessing.cpu_count()
        else:
            n_jobs = self.n_jobs

        # super raises numba import exception if not available
        # so now we know we can use numba

        from numba import set_num_threads

        set_num_threads(n_jobs)

        return self

    def fit_transform(self,X,y=None):
        self._y = y

        self.word_length = min(self.word_length, X.shape[-1], X.shape[0])

        # remember class labels
        self.classes_ = np.unique(y)
        self.n_classes_ = self.classes_.shape[0]
        self._class_dictionary = {}

        for index, class_val in enumerate(self.classes_):
            self._class_dictionary[class_val] = index

        self.sfa = SFAFast(
            window_size=self.window_size,
            word_length=self.word_length,
            alphabet_size=self.alphabet_size,
            norm=self.norm,
            binning_method=self.binning_method,
            anova=self.anova,
            variance=self.variance,
            bigrams=self.bigrams,
            skip_grams=self.skip_grams,
            remove_repeat_words=self.remove_repeat_words,
            lower_bounding=self.lower_bounding,
            save_words=self.save_words,
            feature_selection=self.feature_selection,
            max_feature_count=self.max_feature_count,
            p_threshold=self.p_threshold,
            random_state=self.random_state,
            return_sparse=self.return_sparse,
            return_pandas_data_series=self.return_pandas_data_series,
            build_histogram = self.build_histogram,
            n_jobs=self.n_jobs
        )

        
        X_transform = self.sfa.fit_transform(X,y)

        self.breakpoints = self.sfa.breakpoints
        X_words = self.sfa.words
        X_words_indices = self.words_to_indices(X_words)
        # print(X_words_indices.shape)

        self.train_words = X_words
        self.train_word_indices = X_words_indices
        self.train_hist = X_transform

        return X_transform

    def predict(self,X):
        X_transform = self.sfa.transform(X)

        if self.build_histogram:
            self.predict_hist = X_transform.toarray()
            self.pred_histogram = self.predict_hist.astype(float)
            print(self.pred_histogram.shape)

        self.pred_words = self.sfa.words
        self.pred_word_indices = self.words_to_indices(self.pred_words)

        self.predict_words_bps = self.pred_word_indices

        
        if self.metric in ['hist_euclidean']:

            if not self.build_histogram:
                dist_mat = np.zeros((len(X), len(self.train_word_indices)))
            else:
                dist_mat = euclidean_vectorized(self.predict_hist,self.train_hist)

        elif self.metric in ['symbolic_l1']:
            pred_X = np.squeeze(self.pred_word_indices,axis=1)
            train_X = np.squeeze(self.train_word_indices,axis=1)
            dist_mat = symbol_vectorized(pred_X,train_X)

        elif self.metric in ['sfa_mindist']:
            pred_X = np.squeeze(self.pred_word_indices,axis=1)
            train_X = np.squeeze(self.train_word_indices,axis=1)
            breakpoints = self.sfa.mindist_breakpoints
            dist_mat = sax_mindist(pred_X,train_X,breakpoints) ** np.sqrt(2)

        else:
            raise Exception(f"Sorry, the {self.metric} is not currently supported.")

        self.dist_mat = dist_mat
        
        self.pred_dist_mat = dist_mat
        ind = np.argmin(dist_mat,axis=1)
        ind = ind.T
        pred = self._y[ind]

        return pred


    def sfa_cell(self,r,c,breakpoints):
        partial_dist = 0
        breakpoints_i = breakpoints
        if np.abs(r-c) <= 1:
            partial_dist = 0
        else:
            partial_dist = breakpoints_i[int(max(r,c) - 1)] - breakpoints_i[int(min(r,c))]
        return partial_dist


    def words_to_indices(self,X_wordslist):
        n_instances = len(X_wordslist)
        n_words = len(X_wordslist[0])
        word_bits = self.sfa.word_bits
        letter_bits = self.sfa.letter_bits

        letter = 2**letter_bits
        
        word_indices = np.zeros((n_instances,n_words,self.word_length))
        for i, instance in enumerate(X_wordslist):
            for j, word in enumerate(instance):
                word = X_wordslist[i,j]

                for k in range(self.word_length):
                    ind = word % letter
                    word_indices[i,j,k] = ind
                    word = word // letter

        return word_indices
        