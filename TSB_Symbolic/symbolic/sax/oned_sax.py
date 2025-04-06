# Adapted from tslearn
"""1d-SAX : a Novel Symbolic Representation for Time Series"""

import sys

import numpy as np
import pandas as pd
import scipy.stats

from ..paa.paa_esax import PAAESAX
from ..paa.paa_approx import PAA
from tslearn.piecewise import OneD_SymbolicAggregateApproximation 

class OneDSAX():
    """1d-SAX : a Novel Symbolic Representation for Time Series.


    Parameters
    ----------
    word_length:         int, length of word to shorten window to (using
    PAA) (default 1)
    alphabet_size:       int, number of values to discretise each value
    to (default to 5)
    window_size:         int, size of window for sliding. Input series
    length for whole series transform (default to 12)
    remove_repeat_words: boolean, whether to use numerosity reduction (
    default False)
    save_words:          boolean, whether to use numerosity reduction (
    default False)

    return_pandas_data_series:          boolean, default = True
        set to true to return Pandas Series as a result of transform.
        setting to true reduces speed significantly but is required for
        automatic test.

    Attributes
    ----------
    words:      history = []
    """
    def __init__(
        self,
        word_length=8,
        alphabet_size=4,
        alphabet_size_slope=4,
        window_size=12,
        remove_repeat_words=False,
        save_words=False,
        return_pandas_data_series=True,
    ):
        self.word_length = word_length
        self.alphabet_size = alphabet_size
        self.alphabet_size_slope = alphabet_size_slope
        self.window_size = window_size
        self.remove_repeat_words = remove_repeat_words
        self.save_words = save_words
        self.return_pandas_data_series = return_pandas_data_series
        self.words = []
        self.bp_words = []

        self.word_length = int(word_length / 2)

        self.sax_model = None

    def transform(self, X, y=None):
        """Transform data.

        Parameters
        ----------
        X : 2d numpy array [N_instances,N_timepoints]

        Returns
        -------
        dims: Pandas data frame with first dimension in column zero
        """

        n_instances, dim = X.shape
        # print(self.word_length, self.alphabet_size)

        # X = scipy.stats.zscore(X,axis=1)
        if self.sax_model == None:

            one_d_sax = OneD_SymbolicAggregateApproximation(n_segments=self.word_length,
                                alphabet_size_avg=self.alphabet_size, 
                                alphabet_size_slope=self.alphabet_size_slope, 
                                sigma_l=None)

            self.sax_model = one_d_sax
            
            # print(self.word_length, self.alphabet_size)
            bp_words = self.sax_model.fit_transform(np.expand_dims(X, axis=2)) # np.expand_dims(X, axis=2)

        else:
            bp_words = self.sax_model.transform(np.expand_dims(X, axis=2))
            
        bp_words = bp_words.reshape(n_instances, -1)
        bp_words = np.expand_dims(bp_words, axis=1)

        self.bp_words = bp_words
        self.histogram = None
        self.breakpoints = self.sax_model.breakpoints_avg_
 
        return bp_words

    def fit_transform(self, X, y=None):

        return self.transform(X, y)