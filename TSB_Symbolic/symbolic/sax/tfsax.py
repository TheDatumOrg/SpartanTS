#adapted from sktime:https://github.com/sktime/sktime/blob/v0.25.0/sktime/transformations/panel/dictionary_based/_sax.py
""""""

import sys

import numpy as np
import pandas as pd
import scipy.stats

from ..paa.paa_tfsax import PAATFSAX
from ..paa.paa_approx import PAA

class TFSAX():
    """.

    as described in
    Jessica Lin, Eamonn Keogh, Li Wei and Stefano Lonardi,
    "Experiencing SAX: a novel symbolic representation of time series"
    Data Mining and Knowledge Discovery, 15(2):107-144
    Overview: for each series:
        run a sliding window across the series
        for each window
            shorten the series with PAA (Piecewise Approximate Aggregation)
            discretise the shortened series into fixed bins
            form a word from these discrete values
    by default SAX produces a single word per series (window_size=0).
    SAX returns a pandas data frame where column 0 is the histogram (sparse
    pd.series)
    of each series.

    Parameters
    ----------
    word_length:         int, length of word to shorten window to (using
    PAA) (default 8)
    alphabet_size:       int, number of values to discretise each value
    to (default to 4)
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
        alphabet_size_angle=4,
        window_size=0,
        remove_repeat_words=False,
        save_words=False,
        return_pandas_data_series=True,
        variable_segment=False
    ):
        self.word_length = word_length
        self.alphabet_size = alphabet_size
        self.alphabet_size_angle = alphabet_size_angle
        self.window_size = window_size
        self.remove_repeat_words = remove_repeat_words
        self.save_words = save_words
        self.return_pandas_data_series = return_pandas_data_series
        self.variable_segment=variable_segment
        self.words = []
        self.bp_words = []

        self.word_length = int(word_length/2)*2

        self.mindist_table = self._generate_mindist()
        self.tfdist_table = self._generate_tfdist()

    def transform(self,X,y=None):
        """Transform data.

        Parameters
        ----------
        X : 2d numpy array [N_instances,N_timepoints]

        Returns
        -------
        dims: Pandas data frame with first dimension in column zero
        """
        self.breakpoints = self._generate_breakpoints()
        breakpoints = self._generate_breakpoints()
        breakpoints_angle = self._generate_angle_breakpoints()

        # print(breakpoints)
        n_instances,series_length = X.shape

        if self.window_size==0:
            self.window_size =series_length

        bags = pd.DataFrame()
        self.words = []
        dim = []
        num_windows_per_inst = series_length - self.window_size + 1
        self.bp_words = np.zeros((n_instances,num_windows_per_inst,self.word_length))

        for i in range(n_instances):
            bag = {}
            lastWord = -1

            words = []
            
            num_windows_per_inst = series_length - self.window_size + 1

            split = X[i,np.arange(self.window_size)[None,:] + np.arange(num_windows_per_inst)[:,None]]
            split = scipy.stats.zscore(split,axis=1)


            """calculate the mean for each
            """
            # print(split.shape)
            paatf = PAATFSAX(num_intervals=int(self.word_length/2), variable_segment=self.variable_segment)
            # paa = PAA(num_intervals=3)
            
            # data = pd.DataFrame()
            # data[0] = [pd.Series(x,dtype=np.float32) for x in split]

            data = split
            patterns, trends = paatf.transform(data)
            patterns = np.asarray([a.values for a in patterns.iloc[:,0]])
            
            # pattern2 = paa.transform(data)
            # pattern2 = np.asarray([a.values for a in pattern2.iloc[:,0]])

            assert patterns.shape == trends.shape
            # print(patterns.shape, trends.shape)
            # print(f"sax: {patterns[0, :5]} | trend: {trends[0][:5]}")

            """quantization
            """
            for n in range(patterns.shape[0]):
                pattern = patterns[n,:]
                trend = trends[n,:]
                word_sax, bp_indices_sax = self._create_word(pattern,breakpoints)
                word_trend, bp_indices_trend = self._create_word(trend,breakpoints_angle)
                
                # print(np.concatenate([bp_indices_sax, bp_indices_trend]).shape)
                words.append(word_sax)
                words.append(word_trend)

                self.bp_words[i,n,:] = np.concatenate([bp_indices_sax, bp_indices_trend])
                
                lastWord = self._add_to_bag(bag,word_sax,lastWord)

            # print("after: ", bp_indices_sax[:5], bp_indices_trend[:5])
            if self.save_words:
                self.words.append(words)
                

            dim.append(pd.Series(bag) if self.return_pandas_data_series else bag)

        # histogram = self.create_bag(self.words)

        # self.histogram = histogram
        self.histogram = None

        bags[0] = dim

        return bags

    def fit_transform(self, X, y=None):
        self.transform(X, y)
        return self.bp_words

    def distance(self, X, Y, w, n):

        """Check dimension
        """
        if len(X.shape) == 3:
            X, Y = X[:,0,:], Y[:,0,:]

        """Check precision type
        """
        
        X = X.astype(np.int32)
        Y = Y.astype(np.int32)

        # Split sax and trend feature
        # print(X.shape, Y.shape)
        X_sax, X_tf = X[:, :w], X[:, w:] 
        Y_sax, Y_tf = Y[:, :w], Y[:, w:] 

        assert X_sax.shape == X_tf.shape
        
        # broadcast
        Y_sax_rep = np.repeat(Y_sax[:, None, :], len(X), axis=1)
        X_sax_rep = np.repeat(X_sax[None, :, :], len(Y), axis=0)

        Y_tf_rep = np.repeat(Y_tf[:, None, :], len(X), axis=1)
        X_tf_rep = np.repeat(X_tf[None, :, :], len(Y), axis=0)

        mindist_mat = self.mindist_table[Y_sax_rep, X_sax_rep]
        tfdist_mat  = self.tfdist_table[Y_tf_rep, X_tf_rep]

        dist_mat = np.sqrt(n/w * np.sum(mindist_mat**2 + w/n * tfdist_mat**2, axis=-1))
        # dist_mat = np.sqrt(n/w * np.sum(mindist_mat**2, axis=-1))
        return dist_mat

    def _create_word(self, pattern, breakpoints):
        word = 0
        bp_indices = np.zeros_like(pattern)
        for i in range(int(self.word_length/2)):
            for bp in range(self.alphabet_size):
                if pattern[i] <= breakpoints[bp]:
                    bp_indices[i] = bp
                    word = (word << 2) | bp
                    break

        return word,bp_indices

    def _add_to_bag(self, bag, word, last_word):
        if self.remove_repeat_words and word == last_word:
            return False
        bag[word] = bag.get(word, 0) + 1
        return True

    def _generate_breakpoints(self):
        # Pre-made gaussian curve breakpoints from UEA TSC codebase
        return {
            2: [0, sys.float_info.max],
            3: [-0.43, 0.43, sys.float_info.max],
            4: [-0.67, 0, 0.67, sys.float_info.max],
            5: [-0.84, -0.25, 0.25, 0.84, sys.float_info.max],
            6: [-0.97, -0.43, 0, 0.43, 0.97, sys.float_info.max],
            7: [-1.07, -0.57, -0.18, 0.18, 0.57, 1.07, sys.float_info.max],
            8: [-1.15, -0.67, -0.32, 0, 0.32, 0.67, 1.15, sys.float_info.max],
            9: [-1.22, -0.76, -0.43, -0.14, 0.14, 0.43, 0.76, 1.22, sys.float_info.max],
            10: [
                -1.28,
                -0.84,
                -0.52,
                -0.25,
                0.0,
                0.25,
                0.52,
                0.84,
                1.28,
                sys.float_info.max,
            ],
        }[self.alphabet_size]

    def _generate_angle_breakpoints(self):
        # Pre-made gaussian curve breakpoints from UEA TSC codebase
        return {
            2: [0, sys.float_info.max],
            3: [-5, 5, sys.float_info.max],
            4: [-30, 0, 30, sys.float_info.max],
            5: [-30, -5, 5, 30, sys.float_info.max],
            6: [-30, -5, 0, 5, 30, sys.float_info.max],
        }[self.alphabet_size_angle]
    
    def _generate_mindist(self):

        return {
            
            2: np.array([[0., 0.], [0., 0.]]),
            3: np.array([[0., 0., 0.86], [0., 0., 0], [0.86, 0., 0.]]),
            4: np.array([[0., 0., 0.67, 1.34], 
                         [0., 0., 0., 0.67], 
                         [0.67, 0., 0., 0.], 
                         [1.34, 0.67, 0., 0.]]),
            5: np.array([[0., 0., 0.59, 1.09, 1.68], [0., 0., 0., 0.5, 1.09], [0.59, 0., 0., 0., 0.59], [1.09, 0.5, 0., 0., 0.], [1.68, 1.09, 0.59, 0., 0.]]),
            6: np.array([[0.,   0.,  0.54,  0.97,  1.40, 1.94], 
                         [0.,   0.,    0.,  0.43,  0.86, 1.4], 
                         [0.54, 0.,    0.,    0.,  0.43, 0.97], 
                         [0.97, 0.43,   0.,    0.,   0., 0.54], 
                         [1.4,  0.86, 0.43,    0.,   0., 0.],
                         [1.94, 1.4,  0.97,  0.54,   0., 0.],
                         ]
                         
                         ),


        
        }[self.alphabet_size]

    def _generate_tfdist(self):

        return {
            
            2: np.array([[0., 0.], [0., 0.]]),
            3: np.array([[0., 0., 0.18], [0., 0., 0.], [0.18, 0., 0.]]),
            4: np.array([[0.,   0., 0.58, 1.73], 
                         [0.,   0.,   0., 0.58], 
                         [0.58, 0.,   0.,   0.], 
                         [1.73, 0.58, 0.,   0.]]),
            5: np.array([[0.,   0.,   0.46, 0.70, 1.73], [0., 0., 0., 0.18, 0.70], [0.46, 0., 0., 0., 0.46], [0.70, 0.18, 0., 0., 0.], [1.73, 0.70, 0.46, 0., 0.]]),
            6: np.array([[0.,   0.,   0.46,  0.58, 0.70, 1.73], 
                         [0.,   0.,     0.,  0.09, 0.18, 0.70], 
                         [0.46, 0.,     0.,    0., 0.09, 0.58], 
                         [0.58, 0.09,   0.,    0.,   0., 0.46], 
                         [0.70, 0.18, 0.09,    0.,   0.,    0],
                         [1.73, 0.7,  0.58,  0.46,   0.,   0.]
                         
                         ]),

        
        }[self.alphabet_size]


    def create_feature_names(sfa_words):
        """Create feature names."""
        feature_names = set()
        for t_words in sfa_words:
            for t_word in t_words:
                feature_names.add(t_word)
        return feature_names
    
    def create_bag(self,words):
        bag_of_words = None
        n_instances = len(words)

        breakpoints = self.breakpoints
        word_length = self.word_length

        feature_count = np.uint32(self.alphabet_size ** word_length)
        all_win_words = np.zeros((n_instances,feature_count),dtype=np.uint32)

        # print(all_win_words.shape, len(words[0]))
        for j in range(n_instances):
            if self.remove_repeat_words:
                masked = np.nonzero(words[j])
                all_win_words[j,:] = np.bincount(words[j][masked],minlength=feature_count)

            else:
                all_win_words[j,:] = np.bincount(words[j],minlength=feature_count)
        return all_win_words

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.


        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        # small word length, window size for testing
        params = {"word_length": 2, "window_size": 4}
        return params        


if __name__ == "__main__":

    # print(np.tan(np.deg2rad(30)), np.tan(np.deg2rad(60)))

    # X = np.random.randint(low=0, high=3, size=(2,5))
    # Y = np.random.randint(low=0, high=3, size=(4,5))

    # Z = np.arange(25).reshape(5,5)

    # check = X[:, None, :] != Y[None, :, :]


    # print(X)
    # print(Y)
    # print(check.shape)


    # Y_rep = np.repeat(Y[:, None, :], len(X), axis=1)
    # X_rep = np.repeat(X[None, :, :], len(Y), axis=0)

    # print(X_rep.shape, Y_rep.shape)
    # print(X_rep)

    # print("Z: ", Z)
    # print(Z[Y_rep, X_rep])


    tfsax = TFSAX()

    X = np.random.randint(low=0, high=3, size=(2,4))
    Y = np.random.randint(low=0, high=3, size=(4,4))

    print(X)
    print(Y)
    tfsax.distance(X, Y, tfsax.mindist_table, tfsax.tfdist_table, 2, 2)