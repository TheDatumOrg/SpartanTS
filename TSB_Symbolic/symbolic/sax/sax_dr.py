#adapted from sktime:https://github.com/sktime/sktime/blob/v0.25.0/sktime/transformations/panel/dictionary_based/_sax.py
""""""

import sys

import numpy as np
import pandas as pd
import scipy.stats

from ..paa.paa_sax_dr import PAASAXDR
from ..paa.paa_approx import PAA

class SAXDR():
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
    ):
        self.word_length = word_length
        self.alphabet_size = alphabet_size
        self.alphabet_size_angle = alphabet_size_angle
        self.window_size = window_size
        self.remove_repeat_words = remove_repeat_words
        self.save_words = save_words
        self.return_pandas_data_series = return_pandas_data_series
        self.words = []
        self.bp_words = []

        self.word_length = int(word_length/2)*2
        self.dirdist_table = None

        self.mindist_table = self._generate_mindist()

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

        if self.dirdist_table is None:
            self.vex_max = []
            self.vex_mean = []
            self.cav_min = []
            self.cav_mean = []

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
            paadr = PAASAXDR(num_intervals=int(self.word_length/2))
            # paa = PAA(num_intervals=3)
            
            # data = pd.DataFrame()
            # data[0] = [pd.Series(x,dtype=np.float32) for x in split]

            data = split
            patterns, trend, dr_stat = paadr.transform(data)
            patterns = np.asarray([a.values for a in patterns.iloc[:,0]])
            

            if self.dirdist_table is None:
                
                self.cav_mean.append(dr_stat[0])
                self.cav_min.append(dr_stat[1])
                self.vex_mean.append(dr_stat[2])
                self.vex_max.append(dr_stat[3])
                
                
            # pattern2 = paa.transform(data)
            # pattern2 = np.asarray([a.values for a in pattern2.iloc[:,0]])
            # print("before: ", patterns[0], trend[0])
            # print(dr_stat)


            """quantization
            """
            for n in range(patterns.shape[0]):
                pattern = patterns[n,:]
                word_sax, bp_indices_sax = self._create_word(pattern,breakpoints)
                
                # print(np.concatenate([bp_indices_sax, bp_indices_trend]).shape)
                words.append(word_sax)

                self.bp_words[i,n,:] = np.concatenate([bp_indices_sax, trend[n]])
                
                lastWord = self._add_to_bag(bag,word_sax,lastWord)

            if self.save_words:
                self.words.append(words)
            
            # print("after: ", self.bp_words[i,0,:])

            dim.append(pd.Series(bag) if self.return_pandas_data_series else bag)

        # histogram = self.create_bag(self.words)

        # self.histogram = histogram
        self.histogram = None

        bags[0] = dim


        # build drdist look-up table
        if self.dirdist_table is None:
            
            self.cav_mean = np.array(self.cav_mean)
            self.cav_min  = np.array(self.cav_min)
            self.vex_mean = np.array(self.vex_mean)
            self.vex_max  = np.array(self.vex_max)

            self.dirdist_table = self._generate_dirdist()

            print(self.dirdist_table)

        return bags


    def fit_transform(self, X, y=None):

        self.transform(X, y)
        return self.bp_words


    def distance(self, X, Y, w, n):

        """Check dimension
        """

        assert self.dirdist_table is not None

        if len(X.shape) == 3:
            X, Y = X[:,0,:], Y[:,0,:]

        """Check precision type
        """
        
        X = X.astype(np.int32)
        Y = Y.astype(np.int32)

        # Split sax and trend feature
        print(X.shape, Y.shape)
        X_sax, X_dr = X[:, :w], X[:, w:] 
        Y_sax, Y_dr = Y[:, :w], Y[:, w:] 

        assert X_sax.shape == X_dr.shape
        
        # broadcast
        Y_sax_rep = np.repeat(Y_sax[:, None, :], len(X), axis=1)
        X_sax_rep = np.repeat(X_sax[None, :, :], len(Y), axis=0)

        Y_dr_rep = np.repeat(Y_dr[:, None, :], len(X), axis=1)
        X_dr_rep = np.repeat(X_dr[None, :, :], len(Y), axis=0)

        mindist_mat = self.mindist_table[Y_sax_rep, X_sax_rep]
        dirdist_mat = self.dirdist_table[Y_dr_rep, X_dr_rep]

        dist_mat = np.sqrt(n/w * np.sum(mindist_mat**2, axis=-1)) + np.sqrt(n/w * np.sum(dirdist_mat**2 / w, axis=-1))
        # dist_mat = np.sqrt(n/w * np.sum(mindist_mat**2, axis=-1))
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
            
            4: np.array([[0., 0., 0.67, 1.34], [0., 0., 0., 0.67], [0.67, 0., 0., 0.], [1.34, 0.67, 0., 0.]]),
        
        }[self.alphabet_size]


    def _generate_dirdist(self):

        return np.array([[0, np.mean(self.vex_max - self.vex_mean), np.mean(self.vex_max-self.vex_mean+self.cav_mean-self.cav_min)],
                         [np.mean(self.vex_max - self.vex_mean), 0, np.mean(self.cav_mean-self.cav_min)],
                         [np.mean(self.vex_max-self.vex_mean+self.cav_mean-self.cav_min), np.mean(self.cav_mean - self.cav_min), 0]
                         ])
    
    
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
