import numpy as np
import pandas as pd
# import tsfresh.feature_extraction
from tsfresh.feature_extraction import feature_calculators

class PAASAXVFD():
    def __init__(self,
            num_intervals=8,
            feat_list=None
        ):

        self.num_intervals = num_intervals
        self.feat_list = feat_list

        if self.feat_list is None:
            self.feat_list = ['max', 'min', 'mean', 'median', 'var', 
                              'skew', 'slope', 'range', 'IQR', 'entropy', 
                              'mean_sec_deri_central', 'apEn', 'mean_abs_ch', 'sampEn',
                              'abs_sum_of_ch', 'kurtosis','abs_en', 'bEn']
    def set_num_intervals(self,n):
        self.num_intervals = n

    # todo: looks like this just loops over series instances
    # so should be refactored to work on Series directly
    def transform(self, X, y=None):
        """Transform data.

        Parameters
        ----------
        X : nested numpy array of shape [n_instances, n_timepoints]
            Nested dataframe with multivariate time-series in cells.

        Returns
        -------
        dims: Pandas data frame with first dimension in column zero,
              second in column one etc.
        """
        # Get information about the dataframe
        # num_atts = len(X.iloc[0, 0])
        # col_names = X.columns

        # Check the parameters are appropriate
        # self._check_parameters(num_atts)

        # On each dimension, perform PAA
        dataFrames = []
        # for x in col_names:
        dataFrames.append(self._perform_paa_along_dim(X))

        # Combine the dimensions together
        result = pd.concat(dataFrames, axis=1, sort=False)
        #result.columns = col_names

        return result

    def _perform_paa_along_dim(self, X):
        # X = from_nested_to_2d_array(X, return_numpy=True)
        num_atts = X.shape[1]
        num_insts = X.shape[0]
        dims = pd.DataFrame()
        data = []

        direct_feats_list = []

        for i in range(num_insts):
            series = X[i,:]

            frames = []
            current_frame = 0
            current_frame_size = 0
            frame_length = num_atts / self.num_intervals
            frame_sum = 0

            direct_feats = []
            seg_list = []

            for n in range(num_atts):
                remaining = frame_length - current_frame_size

                # add series item
                seg_list.append(series[n])

                if remaining > 1:
                    frame_sum += series[n]
                    current_frame_size += 1
                else:
                    frame_sum += remaining * series[n]
                    current_frame_size += remaining

                if current_frame_size == frame_length:
                    
                    current_frame += 1

                    frame_sum = (1 - remaining) * series[n]
                    current_frame_size = 1 - remaining

                    assert len(seg_list) > 0
                    frames.extend(self._feat_vec(seg_list))

                    # reset seg_list
                    seg_list = []

            # if the last frame was lost due to double imprecision
            if current_frame == self.num_intervals - 1:
               
                frames.extend(self._feat_vec(seg_list))
                seg_list = []

            data.append(pd.Series(frames))

        dims[0] = data

        return dims

    def _feat_vec(self, X):

        feat_val = []
        
        for feat_name in self.feat_list:

            feat_val.append(self._feat_func(X, feat_name))

        # print(feat_val)
        return feat_val


    def _feat_func(self, X, func_name):
        
        X = np.array(X).reshape(-1,)
        if func_name == 'max':
            return feature_calculators.maximum(X)
        elif func_name == 'min':
            return feature_calculators.minimum(X)
        elif func_name == 'mean':
            return feature_calculators.mean(X)
        elif func_name == 'median':
            return feature_calculators.median(X)   
        elif func_name == 'var':
            return feature_calculators.variance(X)
        elif func_name == 'skew':
            return feature_calculators.skewness(X)
        elif func_name == 'kurtosis':
            return feature_calculators.kurtosis(X)
        elif func_name == 'range':
            return np.max(X) - np.mean(X)
        elif func_name == 'IQR':
            return np.percentile(X, 75) - np.percentile(X, 25)
        elif func_name == 'entropy':
            return self.entropy(X)
        elif func_name == 'bEn':
            return feature_calculators.binned_entropy(X, max_bins=10)
        elif func_name == 'apEn':
            return feature_calculators.approximate_entropy(X,m=3,r=0.2)
        elif func_name == 'sampEn':
            return feature_calculators.sample_entropy(X)
        elif func_name == 'slope':
            return self.slope(X)
        elif func_name == 'abs_en':
            return feature_calculators.abs_energy(X)
        elif func_name == 'abs_sum_of_ch':
            return feature_calculators.absolute_sum_of_changes(X)
        elif func_name == 'mean_abs_ch':
            return feature_calculators.mean_abs_change(X)
        elif func_name == 'mean_sec_deri_central':
            return feature_calculators.mean_second_derivative_central(X)

        else:

            print('no matching func!')
            assert func_name == 'max'
        

    def _check_parameters(self, num_atts):
        """Check parameters of PAA.

        Function for checking the values of parameters inserted into PAA.
        For example, the number of subsequences cannot be larger than the
        time series length.

        Throws
        ------
        ValueError or TypeError if a parameters input is invalid.
        """
        if isinstance(self.num_intervals, int):
            if self.num_intervals <= 0:
                raise ValueError(
                    "num_intervals must have the \
                                  value of at least 1"
                )
            if self.num_intervals > num_atts:
                raise ValueError(
                    "num_intervals cannot be higher \
                                  than the time series length."
                )
        else:
            raise TypeError(
                "num_intervals must be an 'int'. Found '"
                + type(self.num_intervals).__name__
                + "' instead."
            )

    
    def entropy(self, signal, prob="standard"):
        """Computes the entropy of the signal using the Shannon Entropy.

        Description in Article:
        Regularities Unseen, Randomness Observed: Levels of Entropy Convergence
        Authors: Crutchfield J. Feldman David

        Feature computational cost: 1

        Parameters
        ----------
        signal : nd-array
            Input from which entropy is computed
        prob : string
            Probability function (kde or gaussian functions are available)

        Returns
        -------
        float
            The normalized entropy value

        """

        if prob == "standard":
            value, counts = np.unique(signal, return_counts=True)
            p = counts / counts.sum()
        elif prob == "kde":
            p = kde(signal)
        elif prob == "gauss":
            p = gaussian(signal)

        if np.sum(p) == 0:
            return 0.0

        # Handling zero probability values
        p = p[np.where(p != 0)]

        # If probability all in one value, there is no entropy
        if np.log2(len(signal)) == 1:
            return 0.0
        elif np.sum(p * np.log2(p)) / np.log2(len(signal)) == 0:
            return 0.0
        else:
            return -np.sum(p * np.log2(p)) / np.log2(len(signal))


    def slope(self, X):

        max_pt = np.argmax(X)
        min_pt = np.argmin(X)

        max_val = X[max_pt]
        min_val = X[min_pt]

        return (max_val - min_val) / (max_pt - min_pt)




# if __name__ == "__main__":

#     # X = np.random.rand(2,8)
#     X = np.array([[1,2,0,1,0,2, 0,1,2,6,2,1, 2,0,1,2,1,0], [1,2,0,1,0,2, 0,1,2,0,2,1, 2,0,1,2,1,1]])
#     print(X)
#     paa = PAASAXVFD(num_intervals=3, feat_list=['max', 'min', 'mean', 'median'])
    
#     result = paa.transform(X)
#     result = np.asarray([a.values for a in result.iloc[:,0]])
#     print(result)
