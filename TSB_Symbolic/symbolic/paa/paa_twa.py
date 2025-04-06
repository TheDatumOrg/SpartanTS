# Adapted from Sktime transformer
import pandas as pd
import numpy as np

from paa.paa_numba import _perform_paa_along_dim_numba,paa_whole_dataset

class PAATWA():
    def __init__(self,
            num_intervals=8
        ):

        self.num_intervals = num_intervals
    
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
        # result = paa_whole_dataset(X,self.num_intervals)
        self.X_dims = len(X.shape)
        if self.X_dims == 2:
            self.num_insts, self.num_steps = X.shape
        elif self.X_dims == 3:
            self.num_insts, self.num_winds, self.num_steps = X.shape

            X = X.reshape(-1, self.num_steps)

        result = self._perform_paa_along_dim(X)

        if self.X_dims == 3:
            result = result.reshape(self.num_insts, self.num_winds, self.num_intervals)
        # Combine the dimensions together
        # result = pd.concat(dataFrames, axis=1, sort=False)
        #result.columns = col_names

        return result

    def _perform_paa_along_dim(self, X):
        # X = from_nested_to_2d_array(X, return_numpy=True)
        
        num_atts = X.shape[1]
        num_insts = X.shape[0]
        dims = pd.DataFrame()
        data = []
        twa_paa = np.zeros((len(X), self.num_intervals))

        for i in range(num_insts):
            series = X[i,:]

            frames = []
            current_frame = 0
            current_frame_size = 0
            frame_length = num_atts / self.num_intervals
            frame_sum = 0

            ind_list = []
            val_list = []
            twa_list = []

            
            # unequal segments
            for n in range(num_atts):

                ind_list.append(n+1)
                val_list.append(series[n])

                remaining = frame_length - current_frame_size

                # print(f"remaining: {remaining}")

                if remaining > 1:
                    frame_sum += series[n]
                    
                    current_frame_size += 1
                else:
                    frame_sum += remaining * series[n]
                    # print(series[n])
                    current_frame_size += remaining

                if current_frame_size == frame_length:
                    
                    frames.append(frame_sum / frame_length)
                    current_frame += 1
                    twa_list.append(np.sum(np.array(ind_list)[::-1] * np.array(val_list))/np.sum(ind_list))

                    print(f"TWA: {twa_list[-1]} | Index: {ind_list} | Value: {val_list}")
                    # reset
                    frame_sum = (1 - remaining) * series[n]
                    current_frame_size = 1 - remaining

                    ind_list = []
                    val_list = []

                    # print(n, frames)
            # if the last frame was lost due to double imprecision
            if current_frame == self.num_intervals - 1:
                frames.append(frame_sum / frame_length)
                twa_list.append(np.sum(np.array(ind_list)[::-1] * np.array(val_list))/np.sum(ind_list))

            
            twa_paa[i] = twa_list
            
        return twa_paa

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