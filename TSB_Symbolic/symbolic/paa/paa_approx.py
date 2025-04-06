# Adapted from Sktime transformer
import pandas as pd
import numpy as np

from .paa_numba import _perform_paa_along_dim_numba,paa_whole_dataset

class PAA():
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
        result = paa_whole_dataset(X,self.num_intervals)

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

        for i in range(num_insts):
            series = X[i,:]

            frames = []
            current_frame = 0
            current_frame_size = 0
            frame_length = num_atts / self.num_intervals
            frame_sum = 0

            if num_atts % self.num_intervals == 0:
                series_split = np.array_split(series,self.num_intervals)
                frames = [np.mean(interval) for interval in series_split]
            else:
                
                # for each time step
                for n in range(num_atts):
                    remaining = frame_length - current_frame_size

                    if remaining > 1:
                        frame_sum += series[n]
                        current_frame_size += 1
                    else:
                        frame_sum += remaining * series[n]
                        current_frame_size += remaining

                    if current_frame_size == frame_length:
                        frames.append(frame_sum / frame_length)
                        current_frame += 1

                        frame_sum = (1 - remaining) * series[n]
                        current_frame_size = 1 - remaining

                # if the last frame was lost due to double imprecision
                if current_frame == self.num_intervals - 1:
                    frames.append(frame_sum / frame_length)

            data.append(pd.Series(frames))

        dims[0] = data

        return dims

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