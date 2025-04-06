import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class PAASAXDR():
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
        dataFrames.append(self._perform_paa_along_dim(X)[0])

        # Combine the dimensions together
        result = pd.concat(dataFrames, axis=1, sort=False)
        #result.columns = col_names

        return result, self._perform_paa_along_dim(X)[1], self._perform_paa_along_dim(X)[2]

    def _perform_paa_along_dim(self, X):
        # X = from_nested_to_2d_array(X, return_numpy=True)
        num_atts = X.shape[1]
        num_insts = X.shape[0]
        dims = pd.DataFrame()
        data = []

        direct_feats_list = []

        vex_max, vex_mean = [], []
        cav_min, cav_mean = [], []

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
                    frames.append(frame_sum / frame_length)
                    current_frame += 1

                    frame_sum = (1 - remaining) * series[n]
                    current_frame_size = 1 - remaining


                    # direct representation
                    dr_feat = self._direct_feat(seg_list)
                    direct_feats.append(dr_feat)

                    if dr_feat == 2:
                        cav_mean.append(np.mean(seg_list))
                        cav_min.append(np.min(seg_list))
                    elif dr_feat == 0:

                        vex_mean.append(np.mean(seg_list))
                        vex_max.append(np.max(seg_list))

                    
                    # reset seg_list
                    seg_list = []

            # if the last frame was lost due to double imprecision
            if current_frame == self.num_intervals - 1:
                frames.append(frame_sum / frame_length)
                direct_feats.append(self._direct_feat(seg_list))
                seg_list = []

            data.append(pd.Series(frames))
            direct_feats_list.append(direct_feats)


        dims[0] = data
        
        # dr_stat
        if len(cav_mean) == 0:
            cav_mean = 0
        if len(cav_min) == 0:
            cav_min = 0
        if len(vex_mean) == 0:
            vex_mean = 0
        if len(vex_max) == 0:
            vex_max = 0

        dr_stats = (np.mean(cav_mean), np.mean(cav_min), 
                    np.mean(vex_mean), np.mean(vex_max))

        return dims, direct_feats_list, dr_stats

    

    def _direct_feat(self, X):

        N = len(X)
        
        slope = np.array(X[1:]) - np.array(X[:-1])

        pos_slope = np.sum(slope>0)
        neg_slope = np.sum(slope<0)

        # production rules
        if pos_slope > (N / 2):
            # concave
            return 2
        elif neg_slope > (N / 2):
            # convex
            return 0
        else:
            return 1



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

