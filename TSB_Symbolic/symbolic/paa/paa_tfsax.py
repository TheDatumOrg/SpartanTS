# Adapted from Sktime transformer
import numpy as np
import pandas as pd
import ruptures as rpt

class PAATFSAX():
    def __init__(self,
            num_intervals=8,
            variable_segment=False
        ):

        self.num_intervals = num_intervals
        self.variable_segment=variable_segment
    
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
        sax_feat, trend_feat = self._perform_paa_along_dim(X)
        dataFrames.append(sax_feat)

        # Combine the dimensions together
        sax_result = pd.concat(dataFrames, axis=1, sort=False)
        trend_result = trend_feat
        #result.columns = col_names

        return sax_result, trend_result

    def _perform_paa_along_dim(self, X):
        # X = from_nested_to_2d_array(X, return_numpy=True)
        num_atts = X.shape[1]
        num_insts = X.shape[0]
        dims = pd.DataFrame()
        data = []

        trend_feats_list = []

        for i in range(num_insts):
            series = X[i,:]

            frames = []
            current_frame = 0
            current_frame_size = 0
            frame_length = num_atts / self.num_intervals
            frame_sum = 0

            trend_feats = []
            seg_list = []

            if not self.variable_segment:    
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
                        # print(frame_sum / frame_length)
                        current_frame += 1

                        frame_sum = (1 - remaining) * series[n]
                        current_frame_size = 1 - remaining

                        trend_feats.append(self._trend_feat(seg_list))
                        
                        # reset seg_list
                        seg_list = []

                # if the last frame was lost due to double imprecision
                if current_frame == self.num_intervals - 1:
                    frames.append(frame_sum / frame_length)
                    trend_feats.append(self._trend_feat(seg_list))
                    seg_list = []

            else:
                algo = rpt.Binseg(model='l2')
                # algo = rpt.Window(width=int(0.1*len(series)), model='l2')

                algo.fit(series)
                # result = algo.predict(pen=1)

                try:
                    bkps = algo.predict(n_bkps=self.num_intervals-1)
                    segments = np.split(series, bkps[:-1])
                except:
                    print("Segmentation failed. Even segment is used instead.")
                    segments = np.split(series, self.num_intervals)

                if len(segments) != self.num_intervals:
                    # print("Segmentation failed. Even segment is used instead.")
                    print(bkps)
                    segments = np.split(series, self.num_intervals)

                # print(series)
                # print(bkps)
                # print(segments)

                for j in range(len(segments)):
                    
                    frames.append(np.mean(segments[j]))
                    trend_feats.append(self._trend_feat(segments[j]))
                    

            data.append(pd.Series(frames))
            trend_feats_list.append(trend_feats)


        dims[0] = data

        return dims, np.array(trend_feats_list)

    

    def _trend_feat(self, X):

        mean_val = np.mean(X)
        td = (X[-1] - mean_val) - (X[0] - mean_val)
        K = self._trend_point(X)

        tan = td / K

        angle = np.arctan(tan) * 180 / np.pi

        return angle

    def _trend_point(self, X):
        
        K = 0
        for i in range(1, len(X)-1):

            if (X[i] - X[i-1])*(X[i+1] - X[i]) < 0:
                K += 1
            elif ((X[i] - X[i-1])*(X[i+1] - X[i]) == 0) and ((X[i] - X[i-1]) != (X[i+1] - X[i])): 
                K += 1

        return max(K, 1)

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



if __name__ == "__main__":

    
    # """TEST CASE
    # """
    # X = np.array([[1,2,0,1,0,2, 0,1,2,6,2,1, 2,0,1,2,1,0], [1,2,0,1,0,2, 0,1,2,0,2,1, 2,0,1,2,1,1]])
    # # X = np.expand_dims(X, axis=0)
    # paatf = PAATFSAX(num_intervals=3)

    # sax, trend = paatf.transform(X)
    # sax = np.asarray([a.values for a in sax.iloc[:,0]])
    # print(sax)
    # print(trend)

    # print(np.arctan(0.25) * 180 /np.pi)
    a = np.array([1.2675379965738252, 1.0968737637586718, 0.5635477629579723])
    b = np.array([0.2628230971854098, 1.122971443988266, 1.070841243628112])


    ed = np.sqrt(np.sum((a-b)**2))

    tf = 1.73
    

    print(ed, tf)



