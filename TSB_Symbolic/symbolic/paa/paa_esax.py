import numpy as np
import pandas as pd
import ruptures as rpt

class PAAESAX():
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

        for i in range(num_insts):
            series = X[i,:]

            frames = []
            current_frame = 0
            current_frame_size = 0
            frame_length = num_atts / self.num_intervals
            frame_sum = 0

            max_val, min_val, mid_val = -1e6, 1e6, -1e6
            max_pt, min_pt, mid_pt = -1, -1, -1
            s_pt = 0


            
            for n in range(num_atts):
                remaining = frame_length - current_frame_size
                # print(remaining)
                if remaining > 1:
                    frame_sum += series[n]
                    current_frame_size += 1
                else:
                    frame_sum += remaining * series[n]
                    current_frame_size += remaining

                # compare max and min
                if np.nan_to_num(series[n]) > max_val:
                    max_val = np.nan_to_num(series[n])
                    max_pt = n

                if np.nan_to_num(series[n]) < min_val:
                    min_val = np.nan_to_num(series[n])
                    min_pt = n 

                if current_frame_size == frame_length:
                    
                    mid_pt = (s_pt + n) / 2
                    mid_val = frame_sum / frame_length
                    
                    # print(current_frame, series[n], mid_pt, min_pt, max_pt)
                    sort_list = self._sort_order(min_val, max_val, mid_val, min_pt, max_pt, mid_pt)
                    # add num
                    frames.extend(sort_list)
                    current_frame += 1


                    # reset
                    frame_sum = (1 - remaining) * series[n]
                    current_frame_size = 1 - remaining
                    
                    max_val, min_val, mid_val = -1e6, 1e6, -1e6
                    max_pt, min_pt, mid_pt = -1, -1, -1

                    s_pt = n+1

            # if the last frame was lost due to double imprecision
            if current_frame == self.num_intervals - 1:
                
                mid_pt = (s_pt + n) / 2
                # print(mid_pt, "llllll")
                mid_val = frame_sum / frame_length

                sort_list = self._sort_order(min_val, max_val, mid_val, min_pt, max_pt, mid_pt)
                frames.extend(sort_list)
                # frames.append(frame_sum / frame_length)
                
            
            data.append(pd.Series(frames))

        dims[0] = data

        return dims


    def _sort_order(self, min_val, max_val, mid_val, min_pt, max_pt, mid_pt):
        
        assert min_pt > -1 and max_pt > -1 and mid_pt > -1
        if min_pt == max_pt  or min_pt == mid_pt or max_pt == mid_pt:
            return [mid_val, min_val, max_val]

        pt_arr = np.array([min_pt, max_pt, mid_pt])
        ind = np.argsort(pt_arr)
        val_sort = np.array([min_val, max_val, mid_val])[ind]
        return list(val_sort)



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


if __name__ == '__main__':
    

    """Test case
    """
    paae = PAAESAX(num_intervals=2)

    # X = np.array([1,2,0, 1,0,2, 0,1,2, 0,2,1, 2,0,1, 2,1,0])
    X = np.array([[1,2,0,1,    2,0,1,1],
                  [1,0,0,-0.5, -0.4,0.2,1,1.3]]).reshape(2, -1)
    # X = np.expand_dims(X, axis=0)
    # X = np.random.rand(1,9)
    print(X.shape, X)
    result = paae.transform(X)
    result = np.asarray([a.values for a in result.iloc[:,0]])
    print(result)
