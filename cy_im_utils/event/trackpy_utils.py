"""

                                TRACKPY UTILS

"""
from tqdm import tqdm
import numpy as np
import pandas as pd
import trackpy as tp


def imsd_powerlaw_fit(imsd_dict) -> tuple:
    """
    This performs the log-log fit on all the imsd curves
    """
    time = imsd_dict.index.values
    log_t = np.log(time)
    ones = np.ones_like(log_t)
    A_mat = np.vstack([ones, log_t]).T
    b_mat = np.log(imsd_dict.values, where = imsd_dict.values > 0)
    A, n = np.linalg.lstsq(A_mat, b_mat, rcond = -1)[0]
    fits = np.exp(A)[None,:] * time[:,None] ** n[None,:]
    if np.isnan(fits[0]).sum() > 0:
        print("warning -> nans in fit")
    return A, n, fits


def imsd_linear_fit(imsd_dict) -> tuple:
    """
    This performs the linear fit on all the imsd curves
    """
    time = imsd_dict.index.values
    ones = np.ones_like(time)
    A_mat = np.vstack([ones, time]).T
    b_mat = imsd_dict.values
    b, m = np.linalg.lstsq(A_mat, b_mat, rcond = -1)[0]
    fits = m[None,:] * time[:,None] + b[None,:]
    if np.isnan(fits[0]).sum() > 0:
        print("warning -> nans in fit")
    return m, b, fits


class event_tracks:
    """
    This class is for handling the metavision tracks...
    """
    def __init__(self, input_data):
        if isinstance(input_data, str):
            self.data = pd.read_csv(input_data)
        elif isinstance(input_data, pd.DataFrame):
            self.data = input_data
        else:
            print("???")

    def _fetch_valid_indices_(self, min_length: int) -> np.array:
        """
        captures only the "valid" particles that meet the criteria greater than length...
        """
        data = self.data
        unique_elements = data['obj id'].unique()
        valid_indices = np.zeros(len(data['obj id'].values), dtype = bool)
        
        for elem in tqdm(unique_elements):
            indices = data['obj id'] == elem
            nnonzero = indices.sum()
            condition_0 = nnonzero < min_length
            condition_1 = data['x_double'].nunique() != nnonzero
            condition_2 = data['x_double'].nunique() != nnonzero
            # Remove shorter than minimum length
            if condition_0:# or condition_1 or condition_2:
                continue

            # The end of the recording can end on a float that rounds down to
            # the previous value so you need to remove the last value to avoid
            # a rounding error double instance of one time step.......
            remove_last_val = np.ones(nnonzero).astype(bool)
            remove_last_val[-1] = False
            valid_indices[indices] = remove_last_val
        return valid_indices

    def format_data_for_tp(self,
                           min_length: int,
                           update_frequency: float,
                           height: int = 720,
                           affine_matrix: np.array = None
                           ) -> pd.DataFrame:
        """
        This formats the data for processing by trackpy
        """
        valid_indices = self._fetch_valid_indices_(min_length)
        data = self.data
        #valid_indices = np.ones_like(self.data['obj id'].values).astype(bool)

        # Trackpy wants the "frames" to be sequential...
        normalized_frames = data['timestamp'].values - data['timestamp'].values[0]
        dt = 1e6 / update_frequency
        event_frames = np.round(normalized_frames / dt)


        if affine_matrix is not None:
            ones = np.ones_like(data['x_double'].values)
            new_coords = np.dot(affine_matrix, 
                                np.vstack([data['x_double'].values,
                                           data['y_double'].values,
                                           ones]))
        else:
            new_coords = np.vstack([data['x_double'].values,
                                    data['y_double'].values])

        df = pd.DataFrame.from_dict({
                'frame':event_frames[valid_indices].astype(int),
                'particle':data['obj id'][valid_indices].astype(int),
                'timestamp':data['timestamp'][valid_indices],
                'x':new_coords[0,valid_indices],
                'y':new_coords[1,valid_indices],
               }).drop_duplicates(subset = ['frame','particle'],
                                  keep = 'last')
    
        return df
