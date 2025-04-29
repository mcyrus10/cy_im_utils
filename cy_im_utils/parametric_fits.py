import numpy as np
from scipy.optimize import least_squares


def parametric_gaussian(x, params) -> np.array:
    """
    using parameter ordering from gpufit 1d gaussian
    """
    amplitude, center, width, offset =  params
    argx = (x-center) * (x-center) / (2*width*width)
    ex = np.exp(-argx)
    return amplitude * ex + offset


def fit_param_gaussian(data, n_bins, mode = 'normal', density = False) -> tuple:
    data_slice = np.isfinite(data) * (data > 0)
    if mode == 'normal':
        data_handle = data[data_slice]
    elif mode == 'log':
        data_handle = np.log(data[data_slice])
    counts, bins = np.histogram(data_handle,
                               bins = n_bins,
                               density = density)
    if isinstance(n_bins, int):
        iterator = range(n_bins)
    else:
        iterator = range(len(n_bins)-1)
    bins_centered = np.array([(bins[j] + bins[j+1]) / 2 for j in iterator])
    x0 = [1,np.median(data_handle),np.std(data_handle),0]
    bounds = [(0, 0, 0, 0), (np.inf, np.inf, np.inf, np.inf)]
    residual = lambda params, x, data: parametric_gaussian(x, params) - data
    try:
        return bins_centered, counts, least_squares(residual,
                                                    x0,
                                                    bounds = bounds, 
                                                    args = (bins_centered, counts)
                                                    ).x
    except:
        print("---> Error with Least Squares")
        return None
