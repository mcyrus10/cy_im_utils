"""

    WRITTEN BY GEMINI

"""
import numpy as np
from scipy.optimize import curve_fit
from skimage.feature import peak_local_max
from joblib import delayed, Parallel
from functools import partial
import pandas as pd
from tqdm import tqdm

def _2d_gaussian(coords, amplitude, xo, yo, sigma_x, sigma_y, offset):
    """The mathematical model of a 2D Gaussian."""
    x, y = coords
    g = offset + amplitude * np.exp(
        -(((x - xo) ** 2) / (2 * sigma_x ** 2) + ((y - yo) ** 2) / (2 * sigma_y ** 2))
    )
    return g.ravel()



def gaussian_subpixel_peaks_2D(ncc_map, min_distance=10, threshold_abs=0.5, window_radius=2, frame_no: int = -1):
    integer_peaks = peak_local_max(ncc_map, min_distance=min_distance, threshold_abs=threshold_abs)
    subpixel_peaks = []
    
    # Create a grid of coordinates for the local window fitting
    window_size = 2 * window_radius + 1
    x_grid, y_grid = np.meshgrid(np.arange(window_size), np.arange(window_size))
    
    for y_int, x_int in integer_peaks:
        # Check if the window exceeds the map boundaries
        if (x_int - window_radius < 0 or x_int + window_radius >= ncc_map.shape[1] or 
            y_int - window_radius < 0 or y_int + window_radius >= ncc_map.shape[0]):
            subpixel_peaks.append([int(frame_no),float(x_int), float(y_int)])
            continue
            
        # Extract the local 5x5 window around the peak
        window = ncc_map[
            y_int - window_radius : y_int + window_radius + 1,
            x_int - window_radius : x_int + window_radius + 1
        ]
        
        # Initial guesses for the optimizer: [amplitude, x_center, y_center, sigma_x, sigma_y, offset]
        # We guess the center is exactly in the middle of our local window
        initial_guess = (np.max(window), window_radius, window_radius, 1.0, 1.0, np.min(window))
        
        try:
            # Fit the 2D Gaussian to the window
            popt, _ = curve_fit(
                _2d_gaussian, 
                (x_grid, y_grid), 
                window.ravel(), 
                p0=initial_guess, 
                maxfev=400
            )
            
            # popt[1] and popt[2] are the x and y offsets relative to the window corner
            # We subtract window_radius to get the shift relative to the integer center (-2 to +2)
            dx = popt[1] - window_radius
            dy = popt[2] - window_radius
            
            # Constrain extreme failures (if the fit wanders outside the center pixel)
            dx = np.clip(dx, -1.5, 1.5)
            dy = np.clip(dy, -1.5, 1.5)
            
            subpixel_peaks.append([int(frame_no), x_int + dx, y_int + dy])
            
        except RuntimeError:
            # If the optimizer fails to converge, fall back to integer coordinates
            subpixel_peaks.append([int(frame_no), float(x_int), float(y_int)])
            
    return np.array(subpixel_peaks)


def gaussian_subpixel_peaks_2D_parallel_stack(ncc_maps, min_distance=10, threshold_abs=0.5, window_radius=2, n_jobs = -1):
    """
    This function works exactly the same as gaussian_subpixel_peaks_2d_parallel, 
    except it decomposes the operations so that they can be parallelized with
    joblib (delay, parallel)
    """
    func_handle = partial(gaussian_subpixel_peaks_2D, min_distance = min_distance, 
                          threshold_abs = threshold_abs, window_radius = window_radius)
    desc = "iterating over images"
    delayed_ops = [delayed(func_handle)(ncc_map = im, frame_no = frame ) for frame, im in tqdm(enumerate(ncc_maps), desc = desc)]
    results = Parallel(n_jobs = n_jobs)(delayed_ops)
    located = []

    for elem in results:
        located.append(pd.DataFrame({"frame":elem[:,0],"x":elem[:,1],"y":elem[:,2]}))

    return pd.concat(located)