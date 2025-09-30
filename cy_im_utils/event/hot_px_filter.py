import cupy as cp
import numpy as np
from tqdm import tqdm

from sys import path
path.append("/home/mcd4/cy_im_utils")
from cy_im_utils.event.integrate_intensity import _integrate_events_wrapper_


def hot_px_cd_filter(hot_px_map, cd_data, verbose = True, nx: int = 1280, 
        ny: int = 720, batch_size: int = 10_000_000) -> np.array:
    """
    Steps:
        1) converts a hot pixel image (2D boolean array) into complex values (
            x + jy ); Note, you could alternatively cast this as a uint32
            operation where you discretize the pixel array so each pixel has a
            unique coordinate, but this does not afford any memory savings over
            casting as a 64 bit complex....). uint16 cannot represent every
            pixel in the array if you discretize the grid! (720*1280 > 2**16)
        2) convert CD data (n x 4) into complex coordinate array (x + jy)
        3) calculate isin to find intersection between hot pixels and cd data
        4) return boolean slice of cd data excluding all hot pixels
    """
    assert nx*ny < 2**32, "pixels exceed casting to uint32"
    dtype = cp.uint32
    # Convert Hot Pixels into Imaginary Array
    if verbose: 
        print("\tHot px to cp")
    hot_px = None
    hot_px = np.where(hot_px_map)
    #hot_px_cp = cp.array(hot_px[1] + 1j*hot_px[0], dtype = cp.complex64)
    hot_px_cp = cp.array(hot_px[1]*nx + hot_px[0], dtype = dtype)

    # CD Data as Complex Array 
    if verbose: 
        print("\tcd data to cp")

    numel = len(cd_data['x'])
    n_batch = int(np.ceil(numel / batch_size))
    slice_ = cp.zeros(numel, dtype = bool)
    for j in tqdm(range(n_batch), desc = 'cp.isin'):
        local_slice = slice(j*batch_size, (j+1)*batch_size)
        x_cp = cp.array(cd_data['x'][local_slice], dtype = dtype)
        y_cp = cp.array(cd_data['y'][local_slice], dtype = dtype)
        cd_temp = x_cp * nx + y_cp
        slice_[local_slice] = cp.isin(cd_temp, hot_px_cp)

    return ~slice_.get()


def calc_hot_px(cd_data: np.array,
                z: int = 2,
                im_y: int = 720,
                im_x: int = 1280,
                dtype = np.float32,
                ) -> np.array(bool):
    """
    Calculate outlier pixels:
        1) Integrate ENTIRE CD DATA
        2) Calculate median and IQR
        3) Return pixels that are outside med +/- (IQR*Z)

    """
    image_buffer = np.zeros([im_y,im_x], dtype = dtype)
    integrator_fun = _integrate_events_wrapper_(dtype)
    integrator_fun(
            image_buffer,
            cd_data['x'],
            cd_data['y'],
            cd_data['p'].astype(bool),
            omit_neg = False
            )
    med = np.median(image_buffer.flatten())
    iqr = np.subtract(*np.percentile(image_buffer.flatten(),[75,25]))
    print(f"med = {med}; iqr = {iqr}")
    hot_px_map = image_buffer > (med + (iqr * z))
    hot_px_map += image_buffer < (med - (iqr*z))
    
    return hot_px_map
