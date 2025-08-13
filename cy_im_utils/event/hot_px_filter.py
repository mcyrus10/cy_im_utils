import cupy as cp
import numpy as np
from integrate_intensity import _integrate_events_wrapper_


def hot_px_cd_filter(hot_px_map, cd_data) -> np.array:
    """
    Steps:
        1) converts a hot pixel image (2D boolean array) into complex values (
        x + jy )
        2) convert CD data (n x 4) into complex coordinate array (x + jy)
        3) calculate isin to find intersection between hot pixels and cd data
        4) return boolean slice of cd data excluding all hot pixels
    """
    # Convert Hot Pixels into Imaginary Array
    hot_px, cd_complex, temp = None, None, None
    hot_px = np.where(hot_px_map)
    hot_px_complex = cp.array(hot_px[1] + 1j*hot_px[0], dtype = cp.complex64)

    # CD Data as Complex Array 
    cd_complex = cp.array(cd_data['x'] + 1j*cd_data['y'], dtype = cp.complex64)

    # Intersection
    slice_ = cp.isin(cd_complex, hot_px_complex)

    return cd_data[~slice_.get()]


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
