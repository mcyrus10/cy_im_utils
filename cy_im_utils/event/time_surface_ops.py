from cupyx.scipy.ndimage import maximum_filter
from tqdm import tqdm
import cupy as cp
import numpy as np

def calc_time_surface(cd_data, triggers, shape, tau:float = 10000, nx = 1280, ny = 720) -> np.ndarray:
    """
    ??
    """
    print("DEPRECATED")
    timestamp_buffer = cp.zeros([ny, nx], dtype = np.int64)
    temp = np.zeros(shape, dtype = np.float32)
    for j, elem in tqdm(enumerate(triggers[:-1])):
        if j >= temp.shape[0]:
            print("j greater than array size")
            continue
        t_bin = cd_data['t'][triggers[j+1,0]]
        slice_ = slice(*elem)
        timestamp_buffer[:] = 0
        x,y,t = [cd_data[slice_][key] for key in ['x','y','t']]
        timestamp_buffer[y, x] = t
        local_val = cp.exp((timestamp_buffer - t_bin)/tau)
        temp[j] = local_val.get()
    return temp

def calc_time_surface_square(cd_data, triggers, shape, R: int = 1, tau:float = 10000, nx = 1280, ny = 720) -> np.ndarray:
    """
    This follows HOTS paper, to go to single-pixel level surface use R = 1

    """
    timestamp_buffer = cp.zeros(shape[1:], dtype = np.int64)
    time_surface = np.zeros(shape, dtype = np.float32)
    for j, elem in tqdm(enumerate(triggers[:-1])):
        if j >= time_surface.shape[0]:
            print("j greater than array size")
            continue
        slice_ = slice(*elem)
        x,y,t = [cp.array(cd_data[slice_][key]) for key in ['x','y','t']]
        timestamp_buffer[:] = t[0]
        timestamp_buffer[y,x] = t
        timestamp_buffer = maximum_filter(timestamp_buffer, R)
        t1 = t[-1]
        time_surface[j] = cp.exp(-(t1 - timestamp_buffer)/tau).get()
    return time_surface
