from numba import njit, int16, uint16, int64, float64, float32, void, boolean
from tifffile import imwrite
from tqdm import tqdm
import cupy as cp
import h5py
import numpy as np



@njit(void(float32[:,:], uint16[:], uint16[:], boolean[:]))
def _integrate_events_(image, x, y, p) -> None:
    """
    Ultra basic: iterate over all the events and increment/decrement based on
    polarity
    """
    numel = x.size
    for j in range(numel):
        x_ = x[j]
        y_ = y[j]
        p_ = p[j]
        if not p_:
            image[y_,x_] -= 1
        elif p_:
            image[y_,x_] += 1


class hdf5_event_integrator:
    def __init__(self, file_name, nx: int = 1280, ny: int = 720):
        self.data, self.triggers = self.fetch_hdf5(file_name)
        self.nx = nx
        self.ny = ny

    def fetch_hdf5(self, file_name) -> (np.array, np.array):
        """
        Ultra basic hdf5 reader....?
        """
        with h5py.File(file_name, "r") as f:
            data = f['CD']['events'][()]
            triggers = f['EXT_TRIGGER']['events'][()]
        return data, triggers

    def __get__(self, arr):
        """
        helper function for arbitrary array library?
        """
        if isinstance(arr, np.ndarray):
            return arr
        elif isinstance(arr, cp.ndarray):
            return arr.get()
        else:
            print(f"unknown dtype for hdf5 integrator?: {type(arr)}")
            assert False

    def hdf5_to_image_stack(self, 
                            t0,
                            tf,
                            dt,
                            arr_lib = np,
                            thresh = np.inf
                            ):
        ev_keys = ['x','y','p','t']
        x,y,p,t = [np.array(self.data[key], dtype = self.data[key].dtype) 
                        for key in ev_keys]

        t = arr_lib.array(t, dtype = t.dtype)
        image = np.zeros([self.ny,self.nx], dtype = np.float32)
        n_iter = ( tf - t0 ) // dt
        out = np.zeros([n_iter, self.ny, self.nx], dtype = np.float32)
        print(out.shape)
        for j in tqdm(range(n_iter)):
            t0_ = t0 + j*dt
            t1_ = t0_ + dt
            image[:] = 0
            bool_arr = self.__get__((t > t0_) * (t < t1_))
            x_local = x[bool_arr]
            y_local = y[bool_arr]
            p_local = p[bool_arr]
            _integrate_events_(image, x_local, y_local, p_local)
            image[np.abs(image) > thresh] = 0
            out[j] = image.copy()
        return out


@njit(void(int64, int64, int64[:], int64[:,:]))
def fetch_timestep_indices(dt, t0, arr, output):
    """
    This iterates over large ( > 8 GB ) timestamp arrays and finds the indices
    where the timesteps bound the events...?
    """
    n_elem = arr.size
    i_max = output.shape[0]
    i = 0
    t0_bool = False
    for elem in range(n_elem):
        # Find First time stamp
        t_current = arr[elem]
        cond_0 = t_current > (t0 + i*dt)
        cond_1 = t_current > (t0 + (i+1)*dt)
        cond_2 = i < i_max

        # Found first time stamp...?
        if cond_0 and not t0_bool and cond_2:
            output[i,0] = elem
            t0_bool = True

        # Found Next time stamp...?
        if cond_1 and t0_bool and cond_2:
            output[i,1] = elem
            t0_bool = False
            i += 1


@njit(void(int64[:,:], int64[:], int64[:,:]))
def fetch_timestep_triggers(triggers, time_arr, output) -> None:
    """
    Triggers should be a [n x 2] array
    """
    n_elem = time_arr.size
    print("n_elem = ",n_elem)
    i_max = output.shape[0]
    print("i_max = ",i_max)
    i = 0
    t0_bool = False
    for elem in range(n_elem):
        # Find First time stamp
        t_current = time_arr[elem]
        cond_0 = t_current > triggers[i,0]
        cond_1 = t_current > triggers[i,1]
        cond_2 = i < i_max

        # Found first time stamp...?
        if cond_0 and not t0_bool and cond_2:
            output[i,0] = elem
            t0_bool = True

        # Found Next time stamp...?
        if cond_1 and t0_bool and cond_2:
            output[i,1] = elem
            t0_bool = False
            i += 1



def __get__(arr):
    """
    helper function for arbitrary array library?
    """
    if isinstance(arr, np.ndarray):
        return arr
    elif isinstance(arr, cp.ndarray):
        return arr.get()
    else:
        print(f"unknown dtype for hdf5 integrator?: {type(arr)}")
        assert False


def fetch_hdf5_data(file_name) -> (np.array, np.array):
    """
    Ultra basic hdf5 reader....?
    
    The assertion here is to make sure that the first trigger corresponds to
    the start of an exposure. Refer to metavision sdk documentation timing
    interfaces for the EVK4 and SilkyEvCam theres a polarity inversion in the
    trigger-in circuity that flips the meaning of the polarities....
    """
    with h5py.File(file_name, "r") as f:
        data = f['CD']['events'][()]
        triggers = f['EXT_TRIGGER']['events'][()]

    assert triggers['p'][0] == 0, "First event is should be zero (Exposure opening)"

    return data, triggers


def hdf5_to_image_stack(self, 
                        t0,
                        tf,
                        dt,
                        arr_lib = np,
                        thresh = np.inf
                        ):
    ev_keys = ['x','y','p','t']
    x,y,p,t = [np.array(self.data[key], dtype = self.data[key].dtype) 
                    for key in ev_keys]

    t = arr_lib.array(t, dtype = t.dtype)
    image = np.zeros([self.ny,self.nx], dtype = np.float32)
    n_iter = ( tf - t0 ) // dt
    out = np.zeros([n_iter, self.ny, self.nx], dtype = np.float32)
    print(out.shape)
    for j in tqdm(range(n_iter)):
        t0_ = t0 + j*dt
        t1_ = t0_ + dt
        image[:] = 0
        bool_arr = self.__get__((t > t0_) * (t < t1_))
        x_local = x[bool_arr]
        y_local = y[bool_arr]
        p_local = p[bool_arr]
        _integrate_events_(image, x_local, y_local, p_local)
        image[np.abs(image) > thresh] = 0
        out[j] = image.copy()
    return out


def process_triggers(trigger_array, sub_sample = 1, mode = 'exposure') -> np.array:
    """
    This is for controlling how the triggers are sampled. note the stride trick
    is for making the timestamps roll over... so that you cane either have the
    full exposures or fps matched version

    Parmaters
    ----------
        mode : str = 'exposure', 'fps' - Do you want to set the time-bounds as
                     the exposure of the camera or the fps match?
        sub_sample: int - how many sub-divisions you want inside of the given
                    mode

    Returns
    -------
        [n x 2] numpy array with the start and end of each frame

    """
    if mode == 'exposure':
        triggers_on = trigger_array[::2]
        triggers_off = trigger_array[1::2]
    elif mode == 'fps':
        triggers_on = trigger_array[::2][:-1]
        triggers_off = trigger_array[2::2]

    triggers_arr = np.vstack([triggers_on, triggers_off]).T   
    diff = np.diff(triggers_arr, axis = 1)
    local_steps = diff / sub_sample
    n_trigger = triggers_arr.shape[0]
    triggers_out = np.zeros([n_trigger, 1 + sub_sample])
    triggers_out[:,0] = triggers_on
    triggers_out[:,-1] = triggers_off
    for j in range(1,sub_sample):
        triggers_out[:, j] = np.squeeze(triggers_on[:,None] + local_steps * j)

    triggers_out = np.lib.stride_tricks.sliding_window_view(
                                                    triggers_out, 
                                                    (1,2)
                                                    ).reshape(-1,2)
    return triggers_out.astype(np.int64)


def hdf5_integrator(file_name, 
                    sync_mode: str,
                    dt: int,
                    sampling_interval: str = 'exposure',
                    samples_per_trigger: int = 1,
                    thresh = np.inf,
                    nx: int = 1280,
                    ny: int = 720,
                    ) -> None:
    params = {
            "file_name":file_name,
            "sync_mode":sync_mode,
            "dt":dt,
            "sampling_interval":sampling_interval,
            "samples_per_trigger":samples_per_trigger,
            "thresh":thresh,
            "nx":nx,
            "ny":ny,
            }
    for key,val in params.items():
        print(f"{key}: {val}")
    # Load Data
    data, triggers = fetch_hdf5_data(file_name)
    t0 = triggers['t'][0]
    tf = triggers['t'][-1]
    image = np.zeros([ny,nx], dtype = np.float32)
    print(t0,tf)

    # Calculate Each Worker's bounds
    print("\tMode: trigger_sync; ignoring dt argument")
    tr_t = triggers['t']
    # Prep Worker Slicing Array

    processed_triggers = process_triggers(tr_t,
                                samples_per_trigger, 
                                mode = sampling_interval)
    print(processed_triggers)
    timestep_slices = np.zeros_like(processed_triggers, dtype = np.int64)

    print(processed_triggers.dtype, timestep_slices.dtype, data['t'].dtype)
    fetch_timestep_triggers(processed_triggers, data['t'], timestep_slices)

    print("Fetched Time Steps?")

    n_images = timestep_slices.shape[0]
    image_stack = []

    print("integrating images?")
    for j in tqdm(range(n_images), desc = 'worker loop'):
        slice_ = slice(timestep_slices[j,0], timestep_slices[j,1])
        image[:] = 0
        x_ = data['x'][slice_]
        y_ = data['y'][slice_]
        p_ = data['p'][slice_]
        _integrate_events_(image, x_, y_, p_)
        image[np.abs(image) > thresh] = 0
        image_stack.append(image.copy())
    
    # Gather all the Images
    image_stack_global = np.stack(image_stack).astype(np.float32)

    f_name = "/tmp/integrated.tif"
    imwrite(f_name, image_stack_global)


if __name__ == "__main__":
    from pathlib import Path
    f_name = Path("/home/mcd4/Data/october_11_optimized_conditions/event/raw_unsync/condition_5.hdf5")
    f_name = Path("/home/mcd4/Data/october_11_optimized_conditions/event/raw_unsync/default_settings.hdf5")
    assert f_name.is_file(), f"{str(f_name)} doesnt exist "
    sync_mode = 'trigger_sync'
    samples_per_trigger = 1
    thresh = 100
    hdf5_integrator(f_name, 
                        sync_mode = sync_mode, 
                        dt = -1,
                        samples_per_trigger = samples_per_trigger,
                        sampling_interval = 'fps' ,
                        thresh = thresh
                        )
