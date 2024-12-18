from numba import njit, int64, float64, void
import numpy as np

@njit(void(int64, int64, int64[:], int64[:,:]))
def fetch_timestep_indices(dt, t0, arr, output):
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


def test():
    from time import time
    tf = 1_200_000_000
    timestamps = np.linspace(0, tf-1, tf).astype(np.int64)
    print(timestamps.nbytes/1024**3)
    dt = 1000
    t0 = 100
    n_intervals = int((tf-t0) / dt)
    time_0 = time()
    output = np.zeros([n_intervals, 2], dtype = np.int64)
    fetch_timestep_indices(dt, t0, timestamps, output)
    print(time()-time_0)


if __name__ == "__main__":
    test()
