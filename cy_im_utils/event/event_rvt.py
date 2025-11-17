import numpy as np
from numba import njit, prange, void, uint16, int16, int64, float32, float64


def gen_r_kernel(r: float, rmax: int, dtype = np.float32) -> np.array:
    """
    This is taken from the RVT code....creates radius array with unit sum
    """
    a = rmax*2+1
    k = np.zeros([a,a], dtype = dtype)
    for i in range(a):
        for j in range(a):
            rij = ((i-rmax)**2+(j-rmax)**2)**(0.5)
            test_val = int(np.round(rij))
            if test_val <= r and test_val >= r:
                k[i,j] = 1
    return k/np.sum(k)


_signature_dict_ = {
                  'x':uint16[:],
                  'y':uint16[:],
                  'p':int16[:],
                  't':int64[:],
                  'filter':float32[:,:,:],
                  'timestamp':float32[:,:],
                  'time_surface':float32[:,:,:],
                  'im':int16[:,:],
                  'kernels':float32[:,:,:],
                  'r_max':int64,
                  'alpha':float64
                  }
_signature_keys_ = ['x','y','p','t','filter','timestamp','time_surface','im',
                    'kernels', 'r_max','alpha']
_signature_ = void(*(_signature_dict_[key] for key in _signature_keys_))
@njit(_signature_, fastmath = True)
def event_rvt_filter(x, y, p, t, mean_filter_state, timestamps, time_surface,
                     image, kernels, r_max, alpha):
    """
    Basically a combination of "Asynchronous spatial image convolutions for
    event cameras" and "Precision single-particle localization using radial
    variance transform"
    
    BASICS:
        - Approximates Equation 3 from the RVT paper (VoM i.e. Basic mode)
        - Convolves hough cone with event impulses (Equation 12)
            - This estimates the "mean" at a set of radial distances from the
              pixel 
        - Exponentially decays via timestamps (Equation 10)

    Removed noise filter from this.
        - if you are iterating over all the events, this means you have to do
          that twice, but I don't like having these things coupled.......
        - the noise filter is pretty fast so doing the iteration twice with one
          run through being the noise filter is not a killer for efficiency...

    """
    nx,ny = 1280,720
    numel = x.size
    kz, kx, ky = kernels.shape
    kern_z, kern_x, kern_y = np.where(kernels)
    n_kern = len(kern_z)
    for j in range(numel):
        _x_, _y_, _p_, _t_ = x[j], y[j], p[j], t[j]
        x_min = _x_ - r_max < 0
        x_max = _x_ + r_max >= nx
        y_min = _y_ - r_max < 0
        y_max = _y_ + r_max >= ny
        p_cond = _p_ == 1
        if x_min or x_max or y_min or y_max:# or p_cond:
            continue

        image[_x_, _y_] += 1 if _p_ == 1 else -1
        dt_surface = _t_ - time_surface[1,_x_,_y_]
        time_surface[0,_x_,_y_] = np.exp(-alpha*dt_surface)
        time_surface[1,_x_,_y_] = _t_

        slice_x = _x_ + r_max - kern_x
        slice_y = _y_ + r_max - kern_y

        for i in prange(n_kern):
            # Update time stamping: dt is current time - last  time
            dt = _t_ - timestamps[slice_x[i], slice_y[i]]
            timestamps[slice_x[i], slice_y[i]] = _t_
            previous_state  = mean_filter_state[kern_z[i], slice_x[i], slice_y[i]]
            decay = np.exp(-alpha*dt)*previous_state
            mean_filter_state[kern_z[i], slice_x[i], slice_y[i]] = decay + kernels[kern_z[i], kern_x[i], kern_y[i]]
