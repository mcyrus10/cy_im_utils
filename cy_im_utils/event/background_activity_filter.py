from numba import njit, void, uint16, int64, boolean
import numpy as np


@njit(void(uint16, uint16, int64, int64[:,:,:]))
def leaky_integrator(x, y, threshold, hot_pixel_map):
    """
    Identifier for pixel that are over-firing 
    """
    hot_pixel_map[0,x,y] += 1

    if hot_pixel_map[0,x,y] > threshold:
        hot_pixel_map[1,x,y] = 1
    else:
        hot_pixel_map[1,x,y] = 0


@njit(void(uint16[:], uint16[:], int64[:], int64, int64, int64[:,:], int64[:,:,:], boolean[:],uint16, uint16, int64, int64))
def background_activity_filter(ev_x, ev_y, ev_t, kernel_size, filter_length, timestamp_map, hot_pixel_map, events_bin, nx, ny, hot_px_threshold, decay_length):
    """

    True events have spatiotemporal correlations...noise events do not

    """
    numel = ev_x.size
    half_kern = kernel_size // 2
    for i in range(numel):
        x = ev_x[i]
        y = ev_y[i]
        t = ev_t[i]
        if i == 0:
            t0 = t

        leaky_integrator(x,y, hot_px_threshold, hot_pixel_map)

        if hot_pixel_map[1,x,y]:
            continue
        if t-t0 > decay_length:
            for j in range(nx):
                for k in range(ny):
                    hot_pixel_map[0,j,k] = max(0, hot_pixel_map[0,j,k]-1)
            t0 = t

        # Keep From going beyond extent of array
        low_x = max(0, x-half_kern)
        high_x = min(nx, x+half_kern+1)
        
        low_y = max(0, y-half_kern)
        high_y = min(ny, y+half_kern+1)

        slice_ = (slice(low_x, high_x), 
                  slice(low_y, high_y))

        # Events in the local neighborhood
        local_events = timestamp_map[slice_]

        # dt is the shortest time since the last event in the neighborhood
        dt = t - np.min(local_events)

        if dt < filter_length:
            events_bin[i] = True
        
        # Reset timestamp map neighborhood for the current event
        timestamp_map[slice_] = t


