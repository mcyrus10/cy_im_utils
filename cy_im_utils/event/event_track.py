"""

    Tracking Algorithm From: Embedded Vision System for Real-Time Object
    tracking using an asynchronous transient vision sensor"
    


"""
import numpy as np
import pandas as pd
from tqdm import tqdm
from numba import njit, prange, uint16, int16, void, boolean, int64, float64, float32


@njit(float64[:](float64[:], float64))
def numba_min(arr, ref) -> np.array:
    """
    This is meant to replicate np.min(..., inital = ref), which isn't 
    implemented in numba
    """
    min_local = ref
    numel = arr.size
    output = np.ones(arr.size, dtype = np.float64)*ref
    for j in range(numel):
        if arr[j] < ref:
            output[j] = arr[j]
            
    return output
    

#                      x            y        p        t           clust         Rx       Rmax    Rmin      ax        aw      ar       w       filter   kern   nx     ny
@njit(float64[:,:,:](uint16[:], uint16[:], int16[:], int64[:], float64[:,:,:], float64, float64, float64, float64, float64, float64, float64, int64, int64))
def evt_track(
              x_coords, # cd data x coordinates
              y_coords, # cd data y coordinates
              polarities, #cd data polarities
              timestamps, # cd data timestamps
              clusters,   # np.array of clusters 
              R_multiple = 2,  # factor multiplier for R_min 
              R_max = 20,    # maximum search range
              R_min = 10,    # assures minimum dynamic search range...?
              alpha_x = 0.8, # update factor for centroid coordinates (see equation 2)
              alpha_w = 0.8,  # update factor for weight (see equation 5)
              alpha_r = 0.9,  # update factor for radial search distance (see equation 3)
              weight_min = 0.01,  # Weigted frequency cutoff?
              nx: int = 1280,  # image extent
              ny: int = 720, # image extent
             ) -> np.array:
    """
    Implementation of "Embedded Vision System for Real-Time Object tracking
    using an asynchronous transient vision sensor" tracking algorithm
    """
    my_idx = 0
    image = np.zeros((1280,720), dtype = np.int64)
    
    numel = x_coords.size
    for j in range(numel):
        _x_, _y_, _p_, _t_ = x_coords[j], y_coords[j], polarities[j], timestamps[j]
        # -------------------------------------------
        # NO SPATIOTEMPORAL NOISE FILTER -> FILTER BEFORE PASSING TO THIS
        # Note: This is not optimized for speed, but for for clarity and
        #   modularity (it requires iterating over all the events multiple
        #   times
        # -------------------------------------------
        image[_x_, _y_] += _p_ if _p_ == 1 else -1
    
        # -------------------------------------------
        # Distance to each Cluster 
        # -------------------------------------------
        dx = clusters[:, 0, _p_] - float(_x_)
        #print(clusters.shape)
        dy = clusters[:, 1, _p_] - float(_y_)
        R = (dx**2+dy**2)**(1/2)
        #print("?",j ,  R, dx, dy, "?")
    
        R_k = numba_min(R_multiple*clusters[:, 2, _p_], ref = R_max)
    
        R_min_local = np.min(R)
        # Automatically take the FIRST Cluster if there are multiple equidistant?
        candidate_cluster = np.where(R == R_min_local)[0][0]

        # -------------------------------------------
        # APPEND CLUSTER
        # -------------------------------------------
        if R_min_local < R_k[candidate_cluster]:
    
            x_new = clusters[candidate_cluster,0,_p_]*alpha_x + _x_*(1-alpha_x)
            y_new = clusters[candidate_cluster,1,_p_]*alpha_x + _y_*(1-alpha_x)
    
            Rc_new = max(clusters[candidate_cluster,2,_p_]*alpha_r + R[candidate_cluster]*(1-alpha_r), R_min)
            
            dt = _t_ - clusters[candidate_cluster, 3, _p_]
            freq = 0 if dt <= 0 else 1/dt
            weight_new = clusters[candidate_cluster,4,_p_]*alpha_w + freq * (1-alpha_w)
            
            clusters[candidate_cluster, 0, _p_] = x_new   
            clusters[candidate_cluster, 1, _p_] = y_new
            clusters[candidate_cluster, 2, _p_] = Rc_new
            clusters[candidate_cluster, 3, _p_] = _t_
            clusters[candidate_cluster, 4, _p_] = weight_new
            
        # -------------------------------------------
        # NEW CLUSTER
        # -------------------------------------------
        else:
            cluster_unit = np.array((_x_, _y_, R_min, _t_, 0, _t_, my_idx))
            cluster_unit = np.stack((cluster_unit, cluster_unit), axis = 1)[None,:,:]
            clusters = np.vstack((clusters, cluster_unit))
            my_idx += 1
    
        # -------------------------------------------
        # MERGE CLUSTERS?
        # -------------------------------------------
        #if j % 1000 == 0:
        #    pass
    
        
        # -------------------------------------------
        # CLEAR OUT INACTIVE/LOW FREQUENCY?
        # -------------------------------------------
        if j % 100 == 0 and j > 0:
            reset_idx = clusters[:,4,1] <= weight_min
            clusters = clusters[~reset_idx]
            sort_idx = np.argsort(clusters[:,5,1])
            clusters = clusters[sort_idx]

    return clusters


def apply_evt_track(cd_data,
                    trigger_idx,
                    R_min = 20.0,
                    R_max = 20.0,
                    R_multiple = 2.0,
                    plotting_freq  = 250,
                    alpha_x = 0.8,
                    alpha_w = 0.8,
                    alpha_r = 0.9,
                    weight_min = 5e-3,
                    nx = 1280,           # image extent
                    ny = 720,            # image extent
                    ) -> (np.array, pd.DataFrame):
    """
    convenience wrapper for applying the event tracking algorithm...

        Note: this only tracks positive polarities
    """
    image_stack = None
    clusters = np.ones((1, 7, 2)) * -1
    clusters[:,2,:] = R_min
    idx = 0
    n_im = trigger_idx.shape[0]-1
    image_slice = slice(trigger_idx[idx,0], trigger_idx[idx+n_im,1])
    time_step = trigger_idx[idx+n_im,1] - trigger_idx[idx,0]
    t0 = trigger_idx[idx,0]
    n_batch = int(np.ceil(time_step / plotting_freq))
    dtype = np.int16
    image = np.zeros([nx,ny], dtype = dtype).T
    image[:] = 0
    im_max = n_batch
    pts_global = np.array([0,0,0])
    prev_clusters = {}
    print(f"n_im = {n_im}")

    for j in tqdm(range(0,n_im), "iterating over triggers"):
        local_slice = slice(trigger_idx[j,0], trigger_idx[j,1])
        #try:
        clusters = evt_track(cd_data['x'][local_slice],
                             cd_data['y'][local_slice],
                             cd_data['p'][local_slice],
                             cd_data['t'][local_slice],
                             clusters,
                             R_multiple = R_multiple,
                             R_max = R_max,
                             R_min = R_min,
                             alpha_x = alpha_x,
                             alpha_w = alpha_w,
                             alpha_r = alpha_r,
                             weight_min = weight_min,
                             nx = nx,
                             ny = ny,
                            )
        #except ValueError as ve:
        #    print(ve)
        #    continue

        for z in [1]:
            pts = clusters[1:,:2,z][:,::-1]
            z_loc = j*np.ones(pts.shape[0])[:,None]
            pts = np.hstack([z_loc,pts])
            if pts.size == 0:
                break
            pts_global = np.vstack([pts_global, pts])


    _, unique_vals = np.unique(pts_global[:,1:], axis = 0, return_index = True)
    print(len(unique_vals), pts_global.shape[0])
    pts_global = pts_global[unique_vals]
    print(len(unique_vals), pts_global.shape[0])
    data_dict = pd.DataFrame.from_dict({'frame':pts_global[:,0], 'x':pts_global[:,2], 'y':pts_global[:,1]})

    return pts_global, data_dict
