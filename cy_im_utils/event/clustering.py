from tqdm import tqdm
import numpy as np
import cupy as cp
import pandas as pd
from sklearn import cluster
from joblib import Parallel, delayed
from filterpy.kalman import ExtendedKalmanFilter


def fetch_triggered_events(cd_data,
                           eventBin,
                           trigger_idx_0: int = 0,
                           trigger_idx_1: int = -1,
                           acc_time: int = 12195,
                           outliers = np.array([[210,470],[167,218]]),
                           ) -> tuple:
    """
    this function fetches the events inside the triggering window that are
    valid according to the "noise filter"

    Maybe this should be a method of the eb-fb fusion gui...? 
    """
    if trigger_idx_1 < 1000 and trigger_idx_1 != -1:
        print(f"are you sure {trigger_idx_1} (this is the time-based index not the trigger-based index)")
    t_handle = cd_data['t']
    filter_slice =  (t_handle >= t_handle[trigger_idx_0]) * \
                    (t_handle < t_handle[trigger_idx_1]) * \
                    eventBin
    #print(np.sum(filter_slice))
    x_ = cd_data['x'][filter_slice]
    y_ = cd_data['y'][filter_slice]
    t_ = cd_data['t'][filter_slice] / acc_time
    t_ -= t_[0]
    for a,b in outliers:
        slice_ = (y_ != a) * (x_ != b)
        x_ = x_[slice_]
        y_ = y_[slice_]
        t_ = t_[slice_]
        
    return x_, y_, t_


def associate_particles_with_events(track_dict: dict,
                                    x_: np.array,
                                    y_: np.array,
                                    t_: np.array,
                                    batch_size: int = 1_000,
                                    thresh: float = 10,
                                   ) -> np.array:
    """
    This takes the output of trackpy linking and associates individual events
    with that track signal that are within a specified spatio-temporal
    threshold
    """
    n_batch = int(np.ceil(len(x_) / batch_size))
    assoc = np.zeros_like(x_)
    for j, particle in tqdm(enumerate(track_dict['particle'].unique())):
        particle_slice = track_dict['particle'] == particle
        particle_x = track_dict['x'][particle_slice].values
        particle_y = track_dict['y'][particle_slice].values
        particle_t = track_dict['frame'][particle_slice].values
        for q in tqdm(range(n_batch)):
            batch_slice = slice(q*batch_size, (q+1)*batch_size)
            dx = cp.array(particle_x[:,None]) - cp.array(x_[None,batch_slice])
            dy = cp.array(particle_y[:,None]) - cp.array(y_[None,batch_slice])
            dt = cp.array(particle_t[:,None]) - cp.array(t_[None,batch_slice])
            dr = (dx**2 + dy**2 + dt**2)**(1/2)
            bool_arr = cp.where(dr < thresh)[1].get()
            assoc[batch_slice][bool_arr] = particle
    return assoc


def extract_events(slice_idx):
    ds = 10
    
    for j, elem in enumerate(np.unique(assoc)):
        # elem 0 is noise?
        if elem == 0:
            continue
        if elem != slice_idx:
            continue
        slice_ = assoc == elem
        x_temp = x_[slice_]
        y_temp = y_[slice_]
        t_temp = t_[slice_]
        ev_bundle = np.vstack([t_temp, y_temp, x_temp]).T
        inst.viewer.add_points(
                            ev_bundle[::ds],
                            face_color = colors[j],
                            name = f"evs {slice_idx}"
                        )


def dbscan_tracking(trigger_indices, cd_data, fr_init: int = 1, kwargs: dict = {}, mode:str = 'flat') -> pd.DataFrame:
    """

    ULTRA BASIC
        
        - trigger_indices: np.array - the t0 t1 index of the timestamps or triggers
        - cd_data: np.array 
        - kwargs: dict - passed to cluster.DBSCAN -> e.g., eps = 10, min_samples = 5
        - mode: string - (flat or normal) flat means that each event counts for 1 vote in the mean, crazy pixels cannot dominate, otherwise normal will take all the events


    """
    print("[WARN] -> use the parallel version for ~6x speedup ")
    located = []
    for j, elem in tqdm(enumerate(trigger_indices)):
        if j < fr_init:
            continue
        slice_ = slice(*elem)
        x,y,t = [cd_data[slice_][key] for key in ['x','y','t']]
        X = np.vstack([x,y]).T
        labels = cluster.DBSCAN(**kwargs).fit_predict(X)
        located_local = []
        for label in np.unique(labels):
            if label == 1:
                continue
            label_bool = labels == label
            x_slice = x[label_bool]
            y_slice = y[label_bool]
            t_slice = t[label_bool]
            if mode == 'flat':
                x_mean = np.unique(x_slice).mean()
                y_mean = np.unique(y_slice).mean()
                t_mean = np.unique(t_slice).mean()
            elif mode == 'normal':
                x_mean = x_slice.mean()
                y_mean = y_slice.mean()
                t_mean = t_slice.mean()
            n_events = np.sum(label_bool)
            located_local.append([j, x_mean, y_mean, t_mean, n_events])
        located.append(pd.DataFrame(located_local, columns = ['frame','x','y','t','num events']))
    return pd.concat(located)


def dbscan_tracking_par(trigger_indices, cd_data, fr_init: int = 1, kwargs: dict = {}, mode:str = 'flat') -> pd.DataFrame:
    """

    ULTRA BASIC
        
        - trigger_indices: np.array - the t0 t1 index of the timestamps or triggers
        - cd_data: np.array 
        - kwargs: dict - passed to cluster.DBSCAN -> e.g., eps = 10, min_samples = 5
        - mode: string - (flat or normal) flat means that each event counts for 1 vote in the mean, crazy pixels cannot dominate, otherwise normal will take all the events

    """
    def _track_ops_(j,elem):
        if j < fr_init:
            return
        slice_ = slice(*elem)
        x,y,t = [cd_data[slice_][key] for key in ['x','y','t']]
        X = np.vstack([x,y]).T
        labels = cluster.DBSCAN(**kwargs).fit_predict(X)
        located_local = []
        for label in np.unique(labels):
            if label == 1:
                continue
            label_bool = labels == label
            x_slice = x[label_bool]
            y_slice = y[label_bool]
            t_slice = t[label_bool]
            if mode == 'flat':
                x_mean = np.unique(x_slice).mean()
                y_mean = np.unique(y_slice).mean()
                t_mean = np.unique(t_slice).mean()
            elif mode == 'normal':
                x_mean = x_slice.mean()
                y_mean = y_slice.mean()
                t_mean = t_slice.mean()
            n_events = np.sum(label_bool)
            located_local.append([j, x_mean, y_mean, t_mean, n_events])
        return pd.DataFrame(located_local, columns = ['frame','x','y','t','num events'])

    delayed_call = [delayed(_track_ops_)(j,elem) for j, elem in tqdm(enumerate(trigger_indices))]
    located = Parallel(n_jobs = -1)(delayed_call)

    return pd.concat(located)


def interpolate_x_y_timestamps(track_dict, dt) -> pd.DataFrame:
    """

    Once a dataset has been tracked (with the average timestamps -> see above)
    this takes the linked dataset and interpolates the timestamps back onto the
    regular temporal grid (MSD analysis, etc.)

    """
    time_steps = np.arange(0, (track_dict['frame'].values[-1] + 1)*dt, dt)
    new_tracks = []
    for particle, df in tqdm(track_dict.groupby("particle")):
        x,y,t,frame = df[['x','y','t','frame']].values.T
        x_flat = np.interp(time_steps, t, x, left = np.nan, right = np.nan)
        y_flat = np.interp(time_steps, t, y, left = np.nan, right = np.nan)
        frame_flat = np.interp(time_steps, t, frame, left = np.nan, right = np.nan)
        frame_coords = np.where(np.isfinite(frame_flat))[0]
        numel = len(frame_coords)
        new_tracks.append(pd.DataFrame({
            "particle":np.ones(numel)*particle,
            "x":x[frame_coords],
            "y":y[frame_coords],
            "frame":frame_coords,
            }))
    new_tracks = pd.concat(new_tracks)
    return new_tracks
