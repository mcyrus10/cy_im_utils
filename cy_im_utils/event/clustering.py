from tqdm import tqdm
import numpy as np
import cupy as cp

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

