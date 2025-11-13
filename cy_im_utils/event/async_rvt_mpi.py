import numpy as np
import pandas as pd
import pickle
import trackpy as tp
from argparse import ArgumentParser, BooleanOptionalAction
from mpi4py import MPI
from numba import njit, prange, void, uint16, int16, int64, float32, float64
from pathlib import Path
from pickle import dump
from sys import path
from tifffile import imwrite
from tqdm import tqdm
paths = [
        "C:\\Users\\mcd4\\Documents\\cy_im_utils",
        "/home/mcd4/cy_im_utils",
        "/mnt/isgnas/home/mcd4/cy_im_utils"
        ]
for elem in paths:
    path.append(elem)

from cy_im_utils.event.read_hdf5 import __read_hdf5__
from cy_im_utils.event.integrate_intensity import fetch_trigger_indices
from cy_im_utils.event.hot_px_filter import calc_hot_px, hot_px_cd_filter

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
master_rank = size-1
n_workers = master_rank


def gen_r_kernel(r: float, rmax: int, dtype = np.float32) -> np.array:
    """
    This is taken from the RVT code....creates array with uni
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
def __process_events_to_rvt_filter__(x, y, p, t, mean_filter_state, timestamps, 
                                     time_surface, image, kernels, r_max, alpha):
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
    

    """
    nx,ny = 1280,720
    numel = x.size
    kz,kx,ky = kernels.shape
    noise_filter_timestamps = np.zeros((nx,ny), dtype = np.int64)
    noise_kernel = 3
    half_kern = noise_kernel // 2
    noise_filter_length = 1000
    kern_z, kern_x, kern_y = np.where(kernels)
    n_kern = len(kern_z)
    for j  in range(numel):
        _x_, _y_, _p_, _t_ = x[j], y[j], p[j], t[j]
        x_min = _x_ - r_max < 0
        x_max = _x_ + r_max >= nx
        y_min = _y_ - r_max < 0
        y_max = _y_ + r_max >= ny
        p_cond = _p_ == 1
        if x_min or x_max or y_min or y_max:# or p_cond:
            continue
        x_min = max(0, _x_ - half_kern)
        x_max = min(nx-1, _x_ + half_kern+1)
        y_min = max(0, _y_ - half_kern)
        y_max = min(ny-1, _y_ + half_kern+1)

        dt = _t_ - noise_filter_timestamps[x_min:x_max,y_min:y_max]
        noise_filter_timestamps[x_min:x_max,y_min:y_max] = _t_
        if dt.min() > noise_filter_length:
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


def worker(args):
    write_images = args.write_images
    trackpy_params = {
            "minmass":args.tp_minmass,
            "diameter":args.tp_diameter,
            "threshold":args.tp_threshold,
            }

    # RVT Params
    nx, ny = 1280, 720
    r_max = args.r_max
    r_min = args.r_min
    r_step = args.r_step
    radii = np.arange(r_min, r_max, r_step)
    n_radii = len(radii)
    kernels = np.zeros([n_radii, r_max*2+1, r_max*2+1], dtype = np.float32)
    for j, r in enumerate(radii):
        kernels[j] = np.array(gen_r_kernel(r,r_max), dtype = np.float32)

    mean_filter_state = np.zeros([n_radii, nx, ny], dtype = np.float32)
    timestamps = np.zeros([nx,ny], dtype = np.float32)
    image = np.zeros([nx,ny], dtype = np.int16)
    alpha = 1e-5
    print("alpha = ",alpha)
    time_surface = np.zeros([2, nx, ny], dtype = np.float32)
    idx_0 = 0
    idx_1 = idx_0 + n_im
    my_tracks = []
    for j in range(idx_0,idx_1):
        dest = j % n_workers
        if dest != rank:
            continue
        events = comm.recv(source = master_rank, tag = dest)
        image[:] = 0
        time_surface[:] = 0
        mean_filter_state[:] = 0
        __process_events_to_rvt_filter__(
                                         events['x'],
                                         events['y'],
                                         events['p'],
                                         events['t'],
                                         mean_filter_state,
                                         timestamps,
                                         time_surface,
                                         image,
                                         kernels,
                                         r_max,
                                         alpha,
                                        )
        VoM = np.var(mean_filter_state, axis = 0)
        
        if write_images:
            f_name = f"_images_/image_{j:06d}.tif"
            imwrite(f_name, VoM.T)
        
        # Tracking
        tracks = tp.locate(VoM, **trackpy_params)
        frame = np.ones(len(tracks)) * j
        tracks['frame'] = frame
        my_tracks.append(tracks)
    
    print(f"rank: {rank} finished iterating")
    tracks = pd.concat(my_tracks)
    dump(tracks, open(f"_located_/located_frames_{rank:03d}.p","wb"))
    print(f"rank: {rank} finished saving tracks")


def master(args):
    idx_1 = idx_0 + n_im
    for j, idx in tqdm(enumerate(range(idx_0,idx_1)), desc = "master progress"):
        dest = j % n_workers 
        slice_ = slice(trigger_indices[idx,0], 
                       trigger_indices[idx,1])
        cd_data_local = cd_data[slice_]
        comm.send(cd_data_local, dest = dest, tag = dest)


def fetch_data(data_file: Path, 
               super_sampling: int,
               acc_time: int,
               z = 30
               ) -> tuple:
    assert data_file.is_file(), f"file does not exist:\n\t{data_file}"
    if data_file.suffix == ".hdf5":
        print("reading hdf5")
        cd_data = __read_hdf5__(data_file,"CD")
        triggers = __read_hdf5__(data_file,"EXT_TRIGGER")
    elif data_file.suffix == ".p":
        print("reading pickled data")
        data_temp = pickle.load(open(data_file,"rb"))
        cd_data = data_temp['cd_data']
        triggers = data_temp['trigger_data']
    else:
        assert False, f"unknown file extension: {data_file}"

    print(f"filtering hot px with z = {z}")
    hot_px_map = calc_hot_px(cd_data, z = z, dtype = np.float32)
    hot_px_bool = hot_px_cd_filter(hot_px_map, cd_data)
    cd_data = cd_data[hot_px_bool]

    trigger_indices = fetch_trigger_indices(
                                        triggers['t'],
                                        acc_time,
                                        cd_data['t'],
                                        super_sampling
                                        )
    return cd_data, trigger_indices


def arg_parser():
    parser = ArgumentParser()
    parser.add_argument(
                        "--event-file",
                        dest="event_file_name",
                        type = Path,
                        help = "path to event camera file (.hdf5)",
                        default = ""
                        )
    parser.add_argument(
                        "--acc-time",
                        dest="acc_time",
                        type = int,
                        help = "event camera accumulation time (microseconds)",
                        default = 12195 
                        )
    parser.add_argument(
                        "--super-sampling",
                        dest="ss",
                        type = int,
                        help = "event trigger super sampling",
                        default = 1
                        )
    parser.add_argument(
                        "--r-min",
                        dest="r_min",
                        type = int,
                        help = "maximum radius for RVT scanning hough cone",
                        default = None
                        )
    parser.add_argument(
                        "--r-max",
                        dest="r_max",
                        type = int,
                        help = "minimum radius for RVT scanning hough cone",
                        default = None
                        )
    parser.add_argument(
                        "--r-step",
                        dest="r_step",
                        type = int,
                        help = "integer step for radius sweep",
                        default = None
                        )
    parser.add_argument(
                        "--trackpy-threshold",
                        dest="tp_threshold",
                        type = float,
                        help = "trackpy threshold for locating frames",
                        default = None
                        )
    parser.add_argument(
                        "--trackpy-diameter",
                        dest="tp_diameter",
                        type = float,
                        help = "trackpy diameter for locating frames",
                        default = None
                        )
    parser.add_argument(
                        "--trackpy-minmass",
                        dest="tp_minmass",
                        type = float,
                        help = "trackpy minmass for locating frames",
                        default = None
                        )

    parser.add_argument("--write-images",
                        action = BooleanOptionalAction)


    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parser()
    super_sampling = args.ss
    idx_0, n_im = 0, 2000*super_sampling
    cd_data, trigger_indices = None,None
    if rank == master_rank:
        data_file = args.acc_time
        cd_data, trigger_indices = fetch_data(
                data_file = args.event_file_name,
                super_sampling = args.ss,
                acc_time = args.acc_time,
                )
        print("-->",trigger_indices.shape)

    trigger_indices = comm.bcast(trigger_indices, root = master_rank)

    if rank == master_rank:
        master(args)
    elif rank != master_rank:
        worker(args)
