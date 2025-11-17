"""

    Executable for running the event RVT on a dataset with filtering options.
    Use -h to see all the flags

"""
import numpy as np
import pandas as pd
import pickle
import trackpy as tp
from argparse import ArgumentParser, BooleanOptionalAction
from mpi4py import MPI
from pathlib import Path
from pickle import dump
from tifffile import imwrite
from tqdm import tqdm

from cy_im_utils.event.read_hdf5 import __read_hdf5__
from cy_im_utils.event.integrate_intensity import fetch_trigger_indices
from cy_im_utils.event.hot_px_filter import calc_hot_px, hot_px_cd_filter
from cy_im_utils.event.event_rvt import event_rvt_filter, gen_r_kernel
from cy_im_utils.event.event_filter_interpolation import event_filter_interpolation_compiled

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
master_rank = size-1
n_workers = master_rank


def worker(args):
    trackpy_params = {
            "minmass":args.tp_minmass,
            "diameter":args.tp_diameter,
            "threshold":args.tp_threshold,
            }

    # RVT Params
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
    alpha = args.alpha
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
        event_rvt_filter(
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
        
        if j < args.n_im:
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
               args,
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

    z = args.hot_px_z
    print(f"filtering hot px with z = {z}")
    hot_px_map = calc_hot_px(cd_data, z = z, dtype = np.float32)
    hot_px_bool = hot_px_cd_filter(hot_px_map, cd_data)
    cd_data = cd_data[hot_px_bool]

    noise_filter_kwargs = {
            'frame_size':np.array([ny, nx], dtype = np.int64),
            'filter_length':args.evt_filter_length,
            'scale':args.evt_filter_scale,
            'update_factor':args.evt_filter_update_factor,
            'interpolation_method':args.evt_filter_method,
            }
    print(f"Applying Interpolation Filter")
    for key,val in noise_filter_kwargs.items():
        print(f"\t{key}: {val}")
    noise_filter = event_filter_interpolation_compiled(**noise_filter_kwargs)
    noise_filter.processEvents(cd_data)
    cd_data = cd_data[noise_filter.eventsBin]

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
    parser.add_argument(
                        "--event-filter-length",
                        dest="evt_filter_length",
                        type = int,
                        help = "interpolation noise filter length (us)",
                        default = None
                        )
    parser.add_argument(
                        "--event-filter-update-factor",
                        dest="evt_filter_update_factor",
                        type = float,
                        help = "interpolation noise filter update factor (0-1)",
                        default = None
                        )
    parser.add_argument(
                        "--event-filter-scale",
                        dest="evt_filter_scale",
                        type = int,
                        help = "interpolation noise filter scale",
                        default = None
                        )
    parser.add_argument(
                        "--event-filter-method",
                        dest="evt_filter_method",
                        type = int,
                        help = "interpolation noise filter method (0-3)",
                        default = None
                        )
    parser.add_argument(
                        "--n-im-to-write",
                        dest="n_im",
                        type = int,
                        help = "# images to write, set to -1 for no imwrite",
                        default = -1
                        )
    parser.add_argument(
                        "--hot-px-z",
                        dest="hot_px_z",
                        type = float,
                        help = "# standard deviations above mean to filter out",
                        default = 30
                        )
    parser.add_argument(
                        "--alpha",
                        dest="alpha",
                        type = float,
                        help = "decay factor for RVT filter",
                        default = 1e-5
                        )

    return parser.parse_args()


if __name__ == "__main__":
    nx, ny = 1280, 720
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
                args = args
                )
        print("-->",trigger_indices.shape)

    trigger_indices = comm.bcast(trigger_indices, root = master_rank)

    if rank == master_rank:
        master(args)
    elif rank != master_rank:
        worker(args)
