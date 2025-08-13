#!/home/mcd4/miniconda3/envs/openeb/bin/python
from PIL import Image 
from cupyx.scipy.ndimage import affine_transform, median_filter, gaussian_filter, convolve, convolve1d
from enum import Enum 
from functools import partial 
from magicgui import magicgui
from matplotlib.gridspec import GridSpec
from pathlib import Path
from scipy.optimize import least_squares
from scipy.stats import median_abs_deviation
from tifffile import imread
from tqdm import tqdm
import cupy as cp
import diffusive_distinguishability.ndim_homogeneous_distinguishability as hd
import gc
import matplotlib.pyplot as plt
import napari
import numpy as np
import pandas as pd
import pickle
import trackpy as tp

from sys import path, platform
util_paths = [
        "/home/mcd4/cy_im_utils",
        "C:\\Users\\mcd4\\Documents\\cy_im_utils",
        ]
for elem in util_paths:
    path.append(elem)
from cy_im_utils.imgrvt_cuda import rvt
from cy_im_utils.event.trackpy_utils import (imsd_powerlaw_fit, imsd_linear_fit,
                                             fetch_particle_pairs, re_link)
from cy_im_utils.event.integrate_intensity import (_integrate_events_wrapper_,
                                                   fetch_indices_wrapper)
from cy_im_utils.event.event_filter_interpolation import event_filter_interpolation_compiled
from cy_im_utils.event.clustering import *
from cy_im_utils.event.read_hdf5 import __read_hdf5__
from cy_im_utils.event.hot_px_filter import calc_hot_px, hot_px_cd_filter
from cy_im_utils.parametric_fits import parametric_gaussian, fit_param_gaussian
from cy_im_utils.image_quality import mutual_information


def cp_free_mem() -> None:
    """
    Frees cupy's memory pool...
    """
    mempool = cp.get_default_memory_pool()
    mempool.free_all_blocks()


def plot_joint_hist(arr_0,
                    arr_1,
                    bins = 50,
                    figsize = (8,8), 
                    xlabel = "",
                    ylabel = "",
                    mode = 'match_bins') -> None:
    """
    """
    _k_ = 1.4826
    fig = plt.figure(figsize = figsize)
    gs = GridSpec(4,4)
    ax = [fig.add_subplot(gs[1:,:-1])]
    ax.append(fig.add_subplot(gs[0,:-1], sharex = ax[0]))
    ax.append(fig.add_subplot(gs[1:,-1], sharey = ax[0]))

    ax[0].scatter(arr_0, arr_1)
    ax[0].axline((0,0), (1,1), color = 'k', linestyle = '--')
    hist_data = ax[1].hist(arr_0, bins = bins)
    med_0 = np.nanmedian(arr_0)
    mad_0 = median_abs_deviation(arr_0, nan_policy = 'omit')
    rCV_0 = 100*mad_0*_k_/med_0
    label_0 = f"diam: {med_0:0.2f} nm\nrCV: {rCV_0:0.2f}%"
    ax[1].axvline(med_0, color = 'r', linestyle = '--', label = label_0)
    if mode == 'match_bins':
        bins_2 = hist_data[1]
    else:
        bins_2 = bins
    ax[2].hist(arr_1, bins = bins_2, orientation = 'horizontal')
    med_1 = np.nanmedian(arr_1)
    mad_1 = median_abs_deviation(arr_1, nan_policy = 'omit')
    rCV_1 = 100*mad_1*_k_/med_1
    label_1 = f"{med_1:0.2f} nm\nrCV: {rCV_1:0.2f}%"
    ax[2].axhline(med_1, color = 'r', linestyle = '--', label = label_1)
    for a in [ax[1], ax[2]]:
        a.legend()
    ax[0].set(xlabel=xlabel, ylabel=ylabel)
    fig.tight_layout()


def __calc_msd__(
            track_dict,
            fps: float,
            mpp: float,
            track_id: int = -1,
            msd_point_min: int = 0,
            msd_point_max: int = 5,
            max_lagtime: int = 100,
            temperature: float = 295.0,
            bins: int = 50,
            verbose = True
            ):
        T = temperature
        eta = water_viscosity(T, unit = 'K')
        if verbose: 
            print(f"viscosity: {eta} Pa * s")
        imsd_kwargs = {'mpp':mpp, 'fps':fps, 'max_lagtime': max_lagtime}
        track_handle = track_dict
        track_lengths = []
        bay_step = msd_point_min + 1
        if track_id == -1:
            if verbose: 
                print("calculating all tracks")
            imsd = tp.motion.imsd(track_handle, **imsd_kwargs)
            emsd = tp.motion.emsd(track_handle, **imsd_kwargs)
            bay = []
            for elem in track_handle['particle'].unique():
                particle_slice = track_handle['particle'] == elem
                bay.append(__calc_bayesian__(elem, track_handle, fps, mpp,
                    lagtime_step = bay_step))
                track_lengths.append(np.sum(particle_slice))
            bay = np.vstack(bay)
            #print("--> bay shape all elements:",bay.shape)
        else:
            if verbose: 
                print(f"calculating {track_id}")
            particle_slice = track_handle['particle'] == track_id
            if particle_slice.values.sum() == 0:
                assert False, f"Empty Particle ID {track_id}"
            track_lengths.append(np.sum(particle_slice))
            imsd = tp.motion.imsd(track_handle[particle_slice], **imsd_kwargs)
            emsd = imsd
            bay = __calc_bayesian__(track_id,
                                    track_handle,
                                    fps,
                                    mpp,
                                    lagtime_step= bay_step)[None,:]
            #print("--> bay shape 1 element:",bay.shape)

        # Log-Log Fit
        A,n_log,log_fits = imsd_powerlaw_fit(imsd, 
                                             start_index = msd_point_min,
                                             end_index = msd_point_max,
                                             verbose = verbose)

        A_ensemble,n_log_ensemble,log_fits_ensemble = imsd_powerlaw_fit(
                                             emsd, 
                                             start_index = msd_point_min,
                                             end_index = msd_point_max,
                                             verbose = verbose)

        # Linear Fit
        m,b,lin_fits = imsd_linear_fit(imsd, 
                                       start_index = msd_point_min,
                                       end_index = msd_point_max,
                                       verbose = verbose
                                       )

        m_ensemble,b_ensemble,lin_fits_ensemble = imsd_linear_fit(
                                       emsd, 
                                       start_index = msd_point_min,
                                       end_index = msd_point_max,
                                       verbose = verbose
                                       )

        kb = 1.38e-23
        diffusivity_log = np.exp(A)/4
        diam_log = kb * T / (3 * np.pi * eta * diffusivity_log * 1e-12) * 1e9
        diffusivity_lin = m/4
        diam_lin = kb * T / (3 * np.pi * eta * diffusivity_lin * 1e-12) * 1e9
        neg_diams = diam_lin < 0
        bay_diam = kb * T / (3 * np.pi * eta * bay[:,0] * 1e-12) * 1e9

        return  {
                'imsd':imsd,
                'emsd':emsd,
                'bay':bay,
                'bay_diam':bay_diam,
                'A':A,
                'n_log':n_log,
                'log_fits':log_fits,
                'A_ensemble':A_ensemble,
                'n_log_ensemble':n_log_ensemble,
                'log_fits_ensemble':log_fits_ensemble,
                'm':m,
                'b':b,
                'lin_fits':lin_fits,
                'm_ensemble':m_ensemble,
                'b_ensemble':b_ensemble,
                'lin_fits_ensemble':lin_fits_ensemble,
                'diffusivity_log':diffusivity_log,
                'diam_log':diam_log,
                'diffusivity_lin':diffusivity_log,
                'diam_lin':diam_lin,
                'neg_diams': neg_diams,
                'track_lengths': np.array(track_lengths),
                }


def __calc_bayesian__(track_id, 
                      track_handle,
                      fps,
                      mpp,
                      lagtime_step = 1) -> np.array:
    """
    wrapper for the call to hd.estimate_diffusion...

    """
    particle_slice = track_handle['particle'] == track_id
    # Convert displacement from pixels to nm?
    fr_handle = track_handle['frame'][particle_slice].values
    dframe = np.abs(fr_handle[lagtime_step:] - fr_handle[:-lagtime_step])
    frame_drop_bool = dframe != lagtime_step
    x_handle = track_handle['x'][particle_slice].values
    y_handle = track_handle['y'][particle_slice].values
    dx = x_handle[lagtime_step:] - x_handle[:-lagtime_step]
    dy = y_handle[lagtime_step:] - y_handle[:-lagtime_step]
    dr = np.sqrt(dx**2 + dy**2) * mpp
    dr[frame_drop_bool] = np.nan
    posterior, alpha, beta = hd.estimate_diffusion(
            n_dim = 2,
            dt = 1 / fps,
            dr = dr[np.isfinite(dr)]
            )
    bay_diffusivity = np.array([posterior.mean(), posterior.std()])

    return bay_diffusivity


def residual_affine(mat: list, input_pair: np.array, target_pair: np.array) -> np.array:
    """
    This function is used by scipy.optimize.least_squares to fit the affine
    transform matrix

    Note that the conventional x,y cartesian coordinates are switched when
    looking at plt.imshow so the y coordinate is horizontal and x coordinate is
    vertical...

    Parameters:
    -----------
        - mat: array-like; fitted parameters for scale y, horizontal shear, y
          translation, vertical shear, scale x, translation x
        - input_pair: array of event coordinates (y,x,1)
        - target_pair: array of frame coordinates (y,x,1) that map 1-1 to the
          event coordinates
    """
    #print(mat)
    scale_y = mat[0]
    horz_shear = mat[1]
    translation_y = mat[2]
    vert_shear = mat[3]
    scale_x = mat[4]
    translation_x = mat[5]
    tform_array = np.array([
        [scale_y, horz_shear, translation_y],
        [vert_shear, scale_x, translation_x],
        [0, 0, 1]])
    out = np.dot(tform_array, input_pair)
    diff = out - target_pair
    return np.abs(diff).flatten()


def residual_rigid(mat: list, input_pair: np.array, target_pair: np.array) -> np.array:
    """
    This function is used by scipy.optimize.least_squares to fit a rigid
    transform matrix (scale, translation and rotation (NO SHEARING))

    Note that the conventional x,y cartesian coordinates are switched when
    looking at plt.imshow so the y coordinate is horizontal and x coordinate is
    vertical...

    Parameters:
    -----------
        - mat: array-like; fitted parameters for scale y, horizontal shear, y
          translation, vertical shear, scale x, translation x
        - input_pair: array of event coordinates (y,x,1)
        - target_pair: array of frame coordinates (y,x,1) that map 1-1 to the
          event coordinates
    """
    #print(mat)
    #scale = mat[0]
    scale = mat[0]
    translation_y = mat[1]
    translation_x = mat[2]
    theta = mat[3]
    tform_array = np.array([
        [scale * np.cos(theta), np.sin(theta), translation_y],
        [-np.sin(theta), scale * np.cos(theta), translation_x],
        [0, 0, 1]])
    out = np.dot(tform_array, input_pair)
    diff = out - target_pair
    return np.abs(diff).flatten()


def water_viscosity(T, unit = 'C') -> float:
    """
    returns viscosity of water as a function of temperature
    Input temperature unit can be 'C' or 'K'
    Output is Dynamics viscosity in units of Pa*s

    Source for formula
    https://en.wikipedia.org/wiki/Temperature_dependence_of_viscosity, which in
    turn cites: Reid, Robert C.; Prausnitz, John M.; Poling, Bruce E. (1987),
    The Properties of Gases and Liquids, McGraw-Hill Book Company, ISBN
    0-07-051799-1
    """
    if unit == 'C':
        T += 273.15
    elif unit == 'K':
        T = T
    else:
        print ('Temperature unit unknown. C and K are supported')
        return
    A = 1.856e-14 #Pa*s
    B = 4209 # K
    C = 0.04527 # K^-1
    D = -3.376e-5 # K^-2
    return A*np.exp(B/T + C*T +D*T**2)


def fetch_cauchy_kernel(x, gamma, sigma = None, x0 = 0) -> np.array:
    """
    normalized lorentzian kernel (sum = 1)
    """
    kern = 1/(np.pi*gamma*(1 + ((x-x0)/gamma)**2))
    return kern / np.sum(kern)


def fetch_gaussian_kernel(x, sigma, gamma = None, x0 = 0) -> np.array:
    """
    Normalized gaussian kernel (sum = 1)
    """
    denom = np.sqrt(2*np.pi*sigma**2)
    num = np.exp(-(x-x0)**2/(2*sigma**2))
    kern = num / denom
    return kern / np.sum(kern)


def fetch_voigt_kernel(x, sigma, gamma, x0 = 0) -> np.array:
    """
    normalized voigt kernel (sum = 1)
    """
    cauchy = fetch_cauchy_kernel(x, gamma)
    gaussian = fetch_gaussian_kernel(x, sigma)
    kern = np.convolve(cauchy, gaussian, mode = 'same')
    return kern / np.sum(kern)


def fetch_dirac_kernel(x, sigma, gamma, x0 = 0) -> np.array:
    """
    normalized voigt kernel (sum = 1)
    """
    kern = np.zeros_like(x)
    numel = len(kern)
    assert numel % 2 == 1, "Dirac must have odd kernel shape"
    kern[numel // 2] = 1
    return kern


def fetch_square_kernel(x, sigma, gamma, x0 = 0) -> np.array:
    """
    normalized square unweighted integrator (sum = 1)
    """
    kern = np.ones_like(x)
    return kern / np.sum(kern)


def __hdf5_to_numpy__(
                    event_file,
                    cd_data, 
                    acc_time: float, 
                    thresh: float,
                    num_images: int,
                    super_sampling: int,
                    width: int = 720,
                    height: int = 1280,
                    dtype= np.int16,
                    omit_neg: bool = False,
                    ) -> (np.array, np.array):
    """
    This is for loading in an hdf5 file so that the exposure time of the
    frame camera can be matched to the event signal directly.....
    
    Just use the regular raw -> numpy function if you want a finer frame
    rate sampling since this will load the data with gaps!!!
    (discontinuous event data)
    """
    trigger_data = __read_hdf5__(event_file, "EXT_TRIGGER")['t']
    if trigger_data.shape[0] == 0 and num_images != -1:
        print("No triggers found")
        trigger_data = None
    else:
        print(f"trigger shape = {trigger_data.shape} (2x the triggers)")
    trigger_indices = fetch_indices_wrapper(
                                        trigger_data,
                                        acc_time,
                                        cd_data['t'],
                                        super_sampling = super_sampling,
                                        frame_comp_triggers = num_images
                                        )

    print("fetched indices")
    if num_images == -1:
        print("sampling all triggers")
        n_im = trigger_indices.shape[0]
    else:
        print(f"only taking first {num_images} images (subset of total triggers)")
        n_im = num_images
    image_stack = np.zeros([n_im, width, height], dtype = dtype)
    image_buffer = np.zeros([width, height], dtype = dtype)
    integrator_fun = _integrate_events_wrapper_(dtype)

    for j in tqdm(range(n_im), desc = "reading hdf5"):
        id_0, id_1 = trigger_indices[j]
        image_buffer[:] = 0
        slice_ = slice(id_0, id_1, 1)
        integrator_fun(image_buffer, 
                       cd_data['x'][slice_],
                       cd_data['y'][slice_], 
                       cd_data['p'][slice_].astype(bool),
                       omit_neg
                       )
        image_buffer[np.abs(image_buffer) > thresh] = 0
        image_stack[j] = image_buffer.copy()

    return image_stack, trigger_indices


class np_dtype(Enum):
    float32 = np.float32
    float64 = np.float64
    uint32 = np.uint32
    int32 = np.int32
    uint16 = np.uint16
    int16 = np.int16
    uint8 = np.uint8
    int8 = np.int8


class filter_1d_type(Enum):
    gaussian_1D = "Gaussian",partial(fetch_gaussian_kernel)
    cauchy_1D = "Cauchy",partial(fetch_cauchy_kernel)
    voigt_1D = "Voigt",partial(fetch_voigt_kernel)
    dirac_1D = "Dirac",partial(fetch_dirac_kernel)
    square_1D = "Square",partial(fetch_square_kernel)


class median_mode_type(Enum):
    reflect = "reflect"
    constant = "constant"
    nearest = "nearest"
    mirror = "mirror"
    wrap = "wrap"


class transform_type(Enum):
    """
    Fetches the residual function and default x0
    """
    rigid = residual_rigid, [1,0,0,0], 'rigid'
    affine = residual_affine, [1,0,0,1,0,0], 'affine'


def smart_slice(arr: np.array, x0: int, x1: int, y0: int, y1: int) -> np.array:
    """
    This funciton is for padding an array slice with zeros if the slice extends
    beyond the edge of the array
    """
    nx, ny = arr.shape
    dx_0, dx_1, dy_0, dy_1 = 0,0,0,0

    if x1 > nx:
        dx_1 = x1-nx
    if y1 > ny:
        dy_1 = y1-ny
    if x0 < 0:
        dx_0 = -x0
        x0 = 0
    if y0 < 0:
        dy_0 = -y0
        y0 = 0
    slice_ = (slice(x0, x1), slice(y0, y1))
    temp = arr[slice_]
    pad_shape = ((dx_0, dx_1), (dy_0, dy_1))
    return np.pad(temp, pad_shape)


def batch_axial_median(arr, x_batch_size, y_batch_size, size, mode = 'reflect') -> np.array:
    n_im, nx, ny = arr.shape
    out = np.zeros_like(arr)
    batch_x = int(np.ceil(nx / x_batch_size))
    batch_y = int(np.ceil(ny / y_batch_size))
    size_arg = (size, 1, 1)
    for i in tqdm(range(batch_x), desc = "batch median filter GPU"):
        slice_x = slice(i*x_batch_size, (i+1)*x_batch_size)
        for j in range(batch_y):
            slice_y = slice(j*y_batch_size, (j+1)*y_batch_size)
            cp_arr = cp.array(arr[:,slice_x, slice_y], dtype = arr.dtype)
            out[:,slice_x, slice_y] = median_filter(cp_arr, 
                                                    size = size_arg, 
                                                    mode = mode).get()
    return out




class spatio_temporal_registration_gui:
    def __init__(self):
        self.viewer = napari.Viewer()
        self.viewer.title = "Event-Frame Registration GUI"
        self.pull_affine_matrix = None
        self.affine_matrix = None
        self.located_frames = {}
        self.frame_0 = 0
        self.frame_track = {}
        self.track_bool = False
        self.track_holder = {}
        self.reduced_data = {}
        self.super_sampling = 1
        self.hot_px_layer = None
        self.frame_size = np.array([720, 1280], dtype = np.int64)
        self.global_translation = {}
        self.noise_floor  = None

        dock_widgets = {
                'Data Loading': [
                              self._load_frame_(),
                              self._load_event_(),
                              self.__calculate_hot_pixels__(),
                              self._filter_hot_pixels_(),
                              self._preview_event_noise_filter_(),
                              self._apply_event_noise_filter_(),

                              ],
                'Registration': [
                              self._flip_(),
                              self._align_sub_super_sampled_frame_(),
                              self._load_affine_mat_(),
                              self._reset_affine_(),
                              self._fit_affine_(),
                              self._apply_transform_(),
                              self._translate_layer_(),
                              #self._mutual_information_(),
                              #self._zip_centroids_to_points_(),
                              ],
                'Filtering 1':[
                            self._isolate_event_sign_(),
                            #self._combine_event_channels_(),
                            self._abs_of_layer_(),
                            self._preview_rvt_filter_(),
                            self._apply_rvt_to_layer_(),
                            self._apply_gaussian_layer_(),
                            self._apply_median_layer_(),
                            self._apply_conditional_median_layer_(),
                                 ],
                'Filtering 2':[
                            self._filter_1d_(),
                            self._remove_pixels_(),
                            self._subtract_axial_median_gpu_(),
                            self._calc_median_background_(),
                    ],
                'Tracking':[
                            self._preview_track_centroids_(),
                            self._track_batch_locate_(),
                            self._track_link_(),
                            self._calc_msd_(),
                              ],
                'Fusion':[
                            self._estimate_matched_particles_(),
                            self._add_match_points_(),
                            ],
                'Vis':[
                            self._figure_(),
                            self._violin_plot_(),
                            self._joint_hist_(),
                            self._plot_particle_iso_()
                              ],
                'Utils':[
                    self.__free_memory__(),
                    self.__add_empty_image__(),
                    self.__transform_tracks_coordinate_system__(),
                    self.__sub_sample_tracks__(),
                    self.__compute_noise_floor__(),
                    self.__compute__(),
                    self.__unspecified_colormap__(),
                    self.__measure_line_length__(),
                    self.__particles_in_rectangle__(),
                    ]
                }
        tabs = []
        for j,(key,val) in enumerate(dock_widgets.items()):
            handle = self.viewer.window.add_dock_widget(val,
                                               name = key,
                                               add_vertical_stretch = False,
                                               area = 'right'
                                               )
            tabs.append(handle)
            if j > 0:
                self.viewer.window._qt_window.tabifyDockWidget(
                        tabs[0],
                        handle
                        )
        self.total_shift = 0

    #--------------------------------------------------------------------------
    #                               LOADING DATA    
    #--------------------------------------------------------------------------
    def _load_event_(self):
        @magicgui(call_button="load event data",
                  main_window = True,
                  persist = True,
                  layout = 'vertical',
                  event_file = {"label": "Select Event File (.hdf5)"},
                  load_event_bool = {"label": "Load Event"},
                  acc_time = {'label': "Accumulation Time",'max':1e16},
                  num_images = {'label': "Number of Images (-1 for all triggers)",'max':1e16},
                  event_thresh = {'label': "Dead Pixel Threshold",'max':1e16},
                  super_sampling = {'label': "Samples per trigger"},
                  dtype = {'label': "Data type"},
                  omit_negative = {'label': "Omit Negative Polarity"}
                  )
        def inner(
                event_file: Path = Path.home(),
                load_event_bool: bool = True,
                acc_time: float = 0.0,
                event_thresh: float = 0.0,
                num_images: int = -1,
                super_sampling: int = 1,
                dtype: np_dtype = np_dtype.int16,
                omit_negative: bool = False
                ):
            self.event_file = event_file
            self.event_acc_time = acc_time
            self.event_dead_px_thresh = event_thresh
            self.num_event_images = num_images
            self.super_sampling = super_sampling
            self.event_dtype = dtype.value
            self.event_omit_neg = omit_negative
            event_files = event_file.as_posix()
            print(event_files)
            msg = "use hdf5 format (metavision_file_to_hdf5 -i <input>.raw -o <output>.hdf5)"
            assert event_file.suffix == ".hdf5", msg
            print("Reading .hdf5 file")
            self.cd_data = __read_hdf5__(event_file, "CD")

            event_stack, self.trigger_indices = __hdf5_to_numpy__(
                                       self.event_file,
                                       self.cd_data,
                                       acc_time,
                                       thresh = event_thresh,
                                       super_sampling = self.super_sampling,
                                       num_images = self.num_event_images,
                                       dtype = self.event_dtype,
                                       omit_neg = self.event_omit_neg
                                               )

            self.viewer.add_image(event_stack, 
                                  colormap = 'coolwarm', 
                                  name = 'event')
        return inner

    def _load_frame_(self):
        @magicgui(call_button="load frame data",
              main_window = True,
              persist = True,
              layout = 'vertical',
              frame_file = {"label": "Select Frame File (.tif)"},
              load_frame_bool = {"label": "Load Frame"},
                  )
        def inner(
                frame_file: Path = Path.home(),
                load_frame_bool: bool = True,
                ):
            self.frame_file = frame_file
            frame_files = frame_file.as_posix()
            print(frame_files)
            frame_stack = imread(frame_files)
            self.viewer.add_image(frame_stack, colormap = 'viridis',
                                name = 'frame', opacity = 0.4)

        return inner


    #--------------------------------------------------------------------------
    #                            REGISTRATION
    #--------------------------------------------------------------------------
    def _align_sub_super_sampled_frame_(self):
        @magicgui(call_button="Align layer with super sampled",
                  persist = True,
                  layer_name = {'label': "Layer name"},
                  )
        def inner(layer_name: str):
            n_im = self.num_event_images
            if self.super_sampling == 1:
                print((f"Super sampling == 1 --> Matching {layer_name} to event"
                   f" number of images ({n_im})\n--> OPERATING IN PLACE <--"))
                handle = self.__fetch_layer__(layer_name)
                handle.data = handle.data[:n_im]
            else:
                handle = self.__fetch_layer__(layer_name).data
                out_shape = [n_im, handle.shape[1], handle.shape[2]]
                sparse = np.zeros(out_shape, dtype = handle.dtype)
                global_idx = 0
                sparse[0] = handle[0]
                for j in tqdm(range(1,n_im), desc = f"sparsifying {layer_name}"):
                    sparse[j] = handle[global_idx]
                    if j % self.super_sampling == 0:
                        global_idx += 1
                self.viewer.add_image(sparse, 
                                      name = f"{layer_name} sparse", 
                                      colormap = 'gist_earth')
        return inner

    def _load_affine_mat_(self):
        @magicgui(
                call_button="Load affine transform",
                persist = True,
                f_name = {'label': "Affine Transform File Name (.npy)"}
                )
        def inner(f_name: Path):
            affine = np.load(f_name)
            self.affine_matrix = affine
            self.pull_affine_matrix = np.linalg.inv(affine)
            print("Loaded Affine Matrix From File")
        return inner

    def _fit_affine_(self):
        @magicgui(
                  call_button="Fit/Refit Affine Params",
                  mode = {'label':'mode'}
                )
        def inner(
                mode: transform_type
                ):
            handle = self.viewer.layers[-1]
            _points_type_ = napari.layers.points.points.Points
            assert isinstance(handle, _points_type_), "not a points layer"
            data_handle = handle.data
            event_pairs = data_handle[::2,1:]
            if self.affine_matrix is None and self.pull_affine_matrix is None:
                if 'event' in self.global_translation:
                    print("INCORPORATING GLOBAL LAYER TRANSLATION")
                    event_pairs -= self.global_translation['event'][1:]
            frame_pairs = data_handle[1::2,1:]
            assert event_pairs.shape == frame_pairs.shape, "shape mismatch between pairs"
            n_points = event_pairs.shape[0]
            event_pairs = np.hstack([event_pairs, np.ones(n_points)[:,None]]).T
            frame_pairs = np.hstack([frame_pairs, np.ones(n_points)[:,None]]).T
            tform_func, x0, mode_name = mode.value
            out = least_squares(tform_func, x0, args = (event_pairs, frame_pairs))
            self.affine_fit = out

            if mode_name == 'affine':
                composed_mat = np.array([
                    [out.x[0], out.x[1], out.x[2]],
                    [out.x[3], out.x[4], out.x[5]],
                    [0,0,1]
                    ])

            elif mode_name == 'rigid':
                scale = out.x[0]
                sinx = np.sin(out.x[3])
                cosx = np.cos(out.x[3])
                composed_mat = np.array([
                    [scale*cosx, sinx, out.x[1]],
                    [-sinx, scale*cosx, out.x[2]],
                    [0,0,1]
                    ])

            if self.affine_matrix is None and self.pull_affine_matrix is None:
                self.pull_affine_matrix = composed_mat
                self.affine_matrix = np.linalg.inv(composed_mat)
            else:
                print("refining affine matrix")
                self.affine_matrix = self.affine_matrix @ np.linalg.inv(composed_mat)
                self.pull_affine_matrix = np.linalg.inv(self.affine_matrix)

        return inner

    def _apply_transform_(self) -> None:
        """
        still a work in progress, this returns the widget that enables the
        transform operations
        """
        @magicgui(call_button = "Transform")
        def inner(batch_size: int = 50):
            event_handle = self.__fetch_layer__("event").data
            frame_handle = self.__fetch_layer__("frame").data

            assert frame_handle.ndim == 3, "Only 3D time, x, y for frame images"
            assert event_handle.ndim == 3, "not doing colorized raw nonsense any more"
            nz, nx, ny = frame_handle.shape
            _, nx_event, ny_event = event_handle.shape

            tform_pull = cp.eye(4).astype(cp.float32)
            tform_pull[1:,1:] = cp.array(self.pull_affine_matrix, dtype = cp.float32)

            tform_push = cp.eye(4).astype(cp.float32)
            tform_push[1:,1:] = cp.array(self.affine_matrix)

            event_raw_transformed = np.zeros([nz,nx,ny], dtype = np.float32)
            frame_transformed = np.zeros([nz, nx_event, ny_event], dtype = np.float32)

            n_batch = nz // batch_size
            remainder = nz % batch_size
            if remainder > 0:
                n_batch += 1
            cp_free_mem()
            for q in tqdm(range(n_batch)):
                upper_lim = min((q+1)*batch_size, nz)
                local_batch_size = upper_lim - batch_size*q
                slice_ = slice(q*batch_size, upper_lim)
                event_raw_transformed[slice_] = affine_transform(
                    cp.array(event_handle[slice_], dtype = cp.float32),
                    tform_push,
                    output_shape = (local_batch_size,nx,ny),
                    order = 0
                    ).get()
                cp_free_mem()
                frame_transformed[slice_] = affine_transform(
                        cp.array(frame_handle[slice_], dtype = cp.float32),
                        tform_pull,
                        output_shape = (local_batch_size, nx_event, ny_event),
                        order = 3
                        ).get()
                cp_free_mem()

            self.viewer.add_image(frame_transformed, 
                                  visible = False,
                                  name = "frame -> event",
                                  colormap = "viridis"
                                  )
            self.viewer.add_image(event_raw_transformed, 
                                  opacity = 0.4,
                                  name = "event -> frame")
            self.__fetch_layer__("event").visible = False
            self.__fetch_layer__("frame").opacity = 1.0

            # Add Outline of Event camera location wrt frame
            coords = np.array([
                [0,0],
                [0,1280],
                [720,1280],
                [720,0],
                ]).astype(np.float64)
            coords[:,0] -= self.affine_matrix[0,2]
            coords[:,1] -= self.affine_matrix[1,2]
            coords_tformed = coords @ self.pull_affine_matrix[:2,:2].T
            vertices = []
            for j, elem in enumerate(coords_tformed[:-1]):
                vertices.append(np.vstack([elem, coords_tformed[j+1]]))
            vertices.append(np.vstack([coords_tformed[-1], coords_tformed[0]]))
            self.viewer.add_shapes(
                                   vertices,
                                   shape_type = 'line', 
                                   name = 'event overlay',
                                   edge_color = 'black',
                                   edge_width = 3
                                   )
            napari.experimental.link_layers(self.viewer.layers[-2:])
        return inner

    def _reset_affine_(self):
        @magicgui(call_button="reset affine")
        def inner():
            self.pull_affine_matrix = None
            self.affine_matrix = None
            print("Affine Transform set to None")
        return inner

    def _flip_ud_(self):
        @magicgui(call_button="Flip layer Up/Down",
                layer_name = {'label':'Layer name'},
                persist = True  
                )
        def inner(layer_name: str):
            handle = self.__fetch_layer__(layer_name)
            handle.data = handle.data[:,::-1]
            print(f"Flipped {layer_name} UD")
        return inner

    def _flip_lr_(self):
        @magicgui(call_button="flip layer left/right",
                layer_name = {'label':'Layer name'},
                persist = True
                )
        def inner(layer_name: str):
            handle = self.__fetch_layer__(layer_name)
            handle.data = handle.data[:,:,::-1]
            print(f"Flipped {layer_name} LR")
        return inner

    def _flip_(self):
        @magicgui(call_button="Flip layer",
                layer_name = {'label':'Layer name'},
                lr_bool = {'label':"left/right"},
                ud_bool = {'label':"up/down"},
                persist = True
                )
        def inner(layer_name: str, 
                  lr_bool: bool,
                  ud_bool: bool,
                  ):
            handle = self.__fetch_layer__(layer_name)
            if lr_bool:
                handle.data = handle.data[:,:,::-1]
                print(f"Flipped {layer_name} LR")
            if ud_bool:
                handle.data = handle.data[:,::-1]
                print(f"Flipped {layer_name} UD")

        return inner

    def _set_frame_after_shutter_(self):
        """
        This is a slightly subjective point, but is just for finding the point
        where the decay is sufficiently significant that the tracking can work
        properly...
        """
        @magicgui(call_button="Set Frame After Shutter",
                frame_0={'label':'Frame after Shutter','min':0,'max':1e16})
        def inner(frame_0: int = 0):
            self.frame_0 = frame_0
            print(f"first frame set to {self.frame_0}")
        return inner

    def _translate_layer_(self):
        """
        This widget enables translating the canvas of a specific layer so you
        can manually align the features...
        """
        @magicgui(
                call_button="Translate Layer",
                shift_z={'label':'shift z','min':0,'max':10_000},
                shift_y={'label':'shift y','min':-10_000,'max':10_000},
                shift_x={'label':'shift x','min':-10_000,'max':10_000},
                layer_name = {'label':'Layer name'},
                persist = True
                )
        def inner(
                shift_x: int = 0,
                shift_y: int = 0,
                shift_z: int = 0,
                layer_name: str = ""
                ):
            layer = self.__fetch_layer__(layer_name)
            tformed = np.array([shift_z, shift_y, shift_x])
            layer.translate = tformed
            self.global_translation[layer_name] = tformed.copy()
            print(f"translated {layer_name} by {tformed}")
        return inner

    def _mutual_information_(self):
        @magicgui(
                call_button="Mutual Information",
                persist = True,
                layer_1 = {'label':'Event-based Layer'},
                layer_2 = {'label':'Frame-based Layer'},
                n_bins = {'label':"number of bins for 2d hist"},
                local_bool = {'label':'local'}
                )
        def inner(
                layer_1: str,
                layer_2: str,
                n_bins: int, 
                local_bool: bool = True, 
                omit_event_zeros: bool = True
                ):
            if local_bool:
                print("locally applying mutual information (batch size = 1)")
                idx = [self.__fetch_viewer_image_index__()]
            else:
                print("GLOBALLY applying mutual information")
                n_im = self.__fetch_layer__(layer_1).data.shape[0]
                idx = list(range(n_im))

            cp_free_mem()
            log_hist_global = []
            zeros_slice = cp.ones(self.__fetch_layer__(layer_1).data[0].shape,
                                  dtype = bool)
            for j in tqdm(idx, desc = "iterating mutual info metric over idx"):
                layer_1_handle = self.__fetch_layer__(layer_1).data[j]
                layer_2_handle = self.__fetch_layer__(layer_2).data[j]
                layer_1_cp = cp.array(layer_1_handle, 
                                      dtype = layer_1_handle.dtype)
                layer_2_cp = cp.array(layer_2_handle,
                                      dtype = layer_2_handle.dtype)
                if omit_event_zeros:
                    zeros_slice = layer_1_cp != 0
                assert layer_1_cp.shape == layer_2_cp.shape, "shape mismatch"
                #print(layer_1_cp.flatten().shape)
                #print(layer_2_cp.flatten().shape)
                hist_2d, x_edge, y_edge = cp.histogram2d(
                                               layer_1_cp[zeros_slice].flatten(),
                                               layer_2_cp[zeros_slice].flatten(),
                                               bins = n_bins)
                log_hist = cp.zeros_like(hist_2d)
                cp.log(hist_2d, out = log_hist)
                log_hist[hist_2d <= 0] = 0
                log_hist_global.append(log_hist)
            log_hist_global = cp.stack(log_hist_global)
            print(log_hist_global.shape)
            log_hist_global = cp.sum(log_hist_global, axis = 0)
            print(log_hist_global.shape)
            fig,ax = plt.subplots(1,1, figsize = (8,8))
            X_,Y_ = np.meshgrid(x_edge.get(), y_edge.get())
            ax.pcolormesh(X_-0.5, Y_, log_hist_global.get().T)
            ax.set_title(f"Mutual Information:\n{mutual_information(log_hist)}")
            fig.tight_layout()
            plt.show()
        return inner

    def _zip_centroids_to_points_(self):
        @magicgui(
                call_button="Zip centroids to points",
                persist = True,
                centroids_event = {'label':'Event centroids'},
                centroids_frame = {'label':'Frame centroids'},
                )
        def inner(
                centroids_event: str,
                centroids_frame: str,
                ):
            pts_event = self.__fetch_layer__(centroids_event).data.copy()
            pts_frame = self.__fetch_layer__(centroids_frame).data.copy()
            ev_shape = pts_event.shape
            fr_shape = pts_frame.shape
            msg = f"shape mismatch; fr: {fr_shape}, ev: {ev_shape}"
            assert pts_event.shape == pts_frame.shape, msg
            event_sorted = np.zeros_like(pts_event)
            for j, elem in tqdm(enumerate(pts_frame), desc = 'sorting'):
                dx = elem[0] - pts_event[:,0]
                dy = elem[1] - pts_event[:,1]
                dr = np.sqrt(dx**2 + dy**2)
                event_idx = np.where(dr == dr.min())[0][0]
                event_sorted[j] = pts_event[event_idx]
        
            pts_new = []
            for j in range(pts_event.shape[0]):
                pts_new.append(event_sorted[j])
                pts_new.append(pts_frame[j])
            pts_new = np.vstack(pts_new)
            idx = self.__fetch_viewer_image_index__()
            ones = np.ones([pts_new.shape[0], 1]) * idx
            pts_layer = np.hstack([ones, pts_new])
            self.viewer.add_points(pts_layer, name = "centroid points for reg")
        return inner

    #--------------------------------------------------------------------------
    #                               TRACKING    
    #--------------------------------------------------------------------------
    def _preview_track_centroids_(self):
        @magicgui(call_button="Locate Centroids",
                layer_name = {'label':'Layer Name'},
                persist = True,
                minmass = {'label':'minmass', 'max': 1e16, 'step':1e-6},
                diameter = {'label':'Diameter', 'max': 1e16},
                threshold = {'label':'Threshold', 'max': 1e16, 'step':1e-6},
                )
        def inner(
                layer_name: str,
                minmass: float,
                diameter: int,
                threshold: float,
                ):
            test_frame = self.__fetch_viewer_image_index__()
            if layer_name in self.global_translation:
                z_offset = self.global_translation[layer_name][0]
                print(f"offsetting tracked frame {z_offset}")
                test_frame -= z_offset
            track_handle = self.__fetch_layer__(layer_name).data[test_frame].copy()
            if track_handle.ndim == 3:
                print("--> not sure what to do for 4d images?")
                track_handle = np.sum(track_handle.astype(np.float32), axis = -1)
                pass
            elif track_handle.ndim == 2:
                pass
            self.track_dict = {
                    "minmass":minmass,
                    "diameter": diameter,
                    "threshold": threshold,
                    }
            self.track_bool = True
            print(type(track_handle))
            f = tp.locate(
                          np.array(track_handle),
                          diameter,
                          minmass = minmass,
                          threshold = threshold,
                          invert = False,
                          engine = 'numba'
                          )
            self.frame_track[layer_name] = f.copy()
            points_array = f[['y','x']].values
            self.viewer.add_points(
                    points_array,
                    name = f'tracked centroids {test_frame}',
                    symbol = 'x',
                    face_color = 'b'
                    )
        return inner

    def _track_batch_locate_(self):
        @magicgui(call_button="Locate Particles",
                layer_name = {'label':'Layer Name'},
                persist = True,
                mode = {'label':'Mode (batch/locate)'},
                processes = {'label': "Num Processes", "max":1e16},
                batch_size = {'label': "Batch size", "max":1e16}
                )
        def inner(
                layer_name: str,
                mode: str = "batch",
                processes: int = -1,
                batch_size: int = -1
                ):
            assert self.track_bool, ("Set the tracking parameters with "
                                     "'preview' before tracking")
            track_handle = self.__fetch_layer__(layer_name).data
            if layer_name in self.global_translation:
                z_offset = self.global_translation[layer_name][0]
            else:
                z_offset = 0
            print(f"tracking z offset = {z_offset}")

            n_elem = track_handle.shape[0]
            proc = "auto" if processes < 0 else processes
            if track_handle.ndim == 4:
                print("--> not sure what to do for 4d images?")
                track_handle = np.sum(track_handle.astype(np.float32), axis = -1)
                pass
            elif track_handle.ndim == 3:
                pass

            # Multiprocessing
            if mode == "batch":
                # All in 1 batch
                if batch_size == -1:
                    f = tp.batch(
                            np.array(track_handle, dtype = track_handle.dtype), 
                             processes = proc,
                             **self.track_dict
                             )
                # Batches...?
                elif batch_size != -1:
                    f = []
                    n_batch = int(np.ceil(n_elem / batch_size))
                    for j in range(n_batch):
                        slice_ = slice(j*batch_size, (j+1)*batch_size)
                        local_arr = np.array(track_handle[slice_], dtype = track_handle.dtype)
                        local_dict = tp.batch(
                                              local_arr, 
                                              processes = proc,
                                              **self.track_dict
                                              )
                        local_dict['frame'] += j*batch_size
                        f.append(local_dict)
                    f = pd.concat(f)

            # Serial Processing
            elif mode == "locate":
                f = []
                desc =  "locating individual images"
                for j, _image_ in tqdm(enumerate(track_handle), desc = desc):
                    locate_dict = tp.locate(
                                np.array(_image_, dtype = _image_.dtype), 
                                **self.track_dict
                                 )
                    locate_dict['frame'] = j * np.ones(len(locate_dict), dtype = int) 
                    f.append(locate_dict)
                                 
                f = pd.concat(f)
            
            f['frame'] += z_offset
            self.located_frames[layer_name] = f

            # Save Tracks (just in case they were difficult to calculate....)
            if platform == 'linux':
                track_path = Path("/tmp")
            elif platform == 'win32':
                track_path = Path("C:/Users/mcd4/Downloads")
            track_f_name = track_path / "_located_frames_.p"
            print(f"saving located frames to {track_f_name}")
            pickle.dump(f, open(track_f_name,"wb"))
        return inner

    def _track_link_(self):
        @magicgui(call_button="Track Particles",
                layer_name = {'label':'Layer Name'},
                persist = True,
                min_length = {'label':'Filter Stub Length', 'max': 1e16},
                search_range = {'label':'Search Range', 'max': 1e16},
                memory = {'label':'memory', 'max': 1e16},
                drift_bool = {'label':"Correct Drift"},
                omit = {'label':'Particles to omit (comma sep)'}
                )
        def inner(
                layer_name: str,
                min_length: int,
                search_range: int,
                memory: int,
                omit: str,
                drift_bool: bool = False,
                ):
            assert self.track_bool, ("Set the tracking parameters with "
                                     "'preview' before tracking")
            assert self.located_frames is not None, "locate frames first..."
            f = self.located_frames[layer_name]
            t = tp.link(f, search_range = search_range, memory = memory)

            # Remove bad particles
            if omit != "":
                omit_indices = [int(elem) for elem in omit.split(",")]
                remove_bool = np.zeros(len(t), dtype = bool)
                for elem in omit_indices:
                    remove_bool += t['particle'] == elem
                t = t[~remove_bool]
            
            print(t.head)
            t1 = tp.filter_stubs(t, min_length)
            if drift_bool:
                drift = tp.motion.compute_drift(t1)
                t1 = tp.subtract_drift(t1.copy(), drift)
                t1 = t1.droplevel("particle")
            self.track_holder[layer_name] = t1.copy()
            # Add Track Centroids to viewer as points layer
            #slice_handle = ['frame','y','x']
            #self.viewer.add_points(t1[slice_handle], 
            #                       name = "tracked centroids",
            #                       face_color = 'k')

            self.viewer.add_tracks(t1[['particle','frame','y','x']])

        return inner

    def __calc_bayesian__(self, track_id, track_handle, fps, mpp) -> np.array:
        """
        wrapper for the call to hd.estimate_diffusion...

        """
        particle_slice = track_handle['particle'] == track_id
        # Convert displacement from pixels to nm?
        dframe = np.diff(track_handle['frame'][particle_slice].values)
        frame_drop_bool = dframe != 1
        dx = np.diff(track_handle['x'][particle_slice].values)
        dy = np.diff(track_handle['y'][particle_slice].values)
        dr = np.sqrt(dx**2 + dy**2) * mpp
        dr[frame_drop_bool] = np.nan
        posterior, alpha, beta = hd.estimate_diffusion(
                n_dim = 2,
                dt = 1 / fps,
                dr = dr[np.isfinite(dr)]
                )
        bay_diffusivity = np.array([posterior.mean(), posterior.std()])

        return bay_diffusivity

    def _calc_msd_(self):
        @magicgui(
                call_button="Calculate MSD and Bayesian Diam",
                persist = True,
                track_id = {'label':'Track ID (-1 for all tracks)', 'min':-1, 'max': 1e16},
                fps = {'label':'fps', 'max': 1e16},
                mpp = {'label':'micron per pixel', 'max': 1e16, 'step': 1e-4},
                msd_point_min = {'label': 'msd fit idx min', 'max': 1e16},
                msd_point_max = {'label': 'msd fit idx max', 'max': 1e16},
                max_lagtime = {'label': 'max lagtime', 'max': 1e16},
                temperature = {'label': 'Temperature (K)', 'max': 1e16},
                bins = {'label': 'histogram bins', 'min':1e-16, 'max': 1e16},
                layer_name = {'label': 'Layer Name'},
                #exclude_tracks = {'label':'Exclude Idx (comma separated)'}
                )
        def inner(
                fps: float,
                mpp: float,
                layer_name: str,
                track_id: int = -1,
                msd_point_min: int = 0,
                msd_point_max: int = 5,
                max_lagtime: int = 100,
                temperature: float = 295.0,
                bins: int = 50,
                ):
            self.reduced_data[layer_name] = __calc_msd__(
                                        self.track_holder[layer_name],
                                        fps = fps,
                                        mpp = mpp,
                                        track_id = track_id,
                                        msd_point_min = msd_point_min,
                                        msd_point_max = msd_point_max,
                                        max_lagtime = max_lagtime,
                                        temperature = temperature,
                                        bins = bins
                                        )

            self.__plot_msd_bay__(
                    self.reduced_data[layer_name],
                    layer_name,
                    track_id,
                    msd_point_min,
                    msd_point_max,
                    mpp,
                    fps,
                    bins,
                    )

        return inner

    def __plot_msd_bay__(self,
                         data_dict,
                         layer_name,
                         track_id,
                         msd_point_min,
                         msd_point_max,
                         mpp,
                         fps,
                         bins):
        keys = [
                'imsd',
                'bay',
                'bay_diam',
                'A',
                'n_log',
                'log_fits',
                'm',
                'b',
                'lin_fits',
                'diffusivity_log',
                'diam_log',
                'diffusivity_lin',
                'diam_lin',
                'neg_diams',
                'track_lengths'
                ]
        (imsd, bay, bay_diam,
            A, n_log, log_fits, 
            m, b, lin_fits, 
            diffusivity_log, diam_log, 
            diffusivity_lin, diam_lin,
            neg_diams,
            track_length) = [data_dict[key] for key in keys]


        # COMPOSE FIGURE
        fig,ax = plt.subplots(2,3, figsize = (10,10))
        # ---------------------------------------------------------------------
        # Full MSD in linear and log scale
        # ---------------------------------------------------------------------
        for a in ax[0,:-1]:
            a.plot(imsd.index, imsd, color = 'k', alpha = 0.05, 
                   marker = '.', linestyle = '')

            a.set(xlabel = "Lag time (s)", 
                  ylabel = r"$\langle \Delta r^2 \rangle$ [$\mu$m$^2$]")

        # ---------------------------------------------------------------------
        # Fit Region of MSD Curve
        # ---------------------------------------------------------------------
        slice_ = slice(msd_point_min, msd_point_max)
        ax[0,0].set(xscale = 'log', yscale = 'log')
        ax[0,0].plot(imsd.index.values[slice_], log_fits, color = 'r', alpha = 0.05)
        ax[0,1].plot(imsd.index.values[slice_], lin_fits, color = 'r', alpha = 0.05)

        # ---------------------------------------------------------------------
        # Localization Uncertainty? 
        # ---------------------------------------------------------------------
        loc_uncert_ax = ax[0,2].twinx()
        ax[0,2].boxplot(b, positions = [0])
        ax[0,2].set(ylabel = "y-intercept ($\mu$m$^2$)")
        loc_uncert_ax.boxplot(np.sqrt(b[b>0]/4) / mpp, positions = [2])
        loc_uncert_ax.set(ylabel = "Localization uncertainty (px)")


        _k_ = 1.4826
        for j, arr in enumerate([n_log, diam_lin]):
            # Stats for the entire Population
            mean_local = np.nanmean(arr)
            std_local = np.nanstd(arr)
            cv_local = 100 * std_local / mean_local
            mad_local = median_abs_deviation(arr, nan_policy = 'omit')
            med_local = np.nanmedian(arr)
            rcv_local = 100 * mad_local * _k_ / med_local
            label_mean = f"mean:{mean_local:0.2f}\nstd:{std_local:0.2f}\nCV: {cv_local:0.2f} %"
            ax[1,j].axvline(mean_local, label = label_mean, color = 'k', 
                            linestyle = '--')

            label_med = f"median:{med_local:0.2f}\nmad:{mad_local:0.2f}\nrCV: {rcv_local:0.2f} %"
            ax[1,j].axvline(med_local,
                            label = label_med,
                            color = 'b', 
                            linestyle = '--')


        # Gaussian Fits 
        gaussian_fit_linear = fit_param_gaussian(diam_lin, n_bins = bins, 
                                                 mode = 'normal')
        gaussian_fit_log = fit_param_gaussian(n_log, n_bins = bins, 
                                                 mode = 'normal')
        gaussian_fit_bay = fit_param_gaussian(bay_diam, n_bins = bins, 
                                                 mode = 'normal')


        if gaussian_fit_log is not None:
            bins_centered, counts, log_gaussian_fit = gaussian_fit_log
            x_local = np.linspace(bins_centered[0], bins_centered[-1], 1000)
            ax[1,0].plot(x_local, parametric_gaussian(x_local, log_gaussian_fit))
        else:
            print("---> ERROR WITH GAUSSIAN FITTING")

        if gaussian_fit_linear is not None:
            bins_centered, counts, lin_gaussian_fit = gaussian_fit_linear
            x_local = np.linspace(bins_centered[0], bins_centered[-1], 1000)
            ax[1,1].plot(x_local, 
                         parametric_gaussian(x_local, lin_gaussian_fit),
                         )
            _, mean_, std_, _ = lin_gaussian_fit
            cv_ = 100 * std_ / mean_
            print(mean_, std_, cv_)
            ax[1,1].axvline(mean_, color = 'r', 
                    label = f"mean: {mean_:0.2f}\nstd:{std_:0.2f}\ncv: {cv_:0.2f}%")
        else:
            print("---> ERROR WITH GAUSSIAN FITTING OF LINEAR DIAMETERS")

        if gaussian_fit_bay is not None:
            bins_centered, counts, lin_gaussian_bay = gaussian_fit_bay
            x_local = np.linspace(bins_centered[0], bins_centered[-1], 1000)
            ax[1,2].plot(x_local, 
                         parametric_gaussian(x_local, lin_gaussian_bay),
                         )
            _, mean_, std_, _ = lin_gaussian_bay
            cv_ = 100 * std_ / mean_
            print(mean_, std_, cv_)
            ax[1,2].axvline(mean_, color = 'r', 
                    label = f"bay\nmean: {mean_:0.2f}\nstd:{std_:0.2f}\ncv: {cv_:0.2f}%")
        else:
            print("---> ERROR WITH GAUSSIAN FITTING OF Bayesian DIAMETERS")


        _ = ax[1,0].hist(n_log, bins = bins)
        _ = ax[1,1].hist(diam_lin, bins = bins)
        for a in ax[1]:
            a.legend()
        ax[1,0].set(xlabel="MSD Log Exponent (-)", xlim = (0,2))
        ax[1,1].set_xlabel("Particle Diameter msd linear (nm)")
        ax[1,2].set_xlabel("Particle Diameter Bayesian (nm)")
        
        try:
            ax[1,2].hist(bay_diam, bins = bins)
        except:
            print("error with bayesian")
            pass
        fig.tight_layout()
        if Path("/tmp").is_dir():
            dir_local = Path("/tmp")
        else:
            dir_local = Path(".")

        f_name = dir_local / "_delete_me_.png"

        fig.savefig(f_name, dpi = 200)
        self.viewer.add_image(np.asarray(Image.open(f_name)), 
                              name = f"msd {layer_name} {track_id}")

    def _estimate_matched_particles_(self):
        @magicgui(
                call_button="Estimate matched particle pairs",
                track_1 = {'label':'Track 1 Name'},
                track_2 = {'label':'Track 2 Name'},
                thresh = {'label':'Thresh'},
                super_sampling = {'label':'Super Sampling'},
                link_bool = {'label':"Link"},
                tracks_bool = {'label':"Tracks to viewer"},
                mode_bool = {'label':"Mutual frames only"},
                )
        def inner(
                track_1: str,
                track_2: str,
                thresh: float,
                super_sampling: int = 1,
                link_bool: bool = True,
                tracks_bool: bool = True,
                mode_bool: bool = True
                ):
            self.matches = fetch_particle_pairs(
                                                self.track_holder[track_1],
                                                self.track_holder[track_2],
                                                thresh = thresh,
                                                super_sampling = super_sampling
                                                )
            print("finished matching")
            if link_bool:
                temp = re_link(
                               self.track_holder[track_1],
                               self.track_holder[track_2],
                               self.matches,
                               mode = "frame match" if mode_bool else ""
                               )
                handle_1_match = track_1 + " matched"
                handle_2_match = track_2 + " matched"
                self.track_holder[handle_1_match] = temp[0]
                self.track_holder[handle_2_match] = temp[1]
                print(f"added {handle_1_match} to tracks")
                print(f"added {handle_2_match} to tracks")
                if tracks_bool:
                    slice_ = ['particle','frame','y','x']
                    for key in [handle_1_match, handle_2_match]:
                        print(f"adding {key} tracks to viewer")
                        self.viewer.add_tracks(
                                self.track_holder[key][slice_],
                                name = key
                                )
        return inner

    def _add_match_points_(self):
        @magicgui(
                call_button="Add match points to viewer",
                track_1_name = {'label':'Track 1 Name'},
                track_2_name = {'label':'Track 2 Name'},
                )
        def inner(
                track_1_name: str,
                track_2_name: str,
                ):
            print("Recall event needs to be the first track set...")
            assert track_1_name in self.track_holder, f"{track_1_name} not tracked"
            assert track_2_name in self.track_holder, f"{track_2_name} not tracked"

            track_1 = self.track_holder[track_1_name]
            track_2 = self.track_holder[track_2_name]

            assert len(track_1) == len(track_2), "disimilar tracks...????"

            n_points = len(track_1)
            new_pts = np.zeros([n_points*2, 3])
            new_pts[::2,:] = np.vstack([
                                        track_1['frame'].values,
                                        track_1['y'].values,
                                        track_1['x'].values
                                        ]).T
            new_pts[1::2,:] = np.vstack([
                                         track_2['frame'].values,
                                         track_2['y'].values,
                                         track_2['x'].values
                                         ]).T

            self.viewer.add_points(new_pts, name = "match points")
        return inner

    #--------------------------------------------------------------------------
    #                               FILTERS    
    #--------------------------------------------------------------------------
    def _apply_event_noise_filter_(self):
        @magicgui(call_button="Apply Interpolation Event Noise Filter")
        def inner():
            """
            Apply Event Noise filter
                interpolation_methods:
                    - 0: bilinear
                    - 1: bilinear with interval weights
                    - 2: max
                    - 3: distance
            """
            # NOISE FILTER
            noise_filter = event_filter_interpolation_compiled(
                            frame_size = self.frame_size,
                            filter_length = self.filter_length,
                            scale = self.scale,
                            update_factor = self.update_factor,
                            interpolation_method = self.interpolation_method,
                            filtered_ts = None,
                            )
            events = self.cd_data
            noise_filter.processEvents(events)
            self.eventBin = noise_filter.eventsBin
            event_stack,_ = __hdf5_to_numpy__(
                                           self.event_file,
                                           events[noise_filter.eventsBin],
                                           acc_time = self.event_acc_time,
                                           thresh = self.event_dead_px_thresh,
                                           super_sampling = self.super_sampling,
                                           num_images = self.num_event_images,
                                           dtype = self.event_dtype,
                                           omit_neg = self.event_omit_neg
                                           )


            colormap = self.__event_colormap_fetch__(ref_layer = "event")
            self.viewer.add_image(event_stack, 
                                  colormap = colormap, 
                                  name = 'event filtered')

        return inner

    def _preview_event_noise_filter_(self):
        @magicgui(
                call_button="Preview Interpolation Event Noise Filter",
                scale = {'label':'Scale'},
                filter_length = {'label':'filter_length', 'max': 1e16},
                update_factor = {'label':'update_factor', 'step': 1e-6 },
                interpolation_method = {'label':'interpolation method'},
                n_images = {'label':'Images to filter','min':1,'max':1e16},
                persist = True
                )
        def inner(scale: int = 10, 
                  filter_length: int = 1e3,
                  update_factor: float = 0.25,
                  interpolation_method: int = 3,
                  n_images: int = 50
                  ) -> np.array:
            """
            Apply Event Noise filter
                interpolation_methods:
                    - 0: bilinear
                    - 1: bilinear with interval weights
                    - 2: max
                    - 3: distance
            """
            # NOISE FILTER
            print(f"Hard coded image frame size: {self.frame_size}")
            self.interpolation_method = interpolation_method
            self.update_factor = update_factor
            self.scale = scale
            self.filter_length = filter_length

            interpolation_method_str = {
                    0:"bilinear",
                    1:"bilinear with interval weights",
                    2:"max",
                    3:"distance"
                    }[self.interpolation_method]
            print(f"using interpolation method: {interpolation_method_str}")

            noise_filter = event_filter_interpolation_compiled(
                                    frame_size = self.frame_size,
                                    filter_length = self.filter_length,
                                    scale = self.scale,
                                    update_factor = self.update_factor,
                                    interpolation_method = self.interpolation_method,
                                    filtered_ts = None,
                                    )
            cd_slice = slice(self.trigger_indices[0,0], 
                             self.trigger_indices[n_images,0],
                             1)

            events = self.cd_data[cd_slice].copy()
            noise_filter.processEvents(events)
            eventsBin = noise_filter.eventsBin

            #fig,ax = plt.subplots(1,2, sharex = True, sharey = True)
            #ax[0].scatter(events['x'], events['y'])
            #ax[1].scatter(events['x'][eventsBin], events['y'][eventsBin])
            #plt.show()

            event_stack, _ = __hdf5_to_numpy__(
                                           self.event_file,
                                           events[eventsBin],
                                           acc_time = self.event_acc_time,
                                           thresh = self.event_dead_px_thresh,
                                           super_sampling = self.super_sampling,
                                           num_images = n_images,
                                           dtype = self.event_dtype,
                                           omit_neg = self.event_omit_neg
                                           )


            colormap = self.__event_colormap_fetch__(ref_layer = "event")
            self.viewer.add_image(event_stack,
                                  colormap = colormap, 
                                  name = 'event filtered preview (1st acc time)')
            if "event" in self.viewer.layers:
                contrast_limits = self.__fetch_layer__("event").contrast_limits
                self.viewer.layers[-1].contrast_limits = contrast_limits

        return inner

    def _apply_rvt_to_layer_(self):
        @magicgui(call_button="Apply RVT (to previewed layer)")
        def inner():
            layer_handle = self.__fetch_layer__(self.rvt_layer_name).data
            n_im = layer_handle.shape[0]
            temp = np.zeros(layer_handle.shape).astype(np.float32)
            upsample_slice = (
                    slice(0, None, self.rvt_upsample),
                    slice(0, None, self.rvt_upsample)
                    )
            for j in tqdm(range(n_im), desc = "applying rvt"):
                # GPU implementation?
                cp_arr = cp.array(layer_handle[j], dtype = np.float32)
                temp[j] = rvt(
                              cp_arr, 
                              rmin = self.rvt_rmin,
                              rmax = self.rvt_rmax,
                              highpass_size = self.rvt_highpass_size,
                              kind = self.rvt_kind,
                              upsample = self.rvt_upsample,
                              ).get()[upsample_slice]
                # CPU implementation?
            self.viewer.add_image(temp, name = f'{self.rvt_layer_name} RVT' )
        return inner

    def _apply_gaussian_layer_(self):
        @magicgui(
                call_button="Apply 2D Gaussian Filter",
                layer_name = {'label':'Layer Name'},
                sigma = {'label':'sigma','max': 100.0},
                persist = True
                )
        def inner(
                layer_name: str,
                sigma: float = 0.0,
                ):
            layer_handle = self.__fetch_layer__(layer_name).data
            assert layer_handle.ndim == 3, "Gaussian sigma only works for 3d image stacks"
            n_im = layer_handle.shape[0]
            temp = np.zeros_like(layer_handle).astype(np.float32)
            for j in tqdm(range(n_im), desc = "applying gaussian filter"):
                cp_arr = cp.array(layer_handle[j], dtype = np.float32)
                temp[j] = gaussian_filter(cp_arr, sigma = sigma).get()
            self.viewer.add_image(temp, name = f'{layer_name} Gaussian Filtered' )
        return inner

    def _apply_median_layer_(self):
        @magicgui(
                call_button="Apply 2D Median Filter",
                layer_name = {'label':'Layer Name'},
                kernel = {'label':'kernel size','max': 1e16, "step": 2 },
                persist = True
                )
        def inner(
                layer_name: str,
                kernel: int = 3
                ):
            print("Median Operating In Place")
            layer_handle = self.__fetch_layer__(layer_name).data
            if layer_handle.ndim == 3:
                print("applying median to 3D image")
                n_im = layer_handle.shape[0]
                #temp = np.zeros_like(layer_handle).astype(np.float32)
                for j in tqdm(range(n_im), desc = "applying median_filter"):
                    cp_arr = cp.array(layer_handle[j], dtype = np.float32)
                    layer_handle[j] = median_filter(cp_arr, (kernel,kernel)).get()
            elif layer_handle.ndim == 2:
                print("applying median to 2D image")
                cp_arr = cp.array(layer_handle, dtype = np.float32)
                layer_handle = median_filter(cp_arr, (kernel,kernel)).get()

        return inner

    def __calculate_hot_pixels__(self):
        @magicgui(
                call_button="Calculate Hot Pixels",
                layer_name = {'label':'Layer Name'},
                persist = True,
                z = {'label':'z', 'min': 1},
                )
        def inner(
                layer_name: str,
                z: int = 2,
                ):
            hot_px_bool = calc_hot_px(self.cd_data, z = z, dtype = np.float32)
            if 'hot px' in self.viewer.layers:
                del self.viewer.layers['hot px']
            self.viewer.add_image(hot_px_bool, 
                                  name = 'hot px', 
                                  colormap = 'viridis')
            self.hot_px_layer = True
        return inner

    def _filter_hot_pixels_(self):
        @magicgui(
                call_button="Apply CD Hot Pixel Filter",
                )
        def inner():
            assert self.hot_px_layer is not None,"Calculate hot pixel layer first"
            hot_px_map = self.__fetch_layer__('hot px').data
            print("\t---> Filtering CD Data <---")
            self.cd_data = hot_px_cd_filter(hot_px_map, self.cd_data)
            cp_free_mem()

            print("\t---> Updating Event Image Stack <---")
            event_stack, self.trigger_indices = __hdf5_to_numpy__(
                                       self.event_file,
                                       self.cd_data,
                                       acc_time = self.event_acc_time,
                                       thresh = self.event_dead_px_thresh,
                                       super_sampling = self.super_sampling,
                                       num_images = self.num_event_images,
                                       dtype = self.event_dtype,
                                       omit_neg = self.event_omit_neg
                                       )

            handle = self.__fetch_layer__("event")
            handle.data = event_stack

        return inner

    def _apply_conditional_median_layer_(self):
        """
        This is meant to remove pixels that are the only activated pixel inside
        the kernel size. The gaussian/rvt fixates on
        the convolutions are executed on 2d images so the kernel size never
        incorporates the image index dimension
        """
        @magicgui(
                call_button="Remove nonzero outliers",
                layer_name = {'label':'Layer Name'},
                kernel = {'label':'kernel size','max': 100, "step": 2 },
                )
        def inner(
                layer_name: str,
                kernel: int = 11,
                ):
            print(f"Removing {kernel}x{kernel} nonzero outliers")
            cp_free_mem()
            layer_handle = self.__fetch_layer__(layer_name).data
            kernel_cp = cp.ones(kernel**2).reshape(kernel,kernel).astype(np.float32)
            if layer_handle.ndim == 3:
                n_im = layer_handle.shape[0]
                temp = np.zeros_like(layer_handle).astype(np.float32)
                for j in tqdm(range(n_im), desc = "applying outlier filter"):
                    cp_arr = cp.array(layer_handle[j], dtype = np.float32)
                    comp_arr = convolve(cp_arr, kernel_cp)
                    bool_arr = comp_arr == cp_arr
                    temp[j] = cp_arr.copy().get()
                    temp[j][bool_arr.get()] = 0

            elif layer_handle.ndim == 2:
                assert False, "NOT TESTED"

            self.viewer.add_image(temp, 
                    name = f'{layer_name} outlier filtered' ,
                    colormap = self.__fetch_layer__(layer_name).colormap.name
                    )

            cp_free_mem()
        return inner

    def _abs_of_layer_(self):
        @magicgui(
                call_button="Absolute Value of Layer",
                layer_name = {'label':'Layer Name'},
                persist = True
                )
        def inner(
                layer_name: str,
                ):
            self.__fetch_layer__(layer_name).data = np.abs(self.__fetch_layer__(layer_name).data)
            print(f"Applied absolute value to {layer_name}")
        return inner

    def _combine_event_channels_(self):
        @magicgui(
                call_button="|Event| -> grayscale",
                )
        def inner(
                layer_name: str = 'event',
                ):
            layer_handle = self.__fetch_layer__(layer_name).data
            assert layer_handle.ndim == 4, "needs 4 channel event data..."
            nz,nx,ny,_ = layer_handle.shape
            out_pos = np.zeros([nz,nx,ny], dtype = np.uint8)
            #out_neg = np.zeros([nz,nx,ny], dtype = np.uint8)
            none_val = cp.array([52,37,30], dtype = np.uint8)
            print("Hard Coded VOID polarities:")
            print(f"\tvoid polarity: {none_val})")
            for j in tqdm(range(nz)):
                cp_arr = cp.array(layer_handle[j], dtype = cp.uint8)
                out_pos[j] = 255*cp.prod(cp_arr != none_val, axis = -1).astype(cp.uint8).get()
            self.viewer.add_image(out_pos, name = "event combined", colormap = 'gray')
        return inner

    def _diff_layer_(self):
        """
        this is for "approximating" an event image by diffing two images
        """
        @magicgui(call_button="Simulate events from layer (diff layer)")
        def inner(
                layer_name: str = "frame",
                median_kernel: int = 3,
                ):
            frame_handle = self.__fetch_layer__(layer_name).data
            cp_free_mem()
            n_frames = frame_handle.shape[0]
            kernel = (1, median_kernel, median_kernel)
            output = np.zeros(frame_handle.shape, dtype = np.float32)
            print(n_frames, type(n_frames))
            for j in tqdm(range(n_frames-1), desc = "diff frame"):
                slice_ = slice(j,j+2)
                cp_arr = cp.array(frame_handle[slice_], dtype = cp.float32)
                cp_arr = median_filter(cp_arr, kernel)
                diff = cp_arr[1] - cp_arr[0]
                output[j] = diff.get()
            cp_free_mem()
            self.viewer.add_image(output[:-1], name = "diff frame",
                    colormap = "hsv")
        return inner

    def _isolate_event_sign_(self):
        @magicgui(
                    call_button="Isolate frame Event Sign (+/- polarities)",
                    in_place_bool ={"label":"Operate in place"},
                    positive_bool ={"label":"Positive"},
                    negative_bool ={"label":"Negative"},
                    persist = True,
                    )
        def inner(
                layer_name: str,
                in_place_bool: bool,
                positive_bool: bool,
                negative_bool: bool,
                ):
            print("--> iso?")
            layer_handle = self.__fetch_layer__(layer_name)
            iso_dict = {}
            if positive_bool:
                iso_dict['pos'] = [lambda x: x > 0, 'bop orange', 1]
            if negative_bool:
                iso_dict['neg'] = [lambda x: x < 0, 'bop purple', -1]

            for name, (func, colormap, mult) in iso_dict.items():
                bool_arr = func(layer_handle.data)
                if not in_place_bool:
                    temp = np.zeros_like(layer_handle.data)
                    temp[bool_arr] = mult * layer_handle.data[bool_arr].copy()
                    self.viewer.add_image(temp,
                                          name = name, 
                                          colormap = colormap,
                                          blending = 'additive')
                elif in_place_bool and positive_bool:
                    layer_handle.data[layer_handle.data < 0] = 0

                elif in_place_bool and negative_bool:
                    layer_handle.data[layer_handle.data > 0] = 0

            print("Done Isolating")
            print("<-- iso?")
        return inner

    def _preview_rvt_filter_(self):
        @magicgui(
                call_button="Preview RVT",
                persist = True,
                layer_name = {'label':'Layer Name'},
                rmin = {'label':'R Min', 'max': 1e16},
                rmax = {'label':'R Max', 'max': 1e16},
                upsample = {'label':'Up sample'},
                highpass_size = {'label':'Highpass Size (-1 for None)'},
                kind = {'label':'Kind (normalized or basic)'},
                coarse_factor = {'label':"Coarse factor"},
                coarse_mode = {'label':"Coarse mode"},
                median_bool = {'label':'Apply median after RVT (3x3)'}
                )
        def inner(
                layer_name: str,
                rmin: int,
                rmax: int,
                upsample: int = 1,
                highpass_size: float = -1.0,
                kind: str = "normalized",
                coarse_factor: int = 1,
                coarse_mode: str = "add",
                median_bool: bool = True,
                ):
            layer_handle = self.__fetch_layer__(layer_name).data
            image_idx = self.__fetch_viewer_image_index__()
            if highpass_size <= 0:
                highpass_size = None
            self.rvt_rmin: int = rmin
            self.rvt_rmax: int = rmax
            self.rvt_upsample: int = upsample
            self.rvt_highpass_size = highpass_size
            self.rvt_layer_name: str = layer_name
            self.rvt_kind: str = kind
            self.rvt_coarse_factor: int = coarse_factor
            self.rvt_coarse_mode: str = coarse_mode
            cp_arr = cp.array(layer_handle[image_idx], dtype = np.float32)
            upsample_slice = (
                               slice(0, None, upsample),
                               slice(0, None, upsample)
                               )
            temp = rvt(
                      cp_arr, 
                      rmin = self.rvt_rmin,
                      rmax = self.rvt_rmax,
                      highpass_size = self.rvt_highpass_size,
                      kind = self.rvt_kind,
                      coarse_factor = self.rvt_coarse_factor,
                      coarse_mode = self.rvt_coarse_mode,
                      upsample = self.rvt_upsample,
                      )[upsample_slice]
            if median_bool:
                print("applying median to RVT image")
                temp = median_filter(temp, (3,3)).get()
            else:
                temp = temp.get()
            self.viewer.add_image(temp, name = f'{layer_name} RVT preview' )
        return inner

    def _filter_1d_(self):
        @magicgui(
                call_button="Apply 1D filter",
                size_3d = {'label':'size 3d', "step": 2, "min": 3},
                tile_size = {'label':'Tile size', "step": 1, "min": 1},
                sigma = {'label':'sigma', "step": 1e-6},
                gamma = {'label':'gamma', "step": 1e-6},
                filter_1d = {'label':'Kernel type'},
                persist = True
                )
        def inner(
                layer_name: str,
                size_3d: int,
                tile_size: int,
                sigma: float,
                gamma: float,
                filter_1d: filter_1d_type = filter_1d_type.gaussian_1D,
                ):
            handle = self.__fetch_layer__(layer_name)
            msg = "can't operate in place unless its a floatint point dtype"
            assert handle.dtype in [np.float32, np.float64], msg
            n_im, nx, ny = handle.data.shape
            re_slice_idx = size_3d // 2
            # Create Convolution Kernel (along 0-axis)
            x_ = np.linspace(-re_slice_idx, re_slice_idx, size_3d, endpoint = True)
            #kern_name = filter_1d.value[0]
            kernel = filter_1d.value[1](x = x_, sigma = sigma, gamma = gamma)
            kernel = cp.array(kernel, dtype = np.float32)
            assert nx % tile_size == 0, f"tile size doesnt divide {nx} evenly"
            assert ny % tile_size == 0, f"tile size doesnt divide {ny} evenly"
            batch_x = int(nx / tile_size)
            batch_y = int(ny / tile_size)
            for i in tqdm(range(batch_x), desc = "1D filtering"):
                for j in range(batch_y):
                    slice_ = (
                            ...,
                            slice(i*tile_size, (i+1)*tile_size),
                            slice(j*tile_size, (j+1)*tile_size)
                            )
                    cp_arr = cp.array(handle.data[slice_], dtype = cp.float32)
                    filtered = convolve1d(cp_arr, kernel, axis = 0)
                    handle.data[slice_] = filtered.get()
        return inner

    def _subtract_axial_median_gpu_(self):
        @magicgui(
                call_button="Subtract Axial Median",
                persist = True,
                layer_name = {'label':'Layer Name'},
                tile_size = {'label':'Tile Size'},
                dtype = {'label':'Data type'},
                add_to_viewer = {'label':'Add to viewer'}
                )
        def inner(
                layer_name: str,
                tile_size: int,
                dtype: np_dtype,
                add_to_viewer: bool,
                ):
            dtype = dtype.value
            print(f"median dtype = {dtype}")
            handle = self.__fetch_layer__(layer_name).data
            n_images, nx, ny = handle.shape
            n_tile_x = int(np.ceil(nx / tile_size))
            n_tile_y = int(np.ceil(ny / tile_size))
            cp_free_mem()
            med = cp.zeros([nx,ny], dtype = dtype)
            for tile_x in tqdm(range(int(n_tile_x))):
                for tile_y in range(int(n_tile_y)):
                    slice_ = (...,
                            slice(tile_x*tile_size, (tile_x+1)*tile_size,1),
                            slice(tile_y*tile_size, (tile_y+1)*tile_size,1),
                            )
                    layer_cp = cp.array(handle[slice_], dtype = dtype)
                    med_local = cp.median(layer_cp, axis = 0).astype(dtype)
                    med[slice_[1], slice_[2]] = med_local.copy()

            if add_to_viewer:
                self.viewer.add_image(med.get()[None,:,:], name = 'bkgd')
            else:
                print("subtracting background")
                handle = self.__fetch_layer__(layer_name)
                handle.data = handle.data.astype(dtype)
                handle.data -= med.get()
            print(f"{layer_name} -= median (axis = 0)")
            cp_free_mem()

        return inner

    def _calc_median_background_(self):
        @magicgui(
                call_button="Calculate Background (Median)",
                persist = True,
                layer_name = {'label':'Layer Name'},
                tile_size = {'label':'Tile Size'},
                median_size = {'label':'Median size'},
                mode = {'label':'Mode'}
                )
        def inner(
                layer_name: str,
                tile_size: int,
                median_size: int,
                mode: median_mode_type
                ):

            print("mode = ",mode.value)
            temp = batch_axial_median(
                                      self.__fetch_layer__(layer_name).data, 
                                      tile_size,
                                      tile_size,
                                      median_size,
                                      mode = mode.value
                                      )
            cp_free_mem()
            cmap = self.__event_colormap_fetch__(ref_layer = layer_name,
                                                 default_cmap = 'viridis')
            self.viewer.add_image(temp, name = 'bkgd', colormap = cmap)

        return inner

    def _remove_pixels_(self):
        """
        This is for zeroing out hot/crazy pixels in the event camera that throw
        off the contrast...
        """
        @magicgui(
                  call_button="Zero out pixels",
                  layer_name = {'label':"Layer name (-1 for top layer)"},
                  px = {'label':"Pixels (x1,y1,x2,y2,...)"},
                  persist = True
                  )
        def inner(layer_name: str, px: str):
            if layer_name == "-1":
                handle = self.viewer.layers[-1]
            else:
                handle = self.__fetch_layer__(layer_name)
            pixels = np.array([int(val) for val in px.split(",")]).reshape(-1,2)
            for x_, y_ in pixels:
                print(f"zeroing out (x,y): {x_},{y_}")
                handle.data[:,x_,y_] = 0
        return inner

    #--------------------------------------------------------------------------
    #                               Visualization
    #--------------------------------------------------------------------------
    def _violin_plot_(self):
        @magicgui(
                call_button="Violin plot",
                layer_name = {'label': "Layer Name"},
                ground_truth = {'label':"Ground Truth (-1 for None)"}
                )
        def inner(
                layer_name: str,
                ground_truth: float = -1.0,
                ):
            gt = None if ground_truth == -1 else ground_truth
            self.__create_violin_plot__(layer_name, ground_truth = gt)
        return inner

    def _joint_hist_(self):
        @magicgui(
                call_button="Plot Joint Histogram",
                layer_1 = {'label': "Layer 1 Name"},
                layer_2 = {'label': "Layer 2 Name"},
                parameter = {'label': "Parameter"},
                bins = {'label': "Bins", 'min':1,'max':1e16},
                persist = True
                )
        def inner(
                layer_1: str,
                layer_2: str,
                parameter: str,
                bins: int
                ):
            for layer in [layer_1, layer_2]:
                msg = f"{layer} not in reduced"
                assert layer in self.reduced_data, msg
            _x_ = self.reduced_data[layer_1][parameter]
            _y_ = self.reduced_data[layer_2][parameter]
            nx = len(_x_)
            ny = len(_y_)
            msg = f"{layer_1} and {layer_2} have dissmilar shapes: {nx} != {ny}"
            assert nx == ny, msg
            plot_joint_hist(_x_, _y_, bins, xlabel = layer_1, ylabel = layer_2)
            plt.show()

        return inner

    def __create_violin_plot__(self, layer_name, ground_truth = None):
        handle = self.reduced_data[layer_name]
        track_lengths = handle['track_lengths']
        max_length = np.max(track_lengths)
        min_length = np.min(track_lengths)
        length_bins = 2**np.arange(np.log2(min_length), np.log2(max_length), 1)
        _k_ = 1.4826
        fig,ax = plt.subplots(1,2, sharey = True)
        for elem in length_bins:
            slice_ = track_lengths >= elem
            diams_local = handle['diam_lin'][slice_]
            mean_local = np.nanmean(diams_local)
            std_local = np.nanstd(diams_local)
            med_local = np.nanmedian(diams_local)
            cv_local = 100*std_local/mean_local
            mad_med = _k_*100*median_abs_deviation(diams_local, nan_policy = 'omit') / med_local
            n_particles = np.sum(slice_)
            for a in ax:
                a.violinplot(
                        diams_local,
                        positions = [np.log2(elem)],
                        vert = False
                        )
                print(mad_med)
                string = f"med:{med_local} nm\nrCV:{mad_med:0.2f} %\nCV: {cv_local:0.2f} %\n{n_particles} particles"
                xy = (mean_local + std_local*2, np.log2(elem))
                a.annotate(string, xy = xy)
        ax[0].set_xscale("log")
        if ground_truth is not None:
            for a in ax:
                a.axvline(ground_truth, color = 'k', linestyle = '--')
        ax[0].set(ylabel = "log$_2$(track length) >= n")
        for a in ax:
            a.set_xlabel("diameter (nm)")
        fig.tight_layout()
        plt.show()

    def _figure_(self):
        @magicgui(
                call_button="Capture figure",
                crop = {'label': "Crop (x1,x2,y1,y2)"},
                fig_width = {'label': "Image width" },
                dpi = {'label': 'dpi'},
                file_name = {'label':'File name'}
                )
        def inner(crop: str = "",
                fig_width: float = 8.0,
                dpi: int = 150,
                file_name: str = "",
                ):
            img = self.viewer.export_figure()
            if crop != "":
                x1,x2,y1,y2 = [int(elem) for elem in crop.split(",")]
            else:
                x1,x2,y1,y2 = 0,img.shape[0],0,img.shape[1]
            slice_ = (slice(x1,x2,1),slice(y1,y2,1))
            img = img[slice_]
            AR = img.shape[0] / img.shape[1]
            fig,ax = plt.subplots(1,1,figsize = (fig_width, fig_width * AR))
            ax.imshow(img)
            ax.axis(False)
            fig.subplots_adjust(left = 0, right = 1, top = 1, bottom = 0)
            if file_name == "":
                print("no file name specified -> not saving image")
                plt.show()
            else:
                file_name = file_name + ".png"
                fig.savefig(file_name, dpi = dpi)
        return inner

    def _plot_particle_iso_(self):
        @magicgui(
                call_button="Capture figure",
                persist = True,
                track_dataset_name={'label':"Track Dataset Name"},
                image_layer_name={'label':"Image Layer Name"},
                particle_idx={'label':"Particle Index", 'min':0,'max':1e16},
                slice_size={'label':"Slice Size", "min":1,"step":2},
                )
        def inner(
                    track_dataset_name: str,
                    image_layer_name: str,
                    particle_idx: int,
                    slice_size: int,
                ):
            track_handle = self.track_holder[track_dataset_name]
            slice_ = track_handle['particle'].values == particle_idx
            fr_, x_, y_ = np.round(
                    track_handle[['frame','x','y']].values[slice_].T
                    ).astype(int)
            image_handle = self.__fetch_layer__(image_layer_name).data
            im_out = np.zeros([len(fr_), slice_size, slice_size], 
                              dtype = image_handle.dtype)
            x0 = x_ - slice_size // 2
            y0 = y_ - slice_size // 2
            if image_layer_name in self.global_translation:
                fr_diff = self.global_translation[image_layer_name][0]
            else:
                fr_diff = 0
            for j in tqdm(range(len(fr_)-1)):
                #image_slice = (fr_[j] - fr_diff,
                #               slice(y0[j], y0[j]+slice_size),
                #               slice(x0[j], x0[j]+slice_size))
                #im_out[j] = image_handle[image_slice]
                fr_idx = fr_[j] - fr_diff
                im_out[j] = smart_slice(
                                        image_handle[fr_idx],
                                        y0[j], 
                                        y0[j]+slice_size,
                                        x0[j],
                                        x0[j]+slice_size
                                        )

            inst.viewer.add_image(
                    im_out,
                    name = f"{image_layer_name}:{particle_idx} iso",
                    colormap = 'viridis'
                    )

        return inner

    #--------------------------------------------------------------------------
    #                               UTILS (dunder)
    #--------------------------------------------------------------------------
    def __add_empty_image__(self):
        @magicgui(
                call_button="Add Blank Image",
                persist = True,
                layer_name = {'label':'Layer Name'},
                shape = {'label':'Shape (str, csv)'}
                )
        def inner(
                layer_name: str,
                shape: str
                ):
            shape = tuple(int(val) for val in shape.split(","))
            print(f"shape = {shape}")
            zeros = np.zeros(shape)
            self.viewer.add_image(zeros, name = layer_name)
        return inner

    def __compute__(self):
        @magicgui(
                  call_button="Compute lazy array (dask -> numpy)",
                  persist = True
                  )
        def inner(layer_name: str):
            print(f"Computing {layer_name}")
            handle = self.__fetch_layer__(layer_name)
            handle.data = handle.data.compute()
            print(f"{layer_name}: dask -> numpy")
        return inner

    def __compute_noise_floor__(self):
        @magicgui(
                  call_button="Compute noise floor",
                  persist = True
                  )
        def inner(layer_name: str):
            print(f"Computing {layer_name}")
            handle = self.__fetch_layer__(layer_name).data
            medians = np.zeros(handle.shape[0], dtype = handle.dtype)
            for j in tqdm(range(handle.shape[0])):
                cp_arr = cp.array(handle[j], dtype = handle.dtype)
                medians[j] = cp.median(cp_arr).get()
            self.noise_floor = medians
        return inner

    def __event_colormap_fetch__(self,
                                 ref_layer = 'event', 
                                 default_cmap = "coolwarm"
                                 ) -> str:
        if ref_layer in self.viewer.layers:
            return self.__fetch_layer__(ref_layer).colormap.name
        else:
            return default_cmap

    def __fetch_layer__(self, layer_name: str):
        """
        Helper function so that the layers can be re-ordered
        """
        for layer in self.viewer.layers:
            if layer.name == layer_name:
                return layer
        else:
            print(f"Layer {layer_name} not found")
            return 

    def __fetch_viewer_image_index__(self) -> int:
        """
        little hack for fetching the current layer that napari is looking at 
        """
        idx = [elem[1][0] for elem in list(self.viewer.dims) if elem[0] == 'point']
        if len(idx) == 0:
            return 0
        else:
            return int(idx[0])

    def __free_memory__(self):
        @magicgui(
                call_button="Free Memory",
                )
        def inner():
            cp_free_mem()
            gc.collect()
            print("CUDA memory freed and Garbage Collected")
        return inner

    def __measure_line_length__(self):
        @magicgui(call_button="Measure Line Length")
        def inner():
            layer = self.viewer.layers[-1]
            assert isinstance(layer, napari.layers.shapes.shapes.Shapes), "Top layer is not a shapes layer"
            length = np.sum(np.diff(layer.data[0][:,1:], axis = 0)**2)**(1/2)
            print("-"*80)
            print(f"\tLength of line = {length} px")
            print("-"*80)
        return inner

    def __particles_in_rectangle__(self):
        @magicgui(
                 call_button="Fetch Particles inside Rectangle",
                 persist = True,
                 track_key={'label':'Track key'}
                )
        def inner(track_key: str):
            msg =  "Top layer is not a shapes layer"
            assert isinstance(self.viewer.layers[-1], 
                              napari.layers.shapes.shapes.Shapes), msg
            tracks = self.track_holder[track_key]
            handle = inst.viewer.layers[-1].data[0]
            x0,x1 = np.min(handle[:,2]), np.max(handle[:,2])
            y0,y1 = np.min(handle[:,1]), np.max(handle[:,1])
            bool_arr = (tracks['x'].values >= x0) * \
                       (tracks['x'].values <= x1) * \
                       (tracks['y'].values >= y0) * \
                       (tracks['y'].values <= y1)
            parts = np.unique(tracks['particle'][bool_arr])
            print("-"*70)
            print("particles inside rectangle:")
            print(",".join([f"{elem}" for elem in parts]))
            print("-"*70)
        return inner

    def __sub_sample_tracks__(self):
        @magicgui(
                call_button="Sub Sample Tracks",
                persist = True,
                layer_name = {'label':'Layer Name'},
                sub_sampling = {'label':'Sub Sampling','min':1,'max':1e16},
                )
        def inner(
                layer_name: str,
                sub_sampling: int
                ):
            temp = self.track_holder[layer_name].copy()
            fr_bool = temp['frame'] % sub_sampling == 0
            temp = temp[fr_bool]
            temp['frame'] = temp['frame'] / sub_sampling
            layer_out = layer_name + f" ss {sub_sampling}"
            self.track_holder[layer_out] = temp
            print(f"sub sampled {layer_name} by {sub_sampling} --> {layer_out}")
        return inner

    def __transform_tracks_coordinate_system__(self):
        @magicgui(
                call_button="Transform Track Coordinate System",
                persist = True,
                layer_name = {'label':'Layer Name'},
                affine_select = {'label':'Push/Pull'}
                )
        def inner(
                layer_name: str,
                affine_select: str,
                ):
            track_handle = self.track_holder[layer_name]
            track_coords = np.vstack([
                track_handle['y'].values,
                track_handle['x'].values,
                np.ones_like(track_handle['x'].values),
                ])
            if affine_select.lower() == 'push':
                affine = self.affine_matrix
            elif affine_select.lower() == 'pull':
                affine = self.pull_affine_matrix
            tracks_tform = np.dot(affine, track_coords)
            tracks_cpy = track_handle.copy()
            tracks_cpy['y'] = tracks_tform[0]
            tracks_cpy['x'] = tracks_tform[1]
            label = layer_name + " (tform)"
            self.viewer.add_tracks(tracks_cpy[['particle','frame','y','x']],
                                    name = label)
            self.track_holder[label] = tracks_cpy

        return inner

    def __unspecified_colormap__(self):
        @magicgui(
                  call_button="Colormap (not in layer controls)",
                  colormap = {'label':"colormap"},
                  persist = True
                  )
        def inner(layer_name: str, colormap: str):
            handle = self.__fetch_layer__(layer_name)
            handle.colormap = colormap
        return inner


if __name__ == "__main__":
    inst = spatio_temporal_registration_gui()
    napari.run()

