#!/home/mcd4/miniconda3/envs/openeb/bin/python
from PIL import Image 
from cupyx.scipy.ndimage import affine_transform, median_filter, gaussian_filter
from dask_image.imread import imread
from magicgui import magicgui
from metavision_core.event_io import EventsIterator
from metavision_sdk_core import BaseFrameGenerationAlgorithm
from napari.qt.threading import thread_worker
from os import mkdir
from pathlib import Path
from scipy.optimize import least_squares
from tifffile import imwrite
from tqdm import tqdm
import cupy as cp
import dask.array as da
import diffusive_distinguishability.ndim_homogeneous_distinguishability as hd
import gc
import h5py
import matplotlib.pyplot as plt
import napari
import numpy as np
import pandas as pd
import trackpy as tp


from sys import path
path.append("/home/mcd4/cy_im_utils")
from cy_im_utils.imgrvt_cuda import rvt
from cy_im_utils.event.trackpy_utils import imsd_powerlaw_fit, imsd_linear_fit
from cy_im_utils.parametric_fits import parametric_gaussian, fit_param_gaussian


def cp_free_mem() -> None:
    """
    Frees cupy's memory pool...
    """
    mempool = cp.get_default_memory_pool()
    mempool.free_all_blocks()


def residual(mat: list, input_pair: np.array, target_pair: np.array) -> np.array:
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


class spatio_temporal_registration_gui:
    def __init__(self):
        self.viewer = napari.Viewer()
        self.viewer.title = "Event-Frame Registration GUI"
        self.pull_affine_matrix = None
        self.affine_matrix = None
        self.located_frames = None
        self.frame_0 = 0
        self.track_bool = False
        self.track_holder = {}

        dock_widgets = {
                'Registration Ops': [self._load_data_(),
                              self._flip_ud_(),
                              self._flip_lr_(),
                              self._diff_layer_(),
                              self._load_affine_mat_(),
                              self._reset_affine_(),
                              self._fit_affine_(),
                              self._apply_transform_(),
                              self._shift_event_(),
                              self._set_frame_after_shutter_(),
                              self._write_transforms_(),
                              ],
                'Filtering':[
                            self._isolate_event_channels_(),
                            self._combine_event_channels_(),
                            self._tform_event_to_rvt_(),
                            self._apply_gaussian_layer_(),
                            self._apply_median_layer_(),

                ],
                'Tracking Ops':[
                            self._preview_track_centroids_(),
                            self._track_batch_locate_(),
                            self._track_link_(),
                            self._calc_msd_()
                              ],
                'Utils':[
                    self._free_memory_(),
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

    def _free_memory_(self):
        @magicgui(
                call_button="Free Memory",
                )
        def inner():
            cp_free_mem()
            gc.collect()
            print("CUDA memory freed and Garbage Collected")
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
        @magicgui(call_button="Fit/Refit Affine Params")
        def inner():
            handle = self.viewer.layers[-1]
            _points_type_ = napari.layers.points.points.Points
            assert isinstance(handle, _points_type_), "not a points layer"
            data_handle = handle.data
            event_pairs = data_handle[::2,1:]
            frame_pairs = data_handle[1::2,1:]
            assert event_pairs.shape == frame_pairs.shape, "shape mismatch between pairs"
            n_points = event_pairs.shape[0]
            event_pairs = np.hstack([event_pairs, np.ones(n_points)[:,None]]).T
            frame_pairs = np.hstack([frame_pairs, np.ones(n_points)[:,None]]).T
            x0 = [1,0,0,1,0,0]
            out = least_squares(residual, x0, args = (event_pairs, frame_pairs))
            composed_mat = np.array([
                [out.x[0], out.x[1], out.x[2]],
                [out.x[3], out.x[4], out.x[5]],
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
            if frame_handle.ndim == 4:
                nz, nx, ny, _ = frame_handle.shape
                frame_ndim = 4
            elif frame_handle.ndim == 3:
                nz, nx, ny = frame_handle.shape
                frame_ndim = 3

            _, nx_event, ny_event, _ = event_handle.shape

            data_type = event_handle.dtype

            tform_pull = cp.eye(4).astype(cp.float32)
            tform_pull[1:,1:] = cp.array(self.pull_affine_matrix, dtype = cp.float32)

            tform_push = cp.eye(4).astype(cp.float32)
            tform_push[1:,1:] = cp.array(self.affine_matrix)

            event_raw_transformed = np.zeros([nz,nx,ny,3], dtype = np.uint8)
            if frame_ndim == 3:
                frame_transformed = np.zeros([nz, nx_event, ny_event], dtype = np.float32)
            elif frame_ndim == 4:
                frame_transformed = np.zeros([nz, nx_event, ny_event, 3], dtype = np.uint8)

            n_batch = nz // batch_size
            remainder = nz % batch_size
            if remainder > 0:
                n_batch += 1
            cp_free_mem()
            print("potential speedup for 4D affine transform...")
            for q in tqdm(range(n_batch)):
                upper_lim = min((q+1)*batch_size, nz)
                local_batch_size = upper_lim - batch_size*q
                slice_ = slice(q*batch_size, upper_lim)
                for j in range(3):
                    event_raw_transformed[slice_,:,:,j] = affine_transform(
                        cp.array(event_handle[slice_,:,:,j], dtype = cp.float32),
                        tform_push,
                        output_shape = (local_batch_size,nx,ny),
                        order = 0
                        ).get()
                    cp_free_mem()
                    if frame_ndim == 4:
                        frame_transformed[slice_,:,:,j] = affine_transform(
                            cp.array(frame_handle[slice_,:,:,j], dtype = cp.float32),
                            tform_pull,
                            output_shape = (local_batch_size, nx_event, ny_event),
                            order = 3
                                    ).get()
                        cp_free_mem()
                if frame_ndim == 3:
                    frame_transformed[slice_] = affine_transform(
                            cp.array(frame_handle[slice_], dtype = cp.float32),
                            tform_pull,
                            output_shape = (local_batch_size, nx_event, ny_event),
                            order = 3
                            ).get()
                cp_free_mem()

            self.viewer.add_image(event_raw_transformed, 
                                  opacity = 0.4,
                                  name = "event -> frame")
            self.viewer.add_image(frame_transformed, 
                                  visible = False,
                                  name = "frame -> event",
                                  colormap = "hsv"
                                  )
            self.__fetch_layer__("event").visible = False
            self.__fetch_layer__("frame").opacity = 1.0
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
                layer_name = {'label':'Layer name'})
        def inner(layer_name: str = 'event'):
            handle = self.__fetch_layer__(layer_name)
            handle.data = handle.data[:,::-1]
            print(f"Flipped {layer_name} UD")
        return inner

    def _flip_lr_(self):
        @magicgui(call_button="flip event lr")
        def inner():
            handle = self.__fetch_layer__("event")
            handle.data = handle.data[:,:,::-1]
            print("Flipped event LR")
        return inner

    def _load_data_(self):
        @magicgui(call_button="load data sets",
              main_window = True,
              persist = True,
              layout = 'vertical',
              frame_dir = {"label": "Select Frame File (.tif)"},
              load_frame_bool = {"label": "Load Frame"},
              event_dir = {"label": "Select Event File (.raw or .hdf5)"},
              load_event_bool = {"label": "Load Event"},
              fps = {'label': "FPS (0 for hdf5 trigger sync)",'max':1e16}
                  )
        def inner(
                frame_dir: Path = Path.home(),
                load_frame_bool: bool = True,
                event_dir: Path = Path.home(),
                load_event_bool: bool = True,
                fps: int = 0
                ):
            self.frame_dir = frame_dir
            self.event_dir = event_dir
            if load_event_bool:
                event_files = event_dir.as_posix()
                print(event_files)
                # Read a .raw or .hdf5
                event_suffix = event_dir.suffix
                if event_suffix == '.raw':
                    self.delta_t = int(np.round(1e6/fps))
                    print(f"fps: {fps}\tdelta t: {self.delta_t} us")
                    print("Reading .raw file")
                    event_stack = self._load_raw_to_numpy_(str(event_dir), 
                                                           self.delta_t)
                elif event_suffix == '.hdf5':
                    print("Reading .hdf5 file and matching triggers to exposures")
                    event_stack = self._load_hdf5_to_numpy_(str(event_dir))

                inst.viewer.add_image(event_stack, colormap = 'gray', 
                                      name = 'event')

            if load_frame_bool:
                frame_files = frame_dir.as_posix()
                print(frame_files)
                frame_stack = imread(frame_files)
                inst.viewer.add_image(frame_stack, colormap = 'hsv',
                                    name = 'frame', opacity = 0.4)

        return inner

    def _load_hdf5_to_numpy_(self, event_file, mode = 'exposure') -> np.array:
        """
        This is for loading in an hdf5 file so that the exposure time of the
        frame camera can be matched to the event signal directly.....
        
        Just use the regular raw -> numpy function if you want a finer frame
        rate sampling since this will load the data with gaps!!!
        (discontinuous event data)
        """
        with h5py.File(event_file, "r") as f:
            data = f['CD']['events'][()]
            trigger_data = f['EXT_TRIGGER']['events'][()]

        cp_free_mem()
        x, y, t = [cp.array(data[key], dtype = data[key].dtype) for key in ['x','y','t']]
        trigger_time = cp.array(trigger_data['t'], dtype = trigger_data['t'].dtype)
        trigger_0 = trigger_data['p'][0]
        if trigger_0 == 0:
            print("Shifting to first trigger on state")
            trigger_time = trigger_time[1:]

        n_trigger = trigger_time.shape[0]
        if n_trigger % 2 != 0:
            print("Odd number of triggers, subtracting one off")
            n_trigger -= 1
        image_stack = []
        pos_polarity = cp.array(data['p'] > 0, dtype = bool)
        pos_val = cp.array([255,255,255], dtype = cp.uint8 )
        neg_val = cp.array([200,126,64], dtype = cp.uint8 )
        void_val = cp.array([52,37,30], dtype = cp.uint8)
        #if trigger_time[0]
        #plt.figure()
        #plt.plot(trigger_data['t'], trigger_data['p'])
        #plt.show()
        for j in tqdm(range(n_trigger), desc = 'reading hdf5 events'):
            if j % 2 != 0:
                continue
            slice_ = (t >= trigger_time[j]) * (t <= trigger_time[j+1])
            image_holder = cp.full([720,1280,3], void_val)
            for fill, polarity in zip([pos_val, neg_val],[pos_polarity, ~pos_polarity]):
                local_slice = slice_ * polarity
                x_local = x[local_slice]
                y_local = y[local_slice]
                image_holder[y_local, x_local] = fill
            image_stack.append(image_holder.get().copy())
        image_stack = np.stack(image_stack)
        return image_stack

    def _load_raw_to_numpy_(self, event_file, delta_t):
        mv_iterator = EventsIterator(
                event_file,
                delta_t = delta_t,
                mode = "delta_t",
                start_ts = 0,
                )
        height,width = mv_iterator.get_size()
        image_buffer = np.zeros([height, width, 3], dtype = np.uint8)
        print(delta_t, height, width, image_buffer.shape)
        images = []
        for q, ev in tqdm(enumerate(mv_iterator), desc = "raw --> numpy"):
            BaseFrameGenerationAlgorithm.generate_frame(ev, image_buffer)
            images.append(image_buffer.copy())
        return np.stack(images)

    def _diff_layer_(self):
        @magicgui(call_button="Diff Layer")
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
            inst.viewer.add_image(output[:-1], name = "diff frame",
                    colormap = "hsv")
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

    def _shift_event_(self):
        @magicgui(call_button="Shift Event Temporal",
                shift={'label':'shift','min':-10_000,'max':10_000})
        def inner(shift: int = 0):
            transformed = self.__fetch_layer__("event")
            transformed.data = np.roll(transformed.data,
                                       shift = shift, 
                                       axis = 0)
            self.total_shift += shift
        return inner

    def _write_transforms_(self):
        @magicgui(call_button="Write Transformed Images and data")
        def inner():
            frame_transformed_handle = self.__fetch_layer__("frame -> event").data
            event_transformed_handle = self.__fetch_layer__("event -> frame").data
            frame_raw_handle = self.__fetch_layer__("frame").data
            event_raw_handle = self.__fetch_layer__("event").data
            out_dir = Path("./time_synced") / self.microscope/ self.dataset
            if not out_dir.is_dir():
                mkdir(out_dir)
            self.__write_transform_info__(out_dir)
            # Write The Transformed Image Stacks
            f_name_frame_tformed = out_dir / f"frame_transformed.tif"
            f_name_event_tformed = out_dir / f"event_transformed.tif"
            imwrite(f_name_frame_tformed, frame_transformed_handle[self.frame_0:])
            imwrite(f_name_event_tformed, event_transformed_handle[self.frame_0:])

            # Write The Untransformed Image Stacks at the synchronization time
            f_name_frame_raw = out_dir / f"frame_temporal_sync.tif"
            f_name_event_raw = out_dir / f"event_temporal_sync.tif"
            end_index = frame_transformed.shape[0]
            imwrite(f_name_frame_raw, frame_raw_handle[self.frame_0:])
            imwrite(f_name_event_raw, event_raw_handle[self.frame_0:end_index])
            print("Finished Writing Images and Transform Data")

        return inner

    def __extract_folder_metadata__(self, folder_name) -> None:
        """
        this method extracts some of the experiment meta data from the file
        name
        """
        print("-----------> DEPRECATED FUNCTIONALITY")
        components = folder_name.parts
        for elem in components:
            if "cytovia" in elem.lower():
                microscope = 'cytovia'
                break
            elif "evanescent" in elem.lower():
                microscope = 'evanescent'
                break
        else:
            #assert False, "No Microscope in File name"
            microscope = "unknown_instrument"
            data_set = "data_set_temp"

        data_set = str(folder_name.name).split(".tif")[0]
        fps = int(data_set.split("_")[3].split("fps")[0])
        print("fps = ",fps)


        self.microscope = microscope
        self.dataset = data_set
        self.fps = fps
        self.delta_t = int(round(1e6/fps))

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

    def __write_transform_info__(self, out_directory) -> None:
        """
        Writes the synchronization info to a csv file (tab delimited)
        """
        file_name_prefix = f"{self.microscope}_{self.dataset}"
        points = self.__fetch_layer__("Points").data
        frame_shape  = inst.__fetch_layer__("frame").data.shape
        affine_mat = self.affine_matrix
        horz_scale = affine_mat[0,0]
        horz_shear = affine_mat[0,1]
        horz_translation = affine_mat[0,2]
        vert_shear = affine_mat[1,0]
        vert_scale = affine_mat[1,1]
        vert_translation = affine_mat[1,2]
        # This is the alignment time could be + or -
        time_alignment = -1 * self.total_shift * self.delta_t
        time_0 = time_alignment + self.frame_0 * self.delta_t
        duration = (frame_shape[0] - self.frame_0) * self.delta_t
        time_end = time_0 + duration
        params = {
                  "path_to_raw":str(self.event_dir),
                  "path_to_frame":str(self.frame_dir),
                  "time_init":time_0/1e6,
                  "time_end":time_end/1e6,
                  "frame_0":self.frame_0,
                  "horz_scale":horz_scale,
                  "horz_shear":horz_shear,
                  "horz_translation":horz_translation,
                  "vert_shear":vert_scale,
                  "vert_scale":vert_shear,
                  "vert_translation":vert_translation
                  }

        with open(out_directory / f"{file_name_prefix}_sync.csv", 'w') as f:
            for param, val in params.items():
                f.write(f"{param}\t{val}\n")

    def _preview_track_centroids_(self):
        @magicgui(call_button="Locate Centroids",
                layer_name = {'label':'Layer Name'},
                minmass = {'label':'minmass', 'max': 1e16},
                diameter = {'label':'Diameter', 'max': 1e16},
                test_frame = {'label':'Test Frame', 'max': 1e16}
                )
        def inner(
                layer_name: str,
                minmass: float,
                diameter: int,
                test_frame: int
                ):
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
                    }
            self.track_bool = True
            print(type(track_handle))
            f = tp.locate(
                          np.array(track_handle),
                          diameter,
                          minmass = minmass,
                          invert = False)
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
            minmass = self.track_dict['minmass']
            diameter = self.track_dict['diameter']
            track_handle = self.__fetch_layer__(layer_name).data
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
                    f = tp.batch(np.array(track_handle, dtype = np.float32), 
                                 diameter = diameter,
                                 minmass = minmass,
                                 processes = proc)
                # Batches...?
                elif batch_size != -1:
                    f = []
                    n_batch = int(np.ceil(n_elem / batch_size))
                    for j in range(n_batch):
                        slice_ = slice(j*batch_size, (j+1)*batch_size)
                        local_arr = np.array(track_handle[slice_], dtype = np.float32)
                        local_dict = tp.batch(local_arr, 
                                              diameter = diameter,
                                              minmass = minmass,
                                              processes = proc)
                        local_dict['frame'] += j*batch_size
                        f.append(local_dict)
                    f = pd.concat(f)

            # Serial Processing
            elif mode == "locate":
                f = []
                desc =  "locating individual images"
                for j, _image_ in tqdm(enumerate(track_handle), desc = desc):
                    locate_dict = tp.locate(np.array(_image_, dtype = np.float32), 
                                 diameter = diameter,
                                 minmass = minmass)
                    locate_dict['frame'] = j * np.ones(len(locate_dict), dtype = int) 
                    f.append(locate_dict)
                                 
                f = pd.concat(f)
            self.located_frames = f
        return inner

    def _track_link_(self):
        @magicgui(call_button="Track Particles",
                layer_name = {'label':'Layer Name'},
                min_length = {'label':'Filter Stub Length', 'max': 1e16},
                search_range = {'label':'Search Range', 'max': 1e16},
                memory = {'label':'memory', 'max': 1e16},
                )
        def inner(
                layer_name: str,
                min_length: int,
                search_range: int,
                memory: int,
                ):
            assert self.track_bool, ("Set the tracking parameters with "
                                     "'preview' before tracking")
            assert self.located_frames is not None, "locate frames first..."
            f = self.located_frames
            t = tp.link(f, search_range = search_range, memory = memory)
            print(t.head)
            t1 = tp.filter_stubs(t, min_length)
            self.track_holder[layer_name] = t1.copy()
            # Add Track Centroids to viewer as points layer
            slice_handle = ['frame','y','x']
            self.viewer.add_points(t1[slice_handle], 
                                   name = "tracked centroids",
                                   face_color = 'k')

            self.viewer.add_tracks(t1[['particle','frame','y','x']])

        return inner

    def _tform_event_to_rvt_(self):
        @magicgui(
                call_button="Apply RVT",
                layer_name = {'label':'Layer Name'},
                rmin = {'label':'R Min', 'max': 1e16},
                rmax = {'label':'R Max', 'max': 1e16},
                )
        def inner(
                layer_name: str,
                rmin: int,
                rmax: int,
                ):
            layer_handle = self.__fetch_layer__(layer_name).data
            assert layer_handle.ndim == 3, "RVT only works for 3d image stacks"
            n_im = layer_handle.shape[0]
            temp = np.zeros(layer_handle.shape).astype(np.float32)
            for j in tqdm(range(n_im), desc = "applying rvt"):
                # GPU implementation?
                cp_arr = cp.array(layer_handle[j], dtype = np.float32)
                temp[j] = rvt(cp_arr, rmin = rmin, rmax = rmax).get()
                # CPU implementation?
                #cp_arr = np.array(layer_handle[j], dtype = np.float32)
                #temp[j] = rvt(cp_arr, rmin = rmin, rmax = rmax)
            self.viewer.add_image(temp, name = f'{layer_name} RVT' )
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
        dr[ dframe != 1] = np.nan
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
                track_id = {'label':'Track ID (-1 for all tracks)', 'min':-1, 'max': 1e16},
                fps = {'label':'fps', 'max': 1e16},
                mpp = {'label':'micron per pixel', 'max': 1e16, 'step': 1e-4},
                msd_point_min = {'label': 'msd fit idx min', 'max': 1e16},
                msd_point_max = {'label': 'msd fit idx max', 'max': 1e16},
                max_lagtime = {'label': 'max lagtime', 'max': 1e16},
                temperature = {'label': 'Temperature (K)', 'max': 1e16},
                viscosity = {'label': 'Viscosity (pa * s)', 'min':1e-16, 
                             'max': 1e16, 'step':1e-4},
                bins = {'label': 'histogram bins', 'min':1e-16, 'max': 1e16},
                track_key = {'label': 'Track Key (dict)'}

                )
        def inner(
                fps: float,
                mpp: float,
                track_key: str,
                track_id: int = -1,
                msd_point_min: int = 0,
                msd_point_max: int = 5,
                max_lagtime: int = 100,
                temperature: float = 295.0,
                viscosity: float = 0.001,
                bins: int = 50
                ):
            T = temperature
            eta = viscosity
            imsd_kwargs = {'mpp':mpp, 'fps':fps, 'max_lagtime': max_lagtime}
            track_handle = self.track_holder[track_key]
            if track_id == -1:
                print("calculating all tracks")
                imsd = tp.motion.imsd(track_handle, **imsd_kwargs)
                bay = []
                for elem in track_handle['particle'].unique():
                    bay.append(self.__calc_bayesian__(elem, track_handle, fps, mpp))
                bay = np.vstack(bay)
                #print("--> bay shape all elements:",bay.shape)
            else:
                print(f"calculating {track_id}")
                particle_slice = track_handle['particle'] == track_id
                if particle_slice.values.sum() == 0:
                    assert False, f"Empty Particle ID {track_id}"
                imsd = tp.motion.imsd(track_handle[particle_slice], **imsd_kwargs)
                bay = self.__calc_bayesian__(track_id, track_handle, fps, mpp)[None,:]
                #print("--> bay shape 1 element:",bay.shape)

            # Instance's imsd
            self.imsd = imsd

            # Log-Log Fit
            A,n_log,log_fits = imsd_powerlaw_fit(imsd, start_index = msd_point_min,
                                              end_index = msd_point_max)

            # Linear Fit
            m,b,lin_fits = imsd_linear_fit(imsd, start_index = msd_point_min,
                                              end_index = msd_point_max)


            # Remove negative slopes from analysis....
            #remove_neg_slopes = m > 0
            #n_neg = np.sum(remove_neg_slopes)
            #if n_neg > 0:
            #    print(f"Removing {n_neg} trajectories...")
            #m = m[remove_neg_slopes]
            #b = b[remove_neg_slopes]
            #lin_fits = lin_fits[remove_neg_slopes]

            # COMPOSE FIGURE
            fig,ax = plt.subplots(2,3, figsize = (10,10))
            for a in ax[0]:
                a.plot(imsd.index, imsd, color = 'k', alpha = 0.05, 
                       marker = '.', linestyle = '')

                a.set(xlabel = "Lag time (s)", 
                      ylabel = r"$\langle \Delta r^2 \rangle$ [$\mu$m$^2$]")
            slice_ = slice(msd_point_min, msd_point_max)
            ax[0,0].set(xscale = 'log', yscale = 'log')
            ax[0,0].plot(imsd.index.values[slice_], log_fits, color = 'r', alpha = 0.05)
            ax[0,1].plot(imsd.index.values[slice_], lin_fits, color = 'r', alpha = 0.05)
            kb = 1.38e-23
            D_log = np.exp(A)/4
            diam_log = kb * T / (3 * np.pi * eta * D_log * 1e-12) * 1e9
            D_lin = m/4
            diam_lin = kb * T / (3 * np.pi * eta * D_lin * 1e-12) * 1e9
            for j, arr in enumerate([n_log, diam_lin]):
                # Stats for the entire Population
                mean_local = np.nanmean(arr)
                std_local = np.nanstd(arr)
                cv_local = 100 * std_local / mean_local
                label_local = f"mean:{mean_local:0.2f}\nstd:{std_local:0.2f}\nCV: {cv_local:0.2f} %"
                ax[1,j].axvline(mean_local, label = label_local, color = 'k', 
                                linestyle = '--')

                med_local = np.nanmedian(arr)
                ax[1,j].axvline(med_local,
                                label = f"median: {med_local:0.2f}", 
                                color = 'b', 
                                linestyle = '--')


            # Gaussian Fits 
            gaussian_fit_linear = fit_param_gaussian(diam_lin, n_bins = bins, 
                                                     mode = 'normal')
            gaussian_fit_log = fit_param_gaussian(n_log, n_bins = bins, 
                                                     mode = 'normal')
            bay_diam = kb * T / (3 * np.pi * eta * bay[:,0] * 1e-12) * 1e9
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

            bay_diam = kb * T / (3 * np.pi * eta * bay[:,0] * 1e-12) * 1e9
            ax[1,2].hist(bay_diam, bins = bins)
            fig.tight_layout()
            f_name = "/tmp/_delete_me_.png"
            fig.savefig(f_name, dpi = 200)
            self.viewer.add_image(np.asarray(Image.open(f_name)), 
                                name = f"msd {track_id}")
            
        return inner

    def _apply_gaussian_layer_(self):
        @magicgui(
                call_button="Apply 2D Gaussian Filter",
                layer_name = {'label':'Layer Name'},
                sigma = {'label':'sigma','max': 100.0}
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
                kernel = {'label':'kernel size','max': 100, "step": 2 }
                )
        def inner(
                layer_name: str,
                kernel: int = 3
                ):
            layer_handle = self.__fetch_layer__(layer_name).data
            assert layer_handle.ndim == 3, "median_filter only works for 3d image stacks"
            n_im = layer_handle.shape[0]
            temp = np.zeros_like(layer_handle).astype(np.float32)
            for j in tqdm(range(n_im), desc = "applying median_filter"):
                cp_arr = cp.array(layer_handle[j], dtype = np.float32)
                temp[j] = median_filter(cp_arr, (kernel,kernel)).get()
            self.viewer.add_image(temp, name = f'{layer_name} Median Filtered' )
        return inner

    def _convert_event_to_grayscale_(self):
        @magicgui(
                call_button="Apply 2D Median Filter",
                layer_name = {'label':'Layer Name'},
                kernel = {'label':'kernel size','max': 100, "step": 2 }
                )
        def inner(
                layer_name: str,
                kernel: int = 3
                ):
            layer_handle = self.__fetch_layer__(layer_name).data
            assert layer_handle.ndim == 3, "median_filter only works for 3d image stacks"
            n_im = layer_handle.shape[0]
            temp = np.zeros_like(layer_handle).astype(np.float32)
            for j in tqdm(range(n_im), desc = "applying median_filter"):
                cp_arr = cp.array(layer_handle[j], dtype = np.float32)
                temp[j] = median_filter(cp_arr, (kernel,kernel)).get()
            self.viewer.add_image(temp, name = f'{layer_name} Median Filtered' )
        return inner

    def _isolate_event_channels_(self):
        @magicgui(
                call_button="Isolate Event Channels",
                )
        def inner(
                layer_name: str = 'event',
                ):
            layer_handle = self.__fetch_layer__(layer_name).data
            assert layer_handle.ndim == 4, "needs 4 channel event data..."
            nz,nx,ny,_ = layer_handle.shape
            out_pos = np.zeros([nz,nx,ny], dtype = np.uint8)
            out_neg = np.zeros([nz,nx,ny], dtype = np.uint8)
            pos = cp.array([255,255,255], dtype = np.uint8)
            neg = cp.array([200,126,64], dtype = np.uint8)
            print("Hard Coded Positive and Negative polarities:")
            print(f"\tPositive: {pos})")
            print(f"\tNegative: {neg})")
            for j in tqdm(range(nz)):
                cp_arr = cp.array(layer_handle[j], dtype = cp.uint8)
                out_pos[j] = 255*cp.prod(cp_arr == pos, axis = -1).astype(cp.uint8).get()
                out_neg[j] = 255*cp.prod(cp_arr == neg, axis = -1).astype(cp.uint8).get()
            self.viewer.add_image(out_pos, name = "positive", colormap = 'bop purple')
            self.viewer.add_image(out_neg, name = "negative", colormap = 'bop orange')
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
            out_neg = np.zeros([nz,nx,ny], dtype = np.uint8)
            none_val = cp.array([52,37,30], dtype = np.uint8)
            print("Hard Coded VOID polarities:")
            print(f"\tvoid polarity: {none_val})")
            for j in tqdm(range(nz)):
                cp_arr = cp.array(layer_handle[j], dtype = cp.uint8)
                out_pos[j] = 255*cp.prod(cp_arr != none_val, axis = -1).astype(cp.uint8).get()
            self.viewer.add_image(out_pos, name = "event combined", colormap = 'gray')
        return inner


if __name__ == "__main__":
    inst = spatio_temporal_registration_gui()
    napari.run()
