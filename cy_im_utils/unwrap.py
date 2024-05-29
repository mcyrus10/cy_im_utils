from PIL import Image
from cupyx.scipy.interpolate import RegularGridInterpolator as rgi_gpu
from cupyx.scipy.ndimage import median_filter
from enum import Enum
from functools import partial
from magicgui import magicgui
from magicgui.tqdm import tqdm
from multiprocessing import Pool
from napari.qt.threading import thread_worker
from pathlib import Path
from scipy.interpolate import make_interp_spline, RegularGridInterpolator
from scipy.optimize import least_squares
from scipy.signal import find_peaks
from skimage.transform import warp_polar
import cupy as cp
import logging
import matplotlib.pyplot as plt
import napari
import numpy as np


def write_volume(volume: np.array,
                 path: Path,
                 prefix: str,
                 extension: str = 'tif'
                 ) -> None:
    """
    This function will write a reconstruction volume to disk as images, if the
    directory (path) does not exist, it will create the new directory.

    Parameters:
    -----------
    volume: Numpy 3D array
        reconstructed volume
    path: string
        path to the new reconstruction directory
    file_name: string
        name of the output files
    extension: string
        extension of the output files
    """
    nz, nx, ny = volume.shape
    tqdm_writer = tqdm(range(nz), desc="writing images")
    for j in tqdm_writer:
        im = Image.fromarray(volume[j, :, :])
        im_path = path / f"{prefix}_{j:06d}.{extension}"
        im.save(im_path)


def imread(file) -> np.array:
    """
    the hits
    """
    return np.asarray(Image.open(file), dtype=np.float32)


class interp_methods(Enum):
    nearest = 'nearest'
    linear = 'linear'


def color_cycler(x: int) -> np.array:
    """
    this returns the rgba of a color cycling through the bmh style (on 0-1)
    """
    colors = ['348ABD', 'A60628', '7A68A6', '467821', 'D55E00', 'CC79A7',
              '56B4E9', '009E73', 'F0E442', '0072B2']
    n_color = len(colors)
    color_local = colors[x % n_color]
    color_arr = np.array(
                    [int(color_local[i:i+2], 16) for i in (0, 2, 4)] + [255]
                ) / 255
    return color_arr


class gpu_unwrap:
    """
    This executes on the entire volume....
    """
    def __init__(self,
                 points_dict: dict,
                 volume,                    # array-like
                 batch_size: int = 10,
                 sampling: int = 1,
                 interp_method: str = 'linear',
                 ) -> None:
        self.points = points_dict
        keys = sorted(list(points_dict.keys()))
        self.index_0, self.index_1 = keys[0], keys[1]
        self.nz = self.index_1 - self.index_0
        self.nx, self.ny = volume[0].shape
        self.sampling = sampling
        dx_norm_2d, dy_norm_2d, x_new_2d, y_new_2d = self.prep_splines(volume)
        coords_vectorized = self.compute_volume_interp_points(x_new_2d,
                                                              y_new_2d,
                                                              dx_norm_2d,
                                                              dy_norm_2d,
                                                              )
        self.new_vol = self.compute_new_volume(volume,
                                               coords_vectorized,
                                               batch_size=batch_size,
                                               interp_method=interp_method)
        del coords_vectorized
        self.cupy_memory_free()

    def cupy_memory_free(self):
        """
        this clears all the memory off the GPU
        """
        mempool = cp.get_default_memory_pool()
        mempool.free_all_blocks()

    def points_processor(self,
                         points_array: cp.array,
                         ) -> tuple:
        """
        This processes the points retrieved from napari and sequences them so
        the spiral has a continuously increasing angle (not capped at 360), and
        also returns the x and y points from the path (point sequence)
        """
        nx, ny = self.nx, self.ny
        x_ = points_array[:, -1]
        y_ = points_array[:, -2]
        dx = x_ - ny//2
        dy = y_ - nx//2
        angles = cp.rad2deg(cp.arctan(dx/dy))
        angles[dy > 0] += 180
        angles[(dx > 0) * (dy < 0)] += 360
        diff_angles = cp.diff(angles)
        theta = cp.zeros_like(angles)
        n_angles = len(angles)
        rolling_angle = 0
        for i in range(1, n_angles):
            if diff_angles[i-1] > 0:
                d_theta = cp.abs(angles[i]-360 - angles[i-1])
            else:
                d_theta = angles[i-1] - angles[i]
            rolling_angle += d_theta
            theta[i] = rolling_angle
        return x_, y_, theta

    def compute_arc_length(self, spl, theta: np.array):
        """
        numerically integrates the length of the spline and returns the angular
        array such that each point on the spline gets one sample.
        """
        diff = np.inf
        num_samples = 100
        j = 0
        while diff > 1:
            angle_new = cp.linspace(cp.min(theta), cp.max(theta), num_samples)
            x_new, y_new = spl(cp.asnumpy(angle_new)).T
            x_new, y_new = [cp.array(arr) for arr in [x_new, y_new]]
            length = cp.sum(np.sqrt(
                    (x_new[:-1]-x_new[1:])**2 + (y_new[:-1]-y_new[1:])**2)
                            )
            diff = length-num_samples
            num_samples = int(cp.round(length))
            j += 1
            if j > 10:
                logging.warning(f"arc length not converged in {j} iterations")
        return angle_new

    def prep_splines(self, volume) -> tuple:
        """
        ...
        """
        dx_norm_2d, dy_norm_2d = [], []
        x_new_2d, y_new_2d = [], []
        for i, (index_local, val) in enumerate(self.points.items()):
            x_temp = cp.array(val[:, 1], dtype=cp.float32)
            y_temp = cp.array(val[:, 0], dtype=cp.float32)

            x_, y_, theta = self.points_processor(cp.stack([x_temp, y_temp]).T)

            xy = cp.stack([x_, y_]).T.astype(cp.float32)
            boundary_condition = 'natural'
            spl = make_interp_spline(theta.get(),
                                     xy.get(),
                                     bc_type=boundary_condition)
            # =================================================================
            # This should allow you to hard-code the arc length sampling so
            # each row should have the same length?
            if i == 0:
                angle_new = self.compute_arc_length(spl, theta)

            x_new, y_new = cp.array(spl(angle_new.get()).T).astype(cp.float32)
            # =================================================================
            dx_new, dy_new = spl.derivative(1)(angle_new.get()).T
            dx_new, dy_new = [cp.array(arr, dtype=cp.float32) for arr in [dx_new, dy_new]]
            norm_arr = cp.sqrt(dx_new**2+dy_new**2).astype(cp.float32)
            dx_norm = -dy_new / norm_arr
            dy_norm = dx_new / norm_arr
            # =================================================================
            dx_norm_2d.append(dx_norm)
            dy_norm_2d.append(dy_norm)
            x_new_2d.append(x_new)
            y_new_2d.append(y_new)
        return dx_norm_2d, dy_norm_2d, x_new_2d, y_new_2d

    def compute_volume_interp_points(self,
                                     x_new_2d: list,
                                     y_new_2d: list,
                                     dx_norm_2d: list,
                                     dy_norm_2d: list,
                                     ) -> np.array:
        """
        this computes all the interpolation points throughout the volume with
        matrix multiplications. This computes the points at EVERY LAYER by
        interpolating the spline between index_0 and index_1
        I am trying to delete variables and free the GPU memory so it doesn't
        get overloaded...?
        """
        z0 = self.index_0
        nz = self.nz
        index_0 = self.index_0
        index_1 = self.index_1
        sampling = self.sampling

        z_vec = cp.linspace(z0, z0+nz-1, nz)[:, None, None].astype(cp.float32)
        vector_multiplier = cp.linspace(-sampling, sampling, sampling*2
                                        ).astype(cp.float32)
        dx_coords = cp.stack(dx_norm_2d)[:, :, None] @ vector_multiplier[None, :]
        x_coords_template = cp.stack(x_new_2d)[:, :, None] + dx_coords
        x_coords = interpolate(x_coords_template[0][None, :, :],
                               x_coords_template[1][None, :, :],
                               z_vec,
                               index_0,
                               index_1).get()
        del dx_coords, x_coords_template
        self.cupy_memory_free()
        dy_coords = cp.stack(dy_norm_2d)[:, :, None] @ vector_multiplier[None, :]
        y_coords_template = cp.stack(y_new_2d)[:, :, None] + dy_coords
        y_coords = interpolate(y_coords_template[0][None, :, :],
                               y_coords_template[1][None, :, :],
                               z_vec,
                               index_0,
                               index_1).get()
        del dy_coords, y_coords_template
        self.cupy_memory_free()
        z_coords = (z_vec * cp.ones_like(x_coords)).get()
        coords_vectorized = np.stack([z_coords, x_coords, y_coords]
                                     ).transpose(1, 2, 3, 0)
        self.cupy_memory_free()
        return coords_vectorized

    def fetch_interpolator(self, image_stack, index_start, index_end):
        """
        This returns the RegularGridInterpolator which allows you to
        interpolate the volume
        """
        _, nx, ny = image_stack.shape
        nz = index_end - index_start
        x_vector = cp.linspace(0, nx-1, nx)
        y_vector = cp.linspace(0, ny-1, ny)
        z_vector = cp.linspace(index_start, index_end, nz)+self.index_0
        slice_ = slice(index_start, index_end)
        interpolator = rgi_gpu((z_vector, x_vector, y_vector),
                               cp.array(image_stack[slice_], dtype=cp.float32),
                               bounds_error=False,
                               fill_value=None)
        return interpolator

    def compute_new_volume(self,
                           volume: np.array,
                           coords_vectorized: np.array,
                           batch_size: int = 10,
                           interp_method: str = 'linear') -> np.array:
        """
        this slices the batches and interpolates the volume on the GPU
        """
        n_recon, width, height, _ = coords_vectorized.shape
        new_vol = np.zeros([n_recon, width, height], dtype=np.float32)
        n_batches = n_recon // batch_size
        for j in tqdm(range(n_batches)):
            interpolator = self.fetch_interpolator(volume,
                                                   index_start=j*batch_size,
                                                   index_end=(j+1)*batch_size,
                                                   )
            slice_ = slice(j*batch_size, (j+1)*batch_size, 1)
            new_vol[slice_] = interpolator(cp.array(coords_vectorized[slice_],
                                                    dtype=cp.float32),
                                           method=interp_method).get()

        remainder = n_recon % batch_size
        if remainder > 0:
            interpolator = self.fetch_interpolator(
                                        volume,
                                        index_start=n_recon-remainder,
                                        index_end=n_recon
                                        )

            slice_ = slice(n_recon-remainder, n_recon, 1)
            new_vol[slice_] = interpolator(cp.array(coords_vectorized[slice_],
                                                    dtype=cp.float32),
                                           method=interp_method).get()
        del interpolator
        return new_vol


class napari_unwrapper:
    def __init__(self):
        self.viewer = napari.Viewer()
        self.viewer.title = "NIST Unwrapping GUI"
        self.splines = {}
        dock_widgets = {
                        'reset_splines': self.reset_splines(),
                        'polar_transform': self.polar_transform(),
                        'snap_points_to_peaks': self.snap_points_to_peaks(),
                        'snap_points_to_vert_grid': self.snap_points_to_vert_grid(),
                        'polar_points_to_path': self.polar_points_to_path(),
                        'points_to_sprial': self.points_to_spiral(),
                        'center': self.center_x_widget(),
                        'fit_spline': self.fit_spline(),
                        'show_spline': self.show_spline(),
                        'clone': self.clone_path(),
                        'show_layer_unwrapped': self.show_layer_unwrapped(),
                        'unwrap_volume': self.unwrap_volume(),
                        'write_unwrapped': self.write_unwrapped(),
                        }
        for key, val in dock_widgets.items():
            self.viewer.window.add_dock_widget(val,
                                               name=key,
                                               add_vertical_stretch=False,
                                               area='right')

    def _spline_fetch_(self) -> list:
        """

        This function isolates which image frame is in the viewer, and uses it
        to call fetch_spline

        Note: A 'path' group should be highlighted in the napari viewer not points

        """
        assert isinstance(self.viewer.layers[0],
                          napari.layers.image.image.Image), \
            "Layer 0 is not an image layer"
        assert not isinstance(self.viewer.layers[-1],
                          napari.layers.points.points.Points), \
            "Use 'path' (shapes) not 'points' for spline points"
        points = self.viewer.layers[-1].data[0]
        # 3D image or image stack, whatever
        if points.shape[-1] == 3:
            layer_no = int(points[0, 0])
            image = self.viewer.layers[0].data[layer_no]
        else:
            image = self.viewer.layers[0].data
        assert image.ndim == 2, 'image has more than 2 dimensions'

        return list(fetch_spline(points, image)) + [layer_no]

    def _points_to_image_(self) -> np.array:
        """
        This converts the points to an image array so they can be sorted, etc.
        """
        polar_im = self.viewer.layers[-2].data
        image_rep = np.zeros_like(polar_im)
        points_handle = self.viewer.layers[-1].data
        for elem in points_handle:
            x_, y_ = [int(round(elem[j])) for j in range(2)]
            image_rep[x_, y_] = 1
        return image_rep

    def reset_splines(self):
        """
        This widget plots a point in the center of the image
        """
        @magicgui(call_button="Reset Splines")
        def inner():
            self.splines = {}
            logging.info("Resetting Splines Dictionary")
        return inner

    def clone_path(self) -> None:
        @magicgui(call_button="clone point path",
                  destination_layer={'label': 'destination layer',
                                     'min': 0,
                                     'max': 1e6})
        def inner(destination_layer: int = 0):
            temp = inst.viewer.layers[-1].data[0].copy()
            temp[:, 0] = destination_layer
            inst.viewer.add_shapes(temp,
                                   shape_type='path',
                                   name='clone',
                                   edge_color='b')
        return inner

    def center_x_widget(self):
        """
        This widget plots a point in the center of the image
        """
        @magicgui(call_button="Plot Image Center")
        def inner():
            image_temp = self.viewer.layers[0].data
            nx, ny = image_temp.shape[-2:]
            center_coords = np.array([[nx//2, ny//2]])
            self.viewer.add_points(center_coords,
                                   name='center',
                                   symbol='cross',
                                   size=50,
                                   face_color='r',)

        return inner

    def fit_spline(self):
        @magicgui(call_button="fit spline")
        def inner():
            x_, y_, theta, spl, layer_no = self._spline_fetch_()
            self.splines[layer_no] = {'x_': x_,
                                      'y_': y_,
                                      'theta': theta,
                                      'spl': spl}
        return inner

    def show_spline(self):
        @magicgui(call_button="show spline",
                  layer_no={'label': "layer number",
                            'min': -1,
                            'max': 1e8,
                            'value': -1})
        def inner(layer_no: int):
            # if -1 -> fetch last spline
            if layer_no == -1:
                layers = list(self.splines.keys())
                layer_no = layers[-1]
            handle = self.splines[layer_no]
            angles = calculate_arc_length(handle['spl'], handle['theta'])
            self.xy_path = handle['spl'](angles)
            if 'spline' in inst.viewer.layers:
                inst.viewer.layers.remove('spline')
            inst.viewer.add_shapes([self.xy_path[:, ::-1]],
                                   name='spline',
                                   shape_type='path',
                                   edge_color='r')
            inst.viewer.layers.move(-1, -2)
        return inner

    def show_layer_unwrapped(self):
        @magicgui(call_button='unwrap single layer',
                  layer_no={'min': 0, 'max': 1e6, 'value': 0},
                  )
        def inner(layer_no: int,
                  sampling: int = 20,
                  reshape_factor: int = 400,
                  interpolation_method: interp_methods = interp_methods.nearest
                  ):
            points = inst.viewer.layers[-1].data[0]
            im = np.array(inst.viewer.layers[0].data[layer_no])
            y_temp, x_temp = points[:, -2], points[:, -1]
            x_, y_, theta, spl = fetch_spline(points, im)
            angle_new = calculate_arc_length(spl, theta)
            temp = unwrap_layer(im,
                                sampling=sampling,
                                points=np.stack([y_temp, x_temp]).T,
                                angle_new=angle_new,
                                interp_method=interpolation_method.name)
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(im)
            ax[0].scatter(points[:, -1], points[:, -2])
            ax[1].imshow(resample_unroll(temp, reshape_factor).T)
            ax[1].grid(True)
            plt.show()
        return inner

    def unwrap_volume(self):
        """
        This widget takes all the splines (2+) and interpolates between the
        adjacent ones (havent tested, but the sort should work here...)
        """
        @magicgui(call_button='unwrap volume',
                  sampling={'label': 'sampling',
                            'min': 1,
                            'max': 1e6,
                            'value': 15}
                  )
        def inner(sampling: int,
                  batch_size: int = 10,
                  interp_method=interp_methods.linear
                  ):
            points_dict = {}
            indices = sorted(list(self.splines.keys()))
            n_layers = len(indices)
            assert n_layers >= 2, "Need at least 2 layers to interpolate between"
            print("indices:", indices)
            for key in indices:
                val = self.splines[key]
                x_local, y_local = val['x_'], val['y_']
                points_dict[key] = np.stack([y_local, x_local]).T


            # This section takes each set of interpolation rows and
            # interpolates between them adds them to a list that is then
            # concatenated to form the "new_volume)"
            unwrap_holder = []
            for j in range(n_layers-1):
                key_1, key_2 = indices[j], indices[j+1]
                vol_slice = slice(key_1, key_2, 1)
                im = np.array(inst.viewer.layers[0].data[vol_slice])
                local_points_dict = {
                                    key_1: points_dict[key_1],
                                    key_2: points_dict[key_2],
                                    }
                new_volume = gpu_unwrap(points_dict=local_points_dict,
                                        volume=im,                 # array-like
                                        batch_size=batch_size,
                                        sampling=sampling,
                                        interp_method=interp_method.name,
                                        ).new_vol
                unwrap_holder.append(new_volume)
            
            shapes = np.stack([elem.shape for elem in unwrap_holder])
            limiting_dimension = shapes[:, 1].min()
            print(f"limiting size = {limiting_dimension}")
            new_volume = np.vstack(
                [elem[:, :limiting_dimension, :] for elem in unwrap_holder]
                                   ).astype(np.float32)
            self.mute_all()
            if 'unwrapped' in self.viewer.layers:
                self.viewer.layers.remove('unwrapped')

            inst.viewer.add_image(new_volume,
                                  name='unwrapped',
                                  colormap='gist_earth')

        return inner

    def write_unwrapped(self):
        @magicgui(call_button='write unwrapped volume',
                  output_dir={'label': 'output directory',
                              'mode': 'd'}
                  )
        def inner(output_dir: Path = Path.home(),
                  prefix: str = 'unwrapped',
                  extension: str = 'tif'):
            volume = self.viewer.layers['unwrapped'].data
            write_volume(volume, output_dir, prefix, extension)
        return inner

    def mute_all(self) -> None:
        """
        helper function to toggle visibiltiy of all items to off
        """
        for elem in self.viewer.layers:
            elem.visible = False

    def polar_transform(self):
        @magicgui(call_button='Polar Transform Image',
                  layer_no={'min': 0, 'max': 1e6, 'value': 0},
                  angular_samples={'min': 0, 'max': 1e6, 'value': 720},
                  center_x={'min': 0, 'max': 1e6, 'value': 0},
                  center_y={'min': 0, 'max': 1e6, 'value': 0},
                  )
        def inner(layer_no: int,
                  angular_samples: int,
                  center_x: int,
                  center_y: int,
                  ):
            im_handle = self.viewer.layers[0].data[layer_no]
            nx_, ny_ = im_handle.shape
            self.output_shape = (angular_samples, nx_//2)
            self.center = (center_x, center_y)
            polar_image = warp_polar(im_handle,
                                     radius=nx_/2,
                                     center=self.center,
                                     output_shape=self.output_shape
                                     )
            self.mute_all()
            self.viewer.add_image(polar_image, colormap='Spectral')
        return inner

    def snap_points_to_peaks(self):
        @magicgui(call_button='Find Peaks',
                  height={'min': -1e6, 'max': 1e6, 'value': 0.0},
                  vert_spacing={'min': 0, 'max': 1e6, 'value': 25},
                  distance={'min': 0, 'max': 1e6, 'value': 30},
                  )
        def inner(
                height: float,
                vert_spacing: int,
                distance: int
                ):
            all_peaks = []
            polar_im = self.viewer.layers[-1].data
            self.vert_spacing = vert_spacing
            for i, row in enumerate(polar_im):
                if i % vert_spacing != 0:
                    continue
                peaks = find_peaks(row, height=height, distance=distance)
                for j, p in enumerate(peaks[0]):
                    all_peaks.append([i, p])
            inst.viewer.add_points(all_peaks)
        return inner

    def snap_points_to_vert_grid(self):
        @magicgui(call_button='Snap points to vert grid')
        def inner():
            assertion_bool = isinstance(self.viewer.layers[-1], napari.layers.points.points.Points)
            assert assertion_bool, "Top Layer should be a points layer"
            for elem in self.viewer.layers[-1].data:
                if elem[0] % self.vert_spacing != 0:
                    print("repariing point")
                    factor = np.round(elem[0] / self.vert_spacing)
                    elem[0] = factor*self.vert_spacing
        return inner

    def points_to_spiral(self):
        """
        since the peak finding goes from top to bottom the consecutive points
        are arrange radially not angularly so this function re-sorts the points
        (under some conditions) so they follow a sprial clockwise or counter
        clockwise

        Note: Haven't tested clockwise
        """
        @magicgui(call_button='Points to Spiral')
        def inner(counter_clockwise: bool):
            points_type = napari.layers.points.points.Points
            assertion_bool = isinstance(self.viewer.layers[-1], points_type)
            assert assertion_bool, "Top Layer should be a points layer"
            image_rep = self._points_to_image_()
            non_zero = np.nonzero(image_rep)
            unique_rows = len(np.unique(non_zero[0]))
            square_qmark = len(non_zero[1])/unique_rows
            assert square_qmark % 1 == 0, "jagged array"
            square_qmark = int(square_qmark)
            non_zero = np.vstack(non_zero).reshape(2,
                                                   unique_rows,
                                                   square_qmark)
            if counter_clockwise:
                non_zero = non_zero[:, ::-1, :]
            ccw_points = non_zero.transpose(0, 2, 1).reshape(2, -1).T
            self.viewer.add_points(ccw_points)
        return inner

    def polar_points_to_path(self):
        @magicgui(call_button='Polar points to path',
                  layer_no={'min': 0, 'max': 1e6, 'value': 0},
                  )
        def inner(layer_no: int):
            if self.center is None:
                print("Polar Transformed image Center not Set")
                return None
            theta = 2*np.pi*self.viewer.layers[-1].data[:, 0]/self.output_shape[0]
            rad = self.viewer.layers[-1].data[:, 1]
            x_ = np.sin(theta)*rad + self.center[0]
            y_ = np.cos(theta)*rad + self.center[1]
            z_ = np.ones_like(x_)*layer_no
            points = np.array([z_, x_, y_]).T
            self.viewer.add_shapes(points, shape_type='path')
        return inner




def gaussian_1d(params: np.array, x: np.array) -> np.array:
    """
    From gpufit (gauss_1d.cuh)
        params[0]: amplitude
        params[1]: center coordinate
        params[2]: width
        params[3]: offset

        x = x-array

    returns:
        gauss 1d
    """
    argx = (x - params[1]) * (x - params[1]) / (2 * params[2] * params[2])
    ex = np.exp(-argx)
    return params[0] * ex + params[3]


def _residual_(params, x, data):
    model = gaussian_1d(params, x)
    return model - data


def points_processor(points_array: np.array,
                     image_array: np.array
                     ) -> tuple:
    """
    This processes the points retrieved from napari and sequences them so the
    spiral has a continuously increasing angle (not capped at 360), and also
    returns the x and y points from the path (point sequence)
    """
    nx, ny = image_array.shape
    x_ = points_array[:, -1]
    y_ = points_array[:, -2]
    dx = x_ - ny//2
    dy = y_ - nx//2
    angles = np.rad2deg(np.arctan(dx/dy))
    angles[dy > 0] += 180
    angles[(dx > 0) * (dy < 0)] += 360
    diff_angles = np.diff(angles)
    theta = np.zeros_like(angles)
    n_angles = len(angles)
    rolling_angle = 0
    for i in range(1, n_angles):
        if diff_angles[i-1] > 0:
            d_theta = np.abs(angles[i]-360 - angles[i-1])
        else:
            d_theta = angles[i-1] - angles[i]
        rolling_angle += d_theta
        theta[i] = rolling_angle
    return x_, y_, theta


def interpolate(y1, y2, x3, x1, x2):
    """
        |
      y2|               x
        |
      y?|---------- ?         <---- returns y?
        |           |
      y1|     x     |
        |           |
        |           |
        ----------------------
             x1    x3   x2
    """
    return y1 + (x3 - x1) * (y2 - y1) / (x2 - x1)


def fetch_spline(points_array, image, boundary_condition='natural') -> tuple:
    x_, y_, theta = points_processor(points_array, image)
    spl = make_interp_spline(theta, np.stack([x_, y_]).T, bc_type=boundary_condition)
    return x_, y_, theta, spl


def unwrap_layer(im: np.array,
                 points: int,
                 sampling: int,
                 angle_new: np.array = None,
                 boundary_condition: str = 'natural',
                 interp_method: str = 'nearest'
                 ):
    """
    single layer unwrapper, generic, wrap it to go higher
    """
    nx, ny = im.shape
    x_, y_, theta = points_processor(points, im)
    spl = make_interp_spline(theta, np.stack([x_, y_]).T, bc_type=boundary_condition)
    # =====================================================================
    x_new, y_new = spl(angle_new).T
    # =====================================================================
    dx_new, dy_new = spl.derivative(1)(angle_new).T
    norm_arr = np.sqrt(dx_new**2+dy_new**2)
    dx_norm = -dy_new / norm_arr
    dy_norm = dx_new / norm_arr
    vector_multiplier = np.linspace(-sampling, sampling, sampling*2)
    # =====================================================================
    # print(dx_norm)
    dx = dx_norm[:, None] @ vector_multiplier[None, :]
    dy = dy_norm[:, None] @ vector_multiplier[None, :]
    x_coords = x_new[:, None] + dx
    y_coords = y_new[:, None] + dy
    coords_vectorized = np.dstack([y_coords, x_coords])
    # =====================================================================
    x_vector = np.linspace(0, nx-1, nx)
    y_vector = np.linspace(0, ny-1, ny)
    interpolator = RegularGridInterpolator((x_vector, y_vector), im,
                                           bounds_error=False,
                                           fill_value=None)
    # =====================================================================
    new_image = interpolator(coords_vectorized, method=interp_method)
    return new_image


def calculate_arc_length(spl, theta):
    """
    numerically integrates the length of the spline and returns the angular
    array such that each point on the spline gets one sample.
    """
    diff = np.inf
    num_samples = 100
    j = 0
    while diff > 1:
        angle_new = np.linspace(np.min(theta), np.max(theta), num_samples)
        x_new, y_new = spl(angle_new).T
        length = np.sum(np.sqrt(
                (x_new[:-1]-x_new[1:])**2 + (y_new[:-1]-y_new[1:])**2)
                        )
        diff = length-num_samples
        num_samples = int(np.round(length))
        j += 1
        if j > 10:
            logging.warning(f"arc length not converged in {j} iterations")
    return angle_new


def unwrap_wrapper(index: int,
                   index_0: int,
                   index_1: int,
                   points_dict: dict,
                   image_files: list,
                   sampling: int,
                   angle_new: np.array,
                   boundary_condition: str = 'natural',
                   interp_method: str = 'nearest'
                   ):
    """
    this is a handle for the parallel unwrapper to call the unwrap_layer
    primitive
    """
    if isinstance(image_files, list):
        im = imread(image_files[index])
    else:
        im = np.array(image_files[index], dtype=np.float32)
    nx, ny = im.shape
    y_temp = interpolate(points_dict[index_0][:, -2],
                         points_dict[index_1][:, -2],
                         index,
                         index_0,
                         index_1)
    x_temp = interpolate(points_dict[index_0][:, -1],
                         points_dict[index_1][:, -1],
                         index,
                         index_0,
                         index_1)
    points = np.stack([y_temp, x_temp]).T
    return unwrap_layer(im,
                        points,
                        sampling=sampling,
                        angle_new=angle_new,
                        boundary_condition=boundary_condition,
                        interp_method=interp_method)


def par_unwrap(index_0: int,
               index_1: int,
               points_dict: dict,
               sampling: int,
               image_files: list,
               boundary_condition: str = 'natural',
               interp_method: str = 'nearest'
               ) -> np.array:
    """
    this is a wrapper for pooled operations executed in parallel. nice speedup
    """
    if isinstance(image_files, list):
        im_temp = imread(image_files[index_0])
    else:
        im_temp = np.array(image_files[0], dtype=np.float32)
    x_, y_, theta = points_processor(points_dict[index_0], im_temp)
    spl = make_interp_spline(theta,
                             np.stack([x_, y_]).T,
                             bc_type=boundary_condition)
    angle_new = calculate_arc_length(spl, theta)
    partial_unwrap = partial(unwrap_wrapper,
                             index_0=index_0,
                             index_1=index_1,
                             sampling=sampling,
                             points_dict=points_dict,
                             image_files=image_files,
                             angle_new=angle_new,
                             boundary_condition=boundary_condition,
                             interp_method=interp_method
                             )
    with Pool(8) as pool:
        volume_holder = pool.map(partial_unwrap, tqdm(range(index_0, index_1)))

    return np.stack(volume_holder)


def resample_unroll(input_image: np.array,
                    resample_length: int = 200
                    ) -> np.array:
    """
    this function reshapes the unrolled image so that it can be more easily
    viewed (i.e., the aspect ratio makes the image visible) the re-sample
    length is how wide the rows of the image will be.  it does discard some
    pixels to make the image divisible if it is not perfectly divisible by
    resample_length
    """
    nx, ny = input_image.shape
    n_samples = nx // resample_length
    temp = []
    for j in range(n_samples):
        slice_ = slice(j*resample_length, (j+1)*resample_length)
        temp.append(input_image[slice_, :])
    return np.hstack(temp)


if __name__ == "__main__":
    inst = napari_unwrapper()
    napari.run()
