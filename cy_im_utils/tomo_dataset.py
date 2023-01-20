"""

This file has the tomo_dataset class that defines how to read/write and process
frames.
The napari_gui inherits from tomo_dataset and just wraps all the method calls
with button presses, and likewise the jupyter version can operate inside a
notebook

to do:
    - batch fdk cone
    - update all documentation and function/method annotations
    - control digit width in write_image stack
    - control filtration parameters for opposing image stack

"""
try:
    # remote versions can have pip installed, but locally, I am just appending
    # the path
    from cy_im_utils.prep import imread
except ModuleNotFoundError as me:
    from sys import path
    path.append("C:\\Users\\mcd4\\Documents\\cy_im_utils")

from cy_im_utils.prep import radial_zero,field_gpu,imstack_read,\
        center_of_rotation,imread_fit,rotated_crop
from cy_im_utils.recon_utils import ASTRA_General,astra_2d_simple,astra_tomo_handler
from cy_im_utils.sarepy_cuda import remove_all_stripe_GPU
from cy_im_utils.thresholded_median import thresh_median_2D_GPU
from cy_im_utils.visualization import COR_interact,SAREPY_interact,orthogonal_plot,median_2D_interact
from cy_im_utils.gpu_utils import median_GPU_batch
from cy_im_utils.logger import log_setup

from ipywidgets import IntSlider,FloatSlider,HBox,VBox,interactive_output,interact,interact_manual,RadioButtons,Text,IntRangeSlider,interactive,widgets,Layout
from matplotlib.patches import Rectangle
from PIL import Image
from cupyx.scipy.ndimage import median_filter as median_gpu
from dask.array import array as dask_array
from enum import Enum
from functools import partial
try:
    from magicgui import magicgui
    from magicgui.tqdm import tqdm
except ImportError:
    print("--> magicgui not installed")
try:
    from napari.qt.threading import thread_worker
    import napari
except ImportError:
    print("--> napari not installed")
from pathlib import Path
import cupy as cp
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import yaml


class tomo_dataset:
    """
    This is the base class for tomography that can be called inside a script,
    or used with jupyter widgets or napari gui. It uses astra to execute the
    reconstructions
    """
    def __init__(self, config_file):
        logging.info("-"*80)
        logging.info("-- TOMOGRAPHY RECONSTRUCTION --")
        logging.info("-"*80)
        self.previous_settings = {}
        self.config_filename = config_file

        with open(self.config_filename, 'r') as f:
            self.settings = yaml.safe_load(f)

        for key, val in self.settings['paths'].items():
            logging.info(f"converting {val} to pathlib.Path")
            self.settings['paths'][key] = Path(val)

        logging.info(f"{'='*20} SETTINGS: {'='*20}")
        for key, val in self.settings['pre_processing'].items():
            setattr(self, key, val)
        print("------>", self.settings)
        for key, val in self.settings.items():
            if isinstance(val, dict):
                logging.info(f"{key}:")
                for sub_key, sub_val in val.items():
                    logging.info(f"\t{sub_key}:{sub_val}")
            else:
                logging.info(f"{key}:{val}")
        logging.info(f"{'='*20} End SETTINGS {'='*20}")
        self.sare_bool = False
        self.imread = self.fetch_imread_function()
        self.load_field('dark')
        self.load_field('flat')
        self.files = []

    # -------------------------------------------------------------------------
    #                       UTILS
    # -------------------------------------------------------------------------
    def update_config(self) -> None:
        """ This wrapper calls the yaml.safe_dump to re-write the config file,
        such as when files are updated

        Note that when you copy a dictionary, it DOES NOT copy the nested
        layers, they are still referred to as the same memory so any operations
        will modify both the copied and the original.... (this means you cannot
        copy the whole dictionary then change a sub-dictionary because it will
        change the original dictionary's sub-dictionary)
        Therefore, here I am copying the 'paths' sub-dictionary from settings,
        then converting the paths to strings, then copying the full dictionary
        and adding the paths back in. A bit complicated, but this is simpler
        than the config handler class
        """
        logging.info(f"Updating config file ({self.config_filename})")
        yaml_safe_paths = self.settings['paths'].copy()
        for key, val in yaml_safe_paths.items():
            yaml_safe_paths[key] = val.as_posix()
        cpy = self.settings.copy()
        cpy.update({'paths': yaml_safe_paths})
        with open(self.config_filename, 'w') as f:
            yaml.safe_dump(cpy, f)

    def load_transmission_sample(self, image_index: int = 0) -> None:
        """ This is for loading an image for the median to operate on

        Args:
        -----
            image_index: int - transmission image index

        Returns:
        --------
            None

        Side-Effects:
        -------------
            1. Creates self.transmission_sample attribute
            2. writes the tm_global_index attribute (this helps keep track of
                what image self.transmission_sample is so that it doesn't have
                go be re-read each time a function needs that image

        """
        proj_path = self.settings['paths']['projection_path']
        ext = self.settings['pre_processing']['extension']
        proj_files = self.fetch_files(proj_path, ext=ext)
        sample = self.imread(proj_files[image_index])
        if self.settings['pre_processing']['transpose']:
            sample = sample.T
        self.transmission_sample = sample
        self.tm_global_index = image_index

    def fetch_files(self, path_, ext: str = 'tif') -> list:
        """ just a wrapper for fetching a globbed sorted list

        Args:
        -----
            path: pathlib.Path - directory to extract files from
            ext: str - file extension (suffix in pathlib parlance)

        Returns:
            (list) - sorted files from directory matching extension
        """
        return sorted(list(path_.glob(f"*.{ext}")))

    def fetch_combined_image(self,
                             d_theta: float = 180.0,
                             median: int = 3,
                             thresh_kernel: int = 5,
                             thresh_z: float = 1.0
                             ) -> None:
        """ this function composes the 0 + 180 degree image for defining the
        center of rotation.

        Args:
        -----
            mode: str -> right now only auto is availalbe which automatically
                        searches for the 180 degree image and if it is not
                        found, it uses the closest angular value
            median: int -> this median is applied to the composed image to
                        reduce noise
        returns:
        --------
            None

        Side effects:
        -------------
            - creates attribute self.combined_image
            - adds layer 'combined image' to the viewer
        """
        proj_path = self.settings['paths']['projection_path']
        ext = self.settings['pre_processing']['extension']
        proj_files = self.fetch_files(proj_path, ext=ext)
        logging.info("Composing Tilt Correction Image")

        nx, ny = self.flat.shape
        n_images = int(360 // d_theta)
        angles = np.arange(0, 360, d_theta).astype(np.uint16)
        combined = cp.zeros([n_images, nx, ny], dtype=cp.float32)
        image_files = [proj_files[0]]
        angles_local = np.rad2deg(np.array([self._angle_(f) for f in proj_files]))
        for i, angle in enumerate(angles[1:]):
            abs_diff = np.abs(angle-np.array(angles_local, dtype=np.float32))
            file_idx = np.where(abs_diff == np.min(abs_diff))[0][0].astype(np.uint32)
            logging.info(f"\tangle: {angle}; file index: {file_idx}; filename: {str(proj_files[file_idx])}")
            image_files.append(proj_files[file_idx])

        med_kernel = (median, median)
        ff = median_gpu(cp.array(self.flat, dtype=cp.float32), med_kernel)
        df = median_gpu(cp.array(self.dark, dtype=cp.float32), med_kernel)

        for i, f in enumerate(image_files):
            im_temp = median_gpu(cp.asarray(self.imread(f), dtype = cp.float32),
                                 med_kernel)
            transmission = (im_temp-df) / (ff-df)
            attenuation = -cp.log(transmission)
            attenuation[~cp.isfinite(attenuation)] = 0
            attenuation = thresh_median_2D_GPU(attenuation,
                                               thresh_kernel,
                                               thresh_z)
            combined[i] = attenuation
        combined = cp.sum(combined, axis=0).get()

        tp = False
        if 'transpose' in self.settings['pre_processing']:
            tp = self.settings['pre_processing']['transpose']

        combined = combined.T if tp else combined
        self.combined_image = combined

    def is_jupyter(self) -> bool:
        """
        Yanked this from "Parsing Args in Jupyter Notebooks" on YouTube
        This tells you whether you're executing from a notebook or not
        """
        jn = True
        try:
            get_ipython()
        except NameError as err:
            print(err)
            jn = False
        return jn

    def free(self, field: str) -> None:
        """
        This can be used to free the projections if they are taking up too much
        memory

        Args:
        -----
            field : str
                which field to free
        Returns:
        --------
            None

        Side-Effects:
        -------------
            sets field to None (deletes it)

        examples:
            tomo_recon.free_field("projections")
            tomo_recon.free_field("transmission")
        """
        setattr(self, field, None)

    def gpu_curry_loop(self,
                       function,
                       ax_length: int,
                       batch_size: int,
                       tqdm_string: str = ""
                       ) -> None:
        """
        this is a generic method for currying functions to the GPU

        Args:
        -----
            function : python function
                the function only takes arguments for the indices x0 and x1 and
                performs all the operations internally
            ax_len : int
                length of the axis along which the operations are being
                executed (to determine remainder)

        Returns:
        --------
            None

        Side-Effects:
        -------------
            executes the function

        """
        for j in tqdm(range(ax_length//batch_size), desc=tqdm_string):
            function(j*batch_size, (j+1)*batch_size)
        remainder = ax_length % batch_size
        if remainder > 0:
            logging.info(f"remainder = {remainder}")
            function(ax_length-remainder, ax_length)

    def recon_radial_zero(self, radius_offset: int = 0) -> None:
        """
        this function makes all the values outside the radius 0

        args:
        -----
            radius_offset: int (optional)
                how much more than the radius should be zeroed out

        """
        n_frame, detector_width, _ = self.reconstruction.shape
        tqdm_zeroing = tqdm(range(n_frame),
                            desc="zeroing outisde crop radius")
        for f in tqdm_zeroing:
            radial_zero(self.reconstruction[f], radius_offset=radius_offset)

    def _imread_(self, x, dtype=np.float32):
        """ this is a wrapper for using PIL image read that returns the image
        as a numpy array
        """
        with Image.open(x) as im:
            return np.array(im, dtype=dtype)

    def fetch_imread_function(self):
        extension = self.settings['pre_processing']['extension']
        assert extension in ['tif', 'fit'], f'unknown extension: {extension}'
        imread_functions = {
                            'tif': self._imread_,
                            'fit': imread_fit
                            }
        return imread_functions[extension]

    def check_reconstruction_config(self) -> None:
        """ This is to make sure that geometries are compatible so the user does
        not pick something that will make astra cry
        """
        alg_3d = ['FDK_CUDA', 'SIRT3D_CUDA']
        alg_2d = ['FBP_CUDA', 'SIRT_CUDA', 'CGLS_CUDA', 'EM_CUDA', 'SART_CUDA']
        fourier_methods = ['FDK_CUDA', "FBP_CUDA"]
        sirt_methods = ['SIRT3D_CUDA', 'SIRT_CUDA', 'SART_CUDA']
        iter_methods = ['SIRT_CUDA', 'SIRT3D_CUDA', 'SART_CUDA', 'CGLS_CUDA']
        geom_3d = ['parallel3d', 'cone']
        geom_2d = ['parallel', 'fanflat']
        non_par_geom = ['cone', 'fanflat']
        alg = self.settings['recon']['algorithm']
        geom = self.settings['recon']['geometry']
        if alg not in alg_3d+alg_2d:
            assert False, f"{alg} is uknown to check_reconstruction_config"
        if alg in alg_3d:
            assert geom in geom_3d, f'{alg} (3d) incompatilbe with {geom} geometry (2d)'
        elif alg in alg_2d:
            assert geom in geom_2d, f'{alg} (2d) incompatilbe with {geom} geometry (3d)'

        # handle for options dictionary
        if 'options' in self.settings:
            options_h = self.settings['recon']['options']
            if 'FilterType' in options_h:
                if options_h['FilterType'] != 'none' and \
                        alg not in fourier_methods:
                    logging.warning("Skipping FilterType for non Fourier Method")

            if 'MinConstraint' in options_h and alg not in sirt_methods:
                logging.warning("Skipping MinConstraint for non SIRT Method")

        if alg in iter_methods and 'iterations' not in self.settings['recon']:
            logging.warning("0 iterations specified for iterative method")

        if geom in non_par_geom:
            handle = self.settings['recon']
            assert handle['source to origin distance'] > 0.0,\
                "source - detector distance must be greater than 0"
            assert handle['origin to detector distance'] > 0.0, \
                "origin -  detector distance must be greater than 0"

    def search_settings(self,
                        search_key: str,
                        default: "various"
                        ) -> "various":
        """ This can search the configuration through all fields for a specific
        value to see if it exists, if it does not, then the default is
        returned. This is used to auto-populate the widgets with values when an
        existing config is read in.
        Parameters
        ----------
            search_key: str
                the string to match for a parameter in the config
            default: various
                the default value to return if the parameter is not found in
                the config
        Returns
        -------
            either the value from the settings dictionary or the default value
        """
        for key, val in self.settings.items():
            if isinstance(val, dict):
                for sub_key, sub_val in val.items():
                    if search_key == sub_key:
                        return sub_val
            elif isinstance(val, str):
                if search_key == key:
                    return val
        else:
            return default

    def COR_operations(self,
                       cor_y0: int,
                       cor_y1: int,
                       ax: None,
                       ax2: None
                       ) -> None:
        """ This method calculates the tilt and centers the image for
        reconstruction it is used by both the gui and the jupyter widget.

        The algorithm:
            1. Determines the off-axis tilt of the raw cropped image
            2. Applies this rotation and calculates the off-axis tilt
            3. Applies 0.75 x this rotation and calculates the off-axis tilt
            4. Applies 1.25 x this rotation and calculates the off-axis tilt
            5. Linearly interpolates these 3 applied rotations and off-axis
                tilts to estimate the applied rotation to yield 0 degrees
                off-axis tilt
            6. Modifies settings for crop coordinates and COR values (theta)
                in-place

        Parameters:
        -----------
            cor_y0(y1): int - the y-coordinates for the center of rotation
                    calculation

            ax: (optional) array-like - axes for plotting the center of
                rotation images (with the overlay of center of mass)

            ax2: plt.axis - axis for plotting the off-axis tilt of the applied
                    rotations

        Returns:
        --------
            None - modifies 'crop' and 'COR' settings in-place

        """
        keys = ['x0', 'x1', 'y0', 'y1']
        x0, x1, y0, y1 = [self.settings['crop'][key] for key in keys]
        slice_x = slice(x0, x1)
        slice_y = slice(y0, y1)
        cropped_image = self.combined_image[slice_x, slice_y]
        combined_cupy = cp.array(self.combined_image, dtype=cp.float32)
        # Off Axis Estimation of Raw Image
        cor = center_of_rotation(cropped_image, cor_y0, cor_y1, ax=ax[0])
        theta = np.tan(cor[0])*(180/np.pi)
        theta_a = theta*0.75
        theta_b = theta*1.25

        # Rotating raw image by its off-axis amount
        rot = rotated_crop(
                                combined_cupy,
                                -theta,
                                [y0, y1, x0, x1]
                                ).get()

        # Rotating raw image by 0.75*off-axis amount
        rot_a = rotated_crop(
                                combined_cupy,
                                -theta_a,
                                [y0, y1, x0, x1]
                                ).get()

        # Rotating raw image by 1.25*off-axis amount
        rot_b = rotated_crop(
                                combined_cupy,
                                -theta_b,
                                [y0, y1, x0, x1]
                                ).get()

        cor2 = center_of_rotation(rot, cor_y0, cor_y1, ax=[])
        cor2_a = center_of_rotation(rot_a, cor_y0, cor_y1, ax=[])
        cor2_b = center_of_rotation(rot_b, cor_y0, cor_y1, ax=[])

        theta2 = np.tan(cor2[0])*(180/np.pi)
        theta2_a = np.tan(cor2_a[0])*(180/np.pi)
        theta2_b = np.tan(cor2_b[0])*(180/np.pi)

        # Linear fit of applied rotation to final off-axis rotation values
        theta_fit = np.polyfit([theta2, theta2_a, theta2_b],
                               [theta, theta_a, theta_b]
                               , 1)
        theta_final = np.polyval(theta_fit, 0)
        rot_ = rotated_crop(
                                combined_cupy,
                                -theta_final,
                                [y0, y1, x0, x1]
                                ).get()

        cor_final = center_of_rotation(rot_, cor_y0, cor_y1, ax=ax[1])

        theta_qmark_zero = np.tan(cor_final[0])*(180/np.pi)

        xs_local = [theta2, theta2_a, theta2_b, theta_qmark_zero]
        print('theta applied', theta, theta_a, theta_b, theta_final)
        print('theta of rotated', theta2, theta2_a, theta2_b, theta_qmark_zero)

        rot_final = rotated_crop(
                                combined_cupy,
                                -theta_final,
                                [y0, y1, x0, x1]
                                ).get()

        cor2_corrected = center_of_rotation(rot_final,
                                            cor_y0,
                                            cor_y1,
                                            ax=ax[1])

        crop_nx = y1-y0
        dx = int(np.round(cor2_corrected[1]))-crop_nx//2
        y0 += dx
        y1 += dx
        slice_x = slice(x0, x1)
        slice_y = slice(y0, y1)
        crop2 = self.combined_image[slice_x, slice_y]
        crop2rot = rotated_crop(
                                combined_cupy,
                                -theta_final,
                                [y0, y1, x0, x1]
                                ).get()

        cor3 = center_of_rotation(crop2rot, cor_y0, cor_y1, ax=ax[2])

        self.settings['crop'] = {
                                'y0': int(y0),
                                'y1': int(y1),
                                'x0': int(x0),
                                'x1': int(x1)
                                }

        self.settings['COR'] = {
                                'y0': int(cor_y0),
                                'y1': int(cor_y1),
                                'theta': float(theta_final)
                                }

        # Plotting
        if len(ax) == 3:
            if ax[0]:
                ax[0].set_title("Original Image")
            if ax[1]:
                ax[1].set_title(f"rotated = {theta_final:.4f} degrees")
            if ax[2]:
                ax[2].set_title(f"corrected center dx = {dx}")

        if ax2 is not None:
            ax2.scatter([theta, theta_a, theta_b, theta_final], xs_local)
            ax2.plot(np.polyval(theta_fit, xs_local),
                     xs_local,
                     'k--')
            ax2.axhline(0, color='k')
            ax2.set_xlabel("theta applied (deg)")
            ax2.set_ylabel("off-axis of rotated (deg)")

    # -------------------------------------------------------------------------
    #                       PRE PROCESSING
    # -------------------------------------------------------------------------
    def load_field(self, mode: str) -> None:
        """
        wrapper for "field_GPU" in prep.py for loading in flat and dark fields

        args:
            mode : str
                mode can be 'dark' or 'flat'

        examples:
            tomo_recon = tomo_dataset(data_dict)
            tomo_recon.load_field("flat")
            tomo_recon.load_field("dark")
        """
        field_path = self.settings['paths'][f'{mode}_path']

        # This conditional is for the case of no field existing
        if field_path is None:
            logging.info(f"{mode} field is None -> returning 2D array of zeros")
            proj_path = self.settings['paths'][f'projection_path']
            ext_projections = self.settings['pre_processing']['extension']
            proj_files = list(proj_path.glob(f"*{ext_projections}*"))
            proj_image = self.imread(proj_files[0])
            field = np.zeros_like(proj_image, dtype=np.float32)

        # Field does exist
        else:
            keys = list(self.settings['pre_processing'].keys())

            # find the right image reading function
            if f'extension_{mode}' in keys:
                ext = self.settings['pre_processing'][f'extension_{mode}']
            else:
                ext = self.settings['pre_processing']['extension']

            logging.info(f"Reading {mode} field from {field_path}")
            files = list(field_path.glob(f"*{ext}*"))
            logging.info(f"\tnum files = {len(files)}")
            logging.info("\tshape files = "
                              f"{np.array(self.imread(files[0])).shape}")
            nx, ny = np.asarray(self.imread(files[0])).shape
            field = field_gpu(files, 3)
        field = self.median_operations_wrapper(
                                cp.array(field, dtype = np.float32)).get()
        setattr(self, mode, field)

    def median_operations_wrapper(self, image: cp.array) -> cp.array:
        """ Generic wrapper for 2D Median Operations
        """
        prep_handle = self.settings['pre_processing']
        if 'median_xy' in prep_handle.keys():
            kernel = prep_handle['median_xy']
            image = median_gpu(image, (kernel, kernel))
        if 'thresh median kernels' in prep_handle.keys():
            kernels = prep_handle['thresh median kernels']
            z_scores = prep_handle['thresh median z-scores']
            assert len(kernels) == len(z_scores), "Dissimilar kernels and zs"
            for kern, z_sc in zip(kernels, z_scores):
                image = thresh_median_2D_GPU(image, kern, z_sc)
        return image

    def load_projections(self,
                         truncate_dataset: int = 1,
                         ) -> None:
        """
        Testing if its faster to load the images in straight to transmission
        space

        Operations:
            1) flat_ = flat-dark
            2) flat_scale = sum norm_patch pixels in flat_
            3) loop over all images : Normalize + Lambert Beer + Crop + Rotate
                a) load image
                b) subtract dark field (in-place operation)
                c) scale of image-dark (sum over norm)
                d) divide by flat_ (in-place)
                e) multiply by flat_scale/scale (mean of norm --> ~1)
                f) spatial median  (3x3 by default)
                g) crop
                h) rotate
                i) assign to self.transmission array

        Args:
            truncate_dataset : int
                If you want to load in a datset faster this factor downsamples
                the dataset by (every other {truncate_dataset} value
        returns:
            None (operates in-place)

        """
        self.update_config()

        proj_path = self.settings['paths']['projection_path']
        ext = self.settings['pre_processing']['extension']
        all_files = list(
                        proj_path.glob(f"*{ext}*")
                        )[::truncate_dataset]

        new_files = []
        for file_ in all_files:
            if file_ not in self.files:
                new_files.append(file_)
                self.files.append(file_)

        if len(new_files) == 0:
            # If no new files have been added -> do nothing
            logging.info("No New Files")
            return

        # NOTE THE REVERSAL OF X and Y HERE. THIS IS THE ONLY PLACE WHERE THIS
        # DISTINCTION OCCURS
        keys = ['x0', 'x1', 'y0', 'y1']
        x0, x1, y0, y1 = [self.settings['crop'][key] for key in keys]
        crop_patch = [y0, y1, x0, x1]

        logging.info(f"crop_patch = {crop_patch}")
        x0_, x1_, y0_, y1_ = [self.settings['norm'][key] for key in keys]
        norm_patch = [y0_, y1_, x0_, x1_]
        self.norm_x = slice(norm_patch[0], norm_patch[1])
        self.norm_y = slice(norm_patch[2], norm_patch[3])
        theta = self.settings['COR']['theta']

        attn_ny = crop_patch[1]-crop_patch[0]
        attn_nx = crop_patch[3]-crop_patch[2]
        n_proj = len(new_files)

        # Re-shaping this so that n_proj is axis 1 and attn_nx is axis 0?
        temp = np.empty([attn_nx, n_proj, attn_ny], dtype=self.dtype)

        dark_local = cp.array(self.dark)
        flat_local = cp.array(self.flat)
        logging.info(f"New Files Shape = {temp.shape}")

        load_im = lambda f :  cp.asarray(self.imread(f), dtype=self.dtype)

        if self.settings['pre_processing']['transpose']:
            load_im = lambda f:  cp.asarray(self.imread(f),
                                            dtype=self.dtype).T
            dark_local = dark_local.T
            flat_local = flat_local.T

        flat_ = flat_local-dark_local
        flat_scale = cp.sum(flat_[self.norm_y, self.norm_x])
        logging.info(f"flat patch magnitude = {flat_scale}")
        tqdm_imread = tqdm(range(n_proj),
                           desc="Projection -> Transmission Ops")
        """
        In ReconstructCT Source code see line 183 from
        NIF_TomographyReconstruction/NIF_Translate_projection_Tomography.m
        for these operations

        # note -> sep 9, 2022 moved median to before dark field subtraction
        """
        for i in tqdm_imread:
            im = load_im(new_files[i])
            im = self.median_operations_wrapper(im)
            im -= dark_local
            scale = cp.sum(im[self.norm_y, self.norm_x])
            im /= flat_
            im *= flat_scale/scale
            im = rotated_crop(im, -theta, crop_patch)
            temp[:, i, :] = cp.asnumpy(im)

        if hasattr(self, 'transmission'):
            self.transmission = np.hstack([self.transmission, temp])
        else:
            self.transmission = temp
        self.fetch_angles()

    # -------------------------------------------------------------------------
    #                       PROCESSING
    # -------------------------------------------------------------------------
    def remove_all_stripe_ops(self, id0: int, id1: int) -> None:
        """
        SAREPY_CUDA takes sinogram as the index 1 (of 0,1,2) right now!!

        operates in-place

        this is meant to be called by gpu_curry_loop

        args:
        ----
            x0 : int
                index 1 for the slice

            x1 : int
                index 2 for the slice

        """
        transpose = (1, 0, 2)
        slice_ = slice(id0, id1)
        logging.debug(f"indices for slice = {id0},{id1}")
        vol_gpu = cp.asarray(self.transmission[slice_, :, :], dtype=self.dtype)
        vol_gpu = cp.transpose(vol_gpu, transpose)
        vol_gpu = remove_all_stripe_GPU(vol_gpu,
                                        self.settings['SARE']['snr'],
                                        self.settings['SARE']['la_size'],
                                        self.settings['SARE']['sm_size'])
        vol_gpu = cp.transpose(vol_gpu, transpose)
        self.transmission[slice_, :, :] = cp.asnumpy(vol_gpu)

    def remove_all_stripe(self, batch_size: int = 10) -> None:
        """
        operates in-place

        """
        logging.info("REMOVING STRIPE ARTIFACTS")
        n_sino, _, _ = self.transmission.shape
        self.gpu_curry_loop(self.remove_all_stripe_ops,
                            n_sino,
                            batch_size,
                            tqdm_string="Stripe Artifact Removal")

    def attenuation(self, ds_factor: int) -> np.array:
        """ wrapper to compute attenuation array as float32
        """
        return -np.log(
                       self.transmission[:, ::ds_factor, :],
                       where=self.transmission[:, ::ds_factor, :] > 0
                       ).astype(np.float32)

    def reconstruct(self,
                    ds_interval: int = 1,
                    iterations: int = 1,
                    seed=0
                    ) -> None:
        """
        this just translates all the properties over to ASTRA

        Args:
        ----
            ds_interval: int - downsampling interval default is 1 -> no
                            Downsampling
            iterations: int - only for iterative methods
            angles: np.array (optional) - if non-linear
            seed: scalar or np.array - this is the seed for iterative methods
        """
        try:
            del self.reconstruction
        except:
            pass
        logging.info("--- reconstructing ---")
        print("--- RECONSTRUCTING ---")
        self.reconstruction = self._reconstructor_.reconstruct_volume(
                                self.attenuation(ds_interval),
                                self.angles[::ds_interval]
                                )

    def _angle_(self, file_name: Path):
        """ This method converts the file_name to a string then splits it based
        on windows convention (\\), then it uses the delimiter and angle
        position to extract the angle
        """
        if 'filename_delimiter' in self.settings['pre_processing']:
            delimiter = self.settings['pre_processing']['filename_delimiter']
            delimiter = delimiter.replace('"', '')
        else:
            assert False, "No filename_delimiter in config"
        angle_position = int(self.settings['pre_processing']['angle_argument'])
        f_name = str(file_name).split("\\")[-1]
        angle_str = f_name.split(delimiter)[angle_position]
        angle_float = float(angle_str.replace("d", ".").replace("p", ''))
        return np.deg2rad(angle_float)

    def fetch_angles(self) -> None:
        """ This either reads the angles from the file names or returns an
        evenly spaced angular array over the number of files
        """
        files = self.files
        if 'filename_delimiter' in self.settings['pre_processing']:
            self.angles = np.array([self._angle_(f) for f in files],
                                   dtype=np.float32)
        else:
            self.angles = np.linspace(0, 2*np.pi, len(files), endpoint=False)

    # -------------------------------------------------------------------------
    #                       POST PROCESSING
    # -------------------------------------------------------------------------
    def apply_volumetric_median(self, batch_size=250) -> None:
        """ Just a wrapper for the big boy
        med_kernel = z,x,y
        """
        self.update_config()
        try:
            med_kernel = self.settings['post_processing']['median kernel']
        except KeyError as ke:
            print(ke)
            logging.warning("No Volumetric Median Kernel in settings")
            return
        self.reconstruction = median_GPU_batch(self.reconstruction,
                                               med_kernel,
                                               batch_size)

    def serialize(self, arr_name: str, path: Path) -> None:
        """
        Save Array as serialized numpy array

        args:
        -----
            arr_name : str
                string of member name (projections, transmission, etc.)

            path : path_like (string or pathlib.Path)
        """
        f_name = path / f"{self.Name}_{arr_name}.npy"
        logging.info(f"Saving {arr_name} to {f_name}")
        np.save(f_name, getattr(self, arr_name))

    def write_im_stack(self,
                       arr_name: str,
                       directory: Path,
                       ds_interval: int = 1,
                       ) -> None:
        """
        Save Array as Image stack

        Args:
        ----
            arr_name : str
                'transmission' or 'reconstruction'

            directory : pathlib.Path
                path where the image stack will be saved to

            ds_interval : int
                integer by which to downsample the sinograms

        """
        if directory.is_dir():
            logging.warning(f"Deleting {directory}")
            shutil.rmtree(directory)

        os.mkdir(directory)
        logging.info(f"Saving {arr_name} to {directory}")
        arr_handle = getattr(self, arr_name)[:, ::ds_interval, :]
        arr_shape = arr_handle.shape
        assert len(arr_shape) == 3, "This function operates on volumes"
        nx, ny, nz = arr_shape
        tqdm_im_save = tqdm(range(nx), desc=f"saving {arr_name} as images")
        for i in tqdm_im_save:
            im = Image.fromarray(arr_handle[i, :, :])
            f_name = directory / f"{arr_name}_{i:06}.tif"
            im.save(f_name)


def dataset_select_widget(directory=Path(".")):
    """ This funciton is used in the context of a jupyter notebook to provide a
    dropdown menu of *.ini files down one directory...
    """
    config_files = sorted(list(directory.glob("*/*.yml")))
    data_set_select = widgets.Select(options=config_files,
                                     layout=Layout(width='60%', height='200px'),
                                     description='Select Config:'
                                     )
    display(data_set_select)
    return data_set_select


class jupyter_tomo_dataset(tomo_dataset):
    def __init__(self, config=None):
        super().__init__(config)
        assert self.is_jupyter(), "This object only works in a Jupyter environment"

        self._reconstructor_ = astra_tomo_handler(self.settings['recon'])

    def read_dataset(self):
        config = self.cfg
        super().__init__(config)

    # -------------------------------------------------------------------------
    #                    Jupyter VISUALIZATION
    # -------------------------------------------------------------------------
    def SARE_interact(self, figsize: tuple = (12, 5)) -> None:
        """
        Wrapper for SAREPY Interact in visualization

        This visualization tool helps to determine the best settings for the
        SARE filter
        """
        logging.info("Stripe Artifact Interact Started")
        transpose = (1, 0, 2)
        SAREPY_interact(self.settings,
                        np.transpose(self.transmission, transpose),
                        figsize=figsize)

    def median_2D_interact(self,
                           image_index: int = 0,
                           figsize: tuple = (12, 5),
                           kwargs: dict = {}
                           ) -> None:
        """
        Wrapper for median filtering operations

        """
        logging.info("Median 2D Interact Started")
        ext = self.settings['pre_processing']['extension']
        proj_path = self.settings['paths']['projection_path']
        assert proj_path.is_dir(), f"{str(proj_path)} (path) does not exist"
        proj_files = self.fetch_files(proj_path, ext=f"*.{ext}")
        test_image = self.imread(proj_files[image_index])
        median_2D_interact(self.settings,
                           test_image,
                           **kwargs)

    def ORTHO_interact(self, **kwargs) -> None:
        """
        """
        logging.info("Orthogonal Plot Interact Started")
        orthogonal_plot(np.transpose(self.reconstruction,
                                     (2, 1, 0)),  **kwargs)

    def _update_crop_norm_(self,
                           crop_y0: int,
                           crop_dy: int,
                           crop_x0: int,
                           crop_dx: int,
                           norm_y0: int,
                           norm_dy: int,
                           norm_x0: int,
                           norm_dx: int,
                           tpose: str,
                           d_theta: float = 180.0,
                           ) -> None:
        """
        """
        if tpose == 'True':
            self.settings['pre_processing']['transpose'] = True
            if self._tpose_state != True:
                self.fetch_combined_image()
            self._tpose_state = True
        elif tpose == 'False':
            self.settings['pre_processing']['transpose'] = False
            if self._tpose_state != False:
                self.fetch_combined_image()
            self._tpose_state = False

        self.settings['crop'] = {
                                'y0': crop_x0,
                                'y1': crop_x0+crop_dx,
                                'x0': crop_y0,
                                'x1': crop_y0+crop_dy
                                }
        self.settings['norm'] = {
                                'y0': norm_x0,
                                'y1': norm_x0+norm_dx,
                                'x0': norm_y0,
                                'x1': norm_y0+norm_dy
                                }

        self._plot_crop_norm_()

    def _plot_crop_norm_(self) -> None:
        """ This helper function plots the image with the crop and norm window
        in self.ax[0] (for use with cor_calculation)
        """
        self.ax[0].clear()
        self.ax[0].imshow(self.combined_image)
        keys = ['y0', 'y1', 'x0', 'x1']
        if isinstance(self.settings['crop'], dict):
            x0, x1, y0, y1 = [self.settings['crop'][key] for key in keys]
        dy = y1-y0
        dx = x1-x0
        crop_rectangle = Rectangle((x0,y0),dx,dy,
                                   fill=False,
                                   color='r',
                                   linestyle='--',
                                   linewidth=2)
        self.ax[0].add_artist(crop_rectangle)
        # Visualize 0 and 180 degrees with crop patch and normalization
        # patch highlighted
        if isinstance(self.settings['norm'], dict):
            norm_patch = [self.settings['norm'][key] for key in keys]
        if norm_patch[0] != norm_patch[1] and norm_patch[2] != norm_patch[3]:
            x0_, y0_ = norm_patch[0], norm_patch[2]
            dy_ = norm_patch[3]-norm_patch[2]
            dx_ = norm_patch[1]-norm_patch[0]
            norm_rectangle = Rectangle((x0_, y0_), dx_, dy_,
                                       fill=False,
                                       color='w',
                                       linestyle='-',
                                       linewidth=1)
            self.ax[0].add_artist(norm_rectangle)

            self.ax[0].text(norm_patch[0],norm_patch[2],
                            'Norm Patch',
                            verticalalignment='bottom',
                            horizontalalignment='left',
                            color='w',
                            rotation=90)

    def _cor_calculate_wrapper_(self, cor_y0: int, cor_y1: int) -> None:
        """
        """
        self.COR_operations(cor_y0, cor_y1, ax=[[], [], self.ax[1]], ax2=None)

    def COR_interact(self,
                     d_theta=180.0,
                     figsize: tuple = (10, 5),
                     apply_thresh: float = None,
                     med_kernel=3,
                     ) -> None:
        """
        """
        self._tpose_state = False
        self.fetch_combined_image(d_theta = d_theta)
        # This is for calculating vmin and vmax for imshows
        self.fig,self.ax = plt.subplots(1,2, figsize = figsize)
        self.ax[1].axis(False)
        plt.show()

        #---------------------------------------------------------------------------
        # NOTE TO SELF:
        #   These assignements look backward (x0, 0 -> y_max, etc.) but they are
        #   fixing the problem of plt.imshow transposing the image
        #---------------------------------------------------------------------------
        kwargs = {'continuous_update':False,'min':0}
        y_max,x_max = self.combined_image.shape
        max_crop = max([y_max,x_max])
        crop_x0 = IntSlider(description = "crop x$_0$", max=max_crop, **kwargs)
        crop_dx = IntSlider(description = "crop $\Delta$x", max=max_crop, **kwargs)
        crop_y0 = IntSlider(description = "crop y$_0$", max=max_crop, **kwargs)
        crop_dy = IntSlider(description = "crop $\Delta$y", max=max_crop, **kwargs)
        norm_x0 = IntSlider(description = "norm x$_0$", max=max_crop, **kwargs)
        norm_dx = IntSlider(description = "norm $\Delta$x", max=max_crop, **kwargs)
        norm_y0 = IntSlider(description = "norm y$_0$", max=max_crop, **kwargs)
        norm_dy = IntSlider(description = "norm $\Delta$y", max=max_crop, **kwargs)


        cor_y0  = IntSlider(description = "COR y$_0$",  max=max_crop, **kwargs)
        cor_y1  = IntSlider(description = "COR y$_1$",  max=max_crop, **kwargs)


        transpose_init = "False" if 'transpose' not in self.settings['pre_processing'] \
                        else str(self.settings['pre_processing']['transpose'])
        print(f"transpose init = {transpose_init}")
        tpose = RadioButtons(options = ['False','True'],
                            value = transpose_init,
                            description = "Transpose")

        row0 = HBox([tpose])
        row1 = HBox([crop_x0,crop_dx,crop_y0,crop_dy])
        row2 = HBox([norm_x0,norm_dx,norm_y0,norm_dy])
        ui = VBox([row0,row1,row2])

        control_dict = {
                    'crop_y0':crop_y0,
                    'crop_dy':crop_dy,
                    'crop_x0':crop_x0,
                    'crop_dx':crop_dx,
                    'norm_x0':norm_x0,
                    'norm_dx':norm_dx,
                    'norm_y0':norm_y0,
                    'norm_dy':norm_dy,
                    'tpose':tpose
                        }

        crop_norm_partial = partial(self._update_crop_norm_, d_theta = d_theta)
        out = interactive_output(crop_norm_partial, control_dict)
        display(ui,out)

        interact_2 = interactive.factory()
        manual_2 = interact_2.options(manual = True,
                                    manual_name = 'Refresh COR Calc'
                                    )
        out_2 = manual_2(   self._cor_calculate_wrapper_,
                            cor_y0 = cor_y0,
                            cor_y1 = cor_y1,
                            name = "Refresh COR")


class recon_algorithms(Enum):
    FBP_CUDA = "FBP_CUDA"
    FDK_CUDA = "FDK_CUDA"
    CGLS_CUDA = "CGLS_CUDA"
    SART_CUDA = "SART_CUDA"
    # EM_CUDA = "EM_CUDA"                # this just produces zeros
    SIRT3D_CUDA = "SIRT3D_CUDA"
    SIRT_CUDA = "SIRT_CUDA"
    CGLS3D_CUDA = "CGLS3D_CUDA"


class recon_geometry(Enum):
    parallel = "parallel"
    fanflat = "fanflat"
    parallel3d = "parallel3d"
    cone = "cone"


class fbp_fdk_filters(Enum):
    ram_lak = 'ram-lak'
    shepp_logan = 'shepp-logan'
    cosine = 'cosine'
    hamming = 'hamming'
    hann = 'hann'
    none = 'none'
    tukey = 'tukey'
    lanczos = 'lanczos'
    triangular = 'triangular'
    gaussian = 'gaussian'
    barlett_hann = 'barlett-hann'
    blackman = 'blackman'
    nuttall = 'nuttall'
    blackman_harris = 'blackman-harris'
    blackman_nuttall = 'blackman-nuttall'
    flat_top = 'flat-top'
    kaiser = 'kaiser'
    parzen = 'parzen'
    projection = 'projection'
    sinogram = 'sinogram'
    rprojection = 'rprojection'
    rsinogram = 'rsinogram'


class napari_tomo_gui(tomo_dataset):
    """
    This class inherits from tomo_dataset and instantiates a napari instance
    that can modify and reconstruct a dataset
    """
    def __init__(self) -> None:
        log_setup("napari_tomo_gui_logging.log")
        self.sare_bool = False
        self.viewer = napari.Viewer()
        self.viewer.title = "NIST Tomography GUI"
        config_select_widget = {'Config Select': [
                                    self.select_config(),
                                    self.generate_config_widget(),
                                    ]}

        self.config_handles = []
        for key, val in config_select_widget.items():
            self.config_handles.append(
                                       self.viewer.window.add_dock_widget(val,
                                                   name=key,
                                                   add_vertical_stretch=True,
                                                   area='right')
                                      )

    def init_operations(self, config_file) -> None:
        """ Once a config has been selected this widget will be loaded
        This will:
            - fetch combined image
            - add it to viewer
            - load a transmission sample
            - create the widgets

        """
        super().__init__(config_file)
        self.fetch_combined_image()
        self.viewer.add_image(self.combined_image,
                              name='combined image',
                              colormap='Spectral')

        self.load_transmission_sample()
        self.dtype = np.float32

        self.widgets = {
            'Transmission': [
                                self.compose_tilt_image(),
                                self.transpose_widget(),
                                self.select_norm(),
                                self.crop_image(),
                                self.cor_wrapper(),
                                self.median_widget(),
                                self.load_images(),
                                self.show_transmission(),
                                self.reset_transmission(),
                                ],
            'Reconstruction': [
                                self.select_reconstruction_parameters(),
                                self.preview_reconstruction(),
                                self.reconstruct_interact(),
                                self.show_reconstruction(),
                                self.sare_widget(),
                                self.sare_apply(),
                                ],
            'Post Processing': [
                                self.post_median_params_widget(),
                                self.apply_volumetric_median_widget(),
                                self.write_reconstruction_widget(),
                ]
                        }
        for i, (key, val) in enumerate(self.widgets.items()):
            handle = self.viewer.window.add_dock_widget(val,
                                                        name=key,
                                                add_vertical_stretch=True,
                                                        area='right'
                                                        )

            # THIS ADDS THE WIDGETS AS TABS BEHIND THE CONFIG!
            self.viewer.window._qt_window.tabifyDockWidget(
                                                        self.config_handles[0],
                                                        handle)

    def _create_initial_config(self) -> None:
        """ When creating a fresh configuration, this sets up the corresponding
        config file so the settings can be kept track of
        """
        f_name = Path(".") / f"{self.settings['name']}.yml"
        logging.info(f"writing intialized config file to :{str(f_name)}")
        print("--->", self.settings)

        yaml_safe_dict = self.settings.copy()
        for key, val in yaml_safe_dict['paths'].items():
            yaml_safe_dict['paths'][key] = val.as_posix()

        print("--->", yaml_safe_dict)
        with open(f_name, 'w') as file_:
            yaml.safe_dump(yaml_safe_dict, file_)

        self.init_operations(f_name)

    def mute_all(self) -> None:
        """ this suppresses all image layers """
        for elem in self.viewer.layers:
            elem.visible = False

    # -------------------------------------------------------------------------
    #              NAPARI WIDGETS AND FUNCTIONS
    # -------------------------------------------------------------------------
    def select_config(self):
        """ User can select a pre-existing configuration that will
        auto-populate the widget parameters, etc.
        """
        @magicgui(call_button="Load Existing Config",
                  config_file={'label': 'Select Config File (.yml)'},
                  persist=True
                  )
        def inner(config_file=Path.home()):
            self.init_operations(config_file)
        return inner

    def generate_config_widget(self):
        """ if user selects to create a new config -> then this widget is added
        """
        @magicgui(call_button='Generate New Config',
                  main_window=True,               # gives a help option
                  persist=True,   # previous values are automatically reloaded
                  layout='vertical',
                  Name={"label": 'Name of Experiment'},
                  Dark_dir={"label": 'Select Dark Image Directory', 'mode': 'd'},
                  Flat_dir={"label": 'Select Flat Image Directory', 'mode': 'd'},
                  Proj_dir={"label": 'Select Projections Directory', 'mode': 'd'},
                  Extension={"value": "tif"},
                  Delimiter={'label': 'Delimiter for File Naming', "value": "_"},
                  Angle_argument={'label': 'Angle Position in File Name',
                                      "value":1},
                  )
        def inner(
                  Name: str = '',
                  Dark_dir=Path.home(),
                  Flat_dir=Path.home(),
                  Proj_dir=Path.home(),
                  Extension="*.tif",
                  Delimiter="_",
                  Angle_argument: int = 1,
                  ):
            """
            This widget helps to generate a config file that the gui can read.

            Parameters
            ----------
            name: str
                The name of the experiment. This is also the name that is used
                for the configuration file (<name>.ini)
            Dark Image Directory: Path
                Select the directory of the dark images
            Flat Image Directory: Path
                Select the directory of the flat images
            Proj Files: Path
                Select the directory of the projection images
            Extension: str
                Image file extension (e.g., tif or fit)
            Delimiter For File Naming: str
                This delimiter is for splitting the file name so the angular
                position can be read
            Angle Position in File Name: int
                Zero-based indexing of which element in the split file name has
                the angle information
            """
            self.settings = {'name': {},
                             'paths': {},
                             'pre_processing': {}
                             }
            self.settings['name'] = Name
            self.settings['paths']['dark_path'] = Dark_dir
            self.settings['paths']['flat_path'] = Flat_dir
            self.settings['paths']['projection_path'] = Proj_dir
            self.settings['pre_processing']['extension'] = Extension
            self.settings['pre_processing']['filename_delimiter'] = Delimiter
            self.settings['pre_processing']['angle_argument'] = Angle_argument
            self.settings['pre_processing']['dtype'] = 'float32'
            self.settings['pre_processing']['transpose'] = False

            self._create_initial_config()

        return inner

    def compose_tilt_image(self):
        @magicgui(call_button='Compose Combined Image',
                  d_theta={'label': 'd theta',
                           'value': 180.0
                           }
                  )
        def inner(d_theta: float):
            self.fetch_combined_image(d_theta=d_theta)
            if 'combined image' in self.viewer.layers:
                self.viewer.layers.remove('combined image')
            self.viewer.add_image(self.combined_image,
                                  name='combined image',
                                  colormap='Spectral')
        return inner

    def transpose_widget(self):
        """ This gives the option to toggle the transpose
        """
        tp = self.search_settings("transpose", default=False)

        @magicgui(call_button='Apply Transpose')
        def inner(Transpose: bool = tp):
            if Transpose:
                self.settings['pre_processing']['transpose'] = Transpose
                self.combined_image = self.combined_image.T
                if 'combined image' in self.viewer.layers:
                    self.viewer.layers.remove('combined image')
                    self.viewer.add_image(self.combined_image,
                                          name='combined image',
                                          colormap='Spectral',
                                          )
                self.load_transmission_sample()
        return inner

    def select_norm(self):
        """
        If the configuration already has norm parameters:
            add the shape for the norm parameters

        """
        if 'norm' in self.settings:
            keys = ['x0', 'x1', 'y0', 'y1']
            x0, x1, y0, y1 = [self.settings['norm'][key] for key in keys]
            verts = np.array([[x0, y0],
                              [x0, y1],
                              [x1, y1],
                              [x1, y0]], dtype=np.uint32)
            self.viewer.add_shapes(verts, name='Norm', face_color='r')

        @magicgui(call_button="Select norm")
        def inner():
            verts = np.round(self.viewer.layers[-1].data[0][:, -2:]
                             ).astype(np.uint32)
            x0, x1 = np.min(verts[:, 0]), np.max(verts[:, 0])
            y0, y1 = np.min(verts[:, 1]), np.max(verts[:, 1])
            self.settings['norm'] = {'x0': int(x0), 'x1': int(x1),
                                     'y0': int(y0), 'y1': int(y1)}
            self.viewer.layers[-1].name = 'Norm'
            self.viewer.layers['Norm'].face_color = 'r'
        return inner

    def crop_image(self):
        """ This returns the widget that selects the crop portion of the image
        Note: it also mutes the full image and
        """
        if 'crop' in self.settings:
            keys = ['x0', 'x1', 'y0', 'y1']
            x0, x1, y0, y1 = [self.settings['crop'][key] for key in keys]
            verts = np.array([[x0, y0],
                              [x0, y1],
                              [x1, y1],
                              [x1, y0]], dtype=np.uint32)
            self.viewer.add_shapes(verts,
                                   name='Crop',
                                   face_color='b',
                                   opacity=0.2)
            slice_y = slice(y0, y1)
            slice_x = slice(x0, x1)
            crop_image = self.combined_image[slice_x, slice_y]
            self.mute_all()
            self.viewer.add_image(crop_image,
                                  name='cropped image',
                                  colormap='twilight_shifted')

        @magicgui(call_button="Crop Image")
        def inner():
            crop_key = 'Crop'
            if crop_key not in self.viewer.layers:
                self.viewer.layers[-1].name = crop_key
            verts = np.round(self.viewer.layers[crop_key].data[0][:, -2:]
                             ).astype(np.uint32)
            x0, x1 = np.min(verts[:, 0]), np.max(verts[:, 0])
            y0, y1 = np.min(verts[:, 1]), np.max(verts[:, 1])
            self.settings['crop'] = {'x0': x0, 'x1': x1, 'y0': y0, 'y1': y1}
            slice_y = slice(y0, y1)
            slice_x = slice(x0, x1)
            crop_image = self.viewer.layers['combined image'].data[slice_x, slice_y]
            mute_layers = ['combined image', crop_key, 'Norm']
            for key in mute_layers:
                if key not in self.viewer.layers:
                    continue
                self.viewer.layers[key].visible = False
            self.viewer.add_image(crop_image,
                                  colormap='twilight_shifted',
                                  name='cropped image')
            return crop_image
        return inner

    def cor_wrapper(self):
        """

        This is a bit of a modification, but it produces much better results.
        the algorithm:
            1. calculates the off-axis angle for the raw cropped image
            2. it then rotates the cropped image and calculates its off-axis
               angle
            3. then it rotates the cropped image by half the angle calculated
               in step 1 and calculates the off-axis angle
            4. based on those 2 rotational data points it extrapolates to the
               rotation that will make the second angle equal to zero (linear
               fit)

        """
        if 'COR' in self.settings:
            y0 = self.settings['COR']['y0']
            y1 = self.settings['COR']['y1']
            if 'crop' in self.settings:
                crop_h = self.settings['crop']
                x_coord = (crop_h['y1']-crop_h['y0'])/2
            else:
                x_coord = 0
            verts = np.array([[y0, x_coord],
                              [y1, x_coord]])
            self.viewer.add_points(verts, name='COR Points', size=50)

        @magicgui(call_button="Calculate Center of Rotation")
        def inner():
            points_key = "COR Points"
            if points_key not in self.viewer.layers:
                self.viewer.layers[-1].name = points_key
            points = np.round(self.viewer.layers[points_key].data[:, -2:]
                              ).astype(np.uint32)
            verts = np.round(self.viewer.layers['Crop'].data[0][:, -2:]
                             ).astype(np.uint32).copy()
            x0, x1 = np.min(verts[:, 0]), np.max(verts[:, 0])
            y0, y1 = np.min(verts[:, 1]), np.max(verts[:, 1])
            cropped_image = self.viewer.layers['cropped image'].data
            cor_y0, cor_y1 = sorted([points[0, 0], points[1, 0]])
            logging.info(f'y0 = {cor_y0}; y1 = {cor_y1}')

            fig2, ax2 = plt.subplots(1, 1)
            fig, ax = plt.subplots(1, 3, sharex=True, sharey=True)
            self.COR_operations(cor_y0=cor_y0,
                                cor_y1=cor_y1,
                                ax=ax,
                                ax2=ax2)
            fig.tight_layout()
            fig2.tight_layout()
            plt.show()
            # =================================================================
            y0, y1 = [self.settings['crop'][key] for key in ['y0', 'y1']]
            verts[:, 1] = [y0, y1, y1, y0]
            try:
                self.viewer.layers.remove('crop corrected')
            except:
                pass
            self.viewer.add_shapes(verts, name='crop corrected',
                                   face_color='b', visible=False, opacity=0.3)
        return inner

    def median_widget(self):
        median_init = self.search_settings("median_xy", default=1)
        kernels_init = self.search_settings("thresh median kernels",
                                            default='')
        z_scores_init = self.search_settings("thresh median z-scores",
                                             default='')
        print('--->', z_scores_init)
        replace_elements = ["[", "]", " "]
        if kernels_init != '' and z_scores_init != '':
            kernels_init = str(kernels_init)
            z_scores_init = str(z_scores_init)
            for elem in replace_elements:
                kernels_init = kernels_init.replace(elem, "")
                z_scores_init = z_scores_init.replace(elem, "")

        @magicgui(call_button="Preview Median",
                  image_index={'step': 1, 'value': 0, 'max': 1e9},
                  median_size={'step': 2, 'value': median_init}
                  )
        def inner(
                image_index: int = 0,
                median_size: int = 1,
                kernels: str = kernels_init,
                z_scores: str = z_scores_init):
            if image_index != self.tm_global_index:
                self.load_transmission_sample(image_index)
                self.tm_global_index = image_index
            transmission_image = self.transmission_sample
            handle = self.settings['pre_processing']
            handle['median_xy'] = median_size
            med_kernel = (median_size, median_size)
            if kernels != "":
                kernels = [int(elem) for elem in kernels.split(",")]
                z_scores = [float(elem) for elem in z_scores.split(",")]
            else:
                kernels = []
                z_scores = []
            handle['thresh median kernels'] = kernels
            handle['thresh median z-scores'] = z_scores
            med_stack_shape = len(kernels)+2
            nx, ny = transmission_image.shape
            med_image = [transmission_image.copy()]
            if median_size > 1:
                med_image.append(median_gpu(
                                cp.array(med_image[-1], dtype=cp.float32),
                                med_kernel
                                ).get())
            print('kernels = ', kernels, '; z_scores= ', z_scores)
            for kern, z_score in zip(kernels, z_scores):
                print(f'applying {kern} with {z_score}')
                temp = thresh_median_2D_GPU(
                                    cp.array(med_image[-1],dtype=cp.float32),
                                            kern,
                                            z_score).get()
                med_image.append(temp)
            med_image = np.stack(med_image)
            med_layer_name = 'median stack'
            if med_layer_name in self.viewer.layers:
                self.viewer.layers.remove(med_layer_name)

            self.mute_all()
            self.viewer.add_image(med_image,
                                  name=med_layer_name,
                                  colormap='turbo')
        return inner

    def load_images(self):
        @magicgui(call_button="Load Projections to Transmission")
        def inner(sub_sample_transmission: int = 1):
            if not self.sare_bool:
                @thread_worker(connect={'returned': lambda: None})
                def thread_load_images():
                    self.load_projections(truncate_dataset=sub_sample_transmission)
                thread_load_images()
            else:
                logging.warning(("Stripe Filter Has Been Applied, Reset \
                            Transmission to Load more Transmission Files"))
        return inner

    def show_transmission(self):
        @magicgui(call_button="Show Transmission",
                  down_sampling={'value': 1})
        def inner(down_sampling: int):
            try:
                self.viewer.layers.remove('Transmission')
            except:
                pass
            self.mute_all()
            ds = down_sampling
            self.viewer.add_image(
                        dask_array(self.transmission[::ds, ::ds, ::ds].copy()),
                        name='Transmission',
                        colormap='cividis')
        return inner

    def reset_transmission(self):
        @magicgui(call_button="Reset Transmission")
        def inner():
            if 'Transmission' in self.viewer.layers:
                self.viewer.layers.remove('Transmission')
            del self.transmission
            self.files = []
            self.sare_bool = False
        return inner

    def select_reconstruction_parameters(self) -> None:
        """ This widget has all the recon and recon.cfg_options, which have the
        geometry, algorithm, constraints, gpuindex, etc.
        """
        pixel_pitch_init = self.search_settings('pixel pitch', 0.0)
        repro_ratio_init = self.search_settings('reproduction ratio', 1.0)
        algorithm_init = self.search_settings('algorithm', recon_algorithms.FBP_CUDA)
        iterations_init = self.search_settings('iterations', 0)
        odd_init = self.search_settings('origin to detector distance', 0.0)
        sod_init = self.search_settings('source to origin distance', 0.0)
        if isinstance(algorithm_init, str):
            algorithm_init = getattr(recon_algorithms, algorithm_init)

        geometry_init = self.search_settings('geometry', recon_geometry.parallel)
        if isinstance(geometry_init, str):
            geometry_init = getattr(recon_geometry, geometry_init)


        @magicgui(call_button='Select Recon Parameters',
                  Source_detector_distance={
                        'label': 'Source to Detector Distance (mm) (optional)',
                        "value": sod_init+odd_init,
                        'max': 1e9},
                  Origin_detector_distance={
                        'label': 'Origin to Detector Distance (mm) (optional)',
                        "value": odd_init,
                        'max': 1e9},
                  Pixel_pitch={
                            'label': 'Pixel Pitch (mm)',
                            'value': pixel_pitch_init,
                            'step': 0.0001},
                  Reproduction_ratio={'value': repro_ratio_init},

                  Iterations={'value': iterations_init,
                              'min': 0,
                              'max': 1e9,
                              'label': 'Iterations (optional)'
                              },
                  fbp_fdk_seed={'value': True,
                                'label': 'Use FBP/FDK as seed (optional)'
                                },
                  seed_directory={'value': Path.home(),
                                  'mode': 'd',
                                  'label': 'Directory of seed Dataset (optional)'
                                  },
                  fbp_filters={'value': fbp_fdk_filters.ram_lak,
                               'label': 'FBP/FDK Filter (optional)'
                               },
                  Min_constraint={'value': -np.inf,
                                  'label': 'Min Constraint (optional)'
                                  }
                  )
        def inner(
                  Algorithm=algorithm_init,
                  Geometry=geometry_init,
                  Pixel_pitch: float = 1.0,
                  Reproduction_ratio: float = 1.0,
                  Source_detector_distance: float = 0.0,
                  Origin_detector_distance: float = 0.0,
                  Iterations: int = iterations_init,
                  fbp_fdk_seed: bool = True,
                  seed_directory=Path.home(),
                  fbp_filters=fbp_fdk_filters.ram_lak,
                  Min_constraint: float = 0.0,
                  GPUindex: int = 0
                  ):
            """
            This GUI helps to generate a config file that the Napari gui can
            read. The file selectors require a single file to be selected in
            the target directory, and the program finds the parent directory to
            give to the config.

            Parameters
            ----------
            Algorithm: str
                Reconstruction Algorithm
            Geometry: str
                Reconstruction geometry (parallel or cone)
            Pixel_pitch: float
                Camera pixel pitch in mm
            Source_detector_distance: float
                Distance from the source to the detector  in mm
            Origin_detector_distance: float
                Distance from the origin (center of sample) to the detector in
                mm
            Reproduction_ratio: float
                Reproduction ratio...
            Iterations: int (optional)
                If an iterative method is selected, this specifies iterations

            """
            self.settings['recon'] = {}
            self.settings['recon']['pixel pitch'] = Pixel_pitch
            assert Pixel_pitch > 0, "Pixel Pitch must be > 0"
            self.settings['recon']['source to origin distance'] = Source_detector_distance - Origin_detector_distance
            self.settings['recon']['origin to detector distance'] = Origin_detector_distance
            self.settings['recon']['reproduction ratio'] = Reproduction_ratio
            self.settings['recon']['algorithm'] = Algorithm.value
            self.settings['recon']['geometry'] = Geometry.value

            # Optional Arguments
            if Iterations != 0:
                self.settings['recon']['iterations'] = Iterations

            if seed_directory != Path.home():
                self.settings['paths']['seed_path'] = seed_directory
                self.settings['recon']['seed_path'] = seed_directory

            if fbp_fdk_seed and Algorithm.value in ['SIRT_CUDA', 'SIRT3D_CUDA']:
                self.settings['recon']['fbp_fdk_seed'] = fbp_fdk_seed

            # These are options for ASTRA to read
            options = {'GPUindex': GPUindex}
            if Algorithm.value in ['FBP_CUDA', 'FDK_CUDA']:
                options.update({'FilterType': fbp_filters.value})

            if np.isfinite(Min_constraint):
                options.update({'MinConstraint': Min_constraint})

            if options != {}:
                self.settings['recon']['options'] = options

            self.update_config()
            self.check_reconstruction_config()

            self._reconstructor_ = astra_tomo_handler(self.settings['recon'])

        return inner

    def preview_reconstruction(self):
        @magicgui(call_button="Preview 2D Reconstruction",
                  sinogram_index={'label': 'Sinogram Index (row)',
                                  'value': 1,
                                  'min': 0,
                                  'max': 1e9}
                  )
        def inner(sinogram_index: int):
            sinogram = self.transmission[sinogram_index]
            attn = -np.log(sinogram, where=sinogram > 0)
            reconstruction = self._reconstructor_.astra_reconstruct_2D(attn,
                                                                       self.angles)
            self.mute_all()
            name = 'recon preview'
            if name in self.viewer.layers:
                self.viewer.layers.remove(name)
            self.viewer.add_image(reconstruction,
                                  name=name,
                                  colormap='viridis')
        return inner

    def reconstruct_interact(self):
        @magicgui(call_button="Reconstruct",
                  radial_zero={'label': "Cylindrical mask",
                               'value': True}
                  )
        def inner(radial_zero: bool = True):
            @thread_worker(connect={'returned': lambda: None})
            def thread_recon():
                self.reconstruct()
                if radial_zero:
                    self.recon_radial_zero()
            thread_recon()
        return inner

    def show_reconstruction(self):
        @magicgui(call_button="Show Reconstruction",
                  down_sampling={'value': 1})
        def inner(down_sampling: int = 1):
            try:
                self.viewer.layers.remove('Reconstruction')
            except:
                pass

            if hasattr(self, 'reconstruction'):
                self.mute_all()
                ds = down_sampling
                self.viewer.add_image(
                    dask_array(self.reconstruction[::ds, ::ds, ::ds].copy()),
                    name='Reconstruction',
                    colormap='plasma')
            else:
                logging.info("No Reconstruction Exists")
        return inner

    def write_reconstruction_widget(self):
        @magicgui(call_button="Write Reconstruction",
                  Output_dir={
                              'label': 'Output Directory for Reconstruction',
                              'mode': 'd'                  # select a directory
                              }
                  )
        def inner(Output_dir=Path.home(),
                  prefix='reconstruction',
                  extension='tif'):
            nz = self.reconstruction.shape[0]
            for i in tqdm(range(nz)):
                f_name = Output_dir / f'{prefix}_{i:0d}.{extension}'
                Image.fromarray(self.reconstruction[i]).save(f_name)
        return inner

    def sare_widget(self):
        snr_init = self.search_settings('snr', 1.0)
        la_size_init = self.search_settings('la_size', 1)
        sm_size_init = self.search_settings('sa_size', 1)

        @magicgui(call_button='Preview Ring Filter',
                  row={'label': 'Sinogram Index (row)',
                       'value': 0,
                       'min': 0,
                       'max': 10_000},
                  snr={'value': snr_init},
                  la_size={'value': la_size_init, 'step': 2, 'min': 1},
                  sm_size={'value': sm_size_init, 'step': 2, 'min': 1},
                  )
        def inner(row: int,
                  snr: float,
                  la_size: int,
                  sm_size: int,
                  ):
            sinogram_local = self.transmission[row, :, :].copy()

            filtered = remove_all_stripe_GPU(
                    cp.array(sinogram_local[:, None, :], dtype=cp.float32),
                    snr=snr,
                    la_size=la_size,
                    sm_size=sm_size)[:, 0, :].get()

            unfiltered_recon = self._reconstructor_.astra_reconstruct_2D(
                    -np.log(sinogram_local, where=sinogram_local > 0),
                    self.angles)

            filtered_recon = self._reconstructor_.astra_reconstruct_2D(
                    -np.log(filtered, where=filtered > 0),
                    self.angles)

            print('non finites (sinogram_local) = ',
                  np.sum(~np.isfinite(sinogram_local)))
            print('non finites (sinogram filtered) = ',
                  np.sum(~np.isfinite(filtered)))
            print('non finites (unfiltered) = ',
                  np.sum(~np.isfinite(unfiltered_recon)))
            print('non finites (filtered) = ',
                  np.sum(~np.isfinite(filtered_recon)))

            self.mute_all()
            sinogram_layer_name = "sare sinogram stack"
            recon_layer_name = "sare reconstruction stack"
            if sinogram_layer_name in self.viewer.layers:
                self.viewer.layers.remove(sinogram_layer_name)
            if recon_layer_name in self.viewer.layers:
                self.viewer.layers.remove(recon_layer_name)

            sinograms = np.stack([sinogram_local, filtered])
            recons = np.stack([unfiltered_recon, filtered_recon])

            self.viewer.add_image(sinograms,
                                  name=sinogram_layer_name,
                                  colormap='viridis',
                                  visible=False)

            self.viewer.add_image(recons,
                                  name=recon_layer_name,
                                  colormap='viridis')

            keys = ['snr', 'la_size', 'sm_size']
            vals = [snr, la_size, sm_size]
            self.settings['SARE'] = {key: val for key, val in zip(keys, vals)}
        return inner

    def sare_apply(self):
        @magicgui(call_button='Apply Ring Filter (In-Place)')
        def inner(batch_size: int = 10):
            self.update_config()

            @thread_worker(connect={'returned': lambda: None})
            def sarepy_threaded():
                self.remove_all_stripe(batch_size=batch_size)

            sarepy_threaded()
            self.sare_bool = True
        return inner

    def post_median_params_widget(self):
        """ This is for selecting the volumetric median kernel that will be
        applied to the reconstruction volume
        """
        @magicgui(call_button='Preview Volumetric Median',
                  row_apply={'label': 'Row Apply',
                             'value': 0,
                             'step': 1,
                             'max': 10_000,
                             'min': 0},
                  med_z={'label': 'Median Z',
                         'value': 1,
                         'step': 2},
                  med_x={'label': 'Median X',
                         'value': 1,
                         'step': 2},
                  med_y={'label': 'Median Y',
                         'value': 1,
                         'step': 2}

                  )
        def inner(row_apply: int,
                  med_z: int,
                  med_x: int,
                  med_y: int):
            z_offset = med_z // 2
            slice_z = slice(row_apply-z_offset, row_apply+z_offset+1)
            med_kernel = (med_z, med_x, med_y)
            pre_med = self.reconstruction[row_apply, :, :]
            med_temp = median_gpu(
                cp.array(self.reconstruction[slice_z, :, :],
                         dtype=cp.float32), med_kernel).get()

            med_stack = np.stack([pre_med, med_temp[0]])
            self.mute_all()

            if 'post_processing' not in self.settings:
                self.settings['post_processing'] = {}

            self.settings['post_processing']['median kernel'] = med_kernel

            layer_name = 'Volumetric Median'
            if layer_name in self.viewer.layers:
                self.viewer.layers.remove(layer_name)
            self.viewer.add_image(med_stack,
                                  name=layer_name,
                                  colormap='gist_earth'
                                  )
        return inner

    def apply_volumetric_median_widget(self):
        @magicgui(call_button='Apply Volumetric Median')
        def inner(batch_size: int = 250):
            @thread_worker(connect={'returned': lambda: None})
            def thread_apply_vol_median():
                self.apply_volumetric_median(batch_size=batch_size)
            thread_apply_vol_median()
        return inner


if __name__ == "__main__":
    inst = napari_tomo_gui()
    napari.run()
