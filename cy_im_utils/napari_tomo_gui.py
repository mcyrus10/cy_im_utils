""" 
this is a behemoth that can work in jupyter, napari and in a scripted
context. 
"""
from sys import path
path.append("C:\\Users\\mcd4\\Documents\\cy_im_utils")
from cy_im_utils.prep import radial_zero,field_gpu,imstack_read,center_of_rotation,imread_fit
from cy_im_utils.recon_utils import ASTRA_General,astra_2d_simple
from cy_im_utils.sarepy_cuda import *
from cy_im_utils.visualization import COR_interact,SAREPY_interact,orthogonal_plot,median_2D_interact
from cy_im_utils.thresholded_median import thresh_median_2D_GPU

from PIL import Image
from cupyx.scipy.ndimage import rotate as rotate_gpu, median_filter as median_gpu
from magicgui import magicgui
from magicgui.tqdm import tqdm
from pathlib import Path
from enum import Enum
#from tqdm import tqdm
import astra
import configparser
import cupy as cp
import logging
import matplotlib.pyplot as plt
import napari
import numpy as np
import os
import pathlib
import shutil


class tomo_config_handler:
    """
    This is intended to be able to read/write the config file so that you don't
    have to copy/paste any text but just use the interactive functions to
    update the configuration

    This class expects that the name of the dataset, the paths to the
    projections dark and flat are declared and the pre-processing options are
    filled in
    """
    def __init__(self, config_file: str):
        self.config_file = config_file
        self.config = configparser.ConfigParser()
        self.config.read(config_file)
        self.write_data_dict()
    
    def write_data_dict(self) -> None:
        """
        Unpacking config into a single dictionary
        """
        self.data_dict = {}
        for key,val in self.config.items():
            self.data_dict[key] = {}
            for sub_key,sub_val in val.items():
                # Numeric Parsing
                if sub_val.replace(".","1").lstrip("-").isnumeric():
                    sub_val = self.float_int(sub_val)
                # Boolean Parsing
                elif sub_val in ['True','False']:
                    sub_val = self.config.getboolean(key,sub_key)
                # Potentially empty fields
                elif sub_val == 'None':
                    sub_val = None
                # Path Parsing
                elif 'paths' in key:
                    sub_val = pathlib.Path(sub_val)
                # unpack list
                elif '[' in sub_val and ']' in sub_val:
                    temp_list = [e for e in \
                            sub_val.replace("[","").replace("]","").split(",")]
                    for i,elem in enumerate(temp_list):
                        remove_chars =  [" ","'"]
                        elem = "".join([e for e in elem if e not in remove_chars])
                        temp_str = elem.replace(".","1").lstrip("-")
                        if temp_str.isnumeric():
                            temp_list[i] = self.float_int(elem)
                        else:
                            temp_list = []
                    sub_val = temp_list

                self.data_dict[key][sub_key] = sub_val

    def float_int(self, value: str):
        """
        Hack for returning float or integer numbers
        """
        try:
            out = int(value)
        except ValueError as ve:
            out = float(value)
        return out

    def write(self) -> None:
        """
        Wrapper for re-writing the config file after it's updated
        """
        print("WRITING UPDATED CONFIG",self.config_file)
        with open(self.config_file,'w') as config_file:
            self.config.write(config_file)

    def conditional_add_field(self, field: str) -> None:
        """
        Wrapper for adding fields if they do not exist, if they do then this
        will do nothing

        Args:
        -----
            field: str
                name of field to test / add
        """
        if field not in self.config.sections():
            self.config.add_section(field)

    def log_field(self,field) -> None:
        """
        wrapper for 
        """
        logging.info(f"Updating {field} parameters ")
        for key,val in self.config[field].items():
            logging.info(f"\t {key} : {val}")

    def update_COR(self) -> None:
        """
        This method updates the config after the center of rotation has been
        applied so it modifies COR, crop, norm and theta of the config

        """
        # COR and theta
        field = "COR"
        self.conditional_add_field(field)
        self.config[field]['y0'] = str(self.data_dict['COR']['y0'])
        self.config[field]['y1'] = str(self.data_dict['COR']['y1'])
        self.config[field]['theta'] = str(self.data_dict['COR']['theta'])

        self.log_field("COR")

        # Crop and Norm
        for field in ['crop','norm']:
            self.conditional_add_field(field)
            for sub_key in ['x0','x1','y0','y1']:
                sub_val = str(self.data_dict[field][sub_key])
                self.config[field][sub_key] = sub_val

        self.log_field("crop")
        self.log_field("norm")

        self.write()

    def update_SARE(self) -> None:
        """
        This method updates the SARE parameters
        """
        self.conditional_add_field("SARE")
        self.config["SARE"]['snr'] = str(self.data_dict['SARE']['snr'])
        self.config["SARE"]['la_size'] = str(self.data_dict['SARE']['la_size'])
        self.config["SARE"]['sm_size'] = str(self.data_dict['SARE']['sm_size'])

        self.log_field("SARE")
        self.write()

    def update_median(self) -> None:
        """
        This method updates the SARE parameters
        """
        update_keys = [ 
                        'median_xy',
                        'num thresh filters',
                        'thresh median kernels',
                        'thresh median z-scores'
                        ]
        for key in update_keys:
            if key in self.data_dict['pre_processing']:
                self.config["pre_processing"][key] = str(self.data_dict['pre_processing'][key])

        self.log_field("pre_processing")
        self.write()

class napari_tomo_gui:
    """
    This is the big boy: it holds the projections and can facilitate the recon
    operations
    """
    def __init__(   self) -> None:
        self.starting_init()
        self.widgets = [
                        self.select_config(),
                        self.transpose_widget(),
                        self.select_norm(),
                        self.crop_image(),
                        self.cor_wrapper(),
                        self.median_widget(),
                        self.load_images(),
                        self.show_transmission(),
                        self.reconstruct_interact(),
                        self.show_reconstruction(),
                        self.sare_widget(),
                        self.reset()
                        ]
        self.viewer = napari.Viewer()
        self.viewer.window.add_dock_widget(self.widgets, name = 'Tomo Prep')

    #---------------------------------------------------------------------------
    #                       UTILS
    #---------------------------------------------------------------------------
    def starting_init(self) -> None:
        """ getting to square 0 """
        logging.info("Starting new")
        self.transpose = False
        self.files = []
        self.settings = {}
        try:
            del self.reconstruction
        except:
            pass
        try:
            del self.transmission
        except:
            pass
        try:
            del self.combined_image
        except:
            pass

    def load_transmission_sample(self, image_index: int = 0):
        """ This is for loading an image for the median to operate on """
        proj_path = self.settings['paths']['projection_path']
        ext = self.settings['pre_processing']['extension']
        proj_files = self.fetch_files(proj_path, ext = ext)
        if 'tif' in ext:
            imread_fcn = lambda x: np.array(Image.open(x), dtype = np.float32)
        elif 'fit' in ext:
            imread_fcn = imread_fit

        self.transmission_sample = imread_fcn(proj_files[image_index])

    def fetch_files(self, path_, ext: str = 'tif'):
        """ Jesus Christ i've written this line a million times
        """
        return sorted(list(path_.glob(f"*.{ext}")))

    def fetch_combined_image(self, mode = 'auto', median = 3) -> None:
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
        proj_files = self.fetch_files(proj_path, ext = ext)
        print(proj_path.is_dir(),len(proj_files))
        if 'tif' in ext:
            imread_fcn = lambda x: np.array(Image.open(x), dtype = np.float32)
        elif 'fit' in ext:
            imread_fcn = imread_fit

        nx,ny = self.flat.shape
        combined = cp.zeros([2,nx,ny], dtype = cp.float32)
        f_0 = proj_files[0]
        angles = []
        if mode == 'auto':
            """ String matching to find 180 deg image"""
            str_180 = 'p0180d00000'
            for f in proj_files:
                angles.append(self._angle_(f))
                if str_180 in str(f):
                    f_180 = f
                    break
            else:
                print("no image found at 180 Degrees, finding closest")
                abs_diff = np.abs(np.pi-np.array(angles, dtype = np.float32))
                file_idx = np.where(abs_diff == np.min(abs_diff))[0][0].astype(np.uint32)
                print("file index = ",file_idx)
                print("closest file = ",str(proj_files[file_idx]))
                f_180 = proj_files[file_idx]

        ff = cp.array(self.flat, dtype = cp.float32)
        df = cp.array(self.dark, dtype = cp.float32)

        for i,f in enumerate([f_0,f_180]):
            im_temp = cp.asarray(imread_fcn(f), dtype = cp.float32)
            transmission = (im_temp-df)/(ff-df)
            attenuation = -cp.log(transmission)
            attenuation[~cp.isfinite(attenuation)] = 0
            attenuation = median_gpu(attenuation,(median,median))
            combined[i] = attenuation
        combined = cp.sum(combined, axis = 0).get()

        self.combined_image = combined

        self.viewer.add_image(  combined,
                                name = 'combined image',
                                colormap = 'Spectral')
        
    def free(self, field : str) -> None:
        """
        This can be used to free the projections if they are taking up too much
        memory

        args:
        -----
            field : str
                which field to free

        examples:
            tomo_recon.free_field("projections")
            tomo_recon.free_field("transmission")
        """
        setattr(self,field,None)

    def gpu_curry_loop( self,
                        function,
                        ax_length : int,
                        batch_size : int,
                        tqdm_string : str = ""
                        ) -> None:
        """
        this is a generic method for currying functions to the GPU

        args:
        -----
            function : python function
                the function only takes arguments for the indices x0 and x1 and
                performs all the operations internally
            ax_len : int 
                length of the axis along which the operations are being
                executed (to determine remainder)
        """
        for j in tqdm(range(ax_length//batch_size), desc = tqdm_string):
            function(j*batch_size,(j+1)*batch_size)
        remainder = ax_length % batch_size
        if remainder > 0:
            logging.info(f"remainder = {remainder}")
            function(ax_length-remainder,ax_length)
        
    def recon_radial_zero(self, radius_offset: int = 0) -> None:
        """
        this function makes all the values outside the radius eq

        args:
        -----
            radius_offset: int (optional)
                how much more than the radius should be zeroed out

        """
        n_frame,detector_width,_ = self.reconstruction.shape
        tqdm_zeroing = tqdm(range(n_frame), desc = "zeroing outisde crop radius")
        for f in tqdm_zeroing:
            radial_zero(self.reconstruction[f], radius_offset = radius_offset)

    def return_imread_fcn(self, extension: str):
        """
        Ultra obnoxiousness -> datasets with different imread for flat, dark
        and projections
        """
        if 'fit' in  extension:
            return imread_fit
        elif 'tif' in extension or 'tiff' in extension:
            return lambda x: np.array(Image.open(x), dtype = np.float32)
        else:
            assert False, "unknown image extension for imread"

    #---------------------------------------------------------------------------
    #                       PRE PROCESSING
    #---------------------------------------------------------------------------
    def load_field(self, mode : str) -> None:
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
            imread_fcn = self.return_imread_fcn(ext_projections)
            proj_image = imread_fcn(proj_files[0])
            field = np.zeros_like(proj_image, dtype = np.float32)

        # Field does exist
        else:

            keys = list(self.settings['pre_processing'].keys())

            # find the right image reading function
            if f'extension_{mode}' in keys:
                ext = self.settings['pre_processing'][f'extension_{mode}']
            else:
                ext = self.settings['pre_processing']['extension']
            imread_fcn = self.return_imread_fcn(ext)


            logging.info(f"Reading {mode} field from {field_path}")
            files = list(field_path.glob(f"*{ext}*"))
            logging.info(f"\tnum files = {len(files)}")
            logging.info("\tshape files = "
                              f"{np.array(imread_fcn(files[0])).shape}")
            nx,ny = np.asarray(imread_fcn(files[0])).shape
            field = field_gpu(files, 3)
        field = self.median_operations_wrapper(cp.array(field, dtype = np.float32)).get()
        setattr(self,mode,field)

    def median_operations_wrapper(self, image: cp.array) -> None:
        """ Generic wrapper for 2D Median Operations
        """
        prep_handle = self.settings['pre_processing']
        if 'median_xy' in prep_handle.keys():
            kernel = prep_handle['median_xy']
            image = median_gpu(image, (kernel,kernel))
        if 'num thresh filters' in prep_handle.keys():
            kernels = prep_handle['thresh median kernels']
            z_scores = prep_handle['thresh median z-scores']
            for kern,z_sc in zip(kernels,z_scores):
                image = thresh_median_2D_GPU(image, kern, z_sc)
        return image

    def load_projections(self,
                        truncate_dataset : int = 1,
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
                i) -log(image)
                j) remove non-finite
                k) assign to self.transmission array

        Args:
            truncate_dataset : int
                If you want to load in a datset faster this factor downsamples
                the dataset by (every other {truncate_dataset} value
        returns:
            None (operates in-place)

        """
        self.config.update_median()
        self.config.update_COR()

        proj_path = self.settings['paths']['projection_path']
        logging.info(f"Reading Images From {proj_path}")
        ext = self.settings['pre_processing']['extension']
        imread_fcn = self.return_imread_fcn(ext)
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
        nx,ny = np.asarray(imread_fcn(new_files[0])).shape

        crop_patch = self.crop_patch
        norm_patch = self.norm_patch
        self.crop_x = slice(crop_patch[0],crop_patch[1])
        self.crop_y = slice(crop_patch[2],crop_patch[3])
        self.norm_x = slice(norm_patch[0],norm_patch[1])
        self.norm_y = slice(norm_patch[2],norm_patch[3])
        theta = self.settings['COR']['theta']

        attn_ny = crop_patch[1]-crop_patch[0]
        attn_nx = crop_patch[3]-crop_patch[2]
        n_proj = len(new_files)

        # Re-shaping this so that n_proj is axis 1 and attn_nx is axis 0?
        temp = np.empty([attn_nx,n_proj,attn_ny], dtype = self.dtype)

        dark_local = cp.array(self.dark)
        flat_local = cp.array(self.flat)
        logging.info(f"New Files Shape = {temp.shape}")
        logging.info(f"crop_y = {self.crop_y}")
        logging.info(f"crop_x = {self.crop_x}")

        load_im = lambda f :  cp.asarray(imread_fcn(f), dtype = self.dtype)

        if self.transpose:
            load_im = lambda f :  cp.asarray(imread_fcn(f),
                                                        dtype = self.dtype).T
            dark_local = dark_local.T
            flat_local = flat_local.T


        flat_ = flat_local-dark_local
        flat_scale = cp.sum(flat_[self.norm_y,self.norm_x])
        logging.info(f"flat patch magnitude = {flat_scale}")
        tqdm_imread = tqdm(range(n_proj), desc = "Projection -> Transmission Ops")
        """
        In ReconstructCT Source code see line 183 from
        NIF_TomographyReconstruction/NIF_Translate_projection_Tomography.m 
        for these operations

        # note -> sep 9, 2022 moved median to before dark field subtraction
        """
        for i in tqdm_imread:
            im = load_im(new_files[i])
            #im = median_gpu(im,(self.median_spatial, self.median_spatial))
            im = self.median_operations_wrapper(im)
            #im -= self.dark
            im -= dark_local
            scale = cp.sum(im[self.norm_y,self.norm_x])
            im /= flat_
            im *= flat_scale/scale
            #self.norm_mags[i] = cp.mean(im[self.norm_y,self.norm_x]).get()
            im = im[self.crop_y,self.crop_x]
            # Rotate will produce all zeros if it has non-finites
            im[~cp.isfinite(im)] = 0
            im = rotate_gpu(im, -theta, reshape = False)
            # DO THE LOG TRANFORM AFTER THE VO FILTER!!!!!!!
            temp[:,i,:] = cp.asnumpy(im)

        if hasattr(self,'transmission'):
            self.transmission = np.hstack([self.transmission,temp])
        else:
            self.transmission = temp
        self.fetch_angles()

    #--------------------------------------------------------------------------
    #                    VISUALIZATION
    #--------------------------------------------------------------------------
    def COR_interact(self,
                    d_theta : int = 60,
                    angles : list = [],
                    apply_thresh: float = None,
                    med_kernel = 3,
                    ) -> None:
        """
        Wrapper for COR_interact in visualization

        This visualization tool helps you to find the Center of rotation of an
        image stack
        """
        if self.is_jupyter():
            logging.info("COR Interact Started")
            ext = self.settings['pre_processing']['extension']
            imread_fcn = self.return_imread_fcn(ext)
            assert 360 % d_theta== 0, "all angles must be factors of 360"
            if not angles:
                angles = [j*d_theta for j in range(360//d_theta)]
            flat_local = median_gpu(cp.array(self.flat, dtype = cp.float32), (med_kernel,med_kernel))
            dark_local = median_gpu(cp.array(self.dark, dtype = cp.float32), (med_kernel, med_kernel))
            COR_interact(   self.settings,
                            imread_fcn,
                            flat_local,
                            dark_local,
                            angles,
                            apply_thresh = apply_thresh,
                            med_kernel = med_kernel
                            )
        else:
            logging.warning("Interact needs to be executed in a Notebook" 
                            "Environment - This method is not being executed")

    def SARE_interact(self, figsize : tuple = (12,5)) -> None:
        """
        Wrapper for SAREPY Interact in visualization

        This visualization tool helps to determine the best settings for the
        SARE filter
        """
        if self.is_jupyter():
            logging.info("Stripe Artifact Interact Started")
            transpose = (1,0,2)
            SAREPY_interact(self.settings,
                            np.transpose(self.transmission,transpose),
                            figsize = figsize)
        else:
            logging.warning("Interact needs to be executed in a Notebook"
                            "Environment - This method is not being executed")

    def median_2D_interact( self,
                            image_index: int = 0,
                            figsize : tuple = (12,5),
                            kwargs: dict = {}
                            ) -> None:
        """
        Wrapper for median filtering operations

        """
        if self.is_jupyter():
            logging.info("Median 2D Interact Started")
            ext = self.settings['pre_processing']['extension']
            imread_fcn = self.return_imread_fcn(ext)
            proj_path = self.settings['paths']['projection_path']
            assert proj_path.is_dir(), f"{str(proj_path)} (path) does not exist"
            proj_files = sorted(list(proj_path.glob(f"*.{ext}")))
            test_image = imread_fcn(proj_files[image_index])
            median_2D_interact( self.settings,
                                test_image,
                                **kwargs)
        else:
            logging.warning("Interact needs to be executed in a Notebook"
                            "Environment - This method is not being executed")


    def ORTHO_interact(self, **kwargs) -> None:
        """
        Wrapper for COR_interact in visualization

        This visualization tool helps you to find the Center of rotation of an
        image stack
        """
        if self.is_jupyter():
            logging.info("Orthogonal Plot Interact Started")
            orthogonal_plot(np.transpose(self.reconstruction,(2,1,0)), **kwargs)
        else:
            logging.warning("Interact needs to be executed in a Notebook" 
                            "Environment - This method is not being executed")

    #---------------------------------------------------------------------------
    #                       PROCESSING
    #---------------------------------------------------------------------------
    def transmission_GPU(self, batch_size : int = 20) -> None:
        """
        this method executes gpu_ops on all the batches to create
        self.transmission

        args:
        -----
            batch_size : int
                size of mini batches for GPU
        """
        logging.warning("THIS METHOD IS DEPRECATED, USE 'load_projections_to_attn'")
        # in case these have been updated by COR Interact
        crop_patch = self.settings['crop']
        norm_patch = self.settings['norm']
        self.crop_x = slice(crop_patch[0],crop_patch[1])
        self.crop_y = slice(crop_patch[2],crop_patch[3])
        self.norm_x = slice(norm_patch[0],norm_patch[1])
        self.norm_y = slice(norm_patch[2],norm_patch[3])

        attn_nx = crop_patch[1]-crop_patch[0]
        attn_ny = crop_patch[3]-crop_patch[2]
        n_proj = self.projections.shape[0]
        self.transmission = np.empty([n_proj,attn_ny,attn_nx], dtype = self.dtype)
        logging.info(f"transmission shape = {self.transmission.shape}")
        self.gpu_curry_loop(self.gpu_ops, n_proj, batch_size)

    def remove_all_stripe_ops(  self, id0 : int, id1 : int) -> None:
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
        transpose = (1,0,2)
        slice_ = slice(id0,id1)
        logging.debug(f"indices for slice = {id0},{id1}")
        vol_gpu = cp.asarray(self.transmission[slice_,:,:], dtype = self.dtype)
        vol_gpu = cp.transpose(vol_gpu, transpose)
        vol_gpu = remove_all_stripe_GPU(vol_gpu,
                                        self.settings['SARE']['snr'],
                                        self.settings['SARE']['la_size'],
                                        self.settings['SARE']['sm_size'])
        vol_gpu = cp.transpose(vol_gpu, transpose)
        self.transmission[slice_,:,:] = cp.asnumpy(vol_gpu)

    def remove_all_stripe(self, batch_size: int = 10) -> None:
        """
        operates in-place

        """
        logging.info("REMOVING STRIPE ARTIFACTS")
        n_sino,_,_ = self.transmission.shape
        self.gpu_curry_loop(self.remove_all_stripe_ops,
                            n_sino,
                            batch_size,
                            tqdm_string = "Stripe Artifact Removal")

    def attenuation(self, ds_factor: int) -> np.array:
        """ wrapper to compute attenuation array as float32
        """
        return -np.log(
                        self.transmission[:,::ds_factor,:],
                        where = self.transmission[:,::ds_factor,:] > 0
                        ).astype(np.float32)
        

    def reconstruct(self, 
                    ds_interval: int = 1,
                    iterations: int = 1,
                    seed = 0
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
        self.reconstruction = ASTRA_General(
                                            self.attenuation(ds_interval),
                                            self.settings['recon'],
                                            iterations = iterations,
                                            angles = self.angles,
                                            seed = seed
                                            )

    def _angle_(self, file_name: Path):
        """ This method converts the file_name to a string then splits it based
        on windows convention (\\), then it uses the delimiter and angle
        position to extract the angle
        """
        if 'filename_delimiter' in self.settings['pre_processing']:
            delimiter = self.settings['pre_processing']['filename_delimiter']
            delimiter = delimiter.replace('"','')
        else:
            assert False, "No filename_delimiter in config"
        angle_position = int(self.settings['pre_processing']['angle_argument'])
        f_name = str(file_name).split("\\")[-1]
        angle_str = f_name.split(delimiter)[angle_position]
        angle_float = float(angle_str.replace("d",".").replace("p",''))
        return np.deg2rad(angle_float)

    def fetch_angles(self) -> None:
        """ This either reads the angles from the file names or returns an
        evenly spaced angular array over the number of files
        """
        files = self.files
        if 'filename_delimiter' in self.settings['pre_processing']:
            self.angles = np.array([self._angle_(f) for f in files],
                                                        dtype = np.float32)
        else:
            self.angles = np.linspace(0,2*np.pi,len(files), endpoint = False)

    #--------------------------------------------------------------------------
    #                       POST PROCESSING
    #--------------------------------------------------------------------------
    def serialize(self, arr_name : str, path : pathlib.Path ) -> None:
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

    def write_im_stack( self,
                        arr_name : str,
                        directory : pathlib.Path,
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
        arr_handle = getattr(self,arr_name)[:,::ds_interval,:]
        arr_shape = arr_handle.shape
        assert len(arr_shape) == 3, "This function operates on volumes"
        nx,ny,nz = arr_shape
        tqdm_im_save = tqdm(range(nx), desc = f"saving {arr_name} as images")
        for i in tqdm_im_save:
            im = Image.fromarray(arr_handle[i,:,:])
            f_name = directory / f"{arr_name}_{i:06}.tif"
            im.save(f_name)

    #--------------------------------------------------------------------------
    #              Napari Widgets and Functions
    #--------------------------------------------------------------------------
    def mute_all(self):
        """ this suppresses all image layers """
        for elem in self.viewer.layers:
            elem.visible = False

    def select_config(self):
        @magicgui(call_button = "Select Config",
                config_file = {'label':'Select Config File (.ini)'})
        def inner(config_file = Path.home()):
            logging.info("-"*80)
            logging.info("-- TOMOGRAPHY RECONSTRUCTION --")
            logging.info("-"*80)
            self.previous_settings = {}
            self.config = tomo_config_handler(config_file)
            self.settings = self.config.data_dict
            self.files = []
            logging.info(f"{'='*20} SETTINGS: {'='*20}")
            for key,val in self.settings['pre_processing'].items():
                setattr(self,key,val)
            for key,val in self.settings.items():
                logging.info(f"{key}:")
                for sub_key,sub_val in val.items():
                    logging.info(f"\t{sub_key} : {sub_val}")
            logging.info(f"{'='*20} End SETTINGS {'='*20}")

            self.load_field('dark')
            self.load_field('flat')
            self.fetch_combined_image()
            self.load_transmission_sample()

        return inner

    def transpose_widget(self):
        @magicgui(call_button = 'Apply Transpose')
        def inner(Transpose: bool = False):
            if Transpose:
                self.transpose = Transpose
                self.combined_image = self.combined_image.T
                if 'combined image' in self.viewer.layers:
                    self.viewer.layers.remove('combined image')
                    self.viewer.add_image(self.combined_image,
                            name = 'combined image',
                            colormap = 'Spectral',
                            )
        return inner


    def select_norm(self):
        @magicgui(call_button = "Select norm")
        def inner():
            verts = np.round(self.viewer.layers[-1].data[0][:,-2:]
                                                    ).astype(np.uint32)
            x0,x1 = np.min(verts[:,0]), np.max(verts[:,0])
            y0,y1 = np.min(verts[:,1]), np.max(verts[:,1])
            self.norm_patch = [y0,y1,x0,x1]
            if self.transpose:
                keys = ['x0','x1','y0','y1']
            else:
                keys = ['y0','y1','x0','x1']
            self.settings['norm'] = {key:val for key,val in zip(keys,self.norm_patch)}
            self.viewer.layers[-1].name = 'Norm'
            self.viewer.layers['Norm'].face_color = 'r'
        return inner

    def crop_image(self):
        """ This returns the widget that selects the crop portion of the image
        Note: it also mutes the full image and 
        """
        @magicgui(call_button = "Crop Image")
        def inner():
            crop_key = 'Crop'
            if crop_key not in self.viewer.layers:
                self.viewer.layers[-1].name = crop_key
            verts = np.round(self.viewer.layers[crop_key].data[0][:,-2:]).astype(np.uint32)
            y0,y1 = np.min(verts[:,1]), np.max(verts[:,1])
            x0,x1 = np.min(verts[:,0]), np.max(verts[:,0])
            slice_y = slice(y0,y1)
            slice_x = slice(x0,x1)
            crop_image = self.viewer.layers['combined image'].data[slice_x,slice_y]
            mute_layers = ['combined image',crop_key,'Norm']
            for key in mute_layers:
                if key not in self.viewer.layers:
                    continue
                self.viewer.layers[key].visible = False
            self.viewer.add_image(crop_image, colormap = 'twilight_shifted', name = 'cropped image')
            return crop_image
        return inner

    def cor_wrapper(self):
        @magicgui(call_button = "Calculate Center of Rotation")
        def inner():
            points_key = "COR Points"
            if points_key not in self.viewer.layers:
                self.viewer.layers[-1].name = points_key
            points = np.round(self.viewer.layers[points_key].data[:,-2:]).astype(np.uint32)
            verts = np.round(self.viewer.layers['Crop'].data[0][:,-2:]
                                                ).astype(np.uint32).copy()
            x0,x1 = np.min(verts[:,0]), np.max(verts[:,0])
            y0,y1 = np.min(verts[:,1]), np.max(verts[:,1])
            cropped_image = self.viewer.layers['cropped image'].data
            cor_y0,cor_y1 = sorted([points[0,0],points[1,0]])
            print('y0 =',cor_y0,'; y1 =',cor_y1)
            fig,ax = plt.subplots(1,3, sharex = True, sharey = True)
            cor = center_of_rotation(cropped_image,cor_y0,cor_y1, ax = ax[0])
            theta = np.tan(cor[0])*(180/np.pi)
            ax[0].set_title(f"theta = {theta}")
            rot = rotate_gpu(cp.array(cropped_image, dtype = cp.float32),
                            -theta,
                            reshape = False).get()
            cor2 = center_of_rotation(rot, cor_y0, cor_y1, ax = ax[1])
            crop_nx = y1-y0
            dx = int(np.round(cor2[1]))-crop_nx//2
            ax[1].set_title("Re-Fit")
            y0 += dx
            y1 += dx
            slice_x = slice(x0,x1)
            slice_y = slice(y0,y1)
            crop2 = self.viewer.layers['combined image'].data[slice_x,slice_y]
            crop2rot = rotate_gpu(cp.array(crop2, dtype = cp.float32),
                                    -theta,
                                    reshape = False).get()
            cor3 = center_of_rotation(crop2rot,cor_y0,cor_y1, ax = ax[2])
            ax[2].set_title(f"corrected center dx = {dx}")
            fig.tight_layout()
            plt.show()
            verts[:,1] = [y0,y1,y1,y0]
            try:
                self.viewer.layers.remove('crop corrected')
            except:
                pass
            self.viewer.add_shapes(verts, name = 'crop corrected', face_color = 'b', visible = False, opacity = 0.3)
            self.crop_patch = [y0,y1,x0,x1]
            if self.transpose:
                keys = ['x0','x1','y0','y1']
            else:
                keys = ['y0','y1','x0','x1']
            self.settings['crop'] = {key:val for key,val in zip(keys,self.crop_patch)}
            self.settings['COR'] = {key:val for key,val in zip(['y0','y1','theta'],[cor_y0,cor_y1,theta])}
        return inner

    def median_widget(self):
        @magicgui(call_button = "Preview Median",
                median_size = {'step':2,'value':1})
        def inner(median_size: int = 1, kernels: str = '', z_scores: str = ''):
            transmission_image = self.transmission_sample
            handle = self.settings['pre_processing']
            handle['median_xy'] = median_size
            med_kernel = (median_size,median_size)
            if kernels != "":
                kernels = [int(elem) for elem in kernels.split(",")]
                z_scores = [float(elem) for elem in z_scores.split(",")]
            else:
                kernels = []
                z_scores = []
            handle['thresh median kernels'] = kernels
            handle['thresh median z-scores'] = z_scores
            med_stack_shape = len(kernels)+2
            nx,ny = transmission_image.shape
            med_image = [transmission_image.copy()]
            if median_size > 1:
                med_image.append(median_gpu(cp.array(med_image[-1], dtype = cp.float32), 
                                                                med_kernel).get()
                                                                )
            print('kernels = ',kernels,'; z_scores= ',z_scores)
            for kern,z_score in zip(kernels,z_scores):
                temp = thresh_median_2D_GPU(
                            cp.array(med_image[-1], dtype = cp.float32),
                                                        kern,
                                                        z_score).get()
                med_image.append(temp)
            med_image = np.stack(med_image)
            med_layer_name = 'median stack'
            if med_layer_name in self.viewer.layers:
                self.viewer.layers.remove(med_layer_name)

            if self.transpose:
                if med_image.ndim == 2:
                    med_image = med_image.T
                elif med_image.ndim == 3:
                    med_image = np.transpose(med_image,(0,2,1))

            self.mute_all()
            self.viewer.add_image(med_image,
                                    name = med_layer_name,
                                    colormap = 'turbo')
        return inner

    def load_images(self):
        @magicgui(call_button = "Load Projections to Transmission")
        def inner():
            self.load_projections()
        return inner

    def show_transmission(self):
        @magicgui(call_button = "Show Transmission",
                down_sampling = {'value': 1})
        def inner(down_sampling:int):
            try:
                self.viewer.layers.remove('Transmission')
            except:
                pass
            self.mute_all()
            ds = down_sampling
            self.viewer.add_image(  self.transmission[::ds,::ds,::ds].copy(),
                                    name = 'Transmission',
                                    colormap = 'cividis')
        return inner

    def reconstruct_interact(self):
        @magicgui(call_button = "Reconstruct")
        def inner():
            self.reconstruct()
        return inner

    def show_reconstruction(self):
        @magicgui(call_button = "Show Reconstruction",
                    down_sampling = {'value': 1})
        def inner(down_sampling: int):
            try:
                self.viewer.layers.remove('Reconstruction')
            except:
                pass

            if hasattr(self,'reconstruction'):
                self.mute_all()
                ds = down_sampling
                self.viewer.add_image(  self.reconstruction[::ds,::ds,::ds].copy(),
                                        name = 'Reconstruction',
                                        colormap = 'plasma')
            else:
                print("Not Reconstructed Yet")
        return inner
    
    def reset(self):
        @magicgui(call_button = 'Reset')
        def inner():
            self.__init__()
        return inner

    def sare_widget(self):
        @magicgui(call_button = 'Preview Ring Filter',
                row={'value':0},
                snr={'value':1.0},
                la_size={'value':1,'step':2},
                sm_size={'value':1,'step':2},
                )
        def inner(row: int,
                snr: float,
                la_size: int,
                sm_size: int,
                ):
            sinogram_local = self.transmission[row,:,:].copy()
            
            filtered = remove_all_stripe_GPU(
                    cp.array(sinogram_local[:,None,:], dtype = cp.float32),
                    snr = snr,
                    la_size = la_size,
                    sm_size = sm_size)[:,0,:].get()

            sino_args = {
                    'algorithm':'FBP_CUDA',
                    'pixel_size':self.settings['recon']['camera pixel size'],
                    'angles': self.angles,
                    'geometry': 'parallel'
                    }

            unfiltered_recon = astra_2d_simple(
                    -np.log(sinogram_local, where = sinogram_local > 0),
                    **sino_args)

            filtered_recon = astra_2d_simple(
                    -np.log(filtered, where = sinogram_local > 0),
                    **sino_args)

            self.mute_all()
            sinogram_layer_name = "sare sinogram stack"
            recon_layer_name = "sare reconstruction stack"
            if sinogram_layer_name in self.viewer.layers:
                self.viewer.layers.remove(sinogram_layer_name)
            if recon_layer_name in self.viewer.layers:
                self.viewer.layers.remove(recon_layer_name)

            sinograms = np.stack([sinogram_local,filtered])
            recons = np.stack([unfiltered_recon,filtered_recon])

            self.viewer.add_image(  sinograms,
                                    name = sinogram_layer_name,
                                    colormap = 'viridis')

            self.viewer.add_image(  recons,
                                    name = recon_layer_name,
                                    colormap = 'viridis',
                                    visible = False)
            keys = ['snr','la_size','sm_size']
            vals  = [snr,la_size,sm_size]
            self.settings['SARE'] = {key:val for key,val in zip(keys,vals)}
        return inner

class recon_algorithms(Enum):
    FBP_CUDA = "FBP_CUDA"
    FDK_CUDA = "FDK_CUDA"
    SIRT_CUDA = "SIRT_CUDA"

class recon_geometry(Enum):
    parallel = "parallel"
    cone = "cone"

class tomo_config_gen_gui:
    def __init__(self):
        self.generate_config_widget().show(run = True)

    def generate_config_widget(self):
        @magicgui(call_button = 'Generate Config',
                layout = 'vertical',
                dark_files = {"label":'Select File in Dark Directory'},
                flat_files = {"label":'Select File in Flat Directory'},
                ext = {"value":"tif"},
                source_detector = {"value":'0.0 (mm)'},
                origin_detector = {"value":'0.0 (mm)'},
                )
        def inner(dark_files = Path.home(),
                flat_files = Path.home(),
                ext = "*.tif",
                source_detector: str = '0.0',
                origin_detector: str = '0.0',
                algorithm = recon_algorithms.FDK_CUDA,
                geometry = recon_geometry.parallel
                ):
            pass
        return inner

def test_gui():
    inst = napari_tomo_gui()
    napari.run()

def test_config_gen():
    #generate_config_widget.show(run = True)
    tomo_config_gen_gui()

if __name__ == "__main__":
    inst = napari_tomo_gui()
    napari.run()

