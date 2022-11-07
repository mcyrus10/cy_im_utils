
from .prep import radial_zero,field_gpu,imstack_read,center_of_rotation
from .recon_utils import ASTRA_General
from .sarepy_cuda import *
from .visualization import COR_interact,SAREPY_interact,orthogonal_plot,median_2D_interact
from .thresholded_median import thresh_median_2D_GPU


from PIL import Image
from cupyx.scipy.ndimage import rotate as rotate_gpu, median_filter as median_gpu
from functools import partial
from magicgui import magicgui
from numba import njit,cuda
from pathlib import Path
from scipy.ndimage import rotate
from tqdm import tqdm
import astra
import configparser
import cupy as cp
import logging
import matplotlib.pyplot as plt
import napari
import numpy as np
import os
import pathlib
import pickle
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
            for sub_key in ['x0','x1','y0','y1','dx','dy']:
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

class tomo_dataset:
    """
    This is the big boy: it holds the projections and can facilitate the recon
    operations
    """
    def __init__(   self, 
                    #data_dict: dict, 
                    config_file: str, 
                    ) -> None:
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

    #---------------------------------------------------------------------------
    #                       UTILS
    #---------------------------------------------------------------------------
    def update_patch(self) -> None:
        """
        """
        pass


    def is_jupyter(self) -> bool:
        """
        Yanked this from "Parsing Args in Jupyter Notebooks" on YouTube
        This tells you whether you're executing from a notebook or not
        """
        jn = True
        try:
            get_ipython()
        except NameError as err:
            jn = False
        return jn

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
        Ultra obnoxiousness
        """
        if 'fit' in  extension:
            return imread_fit
        elif 'tif' in extension or 'tiff' in extension:
            return lambda x: np.asarray(Image.open(x))
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
                c) scale of image-dark (sum over norm patch)
                d) divide by flat_ (in-place)
                e) multiply by flat_scale/scale (mean of norm patch --> ~1)
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

        #crop_patch = self.settings['crop patch']
        #norm_patch = self.settings['norm patch']
        # new way 
        crop_patch = [self.settings['crop'][k] for k in ['x0','x1','y0','y1']]
        norm_patch = [self.settings['norm'][k] for k in ['x0','x1','y0','y1']]
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

    def gpu_ops(self,x0 : int,x1 : int) -> None:
        """
        In-Place Operations to populate self.transmission; 
        This function is intended to be an argument to self.gpu_curry_loop

        this performs:
            1) slice to GPU
            2) crop
            3) Rotate
            4) Lambert Beer

        args:
        ----
            x0 : int
                index 1 for the slice

            x1 : int
                index 2 for the slice

        """
        logging.warning("THIS METHOD IS DEPRECATED, USE 'load_projections_to_attn'")
        logging.warning("MAKE SURE ORDER OF OPERATIONS IS CORRECT -> "
                "CROP BEFORE ROTATING!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
                "!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        """
        _,height,width = self.transmission.shape
        slice_ = slice(x0,x1)
        batch_size = x1-x0
        projection_gpu = cp.asarray(self.projections[slice_], dtype = self.dtype)
        patch = cp.mean(projection_gpu[:,self.norm_y,self.norm_x], axis = (1,2), dtype = self.dtype)
        projection_gpu -= cp.asarray(self.dark[None,:,:])
        projection_gpu /= cp.asarray(self.flat[None,:,:]-self.dark[None,:,:])
        projection_gpu /= patch.reshape(batch_size,1,1)
        projection_gpu = median_gpu(projection_gpu,(1,self.median_spatial,self.median_spatial))
        projection_gpu = -cp.log(projection_gpu)
        projection_gpu[~cp.isfinite(projection_gpu)] = 0
        projection_gpu = rotate_gpu(projection_gpu, -self.theta, axes = (1,2), reshape = False)
        self.transmission[slice_,:,:] = cp.asnumpy(projection_gpu[:,self.crop_y,self.crop_x])

        """

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
        crop_patch = self.settings['crop patch']
        norm_patch = self.settings['norm patch']
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
        delimiter = self.settings['pre_processing']['filename_delimiter']
        delimiter = delimiter.replace('"','')
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

