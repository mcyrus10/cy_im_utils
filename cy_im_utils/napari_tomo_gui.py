""" 

This file has the tomo_dataset class that defines how to read/write and process
frames. 
The napari_gui inherits from tomo_dataset and just wraps all the method calls
with button presses, and likewise the jupyter version can operate inside a
notebook


"""
from sys import path
path.append("C:\\Users\\mcd4\\Documents\\cy_im_utils")
from cy_im_utils.prep import radial_zero,field_gpu,imstack_read,center_of_rotation,imread_fit
from cy_im_utils.recon_utils import ASTRA_General,astra_2d_simple,astra_tomo_handler
from cy_im_utils.sarepy_cuda import *
from cy_im_utils.thresholded_median import thresh_median_2D_GPU
from cy_im_utils.visualization import COR_interact,SAREPY_interact,orthogonal_plot,median_2D_interact

from PIL import Image
from cupyx.scipy.ndimage import rotate as rotate_gpu, median_filter as median_gpu
from dask.array import array as dask_array
from enum import Enum
from magicgui import magicgui
from magicgui.tqdm import tqdm
from napari.qt.threading import thread_worker
from pathlib import Path
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

from ipywidgets import widgets,Layout


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


        # Transpose (might have changed)
        self.config['pre_processing']['transpose'] =\
                        str(self.data_dict['pre_processing']['transpose'])

        self.log_field("pre_processing")

        # Crop and Norm
        for field in ['crop','norm']:
            self.conditional_add_field(field)
            for sub_key in ['x0','x1','y0','y1']:
                sub_val = str(self.data_dict[field][sub_key])
                self.config[field][sub_key] = sub_val

        self.log_field("crop")
        self.log_field("norm")

        self.write()

    def update_recon_params(self) -> None:
        """
        This method updates the Reconstruction parameters in the config
        """
        # Reconstruction Parameters
        field = "recon"
        self.conditional_add_field(field)
        keys = [
                'camera pixel size',
                'source to origin distance',
                'origin to detector distance',
                'reproduction ratio',
                'recon algorithm',
                'recon geometry',
                'iterations',
                'ng',
                'alpha',
                'seed_path'
                ]
        for key in keys:
            if key not in self.data_dict[field]:
                continue
            self.config[field][key] = str(self.data_dict[field][key])

        self.log_field("recon")

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
                        'thresh median kernels',
                        'thresh median z-scores'
                        ]
        for key in update_keys:
            if key in self.data_dict['pre_processing']:
                self.config["pre_processing"][key] = str(self.data_dict['pre_processing'][key])

        self.log_field("pre_processing")
        self.write()

class tomo_dataset:
    def __init__(self, config_file):
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
        self.sare_bool = False

    #---------------------------------------------------------------------------
    #                       UTILS
    #---------------------------------------------------------------------------
    def load_transmission_sample(self, image_index: int = 0):
        """ This is for loading an image for the median to operate on """
        proj_path = self.settings['paths']['projection_path']
        ext = self.settings['pre_processing']['extension']
        proj_files = self.fetch_files(proj_path, ext = ext)
        if 'tif' in ext:
            imread_fcn = lambda x: np.array(Image.open(x), dtype = np.float32)
        elif 'fit' in ext:
            imread_fcn = imread_fit

        sample = imread_fcn(proj_files[image_index])
        if self.settings['pre_processing']['transpose']:
            sample = imread_fcn(proj_files[image_index]).T
        self.transmission_sample = sample
        self.tm_global_index = image_index

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

        tp = self.search_settings('transpose',False)

        combined = combined.T if tp else combined
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
        this function makes all the values outside the radius 0

        args:
        -----
            radius_offset: int (optional)
                how much more than the radius should be zeroed out

        """
        n_frame,detector_width,_ = self.reconstruction.shape
        tqdm_zeroing = tqdm(range(n_frame),
                                        desc = "zeroing outisde crop radius")
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

    def check_reconstruction_config(self) -> None:
        """ This is to make sure that geometries are compatible so the user does
        not pick something that will make astra cry
        """
        alg_3d = ['FDK_CUDA','SIRT3D_CUDA']
        alg_2d = ['FBP_CUDA','SIRT_CUDA','CGLS_CUDA','EM_CUDA','SART_CUDA']
        fourier_methods = ['FDK_CUDA',"FBP_CUDA"]
        sirt_methods = ['SIRT3D_CUDA','SIRT_CUDA']
        iter_methods = ['SIRT_CUDA','SIRT3D_CUDA','SART_CUDA','CGLS_CUDA']
        geom_3d = ['parallel3d','cone']
        geom_2d = ['parallel','fanflat']
        non_par_geom = ['cone','fanflat']
        alg = self.settings['recon']['recon algorithm'] 
        geom = self.settings['recon']['recon geometry']
        if alg not in alg_3d+alg_2d:
            assert False,f"{alg} is uknown to check_reconstruction_config"
        if alg in alg_3d:
            assert geom in geom_3d,f'{alg} (3d) incompatilbe with {geom} geometry (2d)'
        elif alg in alg_2d:
            assert geom in geom_2d,f'{alg} (2d) incompatilbe with {geom} geometry (3d)'

        if 'FilterType' in self.settings:
            if self.settings['FilterType'] != 'none' and \
                    alg not in fourier_methods:
                logging.warning("Skipping FilterType for non Fourier Method")

        if 'MinConstraint' in self.settings and alg not in sirt_methods:
            logging.warning("Skipping MinConstraint for non SIRT Method")

        if alg in iter_methods and 'iterations' not in self.settings['recon']:
            logging.warning("0 iterations specified for iterative method")

        if geom in non_par_geom:
            handle = self.settings['recon']
            assert handle['source to origin distance'] > 0.0,\
                        "source - detector distance must be greater than 0"
            assert handle['origin to detector distance'] > 0.0, \
                        "origin -  detector distance must be greater than 0"


    #--------------------------------------------------------------------------
    #                       PRE PROCESSING
    #--------------------------------------------------------------------------
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
        field = self.median_operations_wrapper(
                                cp.array(field, dtype = np.float32)).get()
        setattr(self,mode,field)

    def median_operations_wrapper(self, image: cp.array) -> None:
        """ Generic wrapper for 2D Median Operations
        """
        prep_handle = self.settings['pre_processing']
        if 'median_xy' in prep_handle.keys():
            kernel = prep_handle['median_xy']
            image = median_gpu(image, (kernel,kernel))
        if 'thresh median kernels' in prep_handle.keys():
            kernels = prep_handle['thresh median kernels']
            z_scores = prep_handle['thresh median z-scores']
            assert len(kernels) == len(z_scores),"Dissimilar kernels and zs"
            for kern,z_sc in zip(kernels,z_scores):
                image = thresh_median_2D_GPU(image, kern, z_sc)
        return image

    def rotated_crop(   self,
                        image: cp.array,
                        theta: float,
                        crop: list
                        ) -> cp.array:
        """ This pads the array so that the rotation does not introduce zeroes,
        maybe a bit clunky, but whatever
        """
        x0,x1,y0,y1 = crop
        theta_rad = np.deg2rad(theta)
        trig_product = np.abs(np.sin(theta_rad)*np.cos(theta_rad))
        pad_x = np.ceil(trig_product*(y1-y0)).astype(np.uint32)//2
        pad_y = np.ceil(trig_product*(x1-x0)).astype(np.uint32)//2
        x_0,x_1,y_0,y_1 = np.ceil([x0-pad_x,x1+pad_x,y0-pad_y,y1+pad_y]
                                                        ).astype(np.uint32)
        slice_2 = (slice(y_0,y_1),slice(x_0,x_1))
        image_2 = image[slice_2]
        im2_rot = rotate_gpu(image_2,
                         theta,
                         reshape = False)
        slice_3 = (slice(pad_y,pad_y+(y1-y0)),slice(pad_x,pad_x+(x1-x0)))
        return im2_rot[slice_3]

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
        #keys = ['x0','x1','y0','y1']
        #crop_patch = [self.settings['crop'][key] for key in keys]
        #norm_patch = [self.settings['norm'][key] for key in keys]
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
            im = self.median_operations_wrapper(im)
            im -= dark_local
            scale = cp.sum(im[self.norm_y,self.norm_x])
            im /= flat_
            im *= flat_scale/scale
            im = self.rotated_crop(im, -theta, crop_patch)
            temp[:,i,:] = cp.asnumpy(im)

        if hasattr(self,'transmission'):
            self.transmission = np.hstack([self.transmission,temp])
        else:
            self.transmission = temp
        self.fetch_angles()

    #---------------------------------------------------------------------------
    #                       PROCESSING
    #---------------------------------------------------------------------------
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
        #self.reconstruction = ASTRA_General(
        #                                    self.attenuation(ds_interval),
        #                                    self.settings['recon'],
        #                                    iterations = iterations,
        #                                    angles = self.angles,
        #                                    seed = seed
        #                                    )
        logging.info("--- reconstructing ---")
        print("--- RECONSTRUCTING ---")
        self.reconstruction = self._reconstructor_.reconstruct_volume(
                self.attenuation(ds_interval),
                self.angles)

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

def dataset_select_widget(directory = Path(".")):
    config_files = list(sorted(directory.glob("*/*.ini")))
    data_set_select = widgets.Select(options = config_files,
                            layout = Layout(width = '60%',height = '200px'),
                                    description = 'Select Config:'
                                    )
    display(data_set_select)
    return data_set_select

class jupyter_tomo_dataset(tomo_dataset):
    def __init__(self, config = None):
        assert self.is_jupyter(), "This object only works in a Jupyter environment"
        super().__init__(config)
        self._reconstructor_ = astra_tomo_handler(self.settings['recon'])

    def correct_crop_norm(self) -> None:
        """One of these days i'm going to get all this nonsense straigtened out

        The way that cor_interactive works it modifies x0 and y0 so what looks
        like the 'x' axis on the plot is actually 'y' so this is correcting
        that...

        """
        keys = ['y0','y1','x0','x1']
        self.crop_patch = [self.settings['crop'][k] for k in keys]
        self.norm_patch = [self.settings['norm'][k] for k in keys]

    def read_dataset(self):
        config = self.cfg
        super().__init__(config)

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
    #--------------------------------------------------------------------------
    #                    Jupyter VISUALIZATION
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
            flat_local = median_gpu(cp.array(self.flat, dtype = cp.float32),
                                                    (med_kernel,med_kernel))
            dark_local = median_gpu(cp.array(self.dark, dtype = cp.float32),
                                                    (med_kernel, med_kernel))
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

class recon_algorithms(Enum):
    FBP_CUDA = "FBP_CUDA"
    FDK_CUDA = "FDK_CUDA"
    CGLS_CUDA = "CGLS_CUDA"
    SART_CUDA = "SART_CUDA"
    EM_CUDA = "EM_CUDA"

    SIRT3D_CUDA = "SIRT3D_CUDA"
    SIRT_CUDA = "SIRT_CUDA"
    CGLS3D_CUDA = "CGLS3D_CUDA"

class recon_geometry(Enum):
    Parallel = "parallel"
    Fanflat = "fanflat"
    Parallel3d = "parallel3d"
    Cone = "cone"

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
    This is the big boy: it holds the projections and can facilitate the recon
    operations grapically in Napari, interactively in Jupyter or in a script
    """
    def __init__(   self) -> None:
        self.sare_bool = False
        self.viewer = napari.Viewer()
        self.viewer.title = "NIST Tomography GUI"
        config_select_widget = {'Config Select':[
                                    self.select_config(),
                                    self.generate_config_widget(),
                                    ]}

        self.config_handles = []
        for key,val in config_select_widget.items():
            self.config_handles.append(
                            self.viewer.window.add_dock_widget( val,
                                                name = key,
                                                add_vertical_stretch = True,
                                                area = 'right'
                                                )
                            )

    def init_operations(self, config_file) -> None:
        """ Once a config has been selected this widget will be loaded """
        super().__init__(config_file)
        self.load_field('dark')
        self.load_field('flat')
        self.fetch_combined_image()
        self.load_transmission_sample()
        self.dtype = np.float32
        self.files = []

        self.widgets ={
            'Transmission':[
                                self.transpose_widget(),
                                self.select_norm(),
                                self.crop_image(),
                                self.cor_wrapper(),
                                self.median_widget(),
                                self.load_images(),
                                self.show_transmission(),
                                self.reset_transmission(),
                                ],
            'Reconstruction':[
                                self.select_reconstruction_parameters(),
                                self.preview_reconstruction(),
                                self.reconstruct_interact(),
                                self.show_reconstruction(),
                                self.write_reconstruction_widget(),
                                self.sare_widget(),
                                self.sare_apply()
                                ],
                        }
        for i,(key,val) in enumerate(self.widgets.items()):
            handle = self.viewer.window.add_dock_widget( val,
                                                name = key,
                                                add_vertical_stretch = True,
                                                area = 'right'
                                                )

            # THIS ADDS THE WIDGETS AS TABS BEHIND THE CONFIG!
            self.viewer.window._qt_window.tabifyDockWidget(
                                                        self.config_handles[0],
                                                        handle)

    def _create_initial_config(self) -> None:
        """ When creating a fresh configuration, this sets up the corresponding
        config file so the settings can be kept track of
        """
        parser = configparser.ConfigParser()
        for key,val in self.settings.items():
            parser.add_section(key)
            for sub_key,sub_val in val.items():
                parser.set(key,sub_key,str(sub_val))

        f_name = Path(".") / f"{self.settings['general']['name']}.ini"
        print(f"writing config file to :{str(f_name)}")
        with open(f_name ,'w') as file_:
            parser.write(file_)

        self.init_operations(f_name)

    def mute_all(self) -> None:
        """ this suppresses all image layers """
        for elem in self.viewer.layers:
            elem.visible = False

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
        for key,val in self.settings.items():
            for sub_key,sub_val in val.items():
                if search_key == sub_key:
                    return sub_val
        else:
            return default

    #--------------------------------------------------------------------------
    #              NAPARI WIDGETS AND FUNCTIONS
    #--------------------------------------------------------------------------
    def select_config(self):
        """ User can select a pre-existing configuration that will
        auto-populate the widget parameters, etc.
        """
        @magicgui(call_button = "Load Existing Config",
                config_file = {'label':'Select Config File (.ini)'},
                persist = True
                )
        def inner(config_file = Path.home()):
            self.init_operations(config_file)
        return inner

    def generate_config_widget(self):
        """ if user selects to create a new config -> then this widget is added
        """
        @magicgui(call_button = 'Generate New Config',
                main_window = True,               # gives a help option
                persist = True,   # previous values are automatically reloaded
                layout = 'vertical',
                Name = {"label":'Name of Experiment'},
                Dark_dir = {"label":'Select Dark Image Directory','mode':'d'},
                Flat_dir = {"label":'Select Flat Image Directory','mode':'d'},
                Proj_dir = {"label":'Select Projections Directory','mode':'d'},
                Extension = {"value":"tif"},
                Delimiter = {'label':'Delimiter for File Naming',"value":"_"},
                Angle_argument = {'label':'Angle Position in File Name',
                                    "value":1},
                )
        def inner(
                Name: str = '',
                Dark_dir = Path.home(),
                Flat_dir = Path.home(),
                Proj_dir = Path.home(),
                Extension = "*.tif",
                Delimiter = "_",
                Angle_argument:int = 1,
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
            self.settings = {'general':{},
                    'paths':{},
                    'pre_processing':{}
                    }
            self.settings['general']['name'] = Name
            self.settings['paths']['dark_path'] = Dark_dir
            self.settings['paths']['flat_path'] = Flat_dir
            self.settings['paths']['projection_path'] = Proj_dir
            self.settings['pre_processing']['extension'] = Extension
            self.settings['pre_processing']['filename_delimiter'] = Delimiter
            self.settings['pre_processing']['angle_argument'] = Angle_argument
            self.settings['pre_processing']['dtype'] = 'float32'
            self.settings['pre_processing']['transpose'] = False
            self.transpose = False


            self._create_initial_config()

        return inner

    def transpose_widget(self):
        """ This gives the option to toggle the transpose
        """
        tp = self.search_settings("transpose",default = False)

        @magicgui(call_button = 'Apply Transpose')
        def inner(Transpose: bool = tp):
            if Transpose:
                self.transpose = Transpose
                self.settings['pre_processing']['transpose'] = Transpose
                self.combined_image = self.combined_image.T
                if 'combined image' in self.viewer.layers:
                    self.viewer.layers.remove('combined image')
                    self.viewer.add_image(self.combined_image,
                            name = 'combined image',
                            colormap = 'Spectral',
                            )
        return inner

    def select_norm(self):
        """ 
        If the configuration already has norm parameters:
            add the shape for the norm parameters

        """
        if 'norm' in self.settings:
            if self.settings['pre_processing']['transpose']:
                keys = ['y0','y1','x0','x1']
            else:
                keys = ['x0','x1','y0','y1']
            x0,x1,y0,y1 = [self.settings['norm'][key] for key in keys]
            verts = np.array([  [x0,y0],
                                [x0,y1],
                                [x1,y1],
                                [x1,y0]], dtype = np.uint32)
            self.viewer.add_shapes(verts, name = 'Norm', face_color = 'r')
            self.norm_patch = [y0,y1,x0,x1]

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
        if 'crop' in self.settings:
            if self.settings['pre_processing']['transpose']:
                keys = ['y0','y1','x0','x1']
            else:
                keys = ['x0','x1','y0','y1']
            x0,x1,y0,y1 = [self.settings['crop'][key] for key in keys]
            verts = np.array([  [x0,y0],
                                [x0,y1],
                                [x1,y1],
                                [x1,y0]], dtype = np.uint32)
            self.viewer.add_shapes( verts,
                                    name = 'Crop',
                                    face_color = 'b',
                                    opacity = 0.2)
            slice_y = slice(y0,y1)
            slice_x = slice(x0,x1)
            crop_image = self.combined_image[slice_x,slice_y]
            self.mute_all()
            self.viewer.add_image(  crop_image,
                                    name = 'cropped image',
                                    colormap = 'twilight_shifted')
            self.crop_patch = [y0,y1,x0,x1]

        @magicgui(call_button = "Crop Image")
        def inner():
            crop_key = 'Crop'
            if crop_key not in self.viewer.layers:
                self.viewer.layers[-1].name = crop_key
            verts = np.round(self.viewer.layers[crop_key].data[0][:,-2:]
                                                        ).astype(np.uint32)
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
            self.viewer.add_image(  crop_image,
                                    colormap = 'twilight_shifted',
                                    name = 'cropped image')
            return crop_image
        return inner

    def cor_wrapper(self):
        """

        This is a bit of a modification, but it produces much better results.
        the algorithm:
            1. calculates the off-axis angle for the raw cropped image
            2. it then rotates the cropped image and calculates its off-axis angle
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
                x_coord = (self.crop_patch[1]-self.crop_patch[0])/2
            else:
                x_coord = 0
            verts = np.array([  [y0,x_coord],
                                [y1,x_coord]])
            self.viewer.add_points(verts, name = 'COR Points', size = 50)
        
        @magicgui(call_button = "Calculate Center of Rotation")
        def inner():
            points_key = "COR Points"
            combined_cupy = cp.array(self.combined_image, dtype = cp.float32)
            if points_key not in self.viewer.layers:
                self.viewer.layers[-1].name = points_key
            points = np.round(self.viewer.layers[points_key].data[:,-2:]
                                                            ).astype(np.uint32)
            verts = np.round(self.viewer.layers['Crop'].data[0][:,-2:]
                                                ).astype(np.uint32).copy()
            x0,x1 = np.min(verts[:,0]), np.max(verts[:,0])
            y0,y1 = np.min(verts[:,1]), np.max(verts[:,1])
            cropped_image = self.viewer.layers['cropped image'].data
            cor_y0,cor_y1 = sorted([points[0,0],points[1,0]])
            print('y0 =',cor_y0,'; y1 =',cor_y1)

            fig2,ax2 = plt.subplots(1,1)
            fig,ax = plt.subplots(1,3, sharex = True, sharey = True)
            # Iterate until the cor angle of the rotated image is 0 -> then
            # translate
            cor = center_of_rotation(cropped_image,cor_y0,cor_y1, ax = ax[0])
            ax[0].set_title("Original Slice")
            theta = np.tan(cor[0])*(180/np.pi)
            theta_a = theta*0.75
            theta_b = theta*1.25
            rot = self.rotated_crop(
                                    combined_cupy,
                                    -theta,
                                    [y0,y1,x0,x1]
                                    ).get()
            rot_a = self.rotated_crop(
                                    combined_cupy,
                                    -theta_a,
                                    [y0,y1,x0,x1]
                                    ).get()
            rot_b = self.rotated_crop(
                                    combined_cupy,
                                    -theta_b,
                                    [y0,y1,x0,x1]
                                    ).get()


            cor2 = center_of_rotation(rot, cor_y0, cor_y1, ax = [])
            cor2_a = center_of_rotation(rot_a, cor_y0, cor_y1, ax = [])
            cor2_b = center_of_rotation(rot_b, cor_y0, cor_y1, ax = [])

            theta2 = np.tan(cor2[0])*(180/np.pi)
            theta2_a = np.tan(cor2_a[0])*(180/np.pi)
            theta2_b = np.tan(cor2_b[0])*(180/np.pi)

            theta_fit = np.polyfit([theta2,theta2_a,theta2_b],
                                    [theta,theta_a,theta_b]
                                    ,1)
            theta_final = np.polyval(theta_fit,0)
            rot_ = self.rotated_crop(
                                    combined_cupy,
                                    -theta_final,
                                    [y0,y1,x0,x1]
                                    ).get()

            cor_final = center_of_rotation(rot_, cor_y0, cor_y1, ax = ax[1])

            theta_qmark_zero = np.tan(cor_final[0])*(180/np.pi)

            xs_local = [theta2,theta2_a,theta2_b,theta_qmark_zero]
            ax2.scatter([theta,theta_a,theta_b,theta_final],xs_local)
            ax2.plot(   np.polyval(theta_fit, xs_local),
                        xs_local,
                        'k--')
            ax2.set_xlabel("theta applied (deg)")
            ax2.set_ylabel("off-axis of rotated (deg)")
            print('theta applied',theta,theta_a,theta_b,theta_final)
            print('theta of rotated',theta2,theta2_a,theta2_b,theta_qmark_zero)

            rot_final = self.rotated_crop(
                                    combined_cupy,
                                    -theta_final,
                                    [y0,y1,x0,x1]
                                    ).get()

            cor2_corrected = center_of_rotation(rot_final,
                                                cor_y0,
                                                cor_y1,
                                                ax = ax[1])

            crop_nx = y1-y0
            dx = int(np.round(cor2_corrected[1]))-crop_nx//2
            ax[1].set_title(f"rotated = {theta_final:.4f} degrees")
            y0 += dx
            y1 += dx
            slice_x = slice(x0,x1)
            slice_y = slice(y0,y1)
            crop2 = self.viewer.layers['combined image'].data[slice_x,slice_y]
            crop2rot = self.rotated_crop(
                                            combined_cupy,
                                            -theta_final,
                                            [y0,y1,x0,x1]
                                            ).get()
            cor3 = center_of_rotation(crop2rot,cor_y0,cor_y1, ax = ax[2])
            ax[2].set_title(f"corrected center dx = {dx}")
            fig.tight_layout()
            plt.show()
            verts[:,1] = [y0,y1,y1,y0]
            try:
                self.viewer.layers.remove('crop corrected')
            except:
                pass
            self.viewer.add_shapes(verts, name = 'crop corrected',
                        face_color = 'b', visible = False, opacity = 0.3)
            self.crop_patch = [y0,y1,x0,x1]
            if self.transpose:
                keys = ['x0','x1','y0','y1']
            else:
                keys = ['y0','y1','x0','x1']
            self.settings['crop'] = {key:val for key,val in zip(keys,self.crop_patch)}
            self.settings['COR'] = {key:val for key,val in zip(['y0','y1','theta'],
                                                   [cor_y0,cor_y1,theta_final])}
        return inner

    def median_widget(self):
        median_init = self.search_settings("median_xy", default = 1)
        kernels_init = self.search_settings(    "thresh median kernels",
                                                default = '')
        z_scores_init = self.search_settings(   "thresh median z-scores",
                                                default = '')
        print('--->',z_scores_init)
        replace_elements = ["[","]"," "]
        if kernels_init != '' and z_scores_init != '':
            kernels_init = str(kernels_init)
            z_scores_init = str(z_scores_init)
            for elem in replace_elements:
                kernels_init = kernels_init.replace(elem,"")
                z_scores_init = z_scores_init.replace(elem,"")

        @magicgui(call_button = "Preview Median",
                image_index = {'step':1,'value':0,'max':1e9},
                median_size = {'step':2,'value':median_init}
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
                med_image.append(median_gpu(
                                cp.array(med_image[-1], dtype = cp.float32), 
                                med_kernel
                                ).get())
            print('kernels = ',kernels,'; z_scores= ',z_scores)
            for kern,z_score in zip(kernels,z_scores):
                print(f'applying {kern} with {z_score}')
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
            self.viewer.add_image(  med_image,
                                    name = med_layer_name,
                                    colormap = 'turbo')
        return inner

    def load_images(self):
        @magicgui(call_button = "Load Projections to Transmission")
        def inner(sub_sample_transmission:int = 1):
            if not self.sare_bool:
                @thread_worker(connect = {'returned':lambda: None})
                def thread_load_images():
                    self.load_projections(truncate_dataset = sub_sample_transmission)
                thread_load_images()
            else:
                logging.warning(("Stripe Filter Has Been Applied, Reset \
                            Transmission to Load more Transmission Files"))
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
            self.viewer.add_image(  
                        dask_array(self.transmission[::ds,::ds,::ds].copy()),
                                    name = 'Transmission',
                                    colormap = 'cividis')
        return inner

    def reset_transmission(self):
        @magicgui(call_button = "Reset Transmission")
        def inner():
            if 'Transmission' in self.viewer.layers:
                self.viewer.layers.remove('Transmission')
            del self.transmission
            self.files = []
            self.sare_bool = False
        return inner

    def select_reconstruction_parameters(self):
        pixel_pitch_init = self.search_settings('camera pixel size',0.0)
        repro_ratio_init = self.search_settings('reproduction ratio',1.0)

        @magicgui(call_button = 'Select Recon Parameters',
                Source_detector_distance = {
                        'label':'Source to Detector Distance (mm) (optional)',
                            "value":0.0,
                            'max':1e9},
                Origin_detector_distance = {
                        'label':'Origin to Detector Distance (mm) (optional)',
                        "value":0.0,
                        'max':1e9},
                Pixel_pitch = {
                            'label':'Pixel Pitch (mm)',
                            'value':pixel_pitch_init,
                            'step':0.0001},
                Reproduction_ratio = {'value':repro_ratio_init},
                Iterations = {'value':0,
                            'min':0,
                            'max':1e9,
                            'label':'Iterations (optional)'
                            },
                fbp_fdk_seed = {'value':True,
                        'label':'Use FBP/FDK as seed (optional)'
                        },
                seed_directory = {'value':Path.home(),
                        'mode':'d',
                        'label':'Directory of seed Dataset (optional)'
                        },
                fbp_filters = {'value':fbp_fdk_filters.ram_lak,
                        'label':'FBP/FDK Filter (optional)'
                        },
                Min_constraint = {'value':-np.inf,
                        'label':'Min Constraint (optional)'
                        }
                    )
        def inner(
                Algorithm = recon_algorithms.FBP_CUDA,
                Geometry = recon_geometry.Parallel,
                Pixel_pitch: float = 1.0,
                Reproduction_ratio: float = 1.0,
                Source_detector_distance: float = 0.0,
                Origin_detector_distance: float = 0.0,
                Iterations: int = 0,
                fbp_fdk_seed: bool = True,
                seed_directory= Path.home(),
                fbp_filters = fbp_fdk_filters.ram_lak,
                Min_constraint: float = 0.0
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
            self.settings['recon']['camera pixel size'] = Pixel_pitch
            assert Pixel_pitch > 0, "Pixel Pitch must be > 0"
            self.settings['recon']['source to origin distance'] = Source_detector_distance - Origin_detector_distance
            self.settings['recon']['origin to detector distance'] = Origin_detector_distance
            self.settings['recon']['reproduction ratio'] = Reproduction_ratio
            self.settings['recon']['recon algorithm'] = Algorithm.value
            self.settings['recon']['recon geometry'] = Geometry.value

            # Optional Arguments
            if Iterations != 0:
                self.settings['recon']['iterations'] = Iterations

            if seed_directory != Path.home():
                self.settings['paths']['seed_path'] = seed_directory
                self.settings['recon']['seed_path'] = seed_directory

            if fbp_fdk_seed and Algorithm.value in ['SIRT_CUDA','SIRT3D_CUDA']:
                self.settings['recon']['fbp_fdk_seed'] = fbp_fdk_seed

            if Algorithm.value in ['FBP_CUDA','FDK_CUDA']:
                self.settings['recon']['FilterType'] = fbp_filters.value

            if np.isfinite(Min_constraint):
                self.settings['recon']['MinConstraint'] = Min_constraint

            self.config.update_recon_params()
            self.check_reconstruction_config()

            self._reconstructor_ = astra_tomo_handler(self.settings['recon'])

        return inner

    def preview_reconstruction(self):
        @magicgui(call_button = "Preview 2D Reconstruction")
        def inner(  sinogram_index: int):
            sinogram = self.transmission[sinogram_index]
            attn = -np.log(sinogram, where = sinogram > 0)
            reconstruction = self._reconstructor_.astra_reconstruct_2D(attn,
                                                                self.angles)
            self.mute_all()
            name = 'recon preview'
            if name in self.viewer.layers:
                self.viewer.layers.remove(name)
            self.viewer.add_image(reconstruction,
                                name = name,
                                colormap = 'viridis')
        return inner

    def reconstruct_interact(self):
        @magicgui(call_button = "Reconstruct",
                radial_zero = {'label':"Cylindrical mask",
                    'value':True}
                )
        def inner(radial_zero: bool = True):
            @thread_worker(connect = {'returned':lambda: None})
            def thread_recon():
                self.reconstruct()
                if radial_zero:
                    self.recon_radial_zero()
            thread_recon()
        return inner

    def show_reconstruction(self):
        @magicgui(call_button = "Show Reconstruction",
                    down_sampling = {'value': 1})
        def inner(down_sampling: int = 1):
            try:
                self.viewer.layers.remove('Reconstruction')
            except:
                pass

            if hasattr(self,'reconstruction'):
                self.mute_all()
                ds = down_sampling
                self.viewer.add_image(
                    dask_array(self.reconstruction[::ds,::ds,::ds].copy()),
                                      name = 'Reconstruction',
                                      colormap = 'plasma')
            else:
                print("Not Reconstructed Yet")
        return inner

    def write_reconstruction_widget(self):
        @magicgui(call_button = "Write Reconstruction",
                Output_dir={
                    'label':'Output Directory for Reconstruction',
                    'mode':'d'                  # select a directory
                    }
                )
        def inner(Output_dir= Path.home(),
            prefix = 'reconstruction',
            extension = 'tif'):
            nz = self.reconstruction.shape[0]
            for i in tqdm(range(nz)):
                f_name = Output_dir / f'{prefix}_{i:0d}.{extension}'
                Image.fromarray(self.reconstruction[i]).save(f_name)
        return inner

    def sare_widget(self):
        snr_init = self.search_settings('snr',1.0)
        la_size_init = self.search_settings('la_size',1)
        sm_size_init = self.search_settings('sa_size',1)

        @magicgui(call_button = 'Preview Ring Filter',
                row={'value':0,'min':0,'max':10_000},
                snr={'value':snr_init},
                la_size={'value':la_size_init,'step':2,'min':1},
                sm_size={'value':sm_size_init,'step':2,'min':1},
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


            unfiltered_recon = self._reconstructor_.astra_reconstruct_2D(
                    -np.log(sinogram_local, where = sinogram_local > 0),
                    self.angles)

            filtered_recon = self._reconstructor_.astra_reconstruct_2D(
                    -np.log(filtered, where = filtered > 0),
                    self.angles)

            print('non finites (sinogram_local) = ',np.sum(~np.isfinite(sinogram_local)))
            print('non finites (sinogram filtered) = ',np.sum(~np.isfinite(filtered)))
            print('non finites (unfiltered) = ',np.sum(~np.isfinite(unfiltered_recon)))
            print('non finites (filtered) = ',np.sum(~np.isfinite(filtered_recon)))

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
                                    colormap = 'viridis',
                                    visible = False)

            self.viewer.add_image(  recons,
                                    name = recon_layer_name,
                                    colormap = 'viridis')

            keys = ['snr','la_size','sm_size']
            vals  = [snr,la_size,sm_size]
            self.settings['SARE'] = {key:val for key,val in zip(keys,vals)}
        return inner

    def sare_apply(self):
        @magicgui(call_button = 'Apply Ring Filter (In Place)')
        def inner(batch_size: int = 10):
            self.config.update_SARE()

            @thread_worker(connect = {'returned':lambda: None})
            def sarepy_threaded():
                self.remove_all_stripe(batch_size = batch_size)

            sarepy_threaded()
            self.sare_bool = True
        return inner

if __name__ == "__main__":
    inst = napari_tomo_gui()
    napari.run()
    #test_config_gen()

