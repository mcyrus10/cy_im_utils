from .prep import imread_fit,field_gpu
from .visualization import COR_interact,SAREPY_interact
from PIL import Image
from cupyx.scipy.ndimage import rotate as rotate_gpu, median_filter as median_gpu
from pathlib import Path
from tqdm import tqdm
import astra
import cupy as cp
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib
import shutil

class tomo_dataset:
    """
    This is the big boy: it holds the projections and can facilitate all the
    recon operations
    """
    def __init__(self, data_dict):
        logging.info("-"*80)
        logging.info("-- TOMOGRAPHY RECONSTRUCTION --")
        logging.info("-"*80)
        self.settings = data_dict
        self.update_members()

    #---------------------------------------------------------------------------
    #                       UTILS
    #---------------------------------------------------------------------------
    def update_members(self) -> None:
        """
        This updates all the members of tomo_dataset so when the dictionary
        changes everything will be updated
        """
        for key,val in self.settings.items():
            key_ = key.replace(" ","_")
            logging.info(f"{key_} : {val}")
            setattr(self,key_,val)

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
            tomo_recon.free_field("attenuation")
        """
        setattr(self,field,None)

    def check_astra_shape(self) -> None:
        """
        This creates a figure to help confirm that your input to ASTRA has the
        correct shape:
            1) Sinogram is axis 0
            2) Projection is axis 1
            3) Detector col slice is axis 2
        
        """
        nx,ny,nz = self.attenuation.shape
        fig,ax = plt.subplots(2,2)
        ax = ax.flatten()
        ax[0].imshow(self.attenuation[nx//2,:,:])
        ax[0].set_title("Sinogram - (Axis : 0)")
        ax[2].imshow(self.attenuation[:,ny//2,:])
        ax[2].set_title("Projection (Axis : 1)")
        ax[2].plot([0,nx-1],[nz//2,nz//2],'w--')
        ax[2].plot([nx//2,nx//2],[0,nz-1],'w--')
        ax[3].imshow(self.attenuation[:,:,nz//2])
        ax[3].set_title("Detector Col Slice (Axis : 2)")
        [a.axis(False) for a in ax]
        fig.tight_layout()
        fig.suptitle("Astra Expected - (Attenuation Array)")

    def gpu_curry_loop(self, function, ax_length : int, batch_size : int) -> None:
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
        for j in tqdm(range(ax_length//batch_size)):
            function(j*batch_size,(j+1)*batch_size)
        remainder = ax_length % batch_size
        if remainder > 0:
            logging.info(f"remainder = {remainder}")
            function(ax_length-remainder,ax_length)

    #---------------------------------------------------------------------------
    #                       PRE PROCESSING
    #---------------------------------------------------------------------------
    def load_field(self, mode : str) -> None:
        """
        wrapper for "field_GPU" in prep.py

        args:
            mode : str
                mode can be 'dark' or 'flat'

        examples:
            tomo_recon = tomo_dataset(data_dict)
            tomo_recon.load_field("flat")
            tomo_recon.load_field("dark")
        """
        field_path = self.settings[f'{mode} path']
        logging.info(f"Reading {mode} field from {field_path}")
        files = list(field_path.glob("*.tif"))
        nx,ny = np.asarray(self.imread_function(files[0])).shape
        field = field_gpu(files, self.median_spatial)
        if self.transpose:
            field = field.T
        setattr(self,mode,field)

    def load_projections(self, mode : str = 'read', truncate_dataset : int = 1) -> np.array:
        """
        """
        if 'serialized' in mode:
            logging.info(f"Reading Serialized Dataset ({self.serialized_path})")
            self.projections = np.load(self.serialized_path)[::truncate_dataset]
        elif 'read' in mode:
            logging.info(f"Reading Images From {self.projection_path}")
            files = list(self.projection_path.glob("*.tif"))[::truncate_dataset]
            nx,ny = np.asarray(self.imread_function(files[0])).shape
            tqdm_imread = enumerate(tqdm(files, desc = "reading images"))
            self.projections = np.zeros([len(files),nx,ny], dtype = self.dtype)
            for i,f in tqdm_imread:
                self.projections[i] = np.asarray(self.imread_function(f), dtype = self.dtype)
            if self.transpose:
                logging.info("Transposing images")
                self.projections = np.transpose(self.projections,(0,2,1))

    def COR_interact(self, d_theta : int = 60, angles : list = []) -> None:
        """
        Wrapper for COR_interact in visualization
        """
        if self.is_jupyter():
            logging.info("COR Interact Started")
            if not angles:
                angles = [j*d_theta for j in range(360//d_theta)]
            COR_interact(self.settings, self.projections, self.flat, self.dark, angles)
        else:
            logging.warning("Interact needs to be executed in a Notebook Environment - This method is not being executed")

    def SARE_interact(self, figsize : tuple = (12,5)) -> None:
        """
        Wrapper for SAREPY Interact in visualization
        """
        if self.is_jupyter():
            logging.info("Stripe Artifact Interact Started")
            SAREPY_interact(self.settings, self.attenuation, figsize = figsize)
        else:
            logging.warning("Interact needs to be executed in a Notebook Environment - This method is not being executed")


    #---------------------------------------------------------------------------
    #                       PROCESSING
    #---------------------------------------------------------------------------
    def gpu_ops(self,x0 : int,x1 : int) -> None:
        """
        In-Place Operations to populate self.attenuation; 
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
        _,height,width = self.attenuation.shape
        slice_ = slice(x0,x1)
        batch_size = x1-x0
        projection_gpu = cp.asarray(self.projections[slice_], dtype = self.dtype)
        projection_gpu = rotate_gpu(projection_gpu, -self.theta, axes = (1,2), reshape = False)
        projection_gpu -= cp.asarray(self.dark[None,:,:])
        projection_gpu /= cp.asarray(self.flat[None,:,:]-self.dark[None,:,:])
        patch = cp.mean(projection_gpu[:,self.norm_x,self.norm_y], axis = (1,2), dtype = self.dtype)
        projection_gpu /= patch.reshape(batch_size,1,1)
        projection_gpu = median_gpu(projection_gpu,(1,self.median_spatial,self.median_spatial))
        projection_gpu -= cp.log(projection_gpu)
        projection_gpu[~cp.isfinite(projection_gpu)] = 0
        self.attenuation[slice_,:,:] = cp.asnumpy(projection_gpu[:,self.crop_y,self.crop_x])

    def attenuation_GPU(self, batch_size : int = 20) -> None:
        """
        this method executes gpu_ops on all the batches to create
        self.attenuation

        args:
        -----
            batch_size : int
                size of mini batches for GPU
        """
        # in case these have been updated by COR Interact
        crop_patch = self.settings['crop patch']
        norm_patch = self.settings['norm patch']
        self.crop_x = slice(crop_patch[0],crop_patch[1])
        self.crop_y = slice(crop_patch[2],crop_patch[3])
        self.norm_x = slice(norm_patch[0],norm_patch[1])
        self.norm_y = slice(norm_patch[2],norm_patch[3])
        self.theta = self.settings['theta']

        attn_nx = crop_patch[1]-crop_patch[0]
        attn_ny = crop_patch[3]-crop_patch[2]
        n_proj = self.projections.shape[0]
        self.attenuation = np.empty([n_proj,attn_ny,attn_nx], dtype = self.dtype)
        logging.info(f"attenuation shape = {self.attenuation.shape}")
        self.gpu_curry_loop(self.gpu_ops, n_proj, batch_size)

    def remove_all_stripe_ops(  self, id0 : int, id1 : int) -> None:
        """
        operates in-place
        this is meant to be called by gpu_curry_loop

        args:
        ----
            x0 : int
                index 1 for the slice

            x1 : int
                index 2 for the slice

        """
        slice_ = slice(id0,id1)
        vol_gpu = cp.asarray(self.attenuation[:,slice_,:], dtype = self.dtype)
        vol_gpu = remove_all_stripe_GPU(vol_gpu,self.snr,self.la_size,self.sm_size)
        self.attenuation[:,slice_,:] = cp.asnumpy(vol_gpu)

    def remove_all_stripe(  self, batch_size : int = 50) -> None:
        """
        operates in-place

        """
        _,n_row,_ = self.attenuation.shape
        gpu_curry_loop(self.remove_all_stripe_ops, n_row, batch_size)

    def reconstruct(self):
        self.reconstruction = ASTRA_General(self.attenuation, self.settings)

    #--------------------------------------------------------------------------
    #                       POST PROCESSING
    #--------------------------------------------------------------------------
    def serialize(self, arr_name : str, path : pathlib.Path ) -> None:
        """
        Save Array as serialized numpy array

        args:
        -----
            arr_name : str
                string of member name (projections, attenuation, etc.)

            path : path_like (string or pathlib.Path)
        """
        f_name = path / f"{arr_name}.npy"
        logging.info(f"Saving {arr_name} to {f_name}")
        np.save(f_name, getattr(self, arr_name))

    def write_im_stack(self, arr_name : str, directory : pathlib.Path) -> None:
        """
        Save Array as Image stack
        """
        if directory.is_dir():
            logging.warning(f"Deleting {directory}")
            shutil.rmtree(directory)

        os.mkdir(directory)
        logging.info(f"Saving {arr_name} to {directory}")
        arr_handle = getattr(self,arr_name)
        arr_shape = arr_handle.shape
        assert len(arr_shape) == 3, "This function operates on volumes"
        nx,ny,nz = arr_shape
        tqdm_im_save = tqdm(range(nx), desc = f"saving {arr_name} as images")
        for i in tqdm_im_save:
            im = Image.fromarray(arr_handle[i,:,:])
            f_name = directory / f"{arr_name}_{i:06}.tif"
            im.save(f_name)

def ASTRA_General(  attn : np.array, data_dict : dict  ) -> np.array:
    """
    algorithm for cone -> FDK_CUDA
    algorithms for Parallel -> SIRT3D_CUDA, FP3D_CUDA, BP3D_CUDA
    """
    detector_rows,n_projections,detector_cols = attn.shape
    distance_source_origin = data_dict['source to origin distance']
    distance_origin_detector = data_dict['origin to detector distance']
    detector_pixel_size = data_dict['camera pixel size']*data_dict['reproduction ratio']
    algorithm = data_dict['recon algorithm']
    geometry = data_dict['recon geometry']
    angles = np.linspace(0, 2 * np.pi, num = n_projections, endpoint=False)
    #  ---------    PARALLEL BEAM    --------------
    if geometry.lower() == 'parallel':
        proj_geom = astra.create_proj_geom('parallel3d', 1, 1, detector_rows, detector_cols, angles)
        projections_id = astra.data3d.create('-sino', proj_geom, attn)

    #  ---------    CONE BEAM    --------------
    elif geometry.lower() == 'cone':
        proj_geom = astra.create_proj_geom('cone', 1, 1, detector_rows, detector_cols, angles,
                (distance_source_origin + distance_origin_detector) / detector_pixel_size, 0)
        projections_id = astra.data3d.create('-sino', proj_geom, attn)
        
    vol_geom = astra.creators.create_vol_geom(detector_cols, detector_cols, detector_rows)
    reconstruction_id = astra.data3d.create('-vol', vol_geom, data=0)
    alg_cfg = astra.astra_dict(algorithm)
    alg_cfg['ProjectionDataId'] = projections_id
    alg_cfg['ReconstructionDataId'] = reconstruction_id
    alg_cfg['option'] = {'FilterType': 'ram-lak'}
    algorithm_id = astra.algorithm.create(alg_cfg)
    
    #-------------------------------------------------
    astra.algorithm.run(algorithm_id)  # This is slow
    #-------------------------------------------------

    reconstruction = astra.data3d.get(reconstruction_id)
    reconstruction /= detector_pixel_size

    # DELETE OBJECTS TO RELEASE MEMORY
    astra.algorithm.delete(algorithm_id)
    astra.data2d.delete([projections_id,reconstruction_id])
    return reconstruction
