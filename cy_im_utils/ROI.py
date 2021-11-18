from prep import GPU_rotate_inplace
from post import write_volume
import numpy as np

import logging
logger = logging.getLogger(__name__)

class ROI:
    """
    Class for expressing the operations applied to cropping and rotating a volume
    """
    def __init__(self, theta : dict, crop : dict, label : str):
        self.theta = theta
        self.crop = crop
        self.slice = None
        self.label = label
        logger.info(f"ROI : {label}")
        logger.info(f"theta -> xz : {theta['xz']} ; yz : {theta['yz']}")
        logger.info(f"crop 1 -> {crop['crop 1']}")
        logger.info(f"crop 2 -> {crop['crop 2']}")
         
    def process(self, volume : np.array, batch_size : int, in_place : bool = False) -> None:
        """
        Execute slicing operations defined by theta and crop
        
        Parameters
        ----------
        volume : 3d np.array
            volume to be cropped/rotated
            
        batch_size : int
            size of the batches to be moved to the GPU
            
        in_place : bool
            This boolean determines if 'volume_slice' is a copy of 'volume' or if it is a view,
            be careful with this because if it is True then it modifies the original volume so 
            it might distort further ROIs. It is much faster using in_place though!
        
        Returns
        -------
            None; (Assigns self.slice to the transformed volume)
        """
        crop_1 = self.crop['crop 1']
        crop_2 = self.crop['crop 2']
        volume_slice = self.slice_by_crop(volume, crop_1, in_place = in_place)
        for plane,theta in self.theta.items():
            if theta != 0.0:
                GPU_rotate_inplace(volume_slice, plane, theta, batch_size)
        if crop_2:
            volume_slice = self.slice_by_crop(volume_slice, crop_2, in_place = True)
        
        self.slice = volume_slice
    
    def slice_by_crop(self, volume : np.array, crop : list, in_place : bool = False) -> np.array:
        """
        Execute the slice of volume by a crop spec
        
        Parameters
        ----------
        volume : 3d np.array
            volume to be cropped
            
        crop : list
            6 integers corresponding to the corners of the crop [x0,x1,y0,y1,z0,z1]

        in_place : bool
            the bool determines whether or not a copy is returned or a reference (view) 
            of the volume (i.e. the same memory)

        """
        slice_x = slice(crop[0],crop[1],1)
        slice_y = slice(crop[2],crop[3],1)
        slice_z = slice(crop[4],crop[5],1)
        if in_place:
            logger.warning("transforming volume in-place --> ensure rotations don't skew other ROIs")
            return volume[slice_x,slice_y,slice_z]
        else:
            return volume[slice_x,slice_y,slice_z].copy()

    def write_volume(self, path : str, extension : str = "tif"):
        """
        Wrapper for calling write_volume
        """
        write_volume(self.slice, path, self.label, extension )
