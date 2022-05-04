from PIL import Image
from tqdm import tqdm
import numpy as np
import os
from pathlib import Path

def write_volume(   volume : np.array,
                    path : Path,
                    prefix : str,
                    extension : str = 'tif'
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
    nz,nx,ny = volume.shape
    tqdm_writer = tqdm(range(nz), desc = "writing images")
    for j in tqdm_writer:
        im = Image.fromarray(volume[j,:,:])
        im_path= path / f"{prefix}_{j:05d}.{extension}"
        im.save(im_path)
