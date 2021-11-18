from PIL import Image
from tqdm import tqdm
from datetime import datetime
import cupy as cp
import os

def write_volume(volume : cp.array, path : str, prefix : str, extension : str = 'tif') -> None:
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
    if not os.path.isdir(path):
        os.mkdir(path)
    else:
        # DELETE FILES THAT ARE IN HERE!!!!!!!!!!!!!
        pass
    nz,nx,ny = volume.shape
    for j in tqdm(range(nz)):
        im = Image.fromarray(volume[j,:,:])
        im_path= f"{path}\\{prefix}_{j:04d}.{extension}"
        im.save(im_path)

