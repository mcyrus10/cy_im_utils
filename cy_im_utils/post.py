from PIL import Image
from tqdm import tqdm
from datetime import datetime
import cupy as cp
import os

def write_log(data_dict : dict, path : str) -> None:
    with open(f"{path}\\.Log.txt",'w') as f:
        f.write(f"Data set name:\t {data_dict['Name']}\n")
        date_time_object = datetime.now()
        f.write(f"Date:\t {date_time_object}\n")
        f.write("{}\n".format("-"*80))
        for key in data_dict:
            if key != 'Name':
                f.write(f"{key}:\t{data_dict[key]}\n")


def write_volume(volume : cp.array, data_dict : dict, path : str, 
        file_name : str, extension : str = 'tif') -> None:
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
    write_log(data_dict, path)
    nz,nx,ny = volume.shape
    for j in tqdm(range(nz)):
        im = Image.fromarray(volume[j,:,:])
        im_path= f"{path}\\{file_name}_{j:04d}.{extension}"
        im.save(im_path)

