"""
-------------------------------------------------------------------------------
                     Data Reduction Prep Utilities
-------------------------------------------------------------------------------
"""

from PIL import Image
from astropy.io import fits
from cupyx.scipy.ndimage import median_filter as median_filter_gpu
from cupyx.scipy import ndimage as gpu_ndimage
from scipy.ndimage import median_filter
from tqdm import tqdm
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from .gpu_utils import GPU_curry


def imstack_read(files : list, dtype = np.float32) -> np.array: 
    """
    Boilerplate image stack reader: takes a list of file names and writes the
    images to a 3D array (image stack)

    Parameters
    ----------
    files : list
        list of filenames

    dtype : numpy datatype
        datatype for the array

    Returns
    -------
    im_stack : 3D np.array of imagems 
    """
    n_images = len(files)
    h,w = np.asarray(Image.open(files[0]), dtype = dtype).shape
    im_stack = np.zeros([n_images,h,w], dtype = dtype)
    for i,f in tqdm(enumerate(files)):
        im = np.asarray(Image.open(f), dtype = dtype)
        im_stack[i] = im
    return im_stack

def field(files, median_spatial = 3, dtype = np.float32): 
    """
    parameters
    ----------
    files: list 
        Image files for the field (dark/flat)

    median_spatial: int
        Median spatial filter size

    dtype: numpy numerical type
        Data type for the images

    returns
    -------
    numpy array (2D) of the spatial median of the z-median of the images
    """
    ext = files[0].split(".")[-1]
    if ext == 'tif':
        read_fcn = Image.open
    elif ext == 'fit':
        read_fcn = imread_fit

    im0 = np.asarray(read_fcn(files[0]))
    shape = im0.shape
    temp = np.zeros([len(files),shape[0],shape[1]])

    for i,f in tqdm(enumerate(files)):
        temp[i,:,:] = np.asarray(read_fcn(f))

    return median_filter(np.median(temp,axis = 0),median_spatial).astype(dtype)
        
def field_gpu(files, median_spatial = 3, dtype = np.float32): 
    """
    parameters
    ----------
    files: list 
        Image files for the field (dark/flat)
    median_spatial: int
        Median spatial filter size
    dtype: numpy numerical type
        Data type for the images

    returns
    -------
        numpy array (2D) of the spatial median of the z-median of the images
    """
    if '.tif' in files[0].suffix:
        read_fcn = Image.open
    elif '.fit' in files[0].suffix:
        read_fcn = imread_fit

    im0 = np.asarray(read_fcn(files[0]))
    shape = im0.shape
    temp = np.zeros([len(files),shape[0],shape[1]])
    # Determine the file extension

    tqdm_file_read = tqdm(enumerate(files), desc = "reading images")
    for i,f in tqdm_file_read:
        temp[i,:,:] = np.asarray(read_fcn(f))
    
    # median along main axis
    z_median = cp.zeros(shape, dtype = dtype)
    for elem in range(shape[0]):
        med_temp = cp.median(cp.array(temp[:,elem,:]), axis = 0)
        z_median[elem] = med_temp
   
    return median_filter_gpu(z_median,median_spatial).get()
        
def imread_fit(file_name, axis = 0, device = 'gpu', dtype = [np.float32,cp.float32]): 
    """
    Parameters:
    -----------
    file_name: string
        name of the file to open

    axis: int
        axis along which to take the median for the combination

    device: string
        'cpu': compute the median on the CPU
        'gpu': compute the median on the GPU (FASTER)

    dtype: list
        list with first element as cpu dtype, second elemement is gpu dtype

    returns: 
    --------
        2D numpy array of median along specified axis (should be 0)
    """
    im = np.asarray(fits.open(file_name)[0].data.astype(dtype[0]))
    if device == 'gpu':
        im = cp.array(im, dtype = dtype[1])
        return cp.asnumpy(cp.median(im, axis = axis))
    elif device == 'cpu':
        return np.median(im, axis = axis)
    
def get_y_vec(img : np.array, axis = 0) -> np.array:
    """
    Snagged this from a stack overflow post
    """
    n = img.shape[axis]
    s = [1] * img.ndim
    s[axis] = -1
    i = np.arange(n).reshape(s)
    return np.round(np.sum(img * i, axis = axis) / np.sum(img, axis = axis), 1)

def center_of_rotation(image : np.array ,coord_0 : int, coord_1 : int, ax : plt.axis = [], image_center : bool = True): 
    """
    Parameters
    ----------
    image : 2D numpy array
        (ATTENUATION IMAGE) Opposing (0-180 degree) images summed

    coord_0 : int
        Lower bounds (row-wise) for curve fitting

    coord_1 : int
        Upper bounds (row-wise) for curve fitting

    ax: int
        axis which y0 and y1 belong to

    image_center: bool
        For visual inspection of the fit (shows where the center of the image
        is relative to the calculated center of mass and curve fit
    
    Returns
    -------
    numpy array:
       polynomial coefficients for linear fit of the center ofzc rotation as a function of row index
    """
    combined = image.copy()
    combined[combined < 0] = 0               #<-----------------------------------
    combined[~np.isfinite(combined)] = 0     #<-----------------------------------
    height,width = combined.shape       # rows, cols
    axis = 1
    COM = get_y_vec(combined,axis)
    subset2 = COM[coord_0:coord_1]
    y = np.arange(coord_0,coord_1)
    com_fit = np.polyfit(y,subset2,1)
    # Plotting
    if ax:
        ax.plot(np.polyval(com_fit,[0,height-1]),[0,height-1],'k-', linewidth = 1, label = 'Curve Fit')
        ax.plot([0,width],[coord_0,coord_0],'k--', linewidth = 0.5)
        ax.plot([0,width],[coord_1,coord_1],'k--', linewidth = 0.5)
        ax.annotate("",xy = (width//4,coord_0), xytext = (width//4,coord_1), arrowprops = dict(arrowstyle="<->"))
        ax.text(width//10,(coord_1-coord_0)/2+coord_0,"fit\nROI", verticalalignment = 'center', color = 'w')
        ax.scatter(COM,range(height),color = 'r', s = 0.5, label = 'Center of mass')
        if image_center:
            ax.plot([width//2,width//2],[0,height-1],color = 'w', linestyle = (0,(5,5)),label = 'Center of image')


        ax.imshow(combined)
        ax.set_title("Center of Rotation")
        ax.legend()
    return com_fit
    
def attenuation_gpu_batch(input_arr,ff,df,output_arr,id0,id1,batch_size,norm_patch, crop_patch, theta, kernel = 3, dtype = np.float32):
    """
    This is a monster (and probably will need some modifications)
    1) upload batch to GPU
    2) rotate
    3) transpose <------------ NOT NECESSARY SINCE YOU KNOW THE BLOCK STRUCTURE NOW
    4) convert image to transmission space
    5) extract normalization patches
    6) normalize transmission images
    7) spatial median (kernel x kernel) -> improves nans when you take -log
    8) lambert beer
    9) reverse the transpose from 3
    10) crop
    11) insert batch into output array
    Parameters:
    -----------
    input_arr: 3D numpy array 
        input volume array
    ff: 2D cupy array 
        flat field
    df: 2D cupy array 
        dark field
    output_arr: 3D numpy array 
        array to output into
    id0: int
        first index of batch
    id1: int
        final index of batch
    batch_size: int
        size of batch
    norm_patch: list
        list of coordinates of normalization patch (x0,x1,y0,y1)
    crop_patch: list
        list of coordinates of crop patch (x0,x1,y0,y1)
    theta: float
        angle to rotate the volume through
    kernel: int (odd number)
        size of median kernel
    dtype: numpy data type
        data type of all arrays
    """
    n_proj,height,width = input_arr.shape
    projection_gpu = cp.asarray(input_arr[id0:id1], dtype = dtype)
    projection_gpu = rotate_gpu(projection_gpu,theta, axes = (1,2), reshape = False)
    projection_gpu -= df.reshape(1,height,width)
    projection_gpu /= (ff-df).reshape(1,height,width)
    patch = cp.mean(projection_gpu[:,norm_patch[0]:norm_patch[1],norm_patch[2]:norm_patch[3]], axis = (1,2), dtype = dtype)
    projection_gpu /= patch.reshape(batch_size,1,1)
    projection_gpu = median_gpu(projection_gpu, (1,kernel,kernel))
    projection_gpu = -cp.log(projection_gpu)
    #-----------------------------------------------
    #---      make all non-finite values 0?      ---
    projection_gpu[~cp.isfinite(projection_gpu)] = 0
    #-----------------------------------------------
    output_arr[id0:id1] = cp.asnumpy(projection_gpu[:,crop_patch[0]:crop_patch[1],crop_patch[2]:crop_patch[3]])
    
def GPU_rotate_inplace(volume : np.array , plane : str, theta : float, batch_size : int) -> None:
    """
    Note: if you are unsure which plane to rotate about refer to
    visualization.orthogonal_plot which will show the planes labeled
    GPU In Place Rotation of a volume by plane; Note that this does not allow
    reshape, so the in place operation can be achieved (reshape changes the
    dimensions of the array to allow the entirety of the rotated image). This
    means that if the window is not large enough it will crop the edge of the
    image
    
    The plane string controls the slicing functions that generate the
    batch-wise slices stepping through the volume this allows a SINLGE LOOP to
    be created that generically calls slice_x_ and slice_y_ to get these
    slices. The remainder slices are also calculated in the scope of this
    conditional.  I havent added the ability to rotate about the z axis so its
    slice is static for all cases (not a lambda).  This might be abstractable
    to a higher degree but I find this format to be readable and less error
    prone.
    
    parameters
    ----------
    volume: np.array
        volume to be rotated (len volume.shape == 3)
    plane: str
        about which plane to apply rotation: 'xz' -> about y-axis; 'yz' about x-axis
    theta: float
        angle to rotate through
    batch_size: int
        size of batch to send to GPU
        
    Returns:
        None -> Operates in-place on *volume*

    Example Usage:
        GPU_rotate_inplace(volume, 'xy', 2, 50)

    """
    message = "WARNING: This function is deprecated, use GPU_curry in gpu_utils to achieve this"
    logging.info(message)
    print(message)
    nx,ny,nz = volume.shape
    # Only the slices that are unique to the axes are re-defined, so all slices
    # are defined generically here
    slice_x_ = lambda j: slice(0,nx,1)
    slice_y_ = lambda j: slice(0,ny,1)
    slice_z_ = lambda j: slice(0,nz,1)
    slice_x_rem = slice_x_("")
    slice_y_rem = slice_y_("")
    slice_z_rem = slice_z_("")

    if plane.lower() == 'yz' and theta != 0.0:
        slice_x_ = lambda j: slice(j*batch_size,(j+1)*batch_size,1)
        axes = (2,1)
        remainder = nx % batch_size
        slice_x_rem = slice(nx-remainder,nx,1)
        iterator = range(nx//batch_size)
    elif plane.lower() == 'xz' and theta != 0.0:
        slice_y_ = lambda j: slice(j*batch_size,(j+1)*batch_size,1)
        axes = (2,0)
        remainder = ny % batch_size
        slice_y_rem = slice(ny-remainder,ny,1)
        iterator = range(ny//batch_size)
    elif plane.lower() == 'xy' and theta != 0.0:
        slice_z_ = lambda j: slice(j*batch_size,(j+1)*batch_size,1)
        axes = (0,1)
        remainder = nz % batch_size
        slice_z_rem = slice(nz-remainder,nz,1)
        iterator = range(nz//batch_size)
       
    for j in tqdm(iterator):
        slice_x = slice_x_(j)
        slice_y = slice_y_(j)
        slice_z = slice_z_(j)
        volume_gpu = cp.array(volume[slice_x,slice_y,slice_z])
        volume[slice_x,slice_y,slice_z] = cp.asnumpy(gpu_ndimage.rotate(volume_gpu, theta, axes = axes, reshape = False))
    if remainder > 0:
        volume_gpu = cp.array(volume[slice_x_rem,slice_y_rem,slice_z_rem])
        volume[slice_x_rem,slice_y_rem,slice_z_rem] = cp.asnumpy(gpu_ndimage.rotate(volume_gpu, theta, axes = axes, reshape = False))
    del volume_gpu
