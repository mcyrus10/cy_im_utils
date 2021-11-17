#------------------------------------------------------------------------------
#                   Data Reduction Prep Utilities
#------------------------------------------------------------------------------

from PIL import Image
from astropy.io import fits
from cupyx.scipy.ndimage import median_filter as median_filter_gpu
from cupyx.scipy import ndimage as gpu_ndimage
from scipy.ndimage import median_filter
from tqdm import tqdm
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt


def imstack_read(files : list, dtype = np.float32) -> np.array: # {{{
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
# }}}
def field(files, median_spatial = 3, dtype = np.float32): # {{{
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
    # }}}    
def field_gpu(files, median_spatial = 3, dtype = cp.float32): # {{{
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
        cupy array (2D) of the spatial median of the z-median of the images
    """
    ext = files[0].split(".")[-1]
    if ext == 'tif':
        read_fcn = Image.open
    elif ext == 'fit':
        read_fcn = imread_fit

    im0 = np.asarray(read_fcn(files[0]))
    shape = im0.shape
    temp = cp.zeros([len(files),shape[0],shape[1]])
    # Determine the file extension

    for i,f in tqdm(enumerate(files)):
        temp[i,:,:] = cp.asarray(read_fcn(f))

    return median_filter_gpu(cp.median(temp,axis = 0),median_spatial).astype(dtype)
    # }}}    
def imread_fit(file_name, axis = 0, device = 'gpu', dtype = [np.float32,cp.float32]): # {{{
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
    # }}}
def get_y_vec(img, axis = 0):#{{{
    """
    Snagged this from a stack overflow post
    """
    n = img.shape[axis]
    s = [1] * img.ndim
    s[axis] = -1
    i = np.arange(n).reshape(s)
    return np.round(np.sum(img * i, axis = axis) / np.sum(img, axis = axis), 1)
#}}}
def center_of_rotation(image,coord_0,coord_1, ax = [], image_center = True): # {{{
    """
    Parameters
    ----------
    image : 2D numpy array
        (ATTENUATION IMAGE) Opposing (0-180 degree) images summed

    y0 : int
        Lower bounds (row-wise) for curve fitting

    y1 : int
        Upper bounds (row-wise) for curve fitting

    axis: int
        axis which y0 and y1 belong to

    ax: axis handle
        For visual inspection of the fit
    
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


        ax.imshow(combined, cmap = 'gist_ncar')
        ax.set_title("Center of Rotation")
        ax.legend()
    return com_fit
    # }}}
def attenuation_gpu_batch(input_arr,ff,df,output_arr,id0,id1,batch_size,norm_patch,
                          crop_patch, theta, kernel = 3, dtype = np.float32):
    # {{{
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
    # }}}
def GPU_rotate_inplace(volume , plane, theta, batch_size): # {{{
    """
    GPU In Place Rotation of a volume by plane; Note that this does not allow reshape, so the in place 
    operation can be achieved (reshape changes the dimensions of the array to allow the entirety of the 
    rotated image). This means that if the window is not large enough it will crop the edge of the image
    
    parameters
    ----------
    volume: numpy array with ndim = 3
        volume to be rotated
    plane: string; 'xz' or 'yz'
        about which plane to apply rotation: 'xz' -> about y-axis; 'yz' about x-axis
    theta: float
        angle to rotate through
    batch_size: int
        size of batch to send to GPU
    """
    nx,ny,nz = volume.shape
    if plane.lower() == 'yz' and theta != 0.0:
        for j in tqdm(range(nx//batch_size)):
            volume_gpu = cp.array(volume[j*batch_size:(j+1)*batch_size,:,:])
            volume[j*batch_size:(j+1)*batch_size,:,:] = cp.asnumpy(gpu_ndimage.rotate(volume_gpu, theta, axes = (2,1), reshape = False))
        remainder = nx%batch_size
        if remainder > 0:
            volume_gpu = cp.array(volume[-remainder:,:,:])
            volume[-remainder:,:,:] = cp.asnumpy(gpu_ndimage.rotate(volume_gpu, theta, axes = (2,1), reshape = False))
    elif plane.lower() == 'xz' and theta != 0.0:
        for j in tqdm(range(ny//batch_size)):
            volume_gpu = cp.array(volume[:,j*batch_size:(j+1)*batch_size,:])
            volume[:,j*batch_size:(j+1)*batch_size,:] = cp.asnumpy(gpu_ndimage.rotate(volume_gpu, theta, axes = (2,0), reshape = False))
        remainder = ny%batch_size
        if remainder > 0:
            volume_gpu = cp.array(volume[:,-remainder:,:])
            volume[:,-remainder:,:] = cp.asnumpy(gpu_ndimage.rotate(volume_gpu, theta, axes = (2,0), reshape = False))
    else:
        assert False, "Invalid input"
#}}}
