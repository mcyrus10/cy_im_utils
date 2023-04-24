"""

-------------------------------------------------------------------------------
                     Data Reduction Prep Utilities
-------------------------------------------------------------------------------

"""
from PIL import Image
from astropy.io import fits
from cupyx.scipy import ndimage as gpu_ndimage
from cupyx.scipy.ndimage import median_filter as median_filter_gpu
from numba import njit,cuda,prange
from pathlib import Path
from scipy.ndimage import median_filter,gaussian_filter
from tqdm import tqdm
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
import logging
from .gpu_utils import GPU_curry


def imstack_read(files: list,
                 dtype=np.float32,
                 tqdm_read=True
                 ) -> np.array:
    """
    Boilerplate image stack reader: takes a list of file names and writes the
    images to a 3D array (image stack)

    Parameters
    ----------
    files : list - list of filenames
    dtype : numpy datatype - datatype for the array
    tqdm_read : bool - use a tqdm wrapper on the reading

    Returns
    -------
    im_stack : 3D np.array of imagems
    """
    n_images = len(files)
    h, w = np.asarray(Image.open(files[0]), dtype=dtype).shape
    im_stack = np.zeros([n_images, h, w], dtype=dtype)
    if tqdm_read:
        iterator = tqdm(enumerate(files), desc='Reading image files')
    else:
        iterator = enumerate(files)
    for i, f in iterator:
        im = np.asarray(Image.open(f), dtype=dtype)
        im_stack[i] = im
    return im_stack


def field(files, median_spatial=3, dtype=np.float32):
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
    temp = np.zeros([len(files), shape[0], shape[1]])

    for i, f in tqdm(enumerate(files)):
        temp[i, :, :] = np.asarray(read_fcn(f))

    temp = np.median(temp, axis=0)
    print(temp.shape)
    med = int(median_spatial)
    delete_me = gaussian_filter(temp, sigma=1)
    print(delete_me.shape)
    print(med)
    return median_filter(temp, size=med).astype(dtype)


def field_gpu(files, median_spatial: int = 3, dtype=np.float32):
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
    temp = np.zeros([len(files), shape[0], shape[1]])
    # Determine the file extension

    tqdm_file_read = tqdm(enumerate(files), desc="reading images")
    for i, f in tqdm_file_read:
        temp[i, :, :] = np.asarray(read_fcn(f))

    # median along main axis
    z_median = cp.zeros(shape, dtype=dtype)
    for elem in range(shape[0]):
        med_temp = cp.median(cp.array(temp[:, elem, :]), axis=0)
        z_median[elem] = med_temp

    logging.debug(f"z_median shape = {z_median.shape}")
    return median_filter_gpu(z_median, size=median_spatial).get()


def imread_fit(file_name,
               axis=0,
               device='gpu',
               dtype=[np.float32, cp.float32],
               ) -> np.array:
    """
    This function reads in a 'fit' file (which has an odd number of frames
    stacked in a 3d array), and it takes the median along the 0 axis to return
    a single 2D array


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
    with fits.open(file_name) as im:
        im = np.array(im[0].data, dtype = dtype[0])

    if im.ndim == 3:
        if device == 'gpu':
            im = cp.array(im, dtype = dtype[1])
            return cp.asnumpy(cp.median(im, axis = axis))
        else:
            return np.median(im, axis = axis)
    elif im.ndim == 2:
        return im
    else:
        assert False,"Unknown Shape of image (should be 2D or 3D"


def imread(im: Path, dtype=np.float32) -> np.array:
    """ super basic wrapper for reading image to np.array
    """
    with Image.open(im) as im_:
        return np.array(im_, dtype=dtype)


def get_y_vec(img: np.array, axis=0) -> np.array:
    """
    Snagged this from a stack overflow post
    """
    n = img.shape[axis]
    s = [1] * img.ndim
    s[axis] = -1
    i = np.arange(n).reshape(s)
    return np.round(np.sum(img * i, axis=axis) / np.sum(img, axis=axis), 1)


def center_of_rotation(image: np.array,
                       coord_y0: int,
                       coord_y1: int,
                       ax: plt.axis = [],
                       image_center: bool = True):
    """
    Parameters
    ----------
    image : 2D numpy array - (ATTENUATION IMAGE) Opposing (0-180 degree) images
                                summed
    coord_y0 : int - Lower bounds (row-wise) for curve fitting
    coord_y1 : int - Upper bounds (row-wise) for curve fitting
    ax: int - axis which y0 and y1 belong to

    image_center: bool
        For visual inspection of the fit (shows where the center of the image
        is relative to the calculated center of mass and curve fit

    Returns
    -------
    numpy array:
       polynomial coefficients for linear fit of the center ofzc rotation as a
       function of row index
    """
    combined = image.copy()
    combined[combined < 0] = 0               #<-----------------------------------
    combined[~np.isfinite(combined)] = 0     #<-----------------------------------
    height,width = combined.shape       # rows, cols
    axis = 1
    COM = get_y_vec(combined, axis)
    subset2 = COM[coord_y0:coord_y1]
    y = np.arange(coord_y0, coord_y1)
    com_fit = np.polyfit(y, subset2, 1)
    # Plotting
    if ax:
        ax.plot(np.polyval(com_fit, [0, height-1]), [0, height-1],
                                                        'k-',
                                                        linewidth = 1,
                                                        label = 'Curve Fit')
        ax.plot([width//6,5*width//6],[coord_y0,coord_y0],'k--', linewidth = 0.5)
        ax.plot([width//6,5*width//6],[coord_y1,coord_y1],'k--', linewidth = 0.5)
        ax.annotate("",xy = (width//4,coord_y0), xytext = (width//4,coord_y1), 
                                        arrowprops = dict(arrowstyle="<->"))
        ax.text(width//10,(coord_y1-coord_y0)/2+coord_y0,"fit\nROI", 
                                        verticalalignment = 'center',
                                        color = 'w')

        ax.scatter(COM,range(height),color = 'r', s = 0.5, label = 'Center of mass')
        if image_center:
            ax.plot([width//2,width//2],[0,height-1],
                                                    color = 'w',
                                                    linestyle = (0,(5,5)),
                                                    label = 'Center of image')


        ax.imshow(combined)
        ax.set_title("Center of Rotation")
        ax.legend()
    return com_fit


def GPU_rotate_inplace(volume: np.array,
                       plane: str,
                       theta: float,
                       batch_size: int
                       ) -> None:
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
        about which plane to apply rotation: 'xz' -> about y-axis; 'yz' about
        x-axis
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
    nx, ny, nz = volume.shape
    # Only the slices that are unique to the axes are re-defined, so all slices
    # are defined generically here
    slice_x_ = lambda j: slice(0, nx, 1)
    slice_y_ = lambda j: slice(0, ny, 1)
    slice_z_ = lambda j: slice(0, nz, 1)
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
        volume[slice_x,slice_y,slice_z] = cp.asnumpy(gpu_ndimage.rotate(
                                                            volume_gpu,
                                                            theta,
                                                            axes = axes,
                                                            reshape = False)
                                                            )
    if remainder > 0:
        volume_gpu = cp.array(volume[slice_x_rem,slice_y_rem,slice_z_rem])
        volume[slice_x_rem,slice_y_rem,slice_z_rem] = cp.asnumpy(gpu_ndimage.rotate(
                                                            volume_gpu,
                                                            theta,
                                                            axes = axes,
                                                            reshape = False)
                                                            )
    del volume_gpu


def rotated_crop(image: cp.array,
                 theta: float,
                 crop: list
                 ) -> cp.array:
    """ This pads the array so that the rotation does not introduce zeroes,
    maybe a bit clunky, but whatever. Notebook in 'Experimentation' has the
    trig math, explanation, etc.

    Parameters:
    -----------
        image: cp.array - the global image to be crop-rotated
        theta: float - the angle in (degrees) to rotate the image through
        crop: array-like - the coordinates of the crop x0,x1,y0,y1

    Returns:
    --------
        cropped image rotated through theta with the dimensions of the
        specified crop (not reshaped) and padded with the original image (not
        zeros)

    """
    x0, x1, y0, y1 = crop
    theta_rad = np.deg2rad(theta)
    trig_product = np.abs(np.sin(theta_rad)*np.cos(theta_rad))
    pad_x = np.ceil(trig_product*(y1-y0)).astype(np.uint32)//2
    pad_y = np.ceil(trig_product*(x1-x0)).astype(np.uint32)//2
    x_0, x_1, y_0, y_1 = np.ceil([x0-pad_x, x1+pad_x, y0-pad_y, y1+pad_y]
                                 ).astype(np.uint32)

    slice_2 = (slice(y_0, y_1), slice(x_0, x_1))
    image_2 = image[slice_2]
    im2_rot = gpu_ndimage.rotate(image_2,
                                 theta,
                                 reshape=False)
    slice_3 = (slice(pad_y, pad_y+(y1-y0)), slice(pad_x, pad_x+(x1-x0)))
    ret_val = im2_rot[slice_3]
    return ret_val


@njit(parallel=True)
def radial_zero(arr: np.array, radius_offset: int = 0) -> None:
    """
    This function is for making values outside the radius of a circle at the
    center of the image to have a value of 0 so their noise is not incorporated
    into any calculations or predictions
    Args:
    -----
    arr : np.array
        input image (must be square)

    radius_offset : int
        this makes the radius smaller if you want to remove some more of the
        edge

    Returns:
    --------
    None (operates in-place)
    """
    nx, ny = arr.shape
    assert nx == ny, "This function only accepts square images"
    radius = nx//2
    for i in prange(nx):
        for j in prange(ny):
            r = ((i-nx//2)**2+(j-ny//2)**2)**(1./2.)
            if r > radius-radius_offset:
                arr[i, j] = 0


@njit
def make_circular_2d_mask(input_arr: np.array,
                          radius: float,
                          x_center: int,
                          y_center: int
                          ) -> None:
    """
    this is pretty useful
    """
    nx, ny = input_arr.shape
    for i in prange(nx):
        for j in prange(ny):
            if ((i-x_center)**2 + (j-y_center)**2)**(0.5) < radius:
                input_arr[i, j] = True
