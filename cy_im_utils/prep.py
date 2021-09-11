#------------------------------------------------------------------------------
#                   Data Reduction Prep Utilities
#------------------------------------------------------------------------------

from PIL import Image
from astropy.io import fits
from cupyx.scipy.ndimage import median_filter as median_filter_gpu
from scipy.ndimage import median_filter
from tqdm import tqdm
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt



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
    height,width = combined.shape       # rows, cols
    axis = 1
    COM = get_y_vec(combined,axis)
    subset2 = COM[coord_0:coord_1]
    y = np.arange(coord_0,coord_1)
    com_fit = np.polyfit(y,subset2,1)
    # Plotting
    if ax:
        if axis == 0:
            ax.plot([0,width-1],np.polyval(com_fit,[0,width-1]),'k-', linewidth = 1, label = 'Curve Fit')
            ax.plot([coord_0,coord_0],[0,width],'k--', linewidth = 0.5)
            ax.plot([coord_1,coord_1],[0,width],'k--', linewidth = 0.5)
            ax.annotate("",xy = (coord_0,height//4), xytext = (coord_1,height//4), arrowprops = dict(arrowstyle="<->"))
            ax.text(height//10,(coord_1-coord_0)/2+coord_0,"fit\nROI", verticalalignment = 'center', color = 'w')
            ax.scatter(range(width),COM,color = 'r', s = 0.5, label = 'Center of mass')
            if image_center:
                ax.plot([0,width-1],[height//2,height//2],color = 'w', linestyle = (0,(5,5)),label = 'Center of image')

        elif axis == 1:
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
def nan_filter(im): # {{{
    """
    Note This does not take arbitrary kernel sizes right now, only kernel size 3 (it does not 
    handle edges correctly if the kernel is larger).
    whew this is ugly; alternate ideas: disallow same size output and just return the inner 
    part-> all interior nodes! simplifies this drastically, could possibly be on 
    To debug this make a 10x10 grid and put the nans in coordinates where you think it is 
    misbehaving then you can more simply correct the mistakes ()
    
    im: 2d Numpy array
        image array with nans
    
    returns:
    --------
        None (mutates im in place)
    """
    kernel = 3
    not_numbers = np.where(np.isnan(im))
    infs = np.where(im == np.inf)
    nans = np.hstack([not_numbers,infs])
    nx,ny = im.shape
    k = kernel//2
    for x,y in zip(nans[0],nans[1]):
        if (y == 0) and (x == 0):
            #print(\"top left corner\")
            slice_ = im[:k+1,:k+1]
        elif (y == 0) and (x > 0) and (x < nx-1):
            #print(\"top row\")
            slice_ = im[x-k:x+k+1,:k+1]
        elif (y == 0) and (x == nx-1):
            #print(\"top corner\")
            slice_ = im[nx-k-1:,:k+1]
        elif (y > 0) and (y < ny-1) and (x == 0):
            #print(\"left edge\")
            slice_ = im[:k+1,y-k:y+k+1]
        elif (y > 0) and (y < ny-1) and (x == nx-1):
            #print(\"right edge\")
            slice_ = im[nx-k-1:,y-k:y+k+1]
        elif (y == ny-1) and (x > 0) and (x < nx-1):
            #print(\"bottom row\")
            slice_ = im[x-k:x+k+1,ny-k-1:]
        elif (y == ny-1) and (x == 0):
            #print(\"bottom left corner?\")
            slice_ = im[:k+1,ny-k-1:]
        elif (y == ny-1) and (x == nx-1):
            #print('Bottom right corner')
            slice_ = im[-k-1:,-k-1:]
        elif (y < ny-1) and ( y > 0) and (x < nx-1) and (x > 0):
            #print('Interior')
            slice_ = im[x-k:x+k+1,y-k:y+k+1]
        temp = slice_[~np.isnan(slice_)]
        med = np.median(temp)#.flatten())
        im[x,y] = med
        del temp,med,slice_
    # }}}
def nan_filter_gpu(im): # {{{
    """
    Note This does not take arbitrary kernel sizes right now, only kernel size 3 (it does not 
    handle edges correctly if the kernel is larger).
    whew this is ugly; alternate ideas: disallow same size output and just return the inner 
    part-> all interior nodes! simplifies this drastically, could possibly be on 
    To debug this make a 10x10 grid and put the nans in coordinates where you think it is 
    misbehaving then you can more simply correct the mistakes ()
    
    im: 2d Numpy array
        image array with nans
    
    returns:
    --------
        None (mutates im in place)
    """
    kernel = 3
    not_numbers = cp.array(cp.where(cp.isnan(im)))
    nans = cp.where(~cp.isfinite(im))
    nx,ny,nz = im.shape
    k = kernel//2
    for x,y,z in zip(nans[0],nans[1],nans[2]):
        if (y == 0) and (x == 0):
            print("top left corner")
            slice_ = im[:k+1,:k+1,z]
        elif (y == 0) and (x > 0) and (x < nx-1):
            print("top row")
            slice_ = im[x-k:x+k+1,:k+1,z]
        elif (y == 0) and (x == nx-1):
            print("top corner")
            slice_ = im[nx-k-1:,:k+1,z]
        elif (y > 0) and (y < ny-1) and (x == 0):
            print("left edge")
            slice_ = im[:k+1,y-k:y+k+1,z]
        elif (y > 0) and (y < ny-1) and (x == nx-1):
            print("right edge")
            slice_ = im[nx-k-1:,y-k:y+k+1,z]
        elif (y == ny-1) and (x > 0) and (x < nx-1):
            print("bottom row")
            slice_ = im[x-k:x+k+1,ny-k-1:,z]
        elif (y == ny-1) and (x == 0):
            print("bottom left corner?")
            slice_ = im[:k+1,ny-k-1:,z]
        elif (y == ny-1) and (x == nx-1):
            print('Bottom right corner')
            slice_ = im[-k-1:,-k-1:,z]
        elif (y < ny-1) and ( y > 0) and (x < nx-1) and (x > 0):
            print('Interior')
            slice_ = im[x-k:x+k+1,y-k:y+k+1,z]
        med = cp.nanmedian(slice_)
        im[x,y,z] = med
        print(slice_)
        print(med)
        print(im[x,y,z])
        del med,slice_
    # }}}

if __name__=="__main__":
    pass

