from tqdm import tqdm
import cupy as cp
from cupyx.scipy.ndimage import median_filter
import logging
import numpy as np

def GPU_curry(  function ,
                arr : np.array,
                axis : int = 0 ,
                batch_size : int = 20
                ) -> None:
    """
    This is a generic template for dispatching a function over a GPU array that
    won't fit on the GPU ram
    
    Parameters:
    -----------
        function : 
            function that takes the array as an argument (lambda)
        arr : np.array
            array on which to perform operations
        axis : int
            axisis along which to subdivide the array (this is the dimension
            batch_size slices)
        batch_size : int
            size of batches for GPU to process

    Returns:
    --------
        None : operates in-place

    Example Usage:
        
        arr = np.ones(100**3).reshape(100,100,100)
        function = lambda j : cp.array(j)+1
        GPU_curry(function, arr, axis = 0, batch_size = 20

    """
    assert len(arr.shape) == 3, "Must be 3D array"
    nx,ny,nz = arr.shape
    slice_x_ = lambda j: slice(0,nx,1)
    slice_y_ = lambda j: slice(0,ny,1)
    slice_z_ = lambda j: slice(0,nz,1)
    slice_x_rem = slice_x_(0)
    slice_y_rem = slice_y_(0)
    slice_z_rem = slice_z_(0)
    slice_batch = lambda j : slice(j*batch_size,(j+1)*batch_size,1)
    if axis == 0:
        slice_x_ = slice_batch
        remainder = nx % batch_size
        slice_x_rem = slice(nx-remainder,nx,1)
        iterator = range(nx//batch_size)
    elif axis == 1:
        slice_y_ = slice_batch
        remainder = ny % batch_size
        slice_y_rem = slice(ny-remainder,ny,1)
        iterator = range(ny//batch_size)
    elif axis == 2:
        slice_z_ = slice_batch
        remainder = nz % batch_size
        slice_z_rem = slice(nz-remainder,nz,1)
        iterator = range(nz//batch_size)
        
    for j in tqdm(iterator):
        slice_x = slice_x_(j)
        slice_y = slice_y_(j)
        slice_z = slice_z_(j)
        arr[slice_x,slice_y,slice_z] = cp.asnumpy(function(cp.array(arr[slice_x,slice_y,slice_z])))
    if remainder > 0:
        arr[slice_x_rem,slice_y_rem,slice_z_rem] = cp.asnumpy(function(cp.array(arr[slice_x_rem,slice_y_rem,slice_z_rem])))

def z_median(   arr: np.array,
                median_size: int = 11
                ) -> np.array:
    """
    Apply median filter in z-direction for de-noising a volume...

    Don't execute this function in-place since it is sliding over the array it
    at least needs a buffer to execute on that is separate from the full array 

    """
    logging.warning("-"*80)
    logging.warning("THIS MEDIAN FILTER IS DEPRECATED AND SLOW DON'T USE")
    logging.warning("-"*80)
    #return None
    nz,nx,ny = arr.shape
    assert median_size % 2 == 1, "median_size must be odd"
    lower_bound = median_size//2
    out = np.zeros([nz-median_size+1,nx,ny])
    #print("output_shape = ",out.shape)
    tqdm_z_med = tqdm(enumerate(range(lower_bound,nz-lower_bound)), 
                                            desc = "applying z-median filter")
    for i,j in tqdm_z_med:
        z0 = j-lower_bound
        z1 = j+lower_bound+1
        #print(f"z0 = {z0}, z1 = {z1}, diff = {z1-z0}")
        slice_ = slice(z0,z1)
        out[i] = cp.median(cp.array(arr[slice_]), axis = 0).get()
        #out[i] = np.median(np.array(arr[slice_]), axis = 0)
    return out

def z_median_GPU(   input_arr: np.array,
                    batch_size: int = 100,
                    median_size: int = 11
                    ) -> np.array:
    """
    Z median filter (0th axis) -> does not operate in-place since it is
    modifying values as it slides
    it has three slices:
        1) input array slice (slice_input)
        2) output array slice (filtered array)
        3) inner slice (slice of the input array that is not corrupted by the
            edges)
    
    Args:
    -----
        input_arr: np.array
        batch_size: int
            size of batches to execute on GPU
        median_size: int

    Returns:
    --------
        output_arr: np.array
            filtered array with z shape lowered by median_size+1
    """
    med_kernel = (median_size,1,1)
    nz_,nx,ny = input_arr.shape
    # output array has half median chopped off either side
    nz = nz_-median_size+1
    output_arr = np.zeros([nz,nx,ny], dtype = np.float32)
    lower_bound = median_size//2
    assert median_size % 2 == 1, "median_size must be odd"
    
    n_batch = nz // batch_size
    remainder = nz % batch_size
    # this slice gets taken out of the median you have calculated
    slice_inner = slice(lower_bound,batch_size+lower_bound)
    for j in tqdm(range(n_batch)):
        slice_input = slice(j*batch_size,(j+1)*batch_size+lower_bound*2)
        slice_output = slice(j*batch_size,(j+1)*batch_size)
        temp_median = median_filter(cp.array(input_arr[slice_input]),med_kernel)
        output_arr[slice_output] = temp_median[slice_inner].get()

    if remainder > 0:
        slice_input = slice(nz_-remainder-lower_bound*2,nz_)
        slice_output = slice(nz-remainder,nz)
        slice_inner = slice(lower_bound,remainder+lower_bound)
        temp_median = median_filter(cp.array(input_arr[slice_input]),med_kernel)
        output_arr[slice_output] = temp_median[slice_inner].get()
    return output_arr

def median_GPU_batch( input_arr: np.array,
                median_: tuple,
                batch_size: int = 100,
                ) -> np.array:
    """
    For applying arbitrary median shape with the GPU it has three slices:
        1) input array slice (slice_input)
        2) output array slice (filtered array)
        3) inner slice (slice of the input array that is not corrupted by the
            edges)
    
    Args:
    -----
        input_arr: np.array
        median_: tuple (of ints)
            median kernel in each direction (z,x,y)
        batch_size: int
            size of batches to execute on GPU

    Returns:
    --------
        output_arr: np.array
            filtered array with:
                - z shape lowered by median_[0]+1
                - x shape lowered by median_[1]+1
                - y shape lowered by median_[2]+1
    """
    assert median_[0] % 2 == 1, "median_size must be odd"
    assert median_[1] % 2 == 1, "median_size must be odd"
    assert median_[2] % 2 == 1, "median_size must be odd"

    med_kernel = median_
    nz_,nx_,ny_ = input_arr.shape
    # output array has half median chopped off either side
    nz = nz_ - median_[0]+1
    nx = nx_ - median_[1]+1
    ny = ny_ - median_[2]+1
    output_arr = np.zeros([nz,nx,ny], dtype = np.float32)

    z_bound = median_[0] // 2
    x_bound = median_[1] // 2
    y_bound = median_[2] // 2
    
    n_batch = nz // batch_size
    remainder = nz % batch_size
    # These slices get taken from the median you have calculated, if edge
    # effects are small these slices could be the whole length of nx and ny
    slice_z_inner = slice(z_bound,batch_size+z_bound)
    slice_x_inner = slice(x_bound,nx+x_bound)
    slice_y_inner = slice(y_bound,ny+y_bound)
    for j in tqdm(range(n_batch)):
        slice_input = slice(j*batch_size,(j+1)*batch_size+z_bound*2)
        slice_output = slice(j*batch_size,(j+1)*batch_size)
        temp_median = median_filter(cp.array(input_arr[slice_input]),med_kernel)
        output_arr[slice_output] = \
                temp_median[slice_z_inner,slice_x_inner,slice_y_inner].get()

    if remainder > 0:
        slice_input = slice(nz_-remainder-z_bound*2,nz_)
        slice_output = slice(nz-remainder,nz)
        slice_z_inner = slice(z_bound,remainder+z_bound)
        temp_median = median_filter(cp.array(input_arr[slice_input]),med_kernel)
        output_arr[slice_output] = \
                temp_median[slice_z_inner,slice_x_inner,slice_y_inner].get()
    return output_arr



def test(N : int = 1000) -> None:
    arr = np.ones(N**3).reshape(N,N,N)
    fun = lambda j : cp.array(j)+1
    GPU_curry(fun,arr)

if __name__ == "__main__":
    test()

