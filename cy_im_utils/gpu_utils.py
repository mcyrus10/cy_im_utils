from tqdm import tqdm
import cupy as cp
import numpy as np

def GPU_curry(function ,arr : np.array, ax : int = 0 ,batch_size : int = 20) -> None:
    """
    This is a generic template for dispatching a function over a GPU array that
    won't fit on the GPU ram
    
    Parameters:
    -----------
        function : 
            function that takes the array as an argument (lambda)
        arr : np.array
            array on which to perform operations
        ax : int
            axis along which to subdivide the array (this is the dimension
            batch_size slices)
        batch_size : int
            size of batches for GPU to process

    Returns:
    --------
        None : operates in-place

    Example Usage:
        
        arr = np.ones(100**3).reshape(100,100,100)
        function = lambda j : cp.array(j)+1
        GPU_curry(function, arr, ax = 0, batch_size = 20

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
    if ax == 0:
        slice_x_ = slice_batch
        remainder = nx % batch_size
        slice_x_rem = slice(nx-remainder,nx,1)
        iterator = range(nx//batch_size)
    elif ax == 1:
        slice_y_ = slice_batch
        remainder = ny % batch_size
        slice_y_rem = slice(ny-remainder,ny,1)
        iterator = range(ny//batch_size)
    elif ax == 2:
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

def test(N : int = 1000) -> None:
    arr = np.ones(N**3).reshape(N,N,N)
    fun = lambda j : cp.array(j)+1
    GPU_curry(fun,arr)

if __name__ == "__main__":
    test()
