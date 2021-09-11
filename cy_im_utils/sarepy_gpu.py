#------------------------------------------------------------------------------
#
#                       SAREPY GPU FUNCTIONS
#
#------------------------------------------------------------------------------
from cupyx.scipy.ndimage import gaussian_filter, median_filter as median_filter_GPU
from numba import cuda
import cupy as cp


def remove_stripe_based_normalization(sinogram, sigma, in_place = False): # {{{
    """
    This is from SAREPY
    
    I removed the num_chunks since you process these in batches?
    """
    n_proj,n_sino,detector_width = sinogram.shape
    if not in_place:
        sinogram = cp.copy(sinogram)
    else:
        print("Mutating sinograms in place")
    listindex = cp.array_split(cp.arange(n_proj),1)
    for pos in listindex:
        bindex = cp.asnumpy(pos[0])
        eindex = cp.asnumpy(pos[-1] + 1)
        listmean = cp.mean(sinogram[bindex:eindex], axis = 0)
        list_filtered = gaussian_filter(listmean,sigma)
        listcoe = list_filtered - listmean
        matcoe = cp.tile(listcoe, (eindex - bindex, 1)).reshape(n_proj,n_sino,detector_width)
        sinogram[bindex:eindex, :] = sinogram[bindex:eindex,:] + matcoe
    return sinogram
    # }}}
@cuda.jit('void(float32[:,:,:],int64[:,:,:],float32[:,:,:])')
def invert_sort_GPU(input_arr,index_arr,output_arr):# {{{
    """
    This function reverses? (inverts?) the sort after the median
    
    parameters:
    -----------
    input_arr: (float32) 3 dimensional cp.array
        the sorted/filtered sinogram
    index_arr: (int64) 3 dimensional cp.array
        the argsort output of sorting the sinogram
    output_arr: (float32) 3 dimensional cp.array
        the variable for holding the reverse sorted sinogram
    
    """
    n_proj,n_sino,detector_width = input_arr.shape
    i,j,k = cuda.grid(3)
    if i < n_proj and j < n_sino and k < detector_width:
        proj_index = int(index_arr[i,j,k])
        val = input_arr[i,j,k]
        output_arr[proj_index,j,k] = val
# }}}
def remove_stripe_based_sorting(sinogram, size, dim = 1, in_place = False, threads_per_block = (8,8,8)): # {{{
    """
    This is from SAREPY but modified a little bit since the syntax becomes a little unwieldy with 3D arrays.
    Greatest inefficiency, I assume, comes from calling argsort and sort independently, Vo used a nexted list 
    comprehension that achieved this in a compact syntax, but I was unable to figure it out for these data 
    so I am calling them both then using a *numba.cuda.jit* kernel to invert the sort. Note the behavior of
    the cuda kernel is unpredictable if the data types or not enforced float32,int64,float32. I also changed
    the order of the input indices so that it is consistent with ASTRA, this can obviously be changed, but 
    be careful and test the output to make sure it still works correctly
    
    Parameters:
    -----------
    sinogram: cupy 3D array 
        input sinogram stack with index order (projection,sinogram,detector_width)
    size: int
        size of median filter kernel
    dim: int
        dimensionality of the median filter 1 -> 1D median, 2 -> 2D median
    threads_per_block: tuple of 3 integers
        block size for CUDA jit -> Product of these 3 integers must be less than 1024 with my GPU! 
    
    """
    n_proj,n_sino,detector_width = sinogram.shape
    list_index = cp.arange(0.0,n_proj,1.0)
    mat_index = cp.tile(cp.tile(list_index, (detector_width,1)),(n_sino,1)).reshape(sinogram.shape)
    #------------------------------------------------------------------------------
    # THIS REQUIRES TRAVERSING THE ARRAYS TWICE, BUT I DON'T UNDERSTAND THE ORIGINAL 
    # SYNTAX WELL ENOUGH TO REPLICATE IT WITH 3D ARRAYS
    #------------------------------------------------------------------------------
    mat_argsort = cp.argsort(sinogram, axis = 0).astype(cp.int64)
    mat_sort = cp.sort(sinogram,axis = 0).astype(cp.float32)
    
    # Apply Median Filter
    if dim == 2:
        mat_sort = median_filter_GPU(mat_sort, (size,1,size))
    elif dim == 1:
        mat_sort = median_filter_GPU(mat_sort, (1,1,size))

    # Invert Sort
    if not in_place:
        mat_sort_back = cp.zeros_like(sinogram, dtype = cp.float32)
    else:
        mat_sort_back = sinogram
    blockspergrid_x = int((n_proj+threads_per_block[1]-1)/threads_per_block[1])
    blockspergrid_y = int((n_sino+threads_per_block[2]-1)/threads_per_block[2])
    blockspergrid_z = int((detector_width+threads_per_block[0]-1)/threads_per_block[0])
    blocks = (blockspergrid_x,blockspergrid_y,blockspergrid_z)
    invert_sort_GPU[blocks,threads_per_block](
                                       mat_sort,
                                       mat_argsort,
                                       mat_sort_back,
                                      )

    return mat_sort_back
# }}}
def remove_stripe_based_filtering_sorting(sinogram, window, size, dim = 1, in_place = True): # {{{
    """
    UNTESTED !!!!!!!!!!!!!!!!!!!!!!!!!!!!

    input shape is non-trivial for this
    
    pass the gaussian window instead of calculating it inside the function (for now, probably simple to implement)
    
    """
    pad = min(150, int(0.1*sinogram.shape[1]))
    print("check input shape for pad!!")
    if not in_place:
        sinogram = cp.copy(sinogram)
    else:
        print("Mutating sinograms in place")
    sinogram = cp.transpose(sinogram,(0,2,1))
    sino_pad = cp.pad(sinogram,((0,0),(pad,pad)), mode = 'reflect')
    _,_,ncol = sino_pad.shape
    list_sign = cp.power(-1, cp.arange(ncol))
    # in place?
    sino_smooth = cp.copy(sinogram)
    for i, sino_1d in enumerate(sino_pad):
        sino_smooth[i] = cp.real(cp.fft.ifft(cp.fft.fft(sino_1d * list_sign)*window)* list_sign)[pad:ncol-pad]
    sino_sharp = sinogram - sino_smooth
    sino_smooth_cor = cp.transpose(remove_stripe_based_sorting(cp.transpose(sino_smooth),(0,2,1)))
    return cp.transpose(sino_smooth_cor + sino_sharp)
# }}}

if __name__=="__main__":
    pass
