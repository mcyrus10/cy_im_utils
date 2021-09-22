#------------------------------------------------------------------------------
#
#                       SAREPY GPU FUNCTIONS
#
#------------------------------------------------------------------------------
from cupyx.scipy.ndimage import gaussian_filter, median_filter as median_filter_GPU, binary_dilation as binary_dilation_GPU,uniform_filter1d as uniform_filter1d_gpu
from numba import cuda
import cupy as cp
import numpy as np


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
@cuda.jit('void(float32[:,:,:],int32[:,:,:],float32[:,:,:])')
def invert_sort_GPU(input_arr,index_arr,output_arr):# {{{
    """
    This function reverses? (inverts?) the sort after the median
    
    parameters:
    -----------
    input_arr: (float32) 3 dimensional cp.array
        the sorted/filtered sinogram
    index_arr: (uint32) 3 dimensional cp.array
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
@cuda.jit('void(float32[:,:,:],int32[:,:,:],float32[:,:,:])')
def invert_sort_GPU_2(input_arr,index_arr,output_arr):# {{{
    """
    
    THIS ONE IS DIFFERENT FROM invert_sort_GPU BECAUSE IT SEEDS THE SECOND
    INDEX OF OUTPUT_ARR NOT THE FIRST INDEX. YOU COULD CHANGE THE SHAPES TO BE
    CONSISTENT, OR JUST DO THIS

    This function reverses? (inverts?) the sort after the median

    parameters:
    -----------
    input_arr: (float32) 3 dimensional cp.array
        the sorted/filtered sinogram
    index_arr: (uint32) 3 dimensional cp.array
        the argsort output of sorting the sinogram
    output_arr: (float32) 3 dimensional cp.array
        the variable for holding the reverse sorted sinogram
    
    """
    detector_width, n_proj ,n_sino = input_arr.shape
    i,j,k = cuda.grid(3)
    if i < detector_width and j < n_proj and k < n_sino:
        proj_index = int(index_arr[i,j,k])
        val = input_arr[i,j,k]
        output_arr[i,proj_index,k] = val
# }}}
def remove_stripe_based_sorting_GPU(sinogram, size, dim = 1, in_place = False, threads_per_block = (8,8,8)): # {{{
    """
    This is from SAREPY but modified a little bit since the syntax becomes a
    little unwieldy with 3D arrays.  Greatest inefficiency, I assume, comes
    from calling argsort and sort independently, Vo used a nexted list
    comprehension that achieved this in a compact syntax, but I was unable to
    figure it out for these data so I am calling them both then using a
    *numba.cuda.jit* kernel to invert the sort. Note the behavior of the cuda
    kernel is unpredictable if the data types or not enforced
    float32,uint32,float32. I also changed the order of the input indices so
    that it is consistent with ASTRA, this can obviously be changed, but be
    careful and test the output to make sure it still works correctly
    
    Parameters:
    -----------
    sinogram: cupy 3D array 
        input sinogram stack with index order (projection,sinogram,detector_width)
    size: int
        size of median filter kernel
    dim: int
        dimensionality of the median filter 1 -> 1D median, 2 -> 2D median
    threads_per_block: tuple of 3 integers
        block size for CUDA jit -> Product of these 3 integers must be less
        than 1024 with my GPU! 
    
    """
    n_proj,n_sino,detector_width = sinogram.shape
    list_index = cp.arange(0.0,n_proj,1.0)
    mat_index = cp.tile(cp.tile(list_index, (detector_width,1)),(n_sino,1)).reshape(sinogram.shape)
    #------------------------------------------------------------------------------
    # THIS REQUIRES TRAVERSING THE ARRAYS TWICE, BUT I DON'T UNDERSTAND THE ORIGINAL 
    # SYNTAX WELL ENOUGH TO REPLICATE IT WITH 3D ARRAYS
    #------------------------------------------------------------------------------
    mat_argsort = cp.argsort(sinogram, axis = 0).astype(cp.uint32)
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
def remove_large_stripe_GPU(sinogram, snr, size, drop_ratio = 0.1, norm = True, threads_per_block = (8,8,8)): # {{{
    """
    Adapted from Vo et al.
    Parameters:
    -----------
    sinogram: 3D cupy array
        input sinogram stack
    
    snr: float
        Ratio to segment stripes from background noise

    size: int
        window size of the median filter

    drop_ratio: float
        Ratio of pixels to be dropped, which is used to reduce the false
        detection of stripes

    norm: bool
        Apply normalization if true

    Returns:
    --------
    3d cupy array
        filterd sinogram

    """
    sinogram = cp.copy(sinogram).astype(cp.float32)
    drop_ratio = np.clip(drop_ratio, 0.0, 0.8)
    sino_sort = cp.sort(sinogram, axis = 0)
    n_row,n_sino,n_col = sinogram.shape
    n_drop = int(0.5 * drop_ratio * n_row)
    sino_smooth = median_filter_GPU(sino_sort, (1,1,size))
    list1 = cp.mean(sino_sort[n_drop:n_row-n_drop,:,:], axis = 0)
    list2 = cp.mean(sino_smooth[n_drop:n_row-n_drop,:,:], axis = 0)
    list_fact = list1/list2
    list_fact[~cp.isfinite(list_fact)] = 1
    list_mask = detect_stripe_GPU(list_fact,snr)
    # NOT IDEAL, BUT GO WITH IT FOR NOW
    list_mask = [cp.array(binary_dilation_GPU(list_mask[i], iterations = 1), dtype = cp.float32) for i in range(n_sino)]
    list_mask = cp.vstack(list_mask)
    mat_fact = cp.tile(list_fact, (n_row,1)).reshape(sinogram.shape)
    if norm:
        sinogram = sinogram/mat_fact

    transpose_shape = (2,0,1)
    sino_tran = cp.transpose(sinogram,transpose_shape).astype(cp.float32)
    sino_tran_argsort = cp.argsort(sino_tran, axis = 1).astype(cp.uint32)
    sino_cor = cp.empty_like(sino_tran_argsort, dtype = cp.float32)
    sino_smooth_tran = cp.transpose(sino_smooth, transpose_shape).astype(cp.float32)
    detector_width,n_sino,n_projections =  sino_smooth_tran.shape
    
    threads_per_block = (8,8,8)
    blockspergrid_x = int((detector_width+threads_per_block[0]-1)/threads_per_block[0])
    blockspergrid_y = int((n_sino+threads_per_block[1]-1)/threads_per_block[1])
    blockspergrid_z = int((n_projections+threads_per_block[2]-1)/threads_per_block[2])
    blocks = (blockspergrid_x,blockspergrid_y,blockspergrid_z)
    invert_sort_GPU_2[blocks, threads_per_block](
                                                sino_smooth_tran,
                                                sino_tran_argsort,
                                                sino_cor
                                                )
    sino_cor = cp.transpose(sino_cor,(1,2,0))
    listx_miss = cp.where(list_mask>0.0)
    
    # Make this into a CUDA Kernel? <-- THIS LOOP EXECUTES SUPER QUICKLY, I WOULDN'T WORRY ABOUT THIS
    for i in range(n_sino):
        idxs = listx_miss[1][listx_miss[0]==i]
        sinogram[:,i, idxs] = sino_cor[:,i,idxs]

    return sinogram
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
def detect_stripe_GPU(list_data,snr):# {{{
    """
    .... deeper down the rabbit hole 
    (slightly slower than cpu version, probably not worth worrying about)
    
    Parameters:
    -----------
    list_data: cp array of shape n_sinograms,detector_width (n_point)
        This list contains the 
    """
    n_sino,n_point = list_data.shape
    list_sort = cp.sort(list_data,axis = 1)
    listx = cp.arange(0,n_point,1.0)
    ndrop = cp.int16(0.25*n_point)
    slope,intercept = cp.polyfit(listx[ndrop:-ndrop-1], list_sort[:,ndrop:-ndrop-1].T,1)
    y_end = intercept + slope * listx[-1]
    noise_level = cp.abs(y_end - intercept)
    noise_level = cp.clip(noise_level, 1e-6, None)
    val1 = cp.abs(list_sort[:,-1] - y_end) / noise_level
    val2 = cp.abs(intercept - list_sort[:,0]) / noise_level
    list_mask = cp.zeros_like(list_sort, dtype = cp.float32)
    v1_bool = val1>=snr
    v2_bool = val2>=snr
    # YOU CAN DO THIS WITH A CUDA.JIT KERNEL IF IT REALLY IS A BOTTLENECK
    for i in range(n_sino):
        if v1_bool[i]:
            upper_thresh = y_end[i] + noise_level[i] * snr * 0.5
            list_mask[i,list_data[i] > upper_thresh] = 1.0 
        if v2_bool[i]:
            lower_thresh = intercept[i] - noise_level[i] * snr * 0.5
            list_mask[i,list_data[i] <= lower_thresh] = 1.0
    return list_mask
    # }}}
def nd_interp2d(input_arr,live_rows,dead_rows): #{{{
    """
    POSSIBLY CONVERT TO CUDA JIT IF THIS GIVES YOU A PROBLEM WITH BOTTLENECKING

    This function mimics the behaviour of interpolate.interp2d for the specific
    case where y is constant

    Parameters:
    input_arr: 3d cupy array
        input sinogram stack with the shape n_projections, n_sinograms,
        detector_width
    
    live_rows: 2d cupy array
        output of cp.where to determine which rows are 'alive'; detect_stripe
        is used to determine which rows 'live (0)' and which are 'dead (1)'

    live_rows: 2d cupy array
        output of cp.where to determine which rows are 'dead' (see above)

    Returns:
        None -> writes input_arr in place

    """
    _, n_sino, _ = input_arr.shape
    for sino in range(n_sino):
        dr = dead_rows[1][dead_rows[0]==sino]
        lr = live_rows[1][live_rows[0]==sino]
        for c in dr:
            below = lr[lr<c][-1]
            above = lr[lr>c][0]
            input_arr[:,sino,c] = input_arr[:,sino,below] + (c-below) *\
                    (input_arr[:,sino,above]-input_arr[:,sino,below])/(above-below)
    # }}}
def remove_unresponsive_and_fluctuating_stripe_GPU(sinogram,snr,size,residual = False): # {{{
    """
    """
    n_proj,n_sino,detector_width = sinogram.shape
    # Vo used the function np.apply_along_axis when calling uniform_filter1d, but this is redundant
    # as uniform_filter1d takes an axis as an argument. THIS IS ALSO MUCH SLOWER THAN JUST CALLING
    # uniform_filter1d_gpu!
    #sino_smooth = cp.apply_along_axis(uniform_filter1d_gpu, 0, sinogram, 10)     
    sino_smooth = uniform_filter1d_gpu(sinogram, 10, axis = 0)
    list_diff = cp.sum(cp.abs(sinogram-sino_smooth), axis = 0)
    list_diff_bck = median_filter_GPU(list_diff, (1,size))
    list_fact = list_diff/list_diff_bck
    list_fact[~cp.isfinite(list_fact)] = 1                     #<-------------------------- Hack for getting around true divide?
    list_mask = detect_stripe_GPU(list_fact,snr)
    list_mask = cp.array([binary_dilation_GPU(list_mask[i], iterations = 1) for i in range(2)], dtype = cp.float32)
    list_mask[:,0:2] = 0.0
    list_mask[:,-2:] = 0.0
    listx = cp.array(cp.where(list_mask < 1.0))
    listx_miss = cp.array(cp.where(list_mask == 1.0))

    output_mat = cp.copy(sinogram)
    nd_interp2d(output_mat,listx,listx_miss)
    return output_mat
    # }}}
def remove_all_stripe_GPU(sinogram,snr,la_size,sm_size,drop_ratio = 0.1,norm = True, dim = 1):# {{{
    sinogram = remove_unresponsive_and_fluctuating_stripe_GPU(sinogram, snr, la_size)
    sinogram = remove_large_stripe_GPU(sinogram, snr, la_size, drop_ratio, norm)
    return remove_stripe_based_sorting_GPU(sinogram,sm_size, dim = dim)
    # }}}

if __name__=="__main__":
    pass
