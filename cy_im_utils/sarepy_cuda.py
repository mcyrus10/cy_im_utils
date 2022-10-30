#============================================================================
# Copyright (c) 2018 Diamond Light Source Ltd. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#============================================================================
# Author: Nghia T. Vo
# E-mail: nghia.vo@diamond.ac.uk
# Description: Original implementation of stripe artifact removal methods, 
# Nghia T. Vo, Robert C. Atwood, and Michael Drakopoulos, "Superior
# techniques for eliminating ring artifacts in X-ray micro-tomography," Optics
# Express 26, 28396-28412 (2018).
# https://doi.org/10.1364/OE.26.028396
#============================================================================
# Translation to Python GPU (Cuda-enabled )
# Translator: M. Cyrus Daugherty
# E-mail: michael.daugherty@nist.gov
# Description: GPU adaptation of Vo et al.'s SAREPY
# Required Libraries: numba, cupy
#------------------------------------------------------------------------------
#
#                       SAREPY GPU FUNCTIONS
#
#------------------------------------------------------------------------------
from cupyx.scipy.ndimage import gaussian_filter,median_filter,binary_dilation,uniform_filter1d
from numba import cuda
import cupy as cp

@cuda.jit('void(float32[:,:,:],int32[:,:,:],float32[:,:,:])')
def invert_sort_GPU(input_arr : cp.array,
                    index_arr : cp.array,
                    output_arr : cp.array
                    ) -> None:
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

def remove_stripe_based_normalization(  sinogram: cp.array,
                                        sigma: float,
                                        in_place: bool = False
                                        ) -> cp.array:
    """
    Still needs testing for robustness
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

def remove_stripe_based_sorting_GPU(sinogram,
                                    size,
                                    dim = 1,
                                    in_place = False,
                                    threads_per_block = (8,8,8)
                                    ) -> cp.array:
    """
    This is from SAREPY but modified a little bit since the syntax becomes a
    little unwieldy with 3D arrays.  Greatest inefficiency, I assume, comes
    from calling argsort and sort independently, Vo used a nested list
    comprehension that achieved this in a compact syntax, but I was unable to
    figure it out for these data so I am calling them both then using a
    *numba.cuda.jit* kernel to invert the sort. Note the behavior of the cuda
    kernel is unpredictable if the data types are not enforced
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
    #---------------------------------------------------------------------------
    # THIS REQUIRES TRAVERSING THE ARRAYS TWICE, BUT I DON'T UNDERSTAND THE
    # ORIGINAL SYNTAX WELL ENOUGH TO REPLICATE IT WITH 3D ARRAYS
    #---------------------------------------------------------------------------
    mat_argsort = cp.argsort(sinogram, axis = 0).astype(cp.uint32)
    mat_sort = cp.sort(sinogram,axis = 0).astype(cp.float32)
    
    # Apply Median Filter
    if dim == 2:
        print("using 2d filter")
        mat_sort = median_filter(mat_sort, (size,1,size))
    elif dim == 1:
        mat_sort = median_filter(mat_sort, (1,1,size))

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

def remove_large_stripe_GPU(sinogram : cp.array,
                            snr : float,
                            size : int,
                            drop_ratio : cp.array = cp.array(0.1),
                            norm : bool = True,
                            threads_per_block : tuple = (8,8,8)
                            ) -> cp.array:
    """

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
    drop_ratio = cp.clip(drop_ratio, 0.0, 0.8)
    sino_sort = cp.sort(sinogram, axis = 0)
    n_row,n_sino,n_col = sinogram.shape
    n_drop = int(0.5 * drop_ratio * n_row)
    sino_smooth = median_filter(sino_sort, (1,1,size))
    list1 = cp.mean(sino_sort[n_drop:n_row-n_drop,:,:], axis = 0)
    list2 = cp.mean(sino_smooth[n_drop:n_row-n_drop,:,:], axis = 0)
    list_fact = list1/list2
    list_fact[~cp.isfinite(list_fact)] = 1
    list_mask = detect_stripe_GPU(list_fact,snr)
    # NOT IDEAL, BUT GO WITH IT FOR NOW
    list_mask = [cp.array(binary_dilation(list_mask[i], iterations = 1), dtype = cp.float32) for i in range(n_sino)]
    list_mask = cp.vstack(list_mask)
    mat_fact = cp.tile(list_fact, (n_row,1)).reshape(sinogram.shape)
    if norm:
        sinogram = sinogram/mat_fact

    sino_argsort = cp.argsort(sinogram, axis = 0).astype(cp.uint32)
    sino_cor = cp.empty_like(sino_argsort, dtype = cp.float32)
    detector_width,n_sino,n_projections =  sino_smooth.shape
    
    threads_per_block = (8,8,8)
    blockspergrid_x = int((detector_width+threads_per_block[0]-1)/threads_per_block[0])
    blockspergrid_y = int((n_sino+threads_per_block[1]-1)/threads_per_block[1])
    blockspergrid_z = int((n_projections+threads_per_block[2]-1)/threads_per_block[2])
    blocks = (blockspergrid_x,blockspergrid_y,blockspergrid_z)
    invert_sort_GPU[blocks, threads_per_block](
                                                sino_smooth,
                                                sino_argsort,
                                                sino_cor
                                                )
    listx_miss = cp.where(list_mask>0.0)
    
    # Make this into a CUDA Kernel? <-- THIS LOOP EXECUTES SUPER QUICKLY, I WOULDN'T WORRY ABOUT THIS
    for i in range(n_sino):
        idxs = listx_miss[1][listx_miss[0]==i]
        sinogram[:,i, idxs] = sino_cor[:,i,idxs]

    return sinogram

def remove_stripe_based_filtering_sorting(  sinogram,
                                            window,
                                            size,
                                            dim = 1,
                                            in_place = True): 
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

def detect_stripe_GPU( list_data : cp.array ,
                        snr : float
                        ) -> cp.array:
    """
    .... deeper down the rabbit hole 
    (slightly slower than cpu version, probably not worth worrying about)
    
    Parameters:
    -----------
    list_data: cp array of shape n_sinograms,detector_width (n_point)
        This list contains the 

    snr : float
        singal-to-noise ratio

    Returns:
        list_mask : cp.array
            list of integers of rows containing stripes
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

def nd_interp2d(input_arr : cp.array,
                live_rows : cp.array,
                dead_rows : cp.array
                ) -> None:
    """
    POSSIBLY CONVERT TO CUDA JIT IF THIS GIVES YOU A PROBLEM WITH BOTTLENECKING

    This function mimics the behaviour of interpolate.interp2d for the specific
    case where y is constant

    Parameters:
    -----------
    input_arr: 3d cupy array
        input sinogram stack with the shape n_projections, n_sinograms,
        detector_width
    
    live_rows: 2d cupy array
        output of cp.where to determine which rows are 'alive'; detect_stripe
        is used to determine which rows 'live (0)' and which are 'dead (1)'

    live_rows: 2d cupy array
        output of cp.where to determine which rows are 'dead' (see above)

    Returns:
    --------
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

def remove_unresponsive_and_fluctuating_stripe_GPU( sinogram,
                                                    snr: float,
                                                    size: int,
                                                    residual: bool = False
                                                    ): 
    """
    """
    n_proj,n_sino,detector_width = sinogram.shape
    # Vo used the function np.apply_along_axis when calling uniform_filter1d,
    # but this is redundant
    # as uniform_filter1d takes an axis as an argument. THIS IS ALSO MUCH
    # SLOWER THAN JUST CALLING
    # uniform_filter1d!
    #sino_smooth = cp.apply_along_axis(uniform_filter1d, 0, sinogram, 10)     
    sino_smooth = uniform_filter1d(sinogram, 10, axis = 0)
    list_diff = cp.sum(cp.abs(sinogram-sino_smooth), axis = 0)
    list_diff_bck = median_filter(list_diff, (1,size))
    list_fact = list_diff/list_diff_bck
    list_fact[~cp.isfinite(list_fact)] = 1                     #<-------- Hack for getting around true divide?
    list_mask = detect_stripe_GPU(list_fact,snr)
    list_mask = cp.array([binary_dilation(list_mask[i], iterations = 1) for i in range(n_sino)], dtype = cp.float32)
    list_mask[:,0:2] = 0.0
    list_mask[:,-2:] = 0.0
    listx = cp.array(cp.where(list_mask < 1.0))
    listx_miss = cp.array(cp.where(list_mask == 1.0))

    output_mat = cp.copy(sinogram)
    nd_interp2d(output_mat,listx,listx_miss)
    return output_mat

def remove_all_stripe_GPU(  sinogram: cp.array,
                            snr: float,
                            la_size: int,
                            sm_size: int,
                            drop_ratio = cp.array(0.1),
                            norm: bool = True,
                            dim: int = 1):
    sinogram = remove_unresponsive_and_fluctuating_stripe_GPU(sinogram, snr, la_size)
    sinogram = remove_large_stripe_GPU(sinogram, snr, la_size, drop_ratio, norm)
    sinogram = remove_stripe_based_sorting_GPU(sinogram,sm_size, dim = dim)
    return sinogram

def test(): 
    import matplotlib.pyplot as plt
    import numpy as np
    _,ax = plt.subplots(1,2)
    data_path = "D:\\Data\\sinogram_binaries\\sinogram_volume_AAA_bottom_25_spacing.p"
    print(f"Loading Data from {data_path}")
    sinograms = np.load(data_path,allow_pickle = True)
    _,n_sino,_ = sinograms.shape
    idx = n_sino//2
    ax[0].imshow(sinograms[:,idx,:])
    sinograms = cp.asarray(sinograms)
    output = remove_all_stripe_GPU(sinograms, snr = 1.5, la_size = 85, sm_size = 10, drop_ratio = cp.array(0.1), norm = True, dim=1)
    ax[1].imshow(cp.asnumpy(output[:,idx,:]))
    ax[0].set_title("Unfiltered")
    ax[1].set_title("remove_all_stripe")
    plt.show()

if __name__ == "__main__":
    test()
