"""

NEED TO ADD SCIPY LICENSE FOR SSIM AND PSNR!!

"""
from cupyx.scipy.ndimage.filters import uniform_filter
from cupyx.scipy.signal import convolve
from skimage.filters import threshold_multiotsu
import cupy as cp
import cupyx.scipy.ndimage as ndimage_GPU
import numpy as np
import numpy as np
import logging

def crete_blur_metric_GPU(  im_stack: cp.array,
                            h: int = 9
                            ) -> np.array:
    """
    Parameters:
      im_stack: cp.array
        stack for CRETE metric should have shape:
            index 0 = image
            index 1 = row or col
            index 2 = row or col

      h : int
        size of filter (9 in the manuscript)
    """
    assert im_stack.ndim == 3, "This function takes 3D arrays (image stacks)"

    h_v = (1/h)*cp.ones(h).reshape(1,1,h)                       # equation 1
    h_h = cp.transpose(h_v,(0,2,1))                             # equation 1

    B_ver = ndimage_GPU.convolve(im_stack, h_v)                 # equation 1
    B_hor = ndimage_GPU.convolve(im_stack, h_h)                 # equation 1

    D_F_ver = cp.abs(cp.diff(im_stack,1,2))                     # equation 2
    D_F_hor = cp.abs(cp.diff(im_stack,1,1))                     # equation 2
                                                                             
    D_B_ver = cp.abs(cp.diff(B_ver,1,2))                        # equation 2
    D_B_hor = cp.abs(cp.diff(B_hor,1,1))                        # equation 2

    V_ver = cp.maximum(0,D_F_ver[:,1:-1,1:-1]-D_B_ver[:,1:-1,1:-1])
    V_hor = cp.maximum(0,D_F_hor[:,1:-1,1:-1]-D_B_hor[:,1:-1,1:-1])

    s_F_ver = cp.sum(D_F_ver[:,1:-1,1:-1], axis = (1,2))        # equation 4
    s_F_hor = cp.sum(D_F_hor[:,1:-1,1:-1], axis = (1,2))        # equation 4

    s_V_ver = cp.sum(V_ver[:,1:-1,1:-1], axis = (1,2))          # equation 4
    s_V_hor = cp.sum(V_hor[:,1:-1,1:-1], axis = (1,2))          # equation 4

    b_F_ver = (s_F_ver-s_V_ver)/s_F_ver                         # equation 5
    b_F_hor = (s_F_hor-s_V_hor)/s_F_hor                         # equation 5

    blur = np.max([b_F_ver.get(),b_F_hor.get()], axis = 0)      # equation 6
    return blur

def peak_signal_noise_ratio_GPU(image_ref,
                                image_test,
                                data_range = None):
    """
    """
    assert image_ref.dtype == image_test.dtype == 'float32', "must use 32 bit float"
    if data_range is None:
        dmin, dmax = -1,1
        true_min = cp.min(image_ref, axis = (1,2))
        true_max = cp.max(image_ref, axis = (1,2))

        if (true_max > dmax).any() or (true_min < dmin).any():
            raise ValueError(
              "reference image has intensity values outside the range expeted"
              "for its data type. Please manually speify the data_range"
            )
        if (true_min >= 0).any():
            data_range = dmax
        else:
            data_range = dmax-dmin
    err = mean_squared_error_GPU(image_ref,image_test)
    return 10*cp.log10((data_range**2)/err) 

def structural_similarity_GPU(  im1: cp.array,
                im2: cp.array,
                win_size = None,
                data_range = None,
                full = False,
                **kwargs):
    """

    THIS IS TRANSLATED FROM skimage.metrics.structural_similarity source code, I 
    removed some of the functionality (Gaussian_weightsa)

    This is inteded to run a batch of images, not to calculate the ssim of all the 
    images together

    Parameters:
    -----------
    im1 : cp.array
    array 1 for comaparison, has shape:
      index 0 -> number of images
      index 1 -> image rows
      index 2 -> image cols
    im2 : cp.array
    array 2 for comparison. same shape as im1

    win_size : int or None, optional
    Side-length of the sliding window used in comparison. Must be an odd value. 

    data_range: float, optional
    if dtype is not float32 this needs to be supplied

    """
    assert im1.shape == im2.shape, "Image stacks must have same shape"
    assert len(im1.shape) == 3, "Image stacks must have 3 dimensions"
    assert im1.dtype == cp.float32, "Only implemented for float32 currently"
    K1 = kwargs.pop('K1',0.01)
    K2 = kwargs.pop('K2',0.03)
    use_sample_covariance = kwargs.pop('use_sample_covariance', True)
    if win_size is None:
        win_size = 7
    if cp.any((cp.asarray(im1.shape[1:]) - win_size) < 0):
        raise ValueError("win_size exceeds image extent")
    if not (win_size % 2 == 1):
        raise ValueError("Window size must be odd.")
    if data_range is None:
        assert im1.dtype == cp.float32, ("Only implemented for cp.float32."
                                                f" dtype = {im1.dtype}")
        data_range = 2
    ndim = im1.ndim

    filter_func = uniform_filter
    filter_args = {'size': (1,win_size,win_size)}
    #filter_args = {'size': win_size}

    NP = win_size ** ndim

    if use_sample_covariance:
        cov_norm = NP / (NP-1)
    else:
        cov_norm = 1.0

    # Compute (weighted) means
    ux = filter_func(im1, **filter_args)
    uy = filter_func(im2, **filter_args)

    # Compute (weighted) variances and covariances
    uxx = filter_func(im1 * im1, **filter_args)
    uyy = filter_func(im2 * im2, **filter_args)
    uxy = filter_func(im1 * im2, **filter_args)
    vx = cov_norm * (uxx - ux*ux)
    vy = cov_norm * (uyy - uy*uy)
    vxy = cov_norm * (uxy - ux*uy)

    R = data_range
    C1 = (K1 * R) ** 2
    C2 = (K2 * R) ** 2

    A1 = 2 * ux * uy + C1
    A2 = 2 * vxy + C2
    B1 = ux **2 + uy **2 + C1
    B2 = vx + vy + C2

    D = B1 * B2
    S = (A1 * A2) / D
    pad = (win_size - 1) // 2

    _,nx,ny = im1.shape
    slice_x = slice(pad,nx-pad)
    slice_y = slice(pad,ny-pad)
    mssim = S[:,slice_x,slice_y].mean(axis = (1,2))

    return mssim

def mean_squared_error_GPU(imstack_0, imstack_1):
    """
    Ultra basic
    """
    return cp.mean((imstack_0-imstack_1)**2, axis = (1,2))

def iou_calculate(  im0: np.array,
                    im1: np.array,
                    classes: int = 3,
                    nbins: int = 256
                    ) -> np.array:
    """
    Very Coarse Unsupervised IOU, skimage.filters.threshold_multiotsu is pretty
    fast and the code is pretty complex so I am not going to try to do a GPU
    conversion
    """
    n_im,nx,ny = im0.shape
    iou = np.zeros(n_im)
    for i in range(n_im):
        im0_h = im0[i]
        im1_h = im1[i]
        if (im0_h.max() == im0_h.min()) or (im1_h.max() == im1_h.min()):
            iou[i] = np.nan
            logging.warning(f"Blank Image IOU -> returning index {i} = np.nan")
        else:
            thresh_0 = threshold_multiotsu(im0_h, classes = classes, nbins = nbins)
            regions_0 = np.digitize(im0_h, bins = thresh_0)
            thresh_1 = threshold_multiotsu(im1_h, classes = classes, nbins = nbins)
            regions_1 = np.digitize(im1_h, bins = thresh_1)
            diff = regions_0-regions_1
            iou[i] = np.sum(diff == 0)/(nx*ny)
    return iou

def laplacian_blur_GPU(im_stack: cp.array) -> cp.array:
    """
    Calculate variance of Laplacian (3d Cupy arrays)
    """
    kernel = cp.array([[
                        [0,1,0],
                        [1,-4,1],
                        [0,1,0]
                        ]], dtype = cp.float32)
    conv = ndimage_GPU.convolve(im_stack,kernel)
    #print("cupy conv shape = ",conv.shape)
    #print("cupy kern shape = ",kernel.shape)
    #print("--->\n",conv)
    return cp.std(conv, axis = (1,2))**2
