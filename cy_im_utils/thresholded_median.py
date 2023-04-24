import cupy as cp
from cupyx.scipy.ndimage import median_filter as median_filter_GPU
from cupyx.scipy.signal import convolve2d as convolve_GPU


def thresh_median_2D_GPU(image: cp.array,
                         kernel_size: int,
                         z_score_thresh: float = 1.0
                         ) -> cp.array:
    """

    KEEP IT SIMPLE STUPID

    each pixel is compared to its local threshold which is the average inside
    the kernel + the standard deviation multiplied by the z_score_thresh.
    Note this only removes values above the thresh it is not symmetric with
    respect to the average

    Arguments:
    ----------
        image: cp.array - image to be filtered
        kernel_size: int - size of filter kernel
        z_score_thresh: float - threshold is calculated as average + standard
                                deviation multiplied by this factor

    Returns:
    --------
        filtered image

    """
    image_local = image.copy()
    median = median_filter_GPU(image, (kernel_size, kernel_size))
    avg_kernel = cp.ones([kernel_size, kernel_size],
                         dtype=cp.float32)/(1.0*kernel_size**2)
    avg = convolve_GPU(image, avg_kernel, mode='same')
    std = cp.sqrt(convolve_GPU((image-avg)**2, avg_kernel, mode='same'))
    condition = image > (avg + std*z_score_thresh)
    image_local[condition] = median[condition]
    return image_local
