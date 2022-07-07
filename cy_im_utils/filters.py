import cupy as cp
from cupyx.scipy.ndimage.filters import gaussian_filter


def _unsharp_mask_single_channel_GPU(image, radius, amount, vrange):
    """
    from skimage

    single channel implementation of unsharp masking filter...?
    """
    blurred = gaussian_filter(image, sigma = radius, mode = 'reflect')
    result = image + (image - blurred) * amount
    if vrange is not None:
        return cp.clip(result, vrange[0],vrange[1], out = result)
    return result

def unsharp_mask_GPU(image, radius = 1.0, amount = 1.0, multichannel = False,
        preserve_range = True, *, channel_axis = None):
    """
    Taken from skimage
    """
    vrange = None  # Range for valid values; used for clipping.
    float_dtype = cp.float32
    if preserve_range:
        fimg = image.astype(float_dtype, copy=False)
    else:
        fimg = img_as_float(image).astype(float_dtype, copy=False)
        negative = cp.any(fimg < 0)
        if negative:
            vrange = [-1., 1.]
        else:
            vrange = [0., 1.]

    if channel_axis is not None:
        result = cp.empty_like(fimg, dtype=float_dtype)
        for channel in range(image.shape[channel_axis]):
            #sl = utils.slice_at_axis(channel, channel_axis)
            sl = (slice(None),) * channel_axis + (channel,) + (...,)
            result[sl] = _unsharp_mask_single_channel(
                fimg[sl], radius, amount, vrange)
        return result
    else:
        return _unsharp_mask_single_channel_GPU(fimg, radius, amount, vrange)
