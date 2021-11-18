#------------------------------------------------------------------------------
#                       Patch Extraction
#------------------------------------------------------------------------------

def patch_slice(image, patch, arr_im = 'array'):
    '''
    Parameters
    ----------
    image  : 2D numpy array
        image to be sliced
    patch  : list of the form [x0,x1,y0,y1]
        coordinate of slice
    arr_im : string 
        Specify for plotting format array slice, or plotting image (yx or xy)

    Returns
        The slice of the image
    '''
    if arr_im.lower() == 'array':
        return image[patch[0]:patch[1],patch[2]:patch[3]]
    elif arr_im.lower() == 'image':
        return image[patch[2]:patch[3],patch[0]:patch[1]]
