#------------------------------------------------------------------------------
#                               UTILS
#------------------------------------------------------------------------------

import numpy as np
from PIL import Image
from tqdm import tqdm
from scipy.ndimage import median_filter
from skimage import restoration
from glob import glob
import matplotlib.pyplot as plt
import sys
sarepy_path = 'C:\\Users\\mcd4\\Documents\\vo_filter_source\\sarepy'
sys.path.insert(0,sarepy_path)
import sarepy.prep.stripe_removal_original as srm1
import sarepy.prep.stripe_removal_improved as srm2
import sarepy.prep.stripe_removal_former as srm3


class attenuation(): # {{{
    def __init__(self, files, flat_field, dark_field, crop_patch, norm_patch, psf, iterations = 30):
        self.df = dark_field
        self.ff = flat_field
        self.norm_patch = norm_patch
        self.crop_patch = crop_patch
        self.shape = np.array([crop_patch[1]-crop_patch[0],len(files),crop_patch[3]-crop_patch[2]])
        self.images = np.zeros(self.shape)
        self.n_projections = len(files)
        self.detector_rows = crop_patch[1]-crop_patch[0]
        self.detector_cols = crop_patch[3]-crop_patch[2]
        self.cor_correction = False
        self.vo_correction = False
        self.cor_poly = []
        #----------------------
        #self.transmission()
        #----------------------
        for i,f in tqdm(enumerate(files)):
            im = np.asarray(Image.open(f))
            self.images[:,i,:] = self.norm_beer_lambert(im, psf, iterations = iterations)

    #def transmission(self):
    #    self.images = (self.images-self.df)/(self.ff-self.df)

    def norm_beer_lambert(self, image, psf, iterations = 30, RL = False):
        """
        Calculate Transmission, normalize and convert to attenuation space via
        beer-lambert law
        """
        n_pat = self.norm_patch
        cr_pat = self.crop_patch
        pat = self.norm_patch
        transmission = (image-self.df)/(self.ff-self.df)
        # Richardson Lucy
        if RL:
            transmission = restoration.richardson_lucy(transmission_image,psf,iterations = iterations)
        norm = np.mean(transmission[n_pat[0]:n_pat[1],n_pat[2]:n_pat[3]])
        attenuation = -np.log(transmission/norm)
        return attenuation[cr_pat[0]:cr_pat[1],cr_pat[2]:cr_pat[3]]
    
    def get_y_vec(self, img, axis=0):
        n = img.shape[axis]
        s = [1] * img.ndim
        s[axis] = -1
        i = np.arange(n).reshape(s)
        return np.round(np.sum(img * i, axis=axis) / np.sum(img, axis=axis), 1)
    
    def vo_filter(self, row, vo_function, nargs, args):
        """
        there might be a better way to achieve this without if-else
        """
        if nargs == 2:
            return vo_function(self.images[row,:,:],args[0],args[1])
        elif nargs == 3:
            return vo_function(self.images[row,:,:],args[0],args[1],args[2])
        elif nargs == 4:
            return vo_function(self.images[row,:,:],args[0],args[1],args[2],args[3])
        else:
            return None

    
    def norm_patch(self, image, patch):
        return np.sum(self.patch_slice(image, patch, arr_im = 'image').flatten())
        
    def patch_slice(self, image, patch, arr_im = 'array'):
        '''
        image  : 2D numpy array
        patch  : list of the form [x0,x1,y0,y1]
        arr_im : string for DESIRED OUTPUT: array slice or image slice
        '''
        if arr_im.lower() == 'array':
            return image[patch[0]:patch[1],patch[2]:patch[3]]
        elif arr_im.lower() == 'image':
            return image[patch[2]:patch[3],patch[0]:patch[1]]

    def center_of_rotation(self, y0, y1, ax = [], image_center = False):
        """
        Parameters
        ----------
        y0 : int
            Lower bounds (row-wise) for curve fitting
        y1 : int
            Upper bounds (row-wise) for curve fitting
        ax: axis handle
            For visual inspection of the fit
        
        Returns
        -------
        numpy array:
           polynomial coefficients for linear fit of the center of rotation as a function of row index
        """
        combined = self.images[:,0,:]+self.images[:,self.n_projections//2,:]
        combined[combined<0] = 0
        ny,nx = combined.shape       # rows, cols
        COM = self.get_y_vec(combined,1)
        subset = COM[y0:y1]
        y = np.arange(y0,y1)
        com_fit = np.polyfit(y,subset,1)
        # Plotting
        if ax:
            ax.plot(np.polyval(com_fit,[0,ny-1]),[0,ny-1],'k-', linewidth = 1, label = 'Curve Fit')
            ax.plot([0,nx],[y0,y0],'k--', linewidth = 0.5)
            ax.plot([0,nx],[y1,y1],'k--', linewidth = 0.5)
            ax.annotate("",xy = (nx//4,y0), xytext = (nx//4,y1), arrowprops = dict(arrowstyle="<->"))
            ax.text(nx//10,(y1-y0)/2+y0,"fit\nrange", verticalalignment = 'center', color = 'r')
            ax.scatter(COM,range(ny),color = 'r', s = 0.5, label = 'Center of mass')
            ax.imshow(combined, origin = 'upper', cmap = 'gist_ncar')
            ax.set_title("Center of Rotation")
            ax.set_xlim(0,nx)
            if image_center:
                ax.plot([nx//2,nx//2],[0,ny-1],'b--',label = 'Center of image')
            plt.gcf().legend(loc = 'upper right')

        self.cor_poly = com_fit
        return com_fit

    #--------------------------------------------------------------------------
    #
    #           Apply Functions -> Modify the ENTIRE volume
    #   These are blocked by a boolean that will only allow them to be executed
    #   once per instance
    #
    #--------------------------------------------------------------------------
    def apply_vo_filter(self, vo_function, nargs, args):
        if self.vo_correction:
            print("SAREPY correction has already been applied once")
            return None
        else:
            for row in tqdm(range(self.detector_rows)):
                self.images[row,:,:] = self.vo_filter(row, vo_function, nargs, args)
            self.vo_correction = True

    def apply_cor_correction(self):
        if self.cor_correction:
            print("Center of rotation correction has already been applied once")
            return None
        else:
            nrow,nproj,ncol = self.images.shape
            for row in tqdm(range(self.detector_rows)):
                corrected_center = np.polyval(self.cor_poly,row)
                roll_pixels = int(np.round(ncol//2-corrected_center))
                self.images[row,:,:] = np.roll(self.images[row,:,:],roll_pixels)
            self.cor_correction = True


   # }}}
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
    im0 = np.asarray(Image.open(files[0]))
    shape = im0.shape
    temp = np.zeros([len(files),shape[0],shape[1]])
    for i,f in tqdm(enumerate(files)):
        temp[i,:,:] = np.asarray(Image.open(f))
    return median_filter(np.median(temp,axis = 0),median_spatial).astype(dtype)
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
def plot_patch(patch, ax, color = 'k', linestyle = '-' , linewidth  = 0.5): # {{{
    """
    This is just a hack to make a rectangular patch for matplotlib
    """
    ax.plot([patch[0],patch[1]],[patch[2],patch[2]],color = color, linestyle = linestyle, linewidth = linewidth)
    ax.plot([patch[0],patch[1]],[patch[3],patch[3]],color = color, linestyle = linestyle, linewidth = linewidth)
    ax.plot([patch[0],patch[0]],[patch[2],patch[3]],color = color, linestyle = linestyle, linewidth = linewidth)
    ax.plot([patch[1],patch[1]],[patch[2],patch[3]],color = color, linestyle = linestyle, linewidth = linewidth)
    # }}}
def patch_slice(image, patch, arr_im = 'array'): # {{{
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
    # }}}
def center_of_rotation(image,y0,y1,ax = [], image_center = False): # {{{
    """
    Parameters
    ----------
    image : 2D numpy array
        (ATTENUATION IMAGE) Opposing (0-180 degree) images summed
    y0 : int
        Lower bounds (row-wise) for curve fitting
    y1 : int
        Upper bounds (row-wise) for curve fitting
    ax: axis handle
        For visual inspection of the fit
    
    Returns
    -------
    numpy array:
       polynomial coefficients for linear fit of the center of rotation as a function of row index
    """
    combined = image.copy()
    combined[combined < 0] = 0               #<-----------------------------------
    ny,nx = combined.shape       # rows, cols
    COM = get_y_vec(combined,1)
    subset2 = COM[y0:y1]
    y = np.arange(y0,y1)
    com_fit = np.polyfit(y,subset2,1)
    # Plotting
    if ax:
        ax.plot(np.polyval(com_fit,[0,ny-1]),[0,ny-1],'k-', linewidth = 1, label = 'Curve Fit')
        ax.plot([0,nx],[y0,y0],'k--', linewidth = 0.5)
        ax.plot([0,nx],[y1,y1],'k--', linewidth = 0.5)
        ax.annotate("",xy = (nx//4,y0), xytext = (nx//4,y1), arrowprops = dict(arrowstyle="<->"))
        ax.text(nx//10,(y1-y0)/2+y0,"fit\nROI", verticalalignment = 'center')
        ax.scatter(COM,range(ny),color = 'r', s = 0.5, label = 'Center of mass')
        ax.imshow(combined, origin = 'upper', cmap = 'gist_ncar')
        ax.set_title("Center of Rotation")
        ax.set_xlim(0,nx)
        if image_center:
            ax.plot([nx//2,nx//2],[0,ny-1],'b--',label = 'Center of image')
        plt.gcf().legend(loc = 'right')
    return com_fit
    # }}}
def nif_99to01contrast(image): # {{{
    #--------------------------------------------------------------------------
    # This returns the bounds that you can scale the image by, to do so you can
    # use this type of syntax:
    #   example:
    #       im = ....
    #       lowbin,highbin = nif_99to01contrast(im)
    #       im[im<lowbin] = lowbin
    #       im[im>highbin] = highbin
    #--------------------------------------------------------------------------
    hist, edges = np.histogram(image)
    normhist = np.cumsum(hist)/np.sum(hist)
    lowinds = normhist <= 0.01
    highinds = normhist >= 0.99
    low = np.zeros_like(normhist)
    low[lowinds] = normhist[lowinds]
    high = np.ones_like(normhist)
    high[highinds] = normhist[highinds]
    val1percent = np.max(low)
    lowindex = np.argmin(abs(low-val1percent))
    val99percent = np.min(high)
    highindex = np.argmin(abs(high-val99percent))
    lowbin = edges[lowindex]
    highbin = edges[highindex]
    if highbin == lowbin:
        highbin = highbin+1
    return lowbin,highbin
    # }}}

if __name__=="__main__":
    pass

