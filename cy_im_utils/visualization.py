"""


"""
from glob import glob
from ipywidgets import IntSlider,FloatSlider,HBox,VBox,interactive_output,interact,interact_manual
from matplotlib.gridspec import GridSpec
from scipy.ndimage import rotate as rotate_cpu, median_filter
from cupyx.scipy.ndimage import median_filter as median_gpu
from tqdm import tqdm
import matplotlib.pyplot as plt
import logging
import numpy as np

from .sarepy_cuda import remove_all_stripe_GPU
from .prep import *
from .prep import center_of_rotation
from .recon_utils import *


def constrain_contrast(im, quantile_low = 0.01, quantile_high = 0.99): 
    """
    --------------------------------------------------------------------------
    Just let numpy do this
    --------------------------------------------------------------------------
    """
    temp = im.flatten()
    return np.quantile(temp,quantile_low),np.quantile(temp,quantile_high)
    
def nif_99to01contrast(image): 
    """
    --------------------------------------------------------------------------
     This returns the bounds that you can scale the image by, to do so you can
     use this type of syntax:
       example:
           im = ....
           lowbin,highbin = nif_99to01contrast(im)
           im[im<lowbin] = lowbin
           im[im>highbin] = highbin
    --------------------------------------------------------------------------
    """
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

def contrast(image , low : float  = 0.01, high : float = 0.99) -> tuple:
    """
    parameters
    ----------
    image: array-like

    """
    assert low+high == 1, "Qunatiles don't sum to 1"
    temp = image.flatten()
    return np.quantile(temp,low),np.quantile(temp,high)
    
def plot_patch(patch : list, ax , color = 'k', linestyle = '-' , linewidth  = 0.5):
    """
    This is just a hack to make a rectangular patch for matplotlib
    Parameters
    ----------
    patch: list
        Corners of patch in the form [x0,x1,y0,1]
    ax: matplotlib axis
        The axis on which to add the patch
    color: string
        color of box

    returns
    -------
    None
    """
    ax.plot([patch[0],patch[1]],[patch[2],patch[2]],color = color, linestyle = linestyle, linewidth = linewidth)
    ax.plot([patch[0],patch[1]],[patch[3],patch[3]],color = color, linestyle = linestyle, linewidth = linewidth)
    ax.plot([patch[0],patch[0]],[patch[2],patch[3]],color = color, linestyle = linestyle, linewidth = linewidth)
    ax.plot([patch[1],patch[1]],[patch[2],patch[3]],color = color, linestyle = linestyle, linewidth = linewidth)
    return None 
    
def plot_circle(coords, ax, color = 'k', linestyle = '-' , linewidth  = 0.5):
    """
    This is just a hack to make a rectangular patch for matplotlib
    Parameters
    ----------
    coords: list
        x,y,radius
    ax: matplotlib axis
        The axis on which to add the patch
    color: string
        color of box

    returns
    -------
    None
    """
    
    return None 
    
def orthogonal_plot(volume, step = 1, line_color = 'k', lw = 1, ls = (0,(5,5)),
        figsize = (10,10), cmap = 'gray', colorbar = False, grid = False,
        crosshairs = True, view = 'all', refresh = 'refresh', cbar_range = [], 
        theta_max = 15.0): 
    
    """
    This is my ripoff of ImageJ orthogonal views, you can change x,y,z slice
    and xz rotation (i.e. rotation about the y-axis) and the yz roatation
    (about the x-axis). Note the rotation is only between +/- 15 degrees

    parameters
    ----------
        volume : 3D array
        step : controls the interactive interval for stepping through volume
        figsize: figure size
        cmap: colormap

    Example
    -------
        interactive_plot(electrode['A']['cropped'], figsize = (12,4), cmap = 'hsv')

    """
    volume = volume
    shape = volume.shape
    plt.close('all')
    if view == 'all':
        fig,ax = plt.subplots(2,2, figsize = figsize)
        ax = ax.flatten()
    elif view == 'yz' or view == 'xz' or view == 'xy':
        fig = plt.figure(figsize = figsize)
        ax = [plt.gca(),plt.gca(),plt.gca()]
    else:
        assert False, f"unkown view config: {view}"

    plt.show()

    def inner(x,y,z,xz,yz):
        #if refresh == 'refresh':
        #    if view == 'all':
        #        fig,ax = plt.subplots(2,2, figsize = figsize)
        #        ax = ax.flatten()
        #    if view == 'yz' or view == 'xz' or view == 'xy':
        #        fig = plt.figure(figsize = figsize)
        #        ax = [plt.gca(),plt.gca(),plt.gca()]

        [a.clear() for a in ax]
        shape = volume.shape
        if not cbar_range:
            l,h = nif_99to01contrast(volume[:,:,z])
        else:
            l,h = cbar_range

        line_kwargs = {'color': line_color, 'linewidth' : lw, 'linestyle':ls}
        im_kwargs = {'cmap':cmap, 'vmin':l, 'vmax':h}
        if view == 'xy' or view == 'all':
            im = ax[0].imshow(volume[:,:,z].T, **im_kwargs)
            ax[0].set_title("x-y plane")
            if crosshairs:
                ax[0].plot([0,shape[0]-1],[y,y],**line_kwargs)
                ax[0].plot([x,x],[0,shape[1]-1],**line_kwargs)

        if view == 'yz' or view == 'all':
            ax[1].imshow(rotate_cpu(volume[x,:,:], yz, reshape = False),
                                                                **im_kwargs)
            ax[1].set_title("y-z plane")
            if crosshairs:
                ax[1].plot([z,z],[0,shape[1]-1],**line_kwargs)
                ax[1].plot([0,shape[2]-1],[y,y],**line_kwargs)

        if view == 'xz' or view == 'all':
            rotated_image = rotate_cpu(volume[:,y,:], xz, reshape = False)
            ax[2].imshow(rotated_image.T, origin = 'upper', **im_kwargs)
            ax[2].set_title("x-z plane")
            if crosshairs:
                ax[2].plot([0,shape[0]-1],[z,z],**line_kwargs)
                ax[2].plot([x,x],[0,shape[2]-1],**line_kwargs)
 
        if view == 'all':
            ax[3].axis('off')
 
        if colorbar:
            cbar = fig.colorbar(im, ax = ax[0], location = 'left')
            cbar.set_label('Attenuation')

        if grid:
            for a in ax:
                a.grid(True)
                a.grid(which = 'minor', alpha = 1)

    # The Rest of this function sets up the UI
    x_max = volume.shape[0]-1
    y_max = volume.shape[1]-1
    z_max = volume.shape[2]-1
    x = IntSlider(description = "x", continuous_update = False, min=0, max=x_max)
    y = IntSlider(description = "y", continuous_update = False, min=0, max=y_max)
    z = IntSlider(description = "z", continuous_update = False, min=0, max=z_max)
    xz = FloatSlider(description = r"$\theta_{xz}$", continuous_update = False, min=-theta_max,max=theta_max)
    yz = FloatSlider(description = r"$\theta_{yz}$", continuous_update = False, min=-theta_max,max=theta_max)

    row1 = HBox([x,y,z])
    row2 = HBox([xz,yz])
    ui = VBox([row1,row2])
    control_dict = { 'x':x,'y':y,'z':z,'xz':xz,'yz':yz }
    out = interactive_output(inner, control_dict)
    display(ui,out)

def COR_interact(   data_dict : dict,
                    ff : np.array, 
                    df : np.array,
                    angles : list = [0,180],
                    figsize : tuple = (10,5)
                    ) -> None:
    """
    This is still a work in progress. The goal is to have a minimal interface
    to act like ReconstructCT's GUI to help find the crop boundaries and
    normalization patch coordinates. It will modify in-place the input
    dictionary so that the output will be a dictionary with the correct
    cropping coordinates,

    WARNING :   The way matplotlib renders images is not intuitive x and y are
                flipped, so all references to cropping, etc. can become very
                confusing. proceed with caution.

    Args:
    -----------
    data_dict: dictionary
        dictionary with projection path , read_fcn, etc. this gets its
        crop_patch and norm_patch overwritten

    ff : np.array
        flat field

    df : np.array
        dark field

    angles: list
        Which angles to show projections (this can be important if the object
        is not cylindrical)

    figsize: tuple of ints
        figure size
    """
    proj_files = list(data_dict['projection path'].glob(f"*.{data_dict['extension']}"))
    dtype = data_dict['dtype']
    Transpose = data_dict['transpose']


    if data_dict['COR rows']:
        cor_y0,cor_y1 = data_dict['COR rows']
    else:
        cor_y0,cor_y1 = 0,ff.shape[-1]


    n_proj = len(proj_files)
    print(n_proj)
    # Create the combined image in attenuation space to determine center of rotation
    n_angles = len(angles)
    projection_indices = [np.round(a/(360/n_proj)).astype(int) for a in angles]
    combined = np.zeros([n_angles,ff.shape[0],ff.shape[1]])
    med_kernel = data_dict['median spatial']
    df_ = cp.asarray(df)
    ff_ = cp.asarray(ff)
    for i,angle_index in tqdm(enumerate(projection_indices)):
        f = proj_files[angle_index]
        im_temp = cp.asarray(data_dict['imread function'](f), dtype = np.float32)
        if data_dict['transpose']:
            im_temp = im_temp.T
        temp = -cp.log((im_temp-df_)/(ff_-df_))
        temp = median_gpu(cp.array(temp), (med_kernel,med_kernel))
        temp[~cp.isfinite(temp)] = 0
        combined[i,:,:] = cp.asnumpy(temp)

    # TRANSPOSE IF THE COR IS NOT VERTICAL!!
    combined = np.sum(combined, axis = 0)

    x_max,y_max = combined.shape

    # This is for calculating vmin and vmax for imshows
    dist = 0.01
    distribution = combined.flatten()

    plt.close('all')
    fig,ax = plt.subplots(1,2, figsize = figsize)
    plt.show()

    #def inner(crop_y0,crop_y1,crop_x0,crop_x1,norm_x0,norm_x1,norm_y0,norm_y1,cor_y0,cor_y1):
    def inner(crop_y0,crop_dy,crop_x0,crop_dx,norm_x0,norm_dx,norm_y0,norm_dy,cor_y0,cor_y1):
        l,h = np.quantile(distribution,dist),np.quantile(distribution,1.0-dist)
        crop_patch = [crop_y0,crop_y0+crop_dy,crop_x0,crop_x0+crop_dx]
        norm_patch = [norm_y0,norm_y0+norm_dy,norm_x0,norm_x0+norm_dx]
        y0,y1 = cor_y0,cor_y1

        
        ##---------------------------------------------------------------------
        ##  DEBUGGING TIP: OVERRIDING THE CONTROL OF THE INTERACT TO DEBUG
        #crop_patch = [356,1982,731,2385]
        #norm_patch = [111,251,192,2385]
        ##---------------------------------------------------------------------
        #plt.close('all')
        [a.clear() for a in ax]
        #fig,ax = plt.subplots(1,2, figsize = (10,5))
        #plt.show()
        if y0==y1:
            ax[0].imshow(combined, vmin = l, vmax = h)
            plot_patch(crop_patch, ax[0], color = 'r', linestyle = '--', linewidth = 2)
            plot_patch(norm_patch, ax[0], color = 'k')
            # Visualize 0 and 180 degrees with crop patch and normalization patch highlighted
            if norm_patch[0] != norm_patch[1] and norm_patch[2] != norm_patch[3]:
                ax[0].text(norm_patch[0],norm_patch[2],'Norm Patch',
                        verticalalignment = 'bottom', horizontalalignment = 'left',
                        rotation = 90)
            ax[1].axis(False)
        elif y0 != y1 and crop_patch[0] != crop_patch[1] and crop_patch[2] != crop_patch[3]:
            if y1 > crop_patch[3]-crop_patch[2]:
                print("COR y1 exceeds window size")
            else:
                #--------------------------------------------------------------
                # Trying to be Smart about the Figures
                #--------------------------------------------------------------
                ax[0].imshow(combined,  vmin = l, vmax = h)
                plot_patch(crop_patch, ax[0], color = 'r', linestyle = '--')
                plot_patch(norm_patch, ax[0], color = 'k')

                if norm_patch[0] != norm_patch[1] and norm_patch[2] != norm_patch[3]:
                    ax[0].text(norm_patch[0],norm_patch[2],'Norm Patch',
                            verticalalignment = 'bottom', horizontalalignment = 'left',
                            rotation = 90)


                slice_x = slice(crop_patch[0],crop_patch[1])
                slice_y = slice(crop_patch[2],crop_patch[3])
                cor_image = combined[slice_y,slice_x]
                cor_image[~np.isfinite(cor_image)] = 0
                cor = center_of_rotation(cor_image, y0,y1, ax = [])
                theta = np.tan(cor[0])*(180/np.pi)
                rot = rotate_cpu(cor_image,-theta, reshape = False)
                cor2 = center_of_rotation(rot, y0,y1,  ax = [])
                #--------------------------------------------------------
                # MODIFY CROP PATCH IN PLACE CENTERS THE IMAGE ON THE COR
                #crop_nx = crop_patch[1]-crop_patch[0]
                crop_nx = crop_patch[1]-crop_patch[0]
                dx = int(np.round(cor2[1])-crop_nx//2)
                print(f"dx = {dx}")
                print(f"intercept (should be center) = {cor2[1]}")
                crop_patch[0] += dx
                crop_patch[1] += dx
                slice_x_corr = slice(crop_patch[0],crop_patch[1])
                slice_y_corr = slice(crop_patch[2],crop_patch[3])
                #--------------------------------------------------------
                cor_image2 = combined[slice_y_corr,slice_x_corr]
                cor_image2[~np.isfinite(cor_image2)] = 0
                cor_image2 = rotate_cpu(cor_image2,-theta,reshape=False)
                cor3 = center_of_rotation(cor_image2, y0, y1, ax = ax[1], image_center = True)
                plot_patch(crop_patch,ax[0], color = 'b')
                ax[1].set_title(f"Rotated ({theta:.3f} deg) and Cropped")
                ax[0].text(crop_patch[1],crop_patch[2],'Crop Patch Centered', verticalalignment = 'bottom', horizontalalignment = 'left',color = 'b', rotation = 90)
                fig.tight_layout()
                data_dict['crop patch'] = crop_patch
                data_dict['norm patch'] = norm_patch
                data_dict['theta'] = theta
                data_dict['COR rows'] = [cor_y0,cor_y1]
        ax[0].set_title("$\Sigma_{{i=0}}^{{{}}}$ Projection[i$\pi / {{{}}}]$".format(len(angles),len(angles)))
        plt.show()


    #---------------------------------------------------------------------------
    # NOTE TO SELF:
    #   These assignements look backward (x0, 0 -> y_max, etc.) but they are
    #   fixing the problem of plt.imshow transposing the image
    #---------------------------------------------------------------------------
    crop_x0 = IntSlider(description = "crop x0", continuous_update = False, min=0,max=y_max)
    crop_dx = IntSlider(description = "crop dx", continuous_update = False, min=0,max=y_max)
    crop_y0 = IntSlider(description = "crop y0", continuous_update = False, min=0,max=x_max)
    crop_dy = IntSlider(description = "crop dy", continuous_update = False, min=0,max=x_max)
    norm_x0 = IntSlider(description = "norm x0", continuous_update = False, min=0,max=y_max)
    norm_dx = IntSlider(description = "norm dx", continuous_update = False, min=0,max=y_max)
    norm_y0 = IntSlider(description = "norm y0", continuous_update = False, min=0,max=x_max)
    norm_dy = IntSlider(description = "norm dy", continuous_update = False, min=0,max=x_max)
    cor_y0  = IntSlider(description = "COR y0",  continuous_update = False, min=0,max=x_max)
    cor_y1  = IntSlider(description = "COR y1",  continuous_update = False, min=0,max=x_max)

    row1 = HBox([crop_x0,crop_dx,crop_y0,crop_dy])
    row2 = HBox([norm_x0,norm_dx,norm_y0,norm_dy])
    row3 = HBox([cor_y0,cor_y1])
    ui = VBox([row1,row2,row3])

    control_dict = {
                'crop_y0':crop_x0,
                'crop_dy':crop_dx,
                'crop_x0':crop_y0,
                'crop_dx':crop_dy,
                'norm_x0':norm_y0,
                'norm_dx':norm_dy,
                'norm_y0':norm_x0,
                'norm_dy':norm_dx,
                'cor_y0':cor_y0,
                'cor_y1':cor_y1
                    }

    out = interactive_output(inner, control_dict)
    display(ui,out)
    z

def attn_express(image, df, ff, med_kernel, crop_patch, norm_patch):
    norm_x = slice(norm_patch[2],norm_patch[3])
    norm_y = slice(norm_patch[0],norm_patch[1])
    crop_x = slice(crop_patch[2],crop_patch[3])
    crop_y = slice(crop_patch[0],crop_patch[1])
    temp = image.copy()
    patch = temp[norm_x,norm_y]
    temp -= df
    temp /= (ff-df)
    temp = median_gpu()

def SAREPY_interact(data_dict : dict,
                    input_array : np.array,
                    figsize : tuple  = (10,5),
                    snr_max : float = 3.0,
                    sm_size_max : int = 51,
                    sino_row_col : str = 'row',
                    ) -> None: 
    """
    Interactive inspection of remove_all_stripe with sliders to control the
    arguments. It overwrites the values in data_dict so they can be unpacked
    and used for the Vo filter batch.

    Parameters:
    -----------
    input_array: 3d numpy array
        array of attenuation values of shape: 
            axis 0 : projection,
            axis 1 : sinogram,
            axis 2 : detector column slice

    figsize: tuple
        
    snr_max: float
        maximum value for snr slider to take

    sm_size_max: int
        (odd number) maximum value for sm_size slider to take
    """
    # Importing the regular sarepy for 2d images for interactive
    from sys import path
    path.append("C:\\Users\\mcd4\\Documents\\vo_filter_source\\sarepy")
    from sarepy.prep.stripe_removal_original import remove_all_stripe as remove_all_stripe_CPU

    if 'row' in sino_row_col:
        gs = GridSpec(1,5)
        fig = plt.figure(figsize = figsize)
        ax = []
        ax.append(fig.add_subplot(gs[0]))
        ax.append(fig.add_subplot(gs[1], sharex = ax[0], sharey = ax[0]))
        ax.append(fig.add_subplot(gs[2], sharex = ax[0], sharey = ax[0]))
        ax.append(fig.add_subplot(gs[3:]))
        ax[1].yaxis.set_ticklabels([])
        ax[2].yaxis.set_ticklabels([])
        ax[3].yaxis.tick_right()
        ax[0].set_title("unfiltered")
        ax[1].set_title("filtered")
        ax[2].set_title("diff")
        ax[3].set_title("recon (FBP)")
    elif 'col' in sino_row_col:
        gs = GridSpec(5,5)
    n_proj,n_sino,detector_width = input_array.shape
    def inner(frame,snr,la_size,sm_size):
        plt.cla()
        temp = input_array[:,frame,:].copy()
        try:
            filtered = remove_all_stripe_GPU(cp.array(temp[:,None,:]),snr,la_size,sm_size)
            filtered = cp.asnumpy(filtered[:,0,:])
            reco = astra_2d_simple(filtered)
            l,h = contrast(temp)
            ax[0].imshow(temp.T, vmin = l, vmax = h)
            ax[1].imshow(filtered.T, vmin = l, vmax = h)
            ax[2].imshow(temp.T-filtered.T)
            ax[3].imshow(reco)
        except:
            print("SVD did not converge")
        data_dict['signal to noise ratio'] = snr
        data_dict['large filter'] = la_size
        data_dict['small filter'] = sm_size
        
    frame = IntSlider(
                        description = "row",
                        continuous_update = False,
                        min = 0,
                        max = n_sino
                        )
    snr = FloatSlider(  
                        description = "snr",
                        continuous_update = False,
                        min = 0,
                        max = snr_max
                        )
    la_size = IntSlider(
                        description = "la_size",
                        continuous_update = False,
                        min = 1,
                        max = detector_width//2,
                        step = 2
                        )
    sm_size = IntSlider(description = "sm_size",
                        continuous_update = False,
                        min = 1,
                        max = sm_size_max,
                        step = 2)
    
    control_dict = {
                    'frame':frame,
                    'snr':snr,
                    'la_size':la_size,
                    'sm_size':sm_size
                   }
    
    ui = HBox([frame,snr,la_size,sm_size])
    out = interactive_output(inner, control_dict)
    display(ui,out)
    
def dynamic_thresh_plot(im,
                        im_filtered,
                        step = 0.05,
                        alpha = 0.9,
                        fix_upper = True, 
                        n_interval = 2,
                        hist_width = 2,
                        cmap = 'gist_ncar',
                        figsize = (10,5)) -> None:
    """
    This is still a work in progress, I can't figure out how to curry the interactive plot... :(

    This renders a histogram which can be segmented into 3 parts (4 sliders)
    the accompanying plots are the raw image and the respective segments. This
    plot has an 'interact button' which you press to execute the segments (so
    that it does not try to continuously update while you move the slider,
    which makes it super laggy) 

    Parameters
    ----------
    im: 2D numpy array
        raw image

    im_filtered: 2D numpy array
        image with some type of filtering (for pixel clustering); this argument
        can also be the same as 'im', but the histogram segmentation is aided
        by some filtering

    step: float
        how large the steps are in the interactive plot sliders

    alpha: float (0-1)
        degree of transparency of the mask

    fix_upper: bool
        ---> I don't think this works

    n_interval: int
        THE GOAL OF THIS WAS TO BE ABLE TO PASS THE NUMBER OF INTERVALS AS AN
        ARGUMENT AND PRODUCE A PLOT WITH THE CORRESPONDING NUMBER OF SEGMENTS,
        BUT CURRYING THE INNER FUNCTION DOES NOT WORK AS EXPECTED

    hist_width: int
        how many 'effective' gridspec slots the histogram will occupy

    cmap: string
        colormap

    figsize: tuple
        figure size

    """
    fig = plt.figure(figsize = figsize)
    min_ = np.min(im_filtered)
    max_ = np.max(im_filtered)
    interval = (min_,max_,step)
    interact_dict = {f"thresh_{t}":interval for t in range(n_interval+1)}
    @interact_manual(**interact_dict)
    def innermost(thresh_0, thresh_1, thresh_2):#, thresh_3):
        thresh = locals()
        thresh = [thresh[key] for key in thresh if "thresh_" in key]
        
        gs = GridSpec(1,hist_width+n_interval+1)
        ax = [fig.add_subplot(gs[0:hist_width])]
        ax[0].tick_params(axis = 'x', labelbottom = True)
        ax[0].tick_params(axis = 'y', labelleft = True)
        hist = ax[0].hist(im_filtered.flatten(), bins = 250, color = 'k')
        m = np.max(hist[0][1:])
        ax[0].set_ylim(0,m)
        ax.append(fig.add_subplot(gs[0,hist_width]))
        ax[1].imshow(im)
        ax[1].axis(False)
        for i in range(n_interval):
            thresh_im = (im_filtered>thresh[i])*(im_filtered<thresh[i+1])
            ax[0].plot([thresh[i],thresh[i]],[0,m],'r--', linewidth = 1)
            ax.append(fig.add_subplot(gs[0,hist_width+i+1], sharex = ax[-1], sharey = ax[-1]))
            ax[-1].imshow(im, cmap = cmap)
            ax[-1].imshow(thresh_im, cmap = 'bone', alpha = alpha)
            ax[-1].axis(False)

        # Get the last threshold boundary
        ax[0].plot([thresh[-1],thresh[-1]],[0,m],'r--', linewidth = 1)

    return innermost
