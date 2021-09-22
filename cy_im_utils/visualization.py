import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import IntSlider,HBox,VBox,interactive_output
from glob import glob
from scipy.ndimage import rotate as rotate_cpu
# These lines are a little hack to bring in functions that are local to this directory (prep)
import os
from sys import path
real_path = os.path.realpath(__file__)
path.append(os.path.dirname(real_path))
from prep import *
from tqdm import tqdm

def constrain_contrast(im, quantile_low = 0.01, quantile_high = 0.99): #{{{
    #--------------------------------------------------------------------------
    # Just let numpy do this
    #--------------------------------------------------------------------------
    temp = im.flatten()
    return np.quantile(temp,quantile_low),np.quantile(temp,quantile_high)
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
def plot_patch(patch, ax, color = 'k', linestyle = '-' , linewidth  = 0.5):# {{{
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
    # }}}
def plot_circle(coords, ax, color = 'k', linestyle = '-' , linewidth  = 0.5):# {{{
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
    # }}}
def orthogonal_plot(volume, step = 1, line_color = 'k', lw = 1, ls = (0,(5,5)),
        figsize = (10,10), cmap = 'gray', colorbar = False, grid = False,
        crosshairs = True, view = 'all', refresh = 'refresh', cbar_range = []): 
    # {{{
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
    if refresh == 'constant':
        if view == 'all':
            fig,ax = plt.subplots(2,2, figsize = figsize)
            ax = ax.flatten()
        if view == 'yz' or view == 'xz' or view == 'xy':
            fig = plt.figure(figsize = figsize)
            ax = [plt.gca(),plt.gca(),plt.gca()]

    @interact(  x=(0,volume.shape[0]-1,step), 
                y = (0,volume.shape[1]-1,step), 
                z = (0,volume.shape[2]-1, step), 
                yz = (-15.,15.), 
                xz = (-15.,15.))
    def imshow_line(x,y,z,xz,yz):
        if refresh == 'refresh':
            if view == 'all':
                fig,ax = plt.subplots(2,2, figsize = figsize)
                ax = ax.flatten()
            if view == 'yz' or view == 'xz' or view == 'xy':
                fig = plt.figure(figsize = figsize)
                ax = [plt.gca(),plt.gca(),plt.gca()]

        shape = volume.shape

        if not cbar_range:
            l,h = nif_99to01contrast(volume[:,:,z])
        else:
            l,h = cbar_range

        if view == 'xy' or view == 'all':
            im = ax[0].imshow(volume[:,:,z].T, cmap = cmap, vmin = l, vmax = h)
            ax[0].set_title("x-y plane")
            if crosshairs:
                ax[0].plot([0,shape[0]-1],[y,y],color = line_color, linewidth = lw, linestyle = ls)
                ax[0].plot([x,x],[0,shape[1]-1],color = line_color, linewidth = lw, linestyle = ls)

        if view == 'yz' or view == 'all':
            ax[1].imshow(ndimage.rotate(volume[x,:,:], yz, reshape = False), cmap = cmap, vmin = l, vmax = h)
            ax[1].set_title("y-z plane")
            if crosshairs:
                ax[1].plot([z,z],[0,shape[1]-1],color = line_color, linewidth = lw, linestyle = ls)
                ax[1].plot([0,shape[2]-1],[y,y],color = line_color, linewidth = lw, linestyle = ls)

        if view == 'xz' or view == 'all':
            ax[2].imshow(ndimage.rotate(volume[:,y,:].T, xz, reshape = False), cmap = cmap, vmin = l, vmax = h)
            ax[2].set_title("x-z plane")
            if crosshairs:
                ax[2].plot([0,shape[0]-1],[z,z],color = line_color, linewidth = lw, linestyle = ls)
                ax[2].plot([x,x],[0,shape[2]-1],color = line_color, linewidth = lw, linestyle = ls)
 
        if view == 'all':
            ax[3].axis('off')
 
        if colorbar:
            cbar = fig.colorbar(im, ax = ax[0], location = 'left')
            cbar.set_label('Attenuation')

        if grid:
            for a in ax:
                a.grid(True)
                a.grid(which = 'minor', alpha = 1)
    # }}}
def COR_interact(data_dict, angles = [0,180], figsize = (10,5), cmap = 'gist_ncar'): # {{{
    """
    This is still a work in progress. The goal is to have a minimal interface
    to act like ReconstructCT's GUI to help find the crop boundaries and
    normalization patch coordinates. It will modify in-place the input
    dictionary so that the output will be a dictionary with the correct
    cropping coordinates and 

    Parameters:
    -----------
    data_dict: dictionary
        dictionary with projection path , read_fcn, etc. this gets its
        crop_patch and norm_patch overwritten
    ff: 2D numpy array
        flat field
    df: 2D numpy array
        dark field
    figsize: tuple of ints
        figure size

    """
    proj_files = glob(data_dict['projection path'])
    ff_files = glob(data_dict['flat path'])
    df_files = glob(data_dict['dark path'])
    read_fcn = data_dict['imread function']
    dtype = data_dict['dtype']
    Transpose = data_dict['transpose']
    cor_y0,cor_y1 = data_dict['COR rows']

    ff = field_gpu(ff_files, dtype = data_dict['dtype'])
    df = field_gpu(df_files, dtype = data_dict['dtype'])

    n_proj = len(proj_files)
    # Create the combined image in attenuation space to determine center of rotation
    n_angles = len(angles)
    projection_indices = [np.round(a/(360/n_proj)).astype(int) for a in angles]
    combined = np.zeros([n_angles,ff.shape[0],ff.shape[1]])
    volume_temp = np.zeros([n_angles,ff.shape[0],ff.shape[1]])
    for i,angle_index in tqdm(enumerate(projection_indices)):
        f = proj_files[angle_index]
        temp = -np.log((np.asarray(read_fcn(f), dtype = dtype)-df.get())/(ff-df).get())
        temp[~np.isfinite(temp)] = 0
        combined[i,:,:] = temp
        volume_temp[i,:,:] = read_fcn(f)

    # TRANSPOSE IF THE COR IS NOT VERTICAL!!
    combined = np.sum(combined, axis = 0)
    if Transpose:
        combined = combined.T

    x_max,y_max = combined.shape

    # This is for calculating vmin and vmax for imshows
    dist = 0.01
    distribution = combined.flatten()

    def inner(crop_x0,crop_x1,crop_y0,crop_y1,norm_x0,norm_x1,norm_y0,norm_y1,cor_y0,cor_y1):
        l,h = np.quantile(distribution,dist),np.quantile(distribution,1.0-dist)
        crop_patch = [crop_x0,crop_x1,crop_y0,crop_y1]
        norm_patch = [norm_x0,norm_x1,norm_y0,norm_y1]
        y0,y1 = cor_y0,cor_y1

        
        ##---------------------------------------------------------------------
        ##  DEBUGGING TIP: OVERRIDING THE CONTROL OF THE INTERACT TO DEBUG
        #crop_patch = [356,1982,731,2385]
        #norm_patch = [111,251,192,2385]
        ##---------------------------------------------------------------------


        if y0==y1:
            fig,ax = plt.subplots(1,1, figsize = figsize)
            ax = [ax]
            print('l,h = ',l,h)
            ax[0].imshow(combined.T, cmap = cmap, vmin = l, vmax = h)
            plot_patch(crop_patch, ax[0], color = 'r', linestyle = '--', linewidth = 2)
            plot_patch(norm_patch, ax[0], color = 'k')
            # Visualize 0 and 180 degrees with crop patch and normalization patch highlighted
            if norm_patch[0] != norm_patch[1] and norm_patch[2] != norm_patch[3]:
                ax[0].text(norm_patch[0],norm_patch[2],'Norm Patch',
                        verticalalignment = 'bottom', horizontalalignment = 'left',
                        rotation = 90)

        elif y0 != y1 and crop_patch[0] != crop_patch[1] and crop_patch[2] != crop_patch[3]:
            if y1 > crop_patch[3]-crop_patch[2]:
                print("COR y1 exceeds window size")
            else:
                fig,ax = plt.subplots(1,2, figsize = figsize)
                ax = ax.flatten()
                #l,h = nif_99to01contrast(combined[np.isfinite(combined)])
                ax[0].imshow(combined.T, cmap = cmap, vmin = l, vmax = h)
                plot_patch(crop_patch, ax[0], color = 'r', linestyle = '--')
                plot_patch(norm_patch, ax[0], color = 'k')

                if norm_patch[0] != norm_patch[1] and norm_patch[2] != norm_patch[3]:
                    ax[0].text(norm_patch[0],norm_patch[2],'Norm Patch',
                            verticalalignment = 'bottom', horizontalalignment = 'left',
                            rotation = 90)


                cor_image = combined[crop_patch[0]:crop_patch[1],crop_patch[2]:crop_patch[3]].T
                cor_image[~np.isfinite(cor_image)] = 0
                cor = center_of_rotation(cor_image, y0,y1, ax = [])
                theta = np.tan(cor[0])*(180/np.pi)
                rot = rotate_cpu(cor_image,-theta, reshape = False)
                cor2 = center_of_rotation(rot, y0,y1,  ax = [])
                #--------------------------------------------------------
                # MODIFY CROP PATCH IN PLACE CENTERS THE IMAGE ON THE COR
                crop_nx = crop_patch[1]-crop_patch[0]
                dx = int(np.round(cor2[1])-crop_nx//2)
                crop_patch[0]+=dx
                crop_patch[1]+=dx
                #--------------------------------------------------------
                cor_image2 = combined[crop_patch[0]:crop_patch[1],crop_patch[2]:crop_patch[3]].T
                cor_image2[~np.isfinite(cor_image2)] = 0
                cor_image2 = rotate_cpu(cor_image2,-theta,reshape=False)
                cor3 = center_of_rotation(cor_image2, y0,y1, ax = ax[1], image_center = True)
                plot_patch(crop_patch,ax[0], color = 'b')
                ax[1].set_title("Rotated and Cropped")
                ax[0].text(crop_patch[1],crop_patch[2],'Crop Patch Centered', verticalalignment = 'bottom', horizontalalignment = 'left',color = 'b', rotation = 90)
                fig.tight_layout()
                data_dict['crop patch'] = crop_patch
                data_dict['norm patch'] = norm_patch
                data_dict['theta'] = theta
                data_dict['COR rows'] = [cor_y0,cor_y1]
        ax[0].set_title("$\Sigma_{{i=0}}^{{{}}}$ Projection[i$\pi / {{{}}}]$".format(len(angles),len(angles)))


    crop_x0 = IntSlider(description = "crop x0", continuous_update = False, min=0,max=x_max)
    crop_x1 = IntSlider(description = "crop x1", continuous_update = False, min=0,max=x_max)
    crop_y0 = IntSlider(description = "crop y0", continuous_update = False, min=0,max=y_max)
    crop_y1 = IntSlider(description = "crop y1", continuous_update = False, min=0,max=y_max)
    norm_x0 = IntSlider(description = "norm x0", continuous_update = False, min=0,max=x_max)
    norm_x1 = IntSlider(description = "norm x1", continuous_update = False, min=0,max=x_max)
    norm_y0 = IntSlider(description = "norm y0", continuous_update = False, min=0,max=y_max)
    norm_y1 = IntSlider(description = "norm y1", continuous_update = False, min=0,max=y_max)
    cor_y0  = IntSlider(description = "COR y0", continuous_update = False, min=0,max=y_max)
    cor_y1  = IntSlider(description = "COR y1", continuous_update = False, min=0,max=y_max)

    row1 = HBox([crop_x0,crop_x1,crop_y0,crop_y1])
    row2 = HBox([norm_x0,norm_x1,norm_y0,norm_y1])
    row3 = HBox([cor_y0,cor_y1])
    ui = VBox([row1,row2,row3])
    control_dict = {
                'crop_x0':crop_x0,'crop_x1':crop_x1,'crop_y0':crop_y0,'crop_y1':crop_y1,
                'norm_x0':norm_x0,'norm_x1':norm_x1,'norm_y0':norm_y0,'norm_y1':norm_y1,
                'cor_y0':cor_y0,'cor_y1':cor_y1
                    }
    out = interactive_output(inner, control_dict)
    display(ui,out)
    # }}}

if __name__=="__main__":
    pass
