import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact

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

if __name__=="__main__":
    pass

