"""

the hits

"""
from cupyx.scipy.ndimage import median_filter as median_gpu
from glob import glob
from ipywidgets import IntSlider,FloatSlider,HBox,VBox,interactive_output,interact,interact_manual,RadioButtons,Text,IntRangeSlider
from matplotlib.gridspec import GridSpec
from matplotlib.transforms import Affine2D
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib
from skimage.transform import warp
from scipy.ndimage import rotate as rotate_cpu, median_filter
from tqdm import tqdm
import matplotlib.pyplot as plt
import logging
import numpy as np
from .prep import *



class COR_interact:
    def __init__(   self,
                    data_dict, 
                    ff : np.array, 
                    df : np.array,
                    angles: list = [0,180],
                    apply_thresh: float = None 
                    ):
        self.keys = list(data_dict.keys())
        self.proj_path = data_dict['paths']['projection_path']
        self.ext = data_dict['pre_processing']['extension']
        self.proj_files = list(proj_path.glob(f"*{ext}"))
        self.apply_thresh = apply_thresh
        self.df_ = cp.asarray(df)
        self.ff_ = cp.asarray(ff)
 

        if data_dict['COR']:
            cor_y0,cor_y1 = data_dict['COR']['y0'],data_dict['COR']['y1']
        else:
            cor_y0,cor_y1 = 0,ff.shape[-1]
    
        self.fetch_combined_image


    def fetch_combined_image(   self) -> None:
        """ This composes the combined image (0 + 180)...
        """
        angles = self.angles
        n_proj = len(self.proj_files)
        logging.info(f"COR -> num_projection files = {n_proj}")
        # Create the combined image in attenuation space to determine center of rotation
        n_angles = len(angles)
        projection_indices = [np.round(a/(360/n_proj)).astype(int) for a in angles]
        combined = np.zeros([n_angles,ff.shape[0],ff.shape[1]])

        for i,angle_index in tqdm(enumerate(projection_indices)):
            f = self.proj_files[angle_index]
            im_temp = cp.asarray(imread_fcn(f), dtype = np.float32)
            temp = -cp.log((im_temp-df_)/(ff_-df_))
            temp = median_gpu(cp.array(temp), (med_kernel,med_kernel))
            temp[~cp.isfinite(temp)] = 0
            combined[i,:,:] = cp.asnumpy(temp)

        combined = np.sum(combined, axis = 0)
        if self.apply_thresh is not None:
            logging.info(f"Applying threshold to combined image ({apply_thresh})")
            combined[combined < apply_thresh] = 0

        self.x_max,self.y_max = combined.shape
        self.combined = combined


    def update_crop_norm(self, crop_y0, crop_dy, crop_x0, crop_dx, tpose):
        self.combined_local = self.combined.copy()
        if tpose == 'True':
            self.combined_local = combined.T
            self.data_dict['pre_processing']['transpose'] = True
        elif tpose == 'False':
            self.combined_local = combined
            self.data_dict['pre_processing']['transpose'] = False
        self.crop_patch = [crop_y0,crop_y0+crop_dy,crop_x0,crop_x0+crop_dx]
        self.norm_patch = [norm_y0,norm_y0+norm_dy,norm_x0,norm_x0+norm_dx]
        self.y0,self.y1 = cor_y0,cor_y1
    
    def plot_crop_norm(self):
        self.ax[0].imshow(self.combined_local)#, vmin = l, vmax = h)
        x0,y0 = self.crop_patch[2],self.crop_patch[0]
        dx = self.crop_patch[3]-self.crop_patch[2]
        dy = self.crop_patch[1]-self.crop_patch[0]
        crop_rectangle = Rectangle((x0,y0),dx,dy, 
                                                fill = False,
                                                color = 'r',
                                                linestyle = '--',
                                                linewidth = 2)
        self.ax[0].add_artist(crop_rectangle)
        plot_patch(self.norm_patch, ax[0], color = 'w')
        # Visualize 0 and 180 degrees with crop patch and normalization
        # patch highlighted
        if norm_patch[0] != norm_patch[1] and norm_patch[2] != norm_patch[3]:
            ax[0].text(norm_patch[0],norm_patch[2],
                                            'Norm Patch',
                                            verticalalignment = 'bottom',
                                            horizontalalignment = 'left',
                                            color = 'w',
                                            rotation = 90)


    def interact(   self,  
                    data_dict : dict,
                    imread_fcn,
                    figsize : tuple = (10,5),
                    apply_thresh: float = None,
                    med_kernel = 3
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

        WARNING2:   all the try - except statements for KeyErrors are from when I
                    changed from using a data dictionary to using a config file so
                    this is a bit deprecated. All the updated versions should just
                    use the version with KeyError (dict is paired with config) 

        Args:
        -----------
        data_dict: dictionary
            dictionary with projection path , read_fcn, etc. this gets its
            crop_patch and norm_patch overwritten

        ff : np.array - flat field
        df : np.array - dark field
        angles: list - Which angles to show projections (this can be important if
                        the object is not cylindrical)
        figsize: tuple of ints - figure size

        apply_thresh: float
            this will apply a threshold to the combined image (sum of attenuation)
            so the background can be removed if it is uneven
        """

       

        # This is for calculating vmin and vmax for imshows
        dist = 0.01
        distribution = combined.flatten()

        self.fig,self.ax = plt.subplots(1,2, figsize = figsize)
        plt.show()

        def inner(crop_y0,crop_dy,crop_x0,crop_dx,norm_x0,norm_dx,norm_y0,norm_dy,
                                cor_y0,cor_y1, tpose):

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
                self.plot_crop_norm()
                ax[1].axis(False)
            elif (  y0 != y1 
                    and crop_patch[0] != crop_patch[1] 
                    and crop_patch[2] != crop_patch[3]):
                if y1 > crop_patch[3]-crop_patch[2]:
                    print("COR y1 exceeds window size")
                else:
                    #--------------------------------------------------------------
                    # Trying to be Smart about the Figures
                    #--------------------------------------------------------------
                    ax[0].imshow(combined_local,  vmin = l, vmax = h)
                    plot_patch(crop_patch, ax[0], color = 'r', linestyle = '--')
                    plot_patch(norm_patch, ax[0], color = 'w')

                    if (norm_patch[0] != norm_patch[1] and 
                                                norm_patch[2] != norm_patch[3]):
                        ax[0].text(norm_patch[0],norm_patch[2],
                                                'Norm Patch',
                                                color = 'w',
                                                verticalalignment = 'bottom',
                                                horizontalalignment = 'left',
                                                rotation = 90)


                    slice_x = slice(crop_patch[0],crop_patch[1])
                    slice_y = slice(crop_patch[2],crop_patch[3])
                    cor_image = combined_local[slice_y,slice_x]
                    cor_image[~np.isfinite(cor_image)] = 0
                    cor = center_of_rotation(cor_image, y0,y1, ax = [])
                    theta = np.tan(cor[0])*(180/np.pi)
                    rot = rotate_cpu(cor_image,-theta, reshape = False)
                    cor2 = center_of_rotation(rot, y0, y1,   ax = [])
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
                    cor_image2 = combined_local[slice_y_corr,slice_x_corr]
                    cor_image2[~np.isfinite(cor_image2)] = 0
                    cor_image2 = rotate_cpu(cor_image2,-theta,reshape=False)
                    # Adjusted cor image re-cropped to pad
                    cor3 = center_of_rotation(  cor_image2,
                                                y0,
                                                y1,
                                                ax = ax[1],
                                               image_center = True)
                    plot_patch(crop_patch,ax[0], color = 'b')
                    ax[1].set_title(f"Rotated ({theta:.3f} deg) and Cropped")
                    ax[0].text(crop_patch[1],crop_patch[2],
                                        'Crop Patch Centered',
                                        verticalalignment = 'bottom',
                                        horizontalalignment = 'left',
                                        color = 'b',
                                        rotation = 90)
                    fig.tight_layout()
                    data_dict['crop']['x0'] = crop_patch[0]
                    data_dict['crop']['x1'] = crop_patch[1]
                    data_dict['crop']['y0'] = crop_patch[2]
                    data_dict['crop']['y1'] = crop_patch[3]
                    data_dict['crop']['dx'] = crop_patch[1]-crop_patch[0]
                    data_dict['crop']['dy'] = crop_patch[3]-crop_patch[2]
                    data_dict['norm']['x0'] = norm_patch[0]
                    data_dict['norm']['x1'] = norm_patch[1]
                    data_dict['norm']['y0'] = norm_patch[2]
                    data_dict['norm']['y1'] = norm_patch[3]
                    data_dict['norm']['dx'] = norm_patch[1]-norm_patch[0]
                    data_dict['norm']['dy'] = norm_patch[3]-norm_patch[2]
                    data_dict['COR']['y0'] = cor_y0
                    data_dict['COR']['y1'] = cor_y1

                    data_dict['COR']['theta'] = theta
            ax[0].set_title("$\Sigma_{{i=0}}^{{{}}}$ Projection[i$\pi / {{{}}}]$".format(len(angles),len(angles)))
            #plt.show()


        #---------------------------------------------------------------------------
        # NOTE TO SELF:
        #   These assignements look backward (x0, 0 -> y_max, etc.) but they are
        #   fixing the problem of plt.imshow transposing the image
        #---------------------------------------------------------------------------
        kwargs = {'continuous_update':False,'min':0}
        max_crop = max([y_max,x_max])
        crop_x0 = IntSlider(description = "crop x$_0$", max=max_crop, **kwargs)
        crop_dx = IntSlider(description = "crop $\Delta$x", max=max_crop, **kwargs)
        crop_y0 = IntSlider(description = "crop y$_0$", max=max_crop, **kwargs)
        crop_dy = IntSlider(description = "crop $\Delta$y", max=max_crop, **kwargs)
        norm_x0 = IntSlider(description = "norm x$_0$", max=max_crop, **kwargs)
        norm_dx = IntSlider(description = "norm $\Delta$x", max=max_crop, **kwargs)
        norm_y0 = IntSlider(description = "norm y$_0$", max=max_crop, **kwargs)
        norm_dy = IntSlider(description = "norm $\Delta$y", max=max_crop, **kwargs)
        cor_y0  = IntSlider(description = "COR y$_0$",  max=max_crop, **kwargs)
        cor_y1  = IntSlider(description = "COR y$_1$",  max=max_crop, **kwargs)


        transpose_val = "False" if 'transpose' not in data_dict else data_dict['transpose']
        tpose = RadioButtons(options = ['False','True'], value = transpose_val, description = "Transpose")

        row0 = HBox([tpose])
        row1 = HBox([crop_x0,crop_dx,crop_y0,crop_dy])
        row2 = HBox([norm_x0,norm_dx,norm_y0,norm_dy])
        row3 = HBox([cor_y0,cor_y1])
        ui = VBox([row0,row1,row2,row3])

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
                    'cor_y1':cor_y1,
                    'tpose':tpose
                        }

        out = interactive_output(inner, control_dict)
        display(ui,out)
