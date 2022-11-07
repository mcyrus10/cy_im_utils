"""

more modular

"""
from PIL import Image
from cupyx.scipy.ndimage import median_filter as median_gpu
from glob import glob
from ipywidgets import IntSlider,FloatSlider,HBox,VBox,interactive_output,interact,interact_manual,RadioButtons,Text,IntRangeSlider,interactive
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from scipy.ndimage import rotate as rotate_cpu, median_filter
from skimage.transform import warp
from tqdm import tqdm
import logging
import matplotlib
import matplotlib.pyplot as plt
import cupy as cp
import numpy as np
from .prep import center_of_rotation



def imread_fcn(x):
    return np.array(Image.open(x), dtype = np.float32)

class center_of_rotation_interact:
    def __init__(   self,
                    data_dict, 
                    ff : np.array, 
                    df : np.array,
                    angles: list = [0,180],
                    apply_thresh: float = None 
                    ):
        self.data_dict = data_dict
        self.keys = list(data_dict.keys())
        self.proj_path = data_dict['paths']['projection_path']
        self.ext = data_dict['pre_processing']['extension']
        self.proj_files = list(self.proj_path.glob(f"*{self.ext}"))
        self.apply_thresh = apply_thresh
        self.angles = angles
        self.df_ = cp.asarray(df)
        self.ff_ = cp.asarray(ff)
 

        if data_dict['COR']:
            cor_y0,cor_y1 = data_dict['COR']['y0'],data_dict['COR']['y1']
        else:
            cor_y0,cor_y1 = 0,ff.shape[-1]
    
        self.fetch_combined_image()


    def fetch_combined_image(   self,
            med_kernel: int = 3,
            ) -> None:
        """ This composes the combined image (0 + 180)...
        """
        angles = self.angles
        n_proj = len(self.proj_files)
        logging.info(f"COR -> num_projection files = {n_proj}")
        # Create the combined image in attenuation space to determine center of rotation
        n_angles = len(angles)
        projection_indices = [np.round(a/(360/n_proj)).astype(int) for a in angles]
        combined = np.zeros([n_angles,self.ff_.shape[0],self.ff_.shape[1]])

        for i,angle_index in tqdm(enumerate(projection_indices)):
            f = self.proj_files[angle_index]
            im_temp = cp.asarray(imread_fcn(f), dtype = np.float32)
            temp = -cp.log((im_temp-self.df_)/(self.ff_-self.df_))
            temp = median_gpu(cp.array(temp), (med_kernel,med_kernel))
            temp[~cp.isfinite(temp)] = 0
            combined[i,:,:] = cp.asnumpy(temp)

        combined = np.sum(combined, axis = 0)
        if self.apply_thresh is not None:
            logging.info(f"Applying threshold to combined image ({apply_thresh})")
            combined[combined < apply_thresh] = 0

        self.x_max,self.y_max = combined.shape
        self.combined = combined


    def update_crop_norm(   self,
                            crop_y0: int,
                            crop_dy: int,
                            crop_x0: int,
                            crop_dx: int,
                            norm_y0: int,
                            norm_dy: int,
                            norm_x0: int,
                            norm_dx: int,
                            tpose: str
                            ) -> None:
        """
        """
        self.combined_local = self.combined.copy()
        if tpose == 'True':
            self.combined_local = self.combined_local.T
            self.data_dict['pre_processing']['transpose'] = True
        elif tpose == 'False':
            self.combined_local = self.combined_local
            self.data_dict['pre_processing']['transpose'] = False
        self.crop_patch = [crop_y0,crop_y0+crop_dy,crop_x0,crop_x0+crop_dx]
        self.norm_patch = [norm_y0,norm_y0+norm_dy,norm_x0,norm_x0+norm_dx]
        self.plot_crop_norm()
    
    def plot_crop_norm(self):
        self.ax[0].clear()
        self.ax[0].imshow(self.combined_local)#, vmin = l, vmax = h)
        x0,y0 = self.crop_patch[0],self.crop_patch[2]
        dy = self.crop_patch[3]-self.crop_patch[2]
        dx = self.crop_patch[1]-self.crop_patch[0]
        crop_rectangle = Rectangle((x0,y0),dx,dy, 
                                                fill = False,
                                                color = 'r',
                                                linestyle = '--',
                                                linewidth = 2)
        self.ax[0].add_artist(crop_rectangle)
        #plot_patch(self.norm_patch, ax[0], color = 'w')
        # Visualize 0 and 180 degrees with crop patch and normalization
        # patch highlighted
        norm_patch = self.norm_patch
        if norm_patch[0] != norm_patch[1] and norm_patch[2] != norm_patch[3]:
            x0_,y0_ = self.norm_patch[0],self.norm_patch[2]
            dy_ = self.norm_patch[3]-self.norm_patch[2]
            dx_ = self.norm_patch[1]-self.norm_patch[0]
            norm_rectangle = Rectangle((x0_,y0_),dx_,dy_, 
                                            fill = False,
                                            color = 'w',
                                            linestyle = '-',
                                            linewidth = 1)
            self.ax[0].add_artist(norm_rectangle)

            self.ax[0].text(norm_patch[0],norm_patch[2],
                                            'Norm Patch',
                                            verticalalignment = 'bottom',
                                            horizontalalignment = 'left',
                                            color = 'w',
                                            rotation = 90)
    

    def cor_calculate(self,cor_y0,cor_y1):
            ##---------------------------------------------------------------------
            ##  DEBUGGING TIP: OVERRIDING THE CONTROL OF THE INTERACT TO DEBUG
            #crop_patch = [356,1982,731,2385]
            #norm_patch = [111,251,192,2385]
            ##---------------------------------------------------------------------
            #plt.close('all')
            self.ax[1].clear()
            self.ax[1].axis(True)
            #fig,ax = plt.subplots(1,2, figsize = (10,5))
            #plt.show()
            self.y0,self.y1 = cor_y0,cor_y1
            y0,y1 = cor_y0, cor_y1
            crop_patch = self.crop_patch
            norm_patch = self.norm_patch
            if y1 > crop_patch[3]-crop_patch[2]:
                print("COR y1 exceeds window size")
            else:
                #--------------------------------------------------------------
                # Trying to be Smart about the Figures
                #--------------------------------------------------------------
                self.plot_crop_norm()

                slice_x = slice(crop_patch[0],crop_patch[1])
                slice_y = slice(crop_patch[2],crop_patch[3])
                cor_image = self.combined_local[slice_y,slice_x]
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
                cor_image2 = self.combined_local[slice_y_corr,slice_x_corr]
                cor_image2[~np.isfinite(cor_image2)] = 0
                cor_image2 = rotate_cpu(cor_image2,-theta,reshape=False)
                # Adjusted cor image re-cropped to pad
                cor3 = center_of_rotation(  cor_image2,
                                            y0,
                                            y1,
                                            ax = self.ax[1],
                                           image_center = True)
                self.ax[1].set_title(f"Rotated ({theta:.3f} deg) and Cropped")
                xy = (crop_patch[0],crop_patch[2])
                dx = crop_patch[1]-crop_patch[0]
                dy = crop_patch[3]-crop_patch[2]
                adjusted_crop = Rectangle(xy,dx,dy, color = 'b', fill = False)
                self.ax[0].add_artist(adjusted_crop)
                self.ax[0].text(crop_patch[1],crop_patch[2],
                                    'Crop Patch Centered',
                                    verticalalignment = 'bottom',
                                    horizontalalignment = 'left',
                                    color = 'b',
                                    rotation = 90)
                print(f"crop_patch:{crop_patch}")
                print(f"self.crop_patch:{self.crop_patch}")
                self.fig.tight_layout()
                self.data_dict['crop']['x0'] = crop_patch[0]
                self.data_dict['crop']['x1'] = crop_patch[1]
                self.data_dict['crop']['y0'] = crop_patch[2]
                self.data_dict['crop']['y1'] = crop_patch[3]
                self.data_dict['crop']['dx'] = crop_patch[1]-crop_patch[0]
                self.data_dict['crop']['dy'] = crop_patch[3]-crop_patch[2]
                self.data_dict['norm']['x0'] = norm_patch[0]
                self.data_dict['norm']['x1'] = norm_patch[1]
                self.data_dict['norm']['y0'] = norm_patch[2]
                self.data_dict['norm']['y1'] = norm_patch[3]
                self.data_dict['norm']['dx'] = norm_patch[1]-norm_patch[0]
                self.data_dict['norm']['dy'] = norm_patch[3]-norm_patch[2]
                self.data_dict['COR']['y0'] = cor_y0
                self.data_dict['COR']['y1'] = cor_y1
                self.data_dict['COR']['theta'] = theta


    def interact(   self,  
                    figsize : tuple = (10,5),
                    apply_thresh: float = None,
                    med_kernel = 3
                    ) -> None:
        """
        """
        data_dict = self.data_dict
        imread_fcn = lambda x: np.array(Image.open(x), dtype = np.float32)
        # This is for calculating vmin and vmax for imshows
        self.fig,self.ax = plt.subplots(1,2, figsize = figsize)
        self.ax[1].axis(False)
        plt.show()

        #---------------------------------------------------------------------------
        # NOTE TO SELF:
        #   These assignements look backward (x0, 0 -> y_max, etc.) but they are
        #   fixing the problem of plt.imshow transposing the image
        #---------------------------------------------------------------------------
        kwargs = {'continuous_update':False,'min':0}

        max_crop = max([self.y_max,self.x_max])
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
        ui = VBox([row0,row1,row2])

        control_dict = {
                    'crop_y0':crop_x0,
                    'crop_dy':crop_dx,
                    'crop_x0':crop_y0,
                    'crop_dx':crop_dy,
                    'norm_x0':norm_y0,
                    'norm_dx':norm_dy,
                    'norm_y0':norm_x0,
                    'norm_dy':norm_dx,
                    'tpose':tpose
                        }

        out = interactive_output(self.update_crop_norm, control_dict)
        display(ui,out)

        interact_2 = interactive.factory()
        manual_2 = interact_2.options(manual = True,
                                    manual_name = 'Refresh COR Calc'
                                    )
        out_2 = manual_2(   self.cor_calculate,
                            cor_y0 = cor_y0,
                            cor_y1 = cor_y1,
                            name = "Refresh COR")
