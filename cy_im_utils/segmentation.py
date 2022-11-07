""" This is basically my implementation of the bivariate histogram segmentation
from NIF Volume Fusion. To run the interactive functions the instance needs to
be in a jupyter notebook. The processing can be executed externally.
"""
from PIL import Image
from cupyx.scipy.ndimage import median_filter as median_gpu
from functools import partial
from ipywidgets import interactive_output, HBox, VBox, IntSlider,interact_manual,interactive,RadioButtons,Text
from matplotlib.widgets import PolygonSelector
from multiprocessing import Pool
from pathlib import Path
from scipy.ndimage import median_filter
from tqdm import tqdm
import cupy as cp
import logging
import matplotlib as mpl
import matplotlib.path as mpltPath
import matplotlib.pyplot as plt
import numpy as np
import pickle
import logging

def imread(x): return np.array(Image.open(x), dtype = np.float32)

def apply_segmentation( neutron,
                        x_ray,
                        phases,
                        down_sampling = 1,
                        med_kernel: int = 5
                        ):
    """ this can be done with multi-processing or MPI...
    """
    segmented = np.zeros_like(neutron, dtype = np.uint8)
    nx,ny = neutron.shape
    stacked_arr = np.stack([neutron.ravel(),x_ray.ravel()]).T
    temp_array = np.zeros_like(neutron, dtype = np.uint8)
    for j,(label,phase) in enumerate(phases.items()):
        polygon_path = mpltPath.Path(phase)
        mask = polygon_path.contains_points(stacked_arr)
        mask = mask.reshape(nx,ny)
        segmented[mask] = j+1
    unlabeled = segmented == 0
    med_filtered = median_filter(segmented.astype(np.float32), (med_kernel,med_kernel))
    segmented[unlabeled] = med_filtered[unlabeled]
    return segmented

def process_frame(  frame_no: int, 
                    neutron_files: list,    # provide partial
                    x_ray_files: list,      # provide partial
                    phases: dict,           # provide partial
                    pad: tuple,             # provide partial
                    ) -> None:
    """ This is meant to be called by multiprocessing.pool
    """
    neutron = np.pad(imread(neutron_files[frame_no]),pad)
    x_ray = imread(x_ray_files[frame_no])
    return apply_segmentation(  neutron,
                                x_ray,
                                phases)

class bivariate_segmentation:
    def __init__(self,
                 neutron_files: list,
                 x_ray_files: list,
                 method: str, 
                 phases: dict = {},
                 segmentation_cmap: str = 'gist_ncar',
                neutron_array = None,
                x_ray_array = None):
        """ optionally pass neutron_array and x_ray_array to the constructor
        """
        self.method = method
        self.neutron_files = neutron_files
        self.x_ray_files = x_ray_files
        self.phases = phases
        self.colors = 'rgbcmyk'
        self.segmentation_cmap = segmentation_cmap
        self.plot_axis = {'xy':0,'yz':1,'xz':2}
        if neutron_array is None: 
            logging.info("reading neutron files")
            self.neutron = self.imstack_read(neutron_files)
        else:
            self.neutron = neutron_array
        if x_ray_array is None:
            logging.info("reading x-ray files")
            self.x_ray = self.imstack_read(x_ray_files)
        else:
            self.x_ray = x_ray_array

        
    def imread(self, x): return np.array(Image.open(x), dtype = np.float32)

    def contrast(self, arr):
        return np.quantile(arr,0.01), np.quantile(arr, 0.99)

    def plot_registration(  self,
                            image_idx: int = 100,
                            multiplicative: float = 5.0,
                            figsize = (12,4)
                            ) -> None:
        """ this plots a z-frame to visualize the registration between 
        """
        neutron_im = self.neutron[image_idx]
        x_ray_im = self.x_ray[image_idx]
        stacked = np.dstack([multiplicative*neutron_im,x_ray_im,x_ray_im])
        arrs = [neutron_im,x_ray_im,stacked]
        labels = ['neutron','x-ray','composite']
        fig,ax = plt.subplots(1,3, sharex= True, sharey = True, 
                                                            figsize = figsize)
        for i,(label,arr) in enumerate(zip(labels,arrs)):
            ax[i].imshow(arr)
            ax[i].set_title(label)
        fig.tight_layout()    

    def imstack_read(self, files):
        """ same as always
        """
        nz = len(files)
        nx,ny = self.imread(files[0]).shape
        imstack = np.zeros([nz,nx,ny])
        for i in tqdm(range(nz)):
            imstack[i] = self.imread(files[i])
        return imstack
    
    def compute_histogram(  self,
                            down_sampling: int = 100,
                            bins: int = 1000
                            ) -> None:
        """ GPU operation to calculate the histogram
        """
        neutron_slice = cp.array(self.neutron.ravel()[::down_sampling],
                                                        dtype = cp.float32)
        x_ray_slice = cp.array(self.x_ray.ravel()[::down_sampling],
                                                        dtype = cp.float32)
        counts,x_coords,y_coords = cp.histogram2d(  neutron_slice,
                                                    x_ray_slice,
                                                    bins = bins)
        self.x_coords = x_coords.get()
        self.y_coords = y_coords.get()
        self.counts = counts.T.get()
    
    
    def plot_hist(self, ax = None, figsize: tuple = (8,8) , kwargs: dict = {}):
        """ this launches the matplotlib interactive 2d histogram plot. if
        phases have already been added, they are overlain over the histogram
        extra space is added to the edges to make segmenting at the edges
        easier
        """
        if ax is None:
            fig,ax = plt.subplots(1,1)
        ax.pcolorfast(self.x_coords,self.y_coords,self.counts, **kwargs)
        for i,(label,phase) in enumerate(self.phases.items()):
            polygon = mpl.patches.Polygon(  phase,
                                            fill = True,
                                            color = self.colors[i],
                                            alpha = 1/3,
                                            linestyle = (0,(5,5)))
            ax.add_artist(polygon)
        self.selector = PolygonSelector(ax,lambda x: None)
        ax.set_xlabel('neutron attenuation (mm$^{-1}$)')
        ax.set_ylabel('x-ray attenuation (mm$^{-1}$)')
        x_additional = 0.1*(self.x_coords.max()-self.x_coords.min())
        x_min = self.x_coords.min()-x_additional
        x_max = self.x_coords.max()+x_additional
        y_additional = 0.1*(self.y_coords.max()-self.y_coords.min())
        y_min = self.y_coords.min()-y_additional
        y_max = self.y_coords.max()+y_additional
        ax.set_xlim(x_min,x_max)
        ax.set_ylim(y_min,y_max)
        
    def interactive_apply_thresh(self,frame_no, plot_axis) -> None:
        """ This is called by the interactive function
        - If no vertices are selected and no phases created: do nothing
        - If no vertices are selected and phases exist: show phases
        - If no vertices are selected show segmented pixels
            - if phases exist also show phases
        """
        self.global_frame = frame_no
        self.global_axis = plot_axis

        slice_ = self.fetch_axis_slice( frame_no,
                                        self.plot_axis[plot_axis])

        neutron_image = self.neutron[slice_]
        x_ray_image = self.x_ray[slice_]
        for i,arr in enumerate([neutron_image,x_ray_image]):
            self.biv_ax[i+3].clear()
            self.biv_ax[i+3].axis(True)
            #vmin,vmax = self.contrast(arr)
            self.biv_ax[i+3].imshow(arr)#, vmin = vmin, vmax = vmax)
       
        # If vertices are empty (i.e., no vertices selected)
        if self.selector.verts == []:
            print("No Vertices Selected")
            # No Phases assigned
            if self.phases != {}:
                self.interactive_show_segmentation()
        # Vertices are selected
        elif self.selector.verts != []:
            print("")
            self.biv_ax[1].clear()
            self.biv_ax[1].axis(True)

            phases = {"":self.selector.verts}
            self.plot_segmented_frame(frame_no,
                                      ax = self.biv_ax[1],
                                      phases = phases,
                                      axis = self.plot_axis[plot_axis],
                                      imshow_kwargs = {'cmap':'gray'}
                                     )
            if self.phases != {}:
                self.interactive_show_segmentation()
                
    def interactive_show_segmentation(self) -> None:
        """ This populates the third subplot to show the state of the current
        segmentation
        """
        axis = self.plot_axis[self.global_axis]
        self.biv_ax[2].axis(True)
        self.plot_segmented_frame(self.global_frame,
                        ax = self.biv_ax[2],
                        axis = axis,
                        imshow_kwargs = {'cmap' : self.segmentation_cmap})
        # Updating Segmentation Plot
        cmap_ = getattr(mpl.cm, self.segmentation_cmap)
        bounds = np.arange(0,len(self.phases)+2,1)
        norm = mpl.colors.BoundaryNorm(bounds, cmap_.N)
        try:
            self.cbar.remove()
        except:
            pass
        self.cbaxes = plt.gcf().add_axes([0.9,0.1,0.03,0.8])
        mappable = mpl.cm.ScalarMappable(norm = norm, cmap = cmap_)
        self.cbar = plt.gcf().colorbar( mappable,
                                        ticks = bounds+0.5,
                                        shrink = 0.5,
                                        cax = self.cbaxes)
        self.cbar.set_ticklabels(["unlabeld"]+list(self.phases.keys()))

                
            
    def integrate_phase(self,name) -> None:
        """ this is called by the interactive function
            - adds the selected (vertices) to the phase dictionary
            - re-plots the histogram with the vertices overlaid
              (semi-transparent)
            - clears the segmentation  plot(axis 2) and re-lots the new
              segmentation
        """
        self.phases[name] = self.selector.verts
        self.biv_ax[0].clear()
        self.plot_hist(self.biv_ax[0], kwargs = self.hist_kwargs)
        self.interactive_show_segmentation()
            
    def bivariate_interact( self,
                            hist_kwargs: dict = {},
                            seg_kwargs:dict = {},
                            figsize = (12,5)
                            ) -> None:
        """ Calling this method launches the interactive plotting widget to
        segment the volume. 
            - The first column shows the 2d histogram (interactive selection)
            - the second column shows the actively selected voxels
            - the third column shows the current segmentation (all phases)
                - colorbar takes phase names
        """
        #fig = plt.figure(figsize = figsize)
        #n_col = 5
        #ax = [plt.subplot(1,n_col,1)]
        #ax.append(plt.subplot(1,n_col,2))
        #ax.append(plt.subplot(1,n_col,3, sharex = ax[1], sharey = ax[1]))
        #ax.append(plt.subplot(1,n_col,4, sharex = ax[1], sharey = ax[1]))
        #ax.append(plt.subplot(1,n_col,5, sharex = ax[1], sharey = ax[1]))
        gs = mpl.gridspec.GridSpec(2,3)
        fig = plt.figure(figsize = figsize)
        ax = [fig.add_subplot(gs[:,0])]
        ax.append(fig.add_subplot(gs[0,1]))
        ax.append(fig.add_subplot(gs[0,2], sharex = ax[1],sharey = ax[1]))
        ax.append(fig.add_subplot(gs[1,1], sharex = ax[1],sharey = ax[1]))
        ax.append(fig.add_subplot(gs[1,2], sharex = ax[1],sharey = ax[1]))
        for a in ax[1:]:
            a.axis(False)
        self.biv_ax = ax
        self.hist_kwargs = hist_kwargs
        self.plot_hist(self.biv_ax[0], kwargs = self.hist_kwargs)
        fig.subplots_adjust(left = 0.05, right = 0.85)
        max_frames = max(self.neutron.shape)
        frame_no = IntSlider(description = 'Frame',
                             min = 0,
                             max = max_frames,
                             value = max_frames//2
                            )

        plot_axis = RadioButtons(options = ['xy','yz','xz'],
                            description = "Plotting Plane",
                            value = 'xy'
                           )
        
        
        phase_add = Text(placeholder = '_',description = 'Phase Name')
        
        interact = interactive.factory()
        manual = interact.options(manual = True, manual_name = 'View Selection')
        out = manual(   self.interactive_apply_thresh,
                        frame_no = frame_no,
                        plot_axis = plot_axis)
        
        interact_2 = interactive.factory()
        manual_2 = interact.options(manual = True, manual_name = 'Add Phase')
        out_2 = manual_2(self.integrate_phase, name = phase_add)
        
    def fetch_axis_slice(self,frame,axis):
        nz,nx,ny = self.neutron.shape
        if axis == 0:
            slice_ = (frame,slice(0,nx),slice(0,ny))
        elif axis == 1:
            slice_ = (slice(0,nz),frame,slice(0,ny))
        elif axis == 2:
            slice_ = (slice(0,nz),slice(0,nx),frame)
        return slice_

    def apply_segmentation( self,
                            frame,
                            phases,
                            axis: int = 0,
                            median_unlabeled = 1,
                            median_image = 1
                            ) -> np.array:
        """ This takes a given set of vertices closed polygons) and segments
        a single frame
        """
        slice_ = self.fetch_axis_slice(frame, axis)
        neutron = self.neutron[slice_]
        x_ray = self.x_ray[slice_]
        nx,ny = neutron.shape
        stacked_arr = np.stack([neutron.ravel(),x_ray.ravel()]).T
        segmentation = np.zeros([nx,ny], dtype = np.uint8)
        for j,(_,phase) in enumerate(phases.items()):
            polygon_path = mpltPath.Path(phase)
            mask = polygon_path.contains_points(stacked_arr)
            mask = mask.reshape(nx,ny)
            segmentation[mask] = j+1
        # replace unlabeled with median values
        if median_unlabeled > 1:
            unlabeled = segmentation == 0
            med_filtered = median_gpu(cp.array(segmentation, dtype = cp.float32),
                                            median_unlabeled).get()
            segmentation[unlabeled] = med_filtered[unlabeled]
        if median_image > 1:
            segmentation = median_gpu(cp.array(segmentation, dtype= cp.float32),
                                            median_image).get()
        return segmentation
    
    def plot_segmented_frame(   self,
                                frame,
                                axis = 0,
                                ax = None,
                                figsize = (8,8),
                                imshow_kwargs: dict = {},
                                phases: dict = None
                                ) -> None:
        """ this wil plot the current segmentation on a provided axes along
        axis...
        Note phases can be passed as an argument to override the stateful phases
        """
        if phases is None:
            phases = self.phases
        segmentation = self.apply_segmentation(frame,
                            phases,
                            axis = axis)#, median_unlabeled = 5, median_image = 9)
        if ax is None:
            fig,ax = plt.subplots(1,1, figsize = figsize)
        cmap = 'gray' if 'cmap' not in imshow_kwargs else plt.rcParams['image.cmap']
        ax.imshow(segmentation, **imshow_kwargs)
    
    def save_phases(self, f_name) -> None:
        """ wrapper to write the phases to a dictionary
        """
        pickle.dump(self.phases,open(f_name,'wb'))

    def process_volume_serial(self) -> None:
        """ Deprecated
        """
        logging.warning("USE MPI or MULTIPROCESSING TO SPEED THIS UP" )
        nz = self.neutron.shape[0]
        self.full_segmentation = np.zeros_like(self.neutron, dtype = np.uint8)
        for i in tqdm(range(nz)):
            self.full_segmentation[i] = self.apply_segmentation(i,
                                                        phases = self.phases)
            
    def process_volume_parallel(self, 
                                num_proc = None) -> None:
        """ This is a wrapper to call multiprocessing externally for processing
        frames. check the task manager to see that its working correctly.

        Creates a new array member self.segmentation
        """
        if 'fbpconvnet' in self.method.lower():
            pad = ((64,64),(64,64))
        else:
            pad = ((0,0),(0,0))
        process_handle = partial(process_frame,
                                 neutron_files = self.neutron_files,
                                 x_ray_files = self.x_ray_files,
                                 phases = self.phases,
                                 pad = pad,
                                )
        nz = len(self.neutron_files)
        logging.info('Multi Processing Segmentation')
        with Pool(processes = num_proc) as pool:
            self.segmentation = np.stack(pool.map(process_handle, tqdm(range(nz))))
