from PIL import Image
from pathlib import Path
from tqdm import tqdm
import logging
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pygpufit.gpufit as gf

from .visualization import plot_patch,colors

class bragg_dataset:
    def __init__(self,
                 bragg_path: Path,
                 wavelengths: list,
                 extension: str = ".tif",
                 flipud: bool = False
                ) -> None:
        """
        Args:
        -----
            bragg_path: Path - this is the path to the directory that
                                holds all the individual wavelengths'
                                directories
            wavelengths: list - list of strings of all the wavelenghts
            extension: str - file extension to match (.tif, .fit, etc.)
            flipud: bool - this will flip the images up-down-wise, which is
                            necessary for certain reconstructions (e.g.,
                            TV-SIRT)
            
        """
        self.bragg_path = bragg_path
        self.wavelengths = wavelengths 
        self.wavelengths_float = np.array([float(w.replace("d",".")) for w in wavelengths], dtype = np.float32)
        self.extension = extension
        self.n_wavelengths = len(wavelengths)
        #files = self.fetch_wavelength_files(self.wavelengths[0])
        self.flipud = flipud
        self.monochromatic_files = self.fetch_monochromatic_files()
        self.nx,self.ny = self.load_im(self.monochromatic_files[wavelengths[0]][0]).shape
        self.n_reconstructions = len(self.monochromatic_files[wavelengths[0]])

    def fetch_monochromatic_files(self) -> dict:
        """
        This should be faster than globbing each time
        """
        monochromatic_files = {}
        tqdm_monochr = tqdm(self.wavelengths,
                desc = "reading monochromatic file sets by wavelength")
        for w in tqdm_monochr:
            monochromatic_files[w] = sorted(list(
                                self.bragg_path.glob(f"*{w}*{self.extension}")
                                ))
        return monochromatic_files


    def calc_n_reconstructions(self) -> int:
        lambda_0 = self.wavelengths[0]
        files = self.fetch_wavelength_files(self.wavelengths[0])
        return len(files)

    
    def fetch_wavelength_files(self, wavelength: str) -> list:
        """
        Globs the wavelength string and returns the files that are associated
        """
        regex = "*{self.wavelengths}*{self.extension}"
        return list(sorted(self.bragg_path.glob(regex)))
        
    def load_im(self, file, dtype = np.float32) -> np.array:
        """
        Wrapper for loading in an image
        """
        return np.asarray(Image.open(file), dtype = dtype)
    
    def fetch_slice_const_wavelength(   self,
                                        frame: int, dtype = np.float32
                                        ) -> np.array:
        """
        This will return a volume of a single reconstruction image for every
        wavelength: 3D array
        """
        nz,nx,ny = self.n_wavelengths,self.nx,self.ny
        output = np.zeros([nz, nx, ny], dtype = dtype)
        #for i,wavelength in tqdm(enumerate(self.wavelengths)):
        for i,wavelength in enumerate(self.wavelengths):
            try:
                file = self.monochromatic_files[wavelength][frame]
                im_ = self.load_im(file)
            except IndexError as ie:
                logging.warning(f"index error with wavelength : {wavelength}")
                im_ = np.ones([nx,ny], dtype = dtype) * np.nan
            output[i] = im_
            
        if self.flipud:
            return output[:,::-1,:]
        else:
            return output
            
    def transform_3d_to_2d(self, arr: np.array) -> np.array:
        """
        This method converts a 3d array with axis 0 as wavelength and axis 1
        and 2 as spataial dimensions into a 2d array for Gpufit

        Args:
        -----
            arr: np.array - 3D with axis 0 - lambda; axis 1 , 2 x and y

        Returns:
        --------
            reshaped np.array - 2D with len(axis 0) = nx*ny and len(axis 1) =
                                n_lambda
        """
        n_lambda,nx,ny = arr.shape
        return np.reshape(np.transpose(arr,(1,2,0)),(nx*ny,n_lambda)).copy()


    def fit_to_erf( self,
                    data: np.array,
                    wavelengths: np.array,
                    tolerance: float,
                    max_number_iterations: int,
                    initial_parameters: np.array,
                    print_output: bool = True
                    ) -> list:
        """
        """
        assert data.dtype == np.float32, "Data must be float32"
        assert wavelengths.dtype == np.float32, "Wavelengths must be float32"
        number_fits,n_lambda = data.shape
        #initial_parameters = np.tile(initial_parameters, 
        #                             (number_fits,1)
        #                             ).astype(np.float32)
        estimator_id = gf.EstimatorID.MLE
        model_id = gf.ModelID.ERROR_FUNCTION
        out = gf.fit(   data = data,
                        weights = None,
                        model_id = model_id,
                        initial_parameters = initial_parameters,
                        tolerance = tolerance,
                        max_number_iterations = max_number_iterations,
                        parameters_to_fit = None,
                        estimator_id = estimator_id,
                        user_info = wavelengths
                        )

        if print_output: self.print_fit(out, model_id, number_fits, n_lambda)

        return out

    def print_fit(  self,
                    out: list,
                    model_id: int,
                    number_fits: int,
                    size_x: int,
                    ) -> None:
        """
        This is the generic outputting from the basic gpufit tutorial
        """
        parameters,states,chi_squares,number_iterations,execution_time = out
        converged = states == 0
        print("*Gpufit*")

        # print summary
        print(f'\nmodel ID:      {model_id}')
        print(f'number of fits:  {number_fits}')
        print(f'fit size:        {size_x}')
        print(f'mean chi_square: {np.mean(chi_squares[converged]):.2f}')
        print(f'iterations:      {np.mean(number_iterations[converged]):.2f}')
        print(f'time:            {execution_time:.2f} s')

        # get fit states
        number_converged = np.sum(converged)
        print(f'\nratio converged       {number_converged    / number_fits * 100:6.2f} %')
        print(f'ratio max it. exceeded  {np.sum(states == 1) / number_fits * 100:6.2f} %')
        print(f'ratio singular hessian  {np.sum(states == 2) / number_fits * 100:6.2f} %')
        print(f'ratio neg curvature MLE {np.sum(states == 3) / number_fits * 100:6.2f} %')

    def roi_image(  self,
                    frame: int,
                    rois: list,
                    plot_index: int = 0,
                    ax: list = [],
                    linewidth = 2
                    ) -> None:
        """
        Args:
        -----
            frame: int - the reconstruction frame to slice along wavelenghts
            rois: list - list of regions of interest (list of lists). note the
                        format of the rois is:
                            [x0,dx,y0,dy], the roi variable converts this into
                            [x0,x1,y0,y1]
            plot_index: int - the index of the image that gets plotted
            ax: array-like - needs at least two elements one for the image and
                            roi rectangles and one for the plots as a function
                            of wavelength
        """
        slice_volume = self.fetch_slice_const_wavelength(frame)
        ax[0].imshow(slice_volume[plot_index])
        for i,roi_ in enumerate(rois):
            roi = [roi_[0],roi_[0]+roi_[1],roi_[2],roi_[2]+roi_[3]]
            slice_x = slice(roi[2],roi[3],1)
            slice_y = slice(roi[0],roi[1],1)
            slice_1d = np.mean(slice_volume[:,slice_x,slice_y], axis = (1,2))
            ax[1].scatter(self.wavelengths_float,slice_1d, color = colors(i))
            plot_patch(roi, ax[0], color = colors(i), linewidth = linewidth)
