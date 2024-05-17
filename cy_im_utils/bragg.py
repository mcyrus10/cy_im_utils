from PIL import Image
from cupyx.scipy.ndimage import median_filter, gaussian_filter1d
from cupyx.scipy.special import erf as gpu_erf
from pathlib import Path
from scipy.special import erf
from tqdm import tqdm
import cupy as cp
import logging
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pygpufit.gpufit as gf

from .visualization import plot_patch, colors, contrast


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
        self.wavelengths_float = np.array([float(w.replace("d", ".")) for w in wavelengths], dtype=np.float32)
        self.extension = extension
        self.n_wavelengths = len(wavelengths)
        # files = self.fetch_wavelength_files(self.wavelengths[0])
        self.flipud = flipud
        self.monochromatic_files = self.fetch_monochromatic_files()
        self.nx, self.ny = self.load_im(self.monochromatic_files[wavelengths[0]][0]).shape
        self.n_reconstructions = len(self.monochromatic_files[wavelengths[0]])

    def fetch_monochromatic_files(self) -> dict:
        """
        This should be faster than globbing each time
        """
        monochromatic_files = {}
        tqdm_monochr = tqdm(
                        self.wavelengths,
                        desc="reading monochromatic file sets by wavelength")
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

    def load_im(self, file, dtype=np.float32) -> np.array:
        """
        Wrapper for loading in an image
        """
        return np.asarray(Image.open(file), dtype=dtype)

    def fetch_slice_const_wavelength(self,
                                     frame: int, dtype=np.float32
                                     ) -> np.array:
        """
        This will return a volume of a single reconstruction image for every
        wavelength: 3D array
        """
        nz, nx, ny = self.n_wavelengths, self.nx, self.ny
        output = np.zeros([nz, nx, ny], dtype=dtype)
        # for i,wavelength in tqdm(enumerate(self.wavelengths)):
        for i, wavelength in enumerate(self.wavelengths):
            try:
                file = self.monochromatic_files[wavelength][frame]
                im_ = self.load_im(file)
            except IndexError as ie:
                logging.warning(f"index error with wavelength : {wavelength}")
                im_ = np.ones([nx, ny], dtype=dtype) * np.nan
            output[i] = im_

        if self.flipud:
            return output[:, ::-1, :]
        else:
            return output

    def transform_3d_to_2d(self, arr: np.array) -> np.array:
        """
        This method converts a 3d array with axis 0 as wavelength and axis 1
        and 2 as spataial dimensions into a 2d array for Gpufit

        Args:
        -----
            arr: np.array - 3D with axis 0 - lambda axis 1 , 2 x and y

        Returns:
        --------
            reshaped np.array - 2D with len(axis 0) = nx*ny and len(axis 1) =
                                n_lambda
        """
        n_lambda, nx, ny = arr.shape
        return np.reshape(np.transpose(arr, (1, 2, 0)), (nx*ny, n_lambda)).copy()

    def fit_to_erf(self,
                   data: np.array,
                   wavelengths: np.array,
                   tolerance: float,
                   max_number_iterations: int,
                   initial_parameters: np.array,
                   print_output: bool = True,
                   weights: np.array = None,
                   estimator_id=gf.EstimatorID.MLE
                   ) -> list:
        """
        """
        assert data.dtype == np.float32, "Data must be float32"
        assert wavelengths.dtype == np.float32, "Wavelengths must be float32"
        number_fits, n_lambda = data.shape
        # initial_parameters = np.tile(initial_parameters,
        #                             (number_fits,1)
        #                             ).astype(np.float32)
        model_id = gf.ModelID.ERROR_FUNCTION
        out = gf.fit(data=data,
                     weights=weights,
                     model_id=model_id,
                     initial_parameters=initial_parameters,
                     tolerance=tolerance,
                     max_number_iterations=max_number_iterations,
                     parameters_to_fit=None,
                     estimator_id=estimator_id,
                     user_info=wavelengths
                     )

        if print_output:
            self.print_fit(out, model_id, number_fits, n_lambda)

        return out

    def print_fit(self,
                  out: list,
                  model_id: int,
                  number_fits: int,
                  size_x: int,
                  ) -> None:
        """
        This is the generic outputting from the basic gpufit tutorial
        """
        parameters, states, chi_squares, number_iterations, execution_time = out
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

    def roi_image(self,
                  frame: int,
                  rois: list,
                  plot_index: int = 0,
                  ax: list = [],
                  linewidth: float = 2
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
        for i, roi_ in enumerate(rois):
            roi = [roi_[0], roi_[0]+roi_[1], roi_[2], roi_[2]+roi_[3]]
            slice_x = slice(roi[2], roi[3], 1)
            slice_y = slice(roi[0], roi[1], 1)
            slice_1d = np.mean(slice_volume[:, slice_x, slice_y], axis=(1, 2))
            ax[1].scatter(self.wavelengths_float, slice_1d, color=colors(i))
            plot_patch(roi, ax[0], color=colors(i), linewidth=linewidth)


class erf_gauss_fit:
    """
    this is for making the composite erf - gaussian center reconstructions for
    the limestone data
    """
    def __init__(self,
                 inst,
                 frame_idx,
                 var_erf=1e-6,
                 var_gauss=1e-6,
                 long_short='short',
                 short_min=0,
                 short_max=5):
        self.inst = inst
        self.test_frame = inst.fetch_slice_const_wavelength(frame_idx)
        self.long_short = long_short

        if long_short == 'long':
            self.cutoff_idx = inst.wavelengths_float > 5
        elif long_short == 'short':
            cidx_1 = inst.wavelengths_float < short_max
            cidx_2 = inst.wavelengths_float > short_min
            self.cutoff_idx = cidx_1 * cidx_2
        self.lambda_local = inst.wavelengths_float[self.cutoff_idx]
        self.lambda_local_erf = self.lambda_local
        self.shape = self.test_frame.shape
        n_elem = np.sum(self.cutoff_idx)
        self.erf_weights = np.ones(n_elem).astype(np.float32) / var_erf
        self.gauss_weights = np.ones(n_elem-1).astype(np.float32) / var_gauss
        #print('weights shape (init) = ', self.gauss_weights.shape)
        n_lambda, nx, ny = self.shape
        self.data = np.reshape(np.transpose(
                               self.test_frame, (1, 2, 0)), (nx*ny, n_lambda)
                               )[:, self.cutoff_idx].copy()

    def _gpu_fit_wrapper_(self,
                        weights,
                        model_id,
                        data,
                        lambda_local,
                        initial_parameters,
                        tolerance=1e-5,
                        max_number_iterations=50,
                        verbose=False):
        test_frame = self.test_frame
        n_lambda, nx, ny = test_frame.shape
        number_fits = nx*ny
        estimator_id = gf.EstimatorID.LSE
        output = gf.fit(
                        data=data.astype(np.float32),
                        weights=np.tile(weights, (number_fits, 1)),
                        model_id=model_id,
                        initial_parameters=initial_parameters,
                        tolerance=tolerance,
                        max_number_iterations=max_number_iterations,
                        parameters_to_fit=None,
                        estimator_id=estimator_id,
                        user_info=lambda_local.astype(np.float32)
                      )

        # ---------------------------------------------------------------------
        # FYI:
        parameters, states, chi_squares, number_iterations, execution_time = output
        # ---------------------------------------------------------------------
        if verbose:
            ndof = len(lambda_local) - 4
            reduced_chi_squares = chi_squares/ndof
            converged = states == 0
            print("*Gpufit*")

            # print summary
            print(f'\nmodel ID:       {model_id}')
            print(f'number of fits:   {number_fits}')
            # print(f'fit size:        {size_x} x {size_x}')
            print(f'mean chi_square:  {np.mean(chi_squares[converged]):.2f}')
            print(f'median chi_square:{np.median(chi_squares[converged]):.2f}')
            print(f'iterations:       {np.mean(number_iterations[converged]):.2f}')
            print(f'time:             {execution_time:.2f} s')

            # get fit states
            number_converged = np.sum(converged)
            print(f'\nratio converged       {number_converged    / number_fits * 100:6.2f} %')
            print(f'ratio max it. exceeded  {np.sum(states == 1) / number_fits * 100:6.2f} %')
            print(f'ratio singular hessian  {np.sum(states == 2) / number_fits * 100:6.2f} %')
            print(f'ratio neg curvature MLE {np.sum(states == 3) / number_fits * 100:6.2f} %')

        return output

    def erf_fit(self,
                initial_parameters=None,
                verbose: bool = False) -> tuple:
        """
        Wrapper for this call (also calculates uncertainty)
        """
        lambda_local = self.lambda_local
        n_lambda, nx, ny = self.shape
        if initial_parameters is None:
            initial_parameters = init_params_calc(self.data,
                                                  lambda_local,
                                                  visualize=False,
                                                  sigma_sign=True)
        data = self.data
        output = self._gpu_fit_wrapper_(
                            weights=self.erf_weights,
                            data=data,
                            model_id=gf.ModelID.ERROR_FUNCTION,
                            initial_parameters=initial_parameters,
                            lambda_local=lambda_local.astype(np.float32),
                            verbose=verbose
                          )

        params, states, chi_squares, num_iter, exec_time = output
        uncert_frame = erf_param_uncertainty(
                        lambda_local,
                        params.T,
                        iweights=cp.array(self.erf_weights, dtype=cp.float32),
                        ichi_squares=chi_squares)
        self.output_erf = output
        self.erf_uncert = uncert_frame
        return output

    def fetch_data_gaussian(self,
                            plot_smoothing=False,
                            sigma=1,
                            enforce_negativity=False) -> np.array:
        """
        this transposes the data array, applies a filter and diffs the array
        """
        n_lambda, nx, ny = self.shape
        data_gaussian = self.data.copy()
        if plot_smoothing:
            indices = (100, 100)
            print("---> indices hard coded to: ", indices)
            lindex_ = lindex_calc(indices, ny)
            fig, ax = plt.subplots(1, 1)
            ax.plot(self.lambda_gauss, data_gaussian[lindex_][:-1])
        data_gaussian = gaussian_filter1d(
                    cp.array(data_gaussian, dtype=cp.float32),
                    sigma,
                    axis=-1,
                    order=0)
        if plot_smoothing:
            ax.plot(self.lambda_gauss, data_gaussian[lindex_][:-1].get())
        data_gaussian = cp.diff(data_gaussian, axis=1).get()
        if enforce_negativity:
            data_gaussian[data_gaussian > 0] = 0
        self.data_gaussian = data_gaussian
        return data_gaussian

    def chi_squares_thresh_mask(self, thresh=10) -> np.array:
        chi_squares = self.output_erf[2]
        return chi_squares > thresh

    def gaussian_fit(self, sigma, enforce_negativity=True, center=None):
        weights = self.gauss_weights
        lambda_local = self.lambda_local[:-1].copy()

        d_lambda = lambda_local[1]-lambda_local[0]
        lambda_local += d_lambda/2
        self.lambda_gauss = lambda_local
        n_lambda, nx, ny = self.shape
        data_gaussian = self.fetch_data_gaussian(
                                     sigma=sigma,
                                     enforce_negativity=enforce_negativity)
        number_fits = nx*ny
        cent = 4.55 if self.long_short == 'short' else 6.0

        initial_parameters = np.tile([
                                      -0.002,  # Amplitude
                                      cent,    # center
                                      0.01,    # width
                                      0.00     # offset
                                      ], (number_fits, 1)).astype(np.float32)
        if center is not None:
            initial_parameters[:, 1] = center

        output_gaussian = self._gpu_fit_wrapper_(
                                  weights=weights,
                                  model_id=gf.ModelID.GAUSS_1D,
                                  data=data_gaussian,
                                  lambda_local=lambda_local.astype(np.float32),
                                  initial_parameters=initial_parameters)

        self.output_gaussian = output_gaussian
        params, states, chi_squares, num_iter, exec_time = output_gaussian
        uncert_frame = gaussian_param_uncertainty(
                        lambda_local,
                        params.T,
                        iweights=cp.array(weights, dtype=cp.float32),
                        ichi_squares=chi_squares)
        self.gaussian_uncert = uncert_frame
        return output_gaussian

    def _chi_sq_interpolation_(self, thresh) -> cp.array:
        """
        This calculates the values to linearly interpolate between erf and
        Gaussian
        """
        _, nx, ny = self.shape
        chi_squares = cp.array(self.output_erf[2].reshape(nx, ny),
                               dtype=cp.float32)
        gauss_factor = chi_squares / thresh
        erf_factor = 1-gauss_factor
        gauss_factor[gauss_factor > 1] = 1
        erf_factor[erf_factor < 0] = 0
        return erf_factor, gauss_factor

    def compute_composite(self, thresh) -> cp.array:
        """
        Weighted Average up to threshold?
        """
        _, nx, ny = self.shape
        center_erf = cp.array(self.output_erf[0][:, 2].reshape(nx, ny),
                              dtype=cp.float32)
        center_gauss = cp.array(self.output_gaussian[0][:, 1].reshape(nx, ny),
                                dtype=cp.float32)
        erf_factor, gauss_factor = self._chi_sq_interpolation_(thresh)
        return erf_factor * center_erf + gauss_factor * center_gauss

    def compute_weighted_mask(self, thresh) -> np.array:
        _, nx, ny = self.shape
        erf_mask = cp.array(self.output_erf[0][:, 0].reshape(nx, ny),
                            dtype=cp.float32)
        gauss_mask = cp.array(self.output_gaussian[0][:, 3].reshape(nx, ny),
                              dtype=cp.float32)
        erf_factor, gauss_factor = self._chi_sq_interpolation_(thresh)
        return erf_factor * erf_mask + gauss_factor * gauss_mask

    def probe_fig(self, figsize, cmap='Spectral',
                  subplots_adjust_kwargs={},
                  impose_xlim: bool = False,
                  thresh: float = 10):
        """
        After much tomfoolery this is the probing figure that allows you to
        click on a pixel and view the Bragg spectrum of that voxel

        It is bespoke but could be used as a template for some other data
        visualization for 4D Arrays

        """
        fontsize = 14
        n_lambda, nx, ny = self.shape
        center_erf = self.output_erf[0][:, 2].reshape(nx, ny)
        chi_squares_erf = self.output_erf[2].reshape(nx, ny)
        cent_erf_uncert = self.erf_uncert[2].reshape(nx, ny)
        center_gauss = self.output_gaussian[0][:, 1].reshape(nx, ny)
        cent_gauss_uncert = self.gaussian_uncert[1].reshape(nx, ny)
        # combined = center_erf.copy()
        # combined[mask] = center_gauss[mask]
        composite = self.compute_composite(thresh).reshape(nx, ny).get()

        arr_dict = {
                'Reconstruction image': self.test_frame[0],
                #'$\chi ^2$': chi_squares_erf,
                'Uncert (center erf)': cent_erf_uncert,
                #'Uncert (center gauss)': cent_gauss_uncert,
                # '$\sigma$ (erf)': self.output_erf[0][:, 3].reshape(nx, ny),
                #'Diff (composite-erf)': composite-center_erf,
                'Center (of erf)': center_erf,
                'Center (of gaussian)': center_gauss,
                # 'Center (combined (hard thresh))': combined,
                #'Center (composite)': composite,
                   }

        mask_ = self.test_frame[0] > 0.1
        # Image Array
        center_counter = 0
        fig, ax = plt.subplots(2, len(arr_dict), figsize=figsize,
                               sharex=True, sharey=True)
        for i, (key, arr_) in enumerate(arr_dict.items()):
            arr = arr_.copy()
            arr *= mask_
            nan_mask = ~np.isfinite(arr)
            arr[nan_mask] = 0
            ax[0, i].set_title(key, fontsize=fontsize)
            ax[0, i].axis(False)
            ax[1, i].axis(False)
            if center_counter == 0:
                try:
                    vmin, vmax = contrast(arr[arr != 0])
                except:
                    print("failed to make contrast")
            if "Center" in key:
                center_counter += 1

            ax[0, i].imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)

        # Spectrum and Derivative plot array
        ax_spec = [fig.add_subplot(2, 3, 4)]
        ax_spec.append(fig.add_subplot(2, 3, 5))
        ax_spec.append(ax_spec[1].twinx())
        ax_spec.append(fig.add_subplot(2, 3, 6))


        # These are for history so they can be removed...
        vlines = np.zeros(len(arr_dict), dtype=object)
        hlines = np.zeros(len(arr_dict), dtype=object)

        def onclick(event):
            """
            This defines what happens when a subplot is clicked on...
                - plotting the spectrum and difference of the clicked voxel
            """
            index = np.round([event.xdata, event.ydata]).astype(int)
            x_, y_ = index
            lindex = lindex_calc(index, ny)
            x_gauss = self.lambda_gauss
            x_erf = self.lambda_local
            fit_gauss = gaussian_1d(np.array(self.output_gaussian[0][lindex]),
                                    x_gauss)
            fit_erf = error_function(np.array(self.output_erf[0][lindex]),
                                     x_erf)
            cent_erf = center_erf[index[1], index[0]]
            y_erf = error_function(self.output_erf[0][lindex],
                                   np.array([cent_erf]))
            cent_gauss = center_gauss[index[1], index[0]]
            y_gauss = gaussian_1d(self.output_gaussian[0][lindex], cent_gauss)
            
            for z, a_ in enumerate(ax[0]):
                try:
                    vlines[z].remove()
                    hlines[z].remove()
                except:
                    pass
                vlines[z] = a_.axvline(index[0], color='k', linewidth=0.5)
                hlines[z] = a_.axhline(index[1], color='k', linewidth=0.5)

            [a_.cla() for a_ in ax_spec]


            # Erf fits
            for q, a_ in enumerate(ax_spec[:2]):
                a_.scatter(x_erf, self.data[lindex], marker='o',
                           color=colors(1), facecolor='k')
                a_.plot(x_erf, fit_erf, marker='.', color=colors(1))
                a_.errorbar(cent_erf, y_erf,
                            xerr=cent_erf_uncert[index[1], index[0]],
                            capsize=10,
                            color='r')

                a_.axvline(cent_erf, color='r', linewidth=0.5)
                a_.plot(cent_erf, y_erf, 'wd', markerfacecolor='r')
                if q == 0:
                    a_.set_title(f"erf$_{{model}}$\nxy:{x_,y_}; Center = {cent_erf:.3f}")

            # Gaussian Fits
            for q, a_ in enumerate(ax_spec[2:]):
                a_.scatter(x_gauss, self.data_gaussian[lindex], marker='o',
                           color=colors(0), facecolor='k')
                a_.plot(x_gauss, fit_gauss, marker='.', color=colors(0))
                a_.errorbar(cent_gauss, y_gauss,
                            xerr=cent_gauss_uncert[index[1], index[0]],
                            capsize=10,
                            color='b')
                a_.axvline(cent_gauss, color='b', linewidth=0.5)
                a_.plot(cent_gauss, y_gauss, 'wd', markerfacecolor='b')
                if q == 1:
                    a_.set_title(f"gauss$_{{model}}$\nxy:{x_,y_}; Center = {cent_gauss:.3f}")


            ax_spec[3].yaxis.tick_right()
            for a_ in ax_spec:
                a_.set_xlabel("$\lambda$ ($\AA$)")
                d_lambda = self.lambda_local[1] - self.lambda_local[0]
                if impose_xlim:
                    a_.set_xlim(self.lambda_local[0]-2*d_lambda,
                                self.lambda_local[-1]+2*d_lambda)

        fig.subplots_adjust(**subplots_adjust_kwargs)
        fig.canvas.mpl_connect('button_press_event', onclick)

        return fig, ax

    def probe_fig_manuscript(self, figsize, cmap='Spectral',
                  subplots_adjust_kwargs={},
                  impose_xlim: bool = False,
                  thresh: float = 10):
        """
        After much tomfoolery this is the probing figure that allows you to
        click on a pixel and view the Bragg spectrum of that voxel

        It is bespoke but could be used as a template for some other data
        visualization for 4D Arrays

        """
        fontsize = 14
        n_lambda, nx, ny = self.shape
        center_erf = self.output_erf[0][:, 2].reshape(nx, ny)
        chi_squares_erf = self.output_erf[2].reshape(nx, ny)
        cent_erf_uncert = self.erf_uncert[2].reshape(nx, ny)
        center_gauss = self.output_gaussian[0][:, 1].reshape(nx, ny)
        cent_gauss_uncert = self.gaussian_uncert[1].reshape(nx, ny)
        # combined = center_erf.copy()
        # combined[mask] = center_gauss[mask]
        composite = self.compute_composite(thresh).reshape(nx, ny).get()

        arr_dict = {
                'Reconstruction image': self.test_frame[0],
                #'$\chi ^2$': chi_squares_erf,
                #'Uncert (center erf)': cent_erf_uncert,
                #'Uncert (center gauss)': cent_gauss_uncert,
                # '$\sigma$ (erf)': self.output_erf[0][:, 3].reshape(nx, ny),
                #'Diff (composite-erf)': composite-center_erf,
                'Center (of erf)': center_erf,
                #'Center (of gaussian)': center_gauss,
                # 'Center (combined (hard thresh))': combined,
                #'Center (composite)': composite,
                   }

        mask_ = self.test_frame[0] > 0.1
        # Image Array
        center_counter = 0
        fig, ax = plt.subplots(2, len(arr_dict), figsize=figsize,
                               sharex=True, sharey=True)
        for i, (key, arr_) in enumerate(arr_dict.items()):
            arr = arr_.copy()
            arr *= mask_
            nan_mask = ~np.isfinite(arr)
            arr[nan_mask] = 0
            ax[0, i].set_title(key, fontsize=fontsize)
            ax[0, i].axis(False)
            ax[1, i].axis(False)
            if center_counter == 0:
                try:
                    vmin, vmax = contrast(arr[arr != 0])
                except:
                    print("failed to make contrast")
            if "Center" in key:
                center_counter += 1

            ax[0, i].imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)

        # Spectrum and Derivative plot array
        ax_spec = fig.add_subplot(2, 1, 2)
        #ax_spec.append(fig.add_subplot(2, 3, 5))
        #ax_spec.append(ax_spec[1].twinx())
        #ax_spec.append(fig.add_subplot(2, 3, 6))


        # These are for history so they can be removed...
        vlines = np.zeros(len(arr_dict), dtype=object)
        hlines = np.zeros(len(arr_dict), dtype=object)

        def onclick(event):
            """
            This defines what happens when a subplot is clicked on...
                - plotting the spectrum and difference of the clicked voxel
            """
            index = np.round([event.xdata, event.ydata]).astype(int)
            x_, y_ = index
            lindex = lindex_calc(index, ny)
            x_gauss = self.lambda_gauss
            x_erf = self.lambda_local
            fit_gauss = gaussian_1d(np.array(self.output_gaussian[0][lindex]),
                                    x_gauss)
            fit_erf = error_function(np.array(self.output_erf[0][lindex]),
                                     x_erf)
            cent_erf = center_erf[index[1], index[0]]
            y_erf = error_function(self.output_erf[0][lindex],
                                   np.array([cent_erf]))
            cent_gauss = center_gauss[index[1], index[0]]
            y_gauss = gaussian_1d(self.output_gaussian[0][lindex], cent_gauss)
            
            for z, a_ in enumerate(ax[0]):
                try:
                    vlines[z].remove()
                    hlines[z].remove()
                except:
                    pass
                vlines[z] = a_.axvline(index[0], color='k', linewidth=0.5)
                hlines[z] = a_.axhline(index[1], color='k', linewidth=0.5)

            ax_spec.cla()
            ax_spec.set_xlabel("$\lambda$ (nm)")
            ax_spec.set_ylabel("Attenuation (mm$^{-1}$)")

            # Erf fits
            for q, a_ in enumerate([ax_spec]):
                a_.scatter(x_erf/10, self.data[lindex], marker='o',
                           color=colors(1), facecolor='k')
                a_.plot(x_erf/10, fit_erf, marker='.', color=colors(1))
                a_.errorbar(cent_erf/10, y_erf,
                            xerr=cent_erf_uncert[index[1], index[0]]/10,
                            capsize=10,
                            color='r')

                a_.axvline(cent_erf/10, color='r', linewidth=0.5)
                a_.plot(cent_erf/10, y_erf, 'wd', markerfacecolor='r')
                if q == 0:
                    a_.set_title(f"erf$_{{model}}$\nxy:{x_,y_}; Center = {cent_erf/10:.3f} nm")

        fig.subplots_adjust(**subplots_adjust_kwargs)
        fig.canvas.mpl_connect('button_press_event', onclick)

        return fig, ax


    def erf_gauss_fit_wrapper(self,
                              sigma: float,
                              chi_sq_thresh: float,
                              med_kern=(3, 3),
                              conv_thresh=-np.inf
                              ) -> cp.array:
        """
        This just wraps up all the operations
        Parameters:
        -----------
            sigma: float - Gaussian sigma
            chi_sq_thresh: float - This sets the linear interpolation between
                                   the erf and gaussian; above this threshold
                                   the compoiste image is completely the
                                   gaussian, below the threshold is a linear
                                   interpolation between the two fits
            med_kern: tuple 2 elements - the reshaped erf solution is filtered
                                         by this median kernel to seed the
                                         gaussian
            conv_thresh: float - this masks the low attenuation values (~0)
                                 to clean up the solution
        Returns:
        --------
            composite image
        """
        _ = self.erf_fit(verbose=False)
        _, nx, ny = self.shape
        centers_temp = cp.array(self.output_erf[0][:, 2],
                                dtype=cp.float32).reshape(nx, ny)
        centers_temp = median_filter(centers_temp, med_kern).flatten().get()
        _ = self.gaussian_fit(center=centers_temp, sigma=sigma)
        composite_image = self.compute_composite(chi_sq_thresh).reshape(nx, ny)
        mask = median_filter(cp.array(self.output_erf[0][:, 0].reshape(nx, ny),
                                      dtype=cp.float32), (3, 3)) > conv_thresh

        erf_uncert = self.erf_uncert[2, :].reshape(nx, ny)
        gauss_uncert = self.gaussian_uncert[1, :].reshape(nx, ny)
        return {
                'Center': composite_image*mask,
                'Uncert erf': erf_uncert,
                'Uncert gauss': gauss_uncert,
                }


def lindex_calc(index: tuple, ny: int) -> int:
    """
    This is for calculating the linear index of the arrays once they are
    transformed from 3d (wavelength,x,y) to 2d (lindex,wavelength)
    """
    return index[1]*ny + index[0]


def init_params_calc(data_array: np.array,
                     wavelengths: np.array,
                     sigma_sign: bool = False,
                     visualize: bool = False) -> np.array:
    """
    This is basically what Dan's Code does to automatically seed it with
    relatively close number
    """
    n_fits, n_lambda = data_array.shape
    offsets = np.nanmean(data_array, axis=1)
    amplitudes = np.nanstd(data_array, axis=1)
    center = np.mean([wavelengths[0], wavelengths[-1]])
    centers = np.tile(center, n_fits)
    sigma = wavelengths[1]-wavelengths[0]
    if sigma_sign:
        sigma *= -2
    #sigma = -0.02
    #logging.info("hard coding sig")
    #print("--- Hard Coding Sigma -> ", sigma, "---")
    sigmas = np.tile(sigma, n_fits)
    if visualize:
        print(f"center = {center}")
        print(f"sigma = {sigma}")
        fig, ax = plt.subplots(1, 2)
        ax[0].hist(offsets)
        ax[1].hist(amplitudes)
    return np.vstack([offsets,
                      amplitudes,
                      centers,
                      sigmas]).astype(np.float32).T.copy()


def error_function(params: np.array, x: np.array) -> np.array:
    """
    This is taken from Gpufit/Gpufit/models/error_function.cuh

    Args:
    -----
        params: np.array:
                - params[0]: offset    (y)
                - params[1]: amplitude
                - params[2]: center    (x0)
                - params[3]: sigma
    Returns:
    --------
        value (error function as function of x)

    Note: - I expanded the dimensions to allow for arbitrary broadcasting...
    """
    if params.ndim == 1:
        params = params[None, :]
    argx = (x[:, None] - params[:, 2]) / (np.sqrt(2.0) * params[:, 3])
    oneperf = 0.5 * (1 + erf(argx))
    value = params[:, 1] * oneperf + params[:, 0]
    return value.astype(np.float32)


def gaussian_1d(params: np.array, x: np.array) -> np.array:
    """
    From gpufit (gauss_1d.cuh)
        params[0]: amplitude
        params[1]: center coordinate
        params[2]: width
        params[3]: offset

        x = x-array

    returns:
        gauss 1d
    """
    argx = (x - params[1]) * (x - params[1]) / (2 * params[2] * params[2])
    ex = np.exp(-argx)
    return params[0] * ex + params[3]


def cp_free_mem():
    """
    frees memory cupy is holding...
    """
    mempool = cp.get_default_memory_pool()
    mempool.free_all_blocks()


def erf_param_uncertainty(ix, ip, iweights, ichi_squares):
    """
    This is for erf bragg edge stuff
    Parameters:
    -----------
        ix = x values (independent variable)
        ip = parameters (fitted parameters with gpufit) shape =
                (num_parameters, num_fits)
        iweights = weights
        ichi_squares = chi_squares from gpufit

    Returns:
    --------
        - parameter_uncertainty (np.array)

    Notes:
        Refer to Dan's Code if anything breaks and bring it back to working
        form, I eliminated some of the for loop traversals with vectorized
        operations namely:
            1) jacobian operations
                - modify shape of argx
                - all operations are vectorized so no looping over x-array
            2) jacobian *= weights...
            3) uncert = cp.einsum(...)
            4) parameter_uncert = chi_squares[None,:]...

        This is maybe a bit less readable, but may give a slight performance
        increase

    """
    ix = cp.array(ix, dtype=cp.float32)
    ichi_squares = cp.array(ichi_squares, dtype=cp.float32)
    nx = ix.shape
    num_p = ip.shape
    parameter_uncert = cp.zeros([num_p[0], num_p[1]], dtype=cp.float32)  # %The output, on RAM
    idx = cp.arange(0, num_p[1], 1).astype(cp.int32)
    nchunk = len(idx)

    p = cp.array(ip)[:, idx]

    jacobian = cp.ones([nx[0], num_p[0], nchunk], dtype=cp.float32)
    argx = (ix[:, None] - p[2, :]) / (np.sqrt(2.0) * p[3, :])
    jacobian[:, 1, :] = 0.5 * (1 + gpu_erf(argx))
    jacobian[:, 2, :] = -0.5 * p[1, :] * cp.exp(-argx * argx) / (np.sqrt(2.0)*p[3, :])
    jacobian[:, 3, :] = jacobian[:, 2, :] * (ix[:, None]-p[2, :]) / p[3, :]

    covariance = cp.transpose(jacobian,  (1, 0, 2)).copy()

    # apply the weights_, assuming each measurement and parameter are
    # uncorrelated, W(i,x) = sigma(i,x) * Ones, i is measurement, x pixel
    if iweights is not None:
        jacobian *= iweights[:, None, None]

    covariance = cp.matmul(covariance.transpose(2, 0, 1),
                           jacobian.transpose(2, 0, 1))
    covariance = cp.linalg.inv(covariance).transpose(1, 2, 0)

    # covariance has a shape (m,m,n) and this einsum specifies to take the m,m
    # diagonal for each n
    uncert = cp.abs(cp.einsum('iij->ij', covariance))

    # gpufit returns chi^2, reduce it by # of degrees of freedom
    ndof = 1 / (nx[0] - num_p[0])  
    chi_squares_ = cp.sqrt(cp.array(ichi_squares, dtype=cp.float32)[idx]*ndof)

    parameter_uncert = chi_squares_[None, :]*cp.sqrt(uncert)

    return cp.asnumpy(parameter_uncert)


def gaussian_param_uncertainty(ix, ip, iweights, ichi_squares):
    """
    This is for erf bragg edge stuff
    Parameters:
    -----------
        ix = x values (independent variable)
        ip = parameters (fitted parameters with gpufit) shape =
                (num_parameters, num_fits)
        iweights = weights
        ichi_squares = chi_squares from gpufit

    Returns:
    --------
        - parameter_uncertainty (np.array)

    Notes:
        Refer to Dan's Code if anything breaks and bring it back to working
        form, I eliminated some of the for loop traversals with vectorized
        operations namely:
            1) jacobian operations
                - modify shape of argx
                - all operations are vectorized so no looping over x-array
            2) jacobian *= weights...
            3) uncert = cp.einsum(...)
            4) parameter_uncert = chi_squares[None,:]...

        This is maybe a bit less readable, but may give a slight performance
        increase

    """
    ix = cp.array(ix, dtype=cp.float32)
    ichi_squares = cp.array(ichi_squares, dtype=cp.float32)
    nx = ix.shape
    num_p = ip.shape
    parameter_uncert = cp.zeros([num_p[0], num_p[1]], dtype=cp.float32)  # %The output, on RAM
    idx = cp.arange(0, num_p[1], 1).astype(cp.int32)
    nchunk = len(idx)

    p = cp.array(ip)[:, idx]

    jacobian = cp.ones([nx[0], num_p[0], nchunk], dtype=cp.float32)

    # Trying to keep this consistent with guass_1d.cuh
    term_1 = ix[:, None] - p[1, :]
    argx = (term_1 * term_1) / (2 * p[2, :] * p[2, :])
    ex = cp.exp(-argx)
    jacobian[:, 0, :] = ex
    jacobian[:, 1, :] = p[0, :] * ex * term_1 / (p[2, :] * p[2, :])
    jacobian[:, 2, :] = p[0, :] * ex * term_1 * term_1 / (p[2, :] * p[2, :] * p[2, :])

    covariance = cp.transpose(jacobian,  (1, 0, 2)).copy()

    # apply the weights_, assuming each measurement and parameter are
    # uncorrelated, W(i,x) = sigma(i,x) * Ones, i is measurement, x pixel
    if iweights is not None:
        jacobian *= iweights[:, None, None]

    covariance = cp.matmul(covariance.transpose(2, 0, 1), jacobian.transpose(2, 0, 1))
    covariance = cp.linalg.inv(covariance).transpose(1, 2, 0)

    # covariance has a shape (m,m,n) and this einsum specifies to take the m,m
    # diagonal for each n
    uncert = cp.abs(cp.einsum('iij->ij', covariance))

    ndof = 1/(nx[0]-num_p[0])  # gpufit returns chi^2, reduce it by # of degrees of freedom
    chi_squares_ = cp.sqrt(cp.array(ichi_squares, dtype=cp.float32)[idx]*ndof)

    parameter_uncert = chi_squares_[None, :]*cp.sqrt(uncert)

    return cp.asnumpy(parameter_uncert)
