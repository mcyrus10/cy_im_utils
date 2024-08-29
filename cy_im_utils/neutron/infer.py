from .visualization import colors
from pygpufit import gpufit as gf
from matplotlib import pyplot as plt
from scipy.ndimage.filters import median_filter
from cupyx.scipy.ndimage.filters import median_filter as median_filter_GPU
import cupy as cp
import numpy as np

class phase_fit:
    def __init__(self, image_stack, phase_steps, period: float, wavelength: float = 1, z: float = 1):
        self.shape = image_stack.shape
        self.xi = wavelength*z/period
        self.phase_steps = phase_steps
        self.image_stack = image_stack
        self.period = period
        self.model_id = gf.ModelID.SINECURVE
        assert self.shape[0] == len(phase_steps),"number of phase steps doesn't match number of images"
    
    def gpufit_wrapper( self,
                        data,
                        phase_steps,
                        initial_parameters,
                        tolerance,
                        max_number_iterations
                        ) -> tuple:
        """self explanatory
        """
        estimator_id = gf.EstimatorID.MLE
        n_fits = data.shape[0]
        constraints = np.zeros([n_fits,8], dtype = np.float32)
        constraints[:,2] = 0.5*self.period
        constraints[:,3] = 1.5*self.period
        constraints[:,4] = -2*np.pi
        constraints[:,5] = 2*np.pi
        constraint_types = np.array([
                gf.ConstraintType.FREE,
                gf.ConstraintType.LOWER_UPPER,
                gf.ConstraintType.LOWER_UPPER,
                gf.ConstraintType.LOWER], dtype = np.int32)
        return gf.fit_constrained( 
                        data = data.astype(np.float32).copy(),
                        weights = None,
                        model_id = self.model_id,
                        constraints = constraints.astype(np.float32).copy(),
                        constraint_types = constraint_types.astype(np.int32).copy(),
                        initial_parameters = initial_parameters.astype(np.float32).copy(),
                        tolerance = tolerance,
                        max_number_iterations = max_number_iterations,
                        parameters_to_fit = None,
                        estimator_id = estimator_id,
                        user_info = phase_steps.astype(np.float32).copy()
                      )

    def fit(self,
            verbose: bool = True,
            tolerance: float = 0.00001,
            max_number_iterations: int = 50,
            n_refit: int = 0):
        n_p,nx,ny = self.shape
        amplitude = (np.max(self.image_stack, axis = 0) - np.min(self.image_stack, axis = 0)).flatten()/2
        offset = np.mean(self.image_stack, axis = 0).flatten()
        period = (np.ones_like(self.image_stack[0])*self.period).flatten()
        phase = (np.ones_like(self.image_stack[0])*1).flatten()
        self.initial_parameters = np.vstack([amplitude,period,phase,offset]).T
        self.data_flat = np.transpose(self.image_stack,(1,2,0)).reshape(nx*ny,n_p)
        # Initial Fit
        out = self.gpufit_wrapper(self.data_flat, self.phase_steps,
                self.initial_parameters, tolerance, max_number_iterations)
        self.parameters = out[0]
        self.states = out[1]
        self.chi_squares = out[2]
        self.number_iterations = out[3]
        self.execution_time = out[4]
        self.pos_amplitudes()
        self.constratin_phases()

        if verbose:
            print("After First Fit:")
            self.fit_summary()
       
        

        for j in range(n_refit):
            self.refit(tolerance = tolerance,
                    max_number_iterations = max_number_iterations)

            if verbose:
                print("-"*40)
                print(f"After Re-Fit {j}:")
                self.fit_summary()
 
    def pos_amplitudes(self) -> None:
        """ This makes all the amplitudes positive and flips the phase so it still matches
        """
        neg_amp_indices = self.parameters[:,0] < 0
        self.parameters[neg_amp_indices,0] *= -1
        self.parameters[neg_amp_indices,2] -= np.pi

    def constratin_phases(self) -> None:
        """ This shifts all the phases so they are non-negative, and with the
        2*pi fitting constraint, ensures all phases are between 0 and 2*pi
        """
        neg_phases = self.parameters[:,2] < 0
        self.parameters[neg_phases] += 2*np.pi


    def fetch_median_params_GPU(self, median_size: int) -> tuple:
        """ This applies median on the GPU to initialize new guesses for the
        fitting parameters
        """
        _,nx,ny = self.shape
        amplitudes = cp.array(self.parameters[:,0].reshape(nx,ny), dtype = cp.float32)
        periods =    cp.array(self.parameters[:,1].reshape(nx,ny), dtype = cp.float32)
        phases =     cp.array(self.parameters[:,2].reshape(nx,ny), dtype = cp.float32)
        offsets =    cp.array(self.parameters[:,3].reshape(nx,ny), dtype = cp.float32)
        params_cp = cp.stack([amplitudes,periods,phases,offsets])
        med_kernel_GPU = (1, median_size,median_size)
        params_cp = median_filter_GPU(params_cp,med_kernel_GPU)
        return [cp.asnumpy(params_cp[j].flatten()) for j in range(4)]

    def refit(self, median_size = 9, tolerance: float = 0.000001,
                        max_number_iterations: int = 50) -> None:
        """After the fit has converged, this will re-fit the non-converged
        states by seeding the initial guesses with median of its neighbors
        """
        non_converged = self.states != 0
        _,nx,ny = self.shape
        med_kernel = (median_size,median_size)
        amplitudes,periods,phases,offsets = self.fetch_median_params_GPU(median_size = median_size)
        refit_init_params = np.vstack([amplitudes,periods,phases,offsets]).T[non_converged,:]
        self.initial_parameters[non_converged] = refit_init_params
        refit_data_slice = self.data_flat[non_converged,:]
        out_refit = self.gpufit_wrapper(refit_data_slice, 
                    self.phase_steps,
                    refit_init_params, 
                    tolerance, 
                    max_number_iterations)
        self.parameters[non_converged] = out_refit[0]
        self.states[non_converged] = out_refit[1]
        self.chi_squares[non_converged] = out_refit[2]
        self.number_iterations[non_converged] = out_refit[3]
        self.pos_amplitudes()
        self.constratin_phases()
            
    def fit_summary(self) -> None:
        """ this prints the summary of the fit 
        """
        number_fits = self.data_flat.shape[0]
        converged = self.states == 0
        print("*Gpufit*")

        # print summary
        print(f'\nmodel ID:        {self.model_id}')
        print(f'number of fits:  {number_fits}')
        #print(f'fit size:        {size_x} x {size_x}')
        print(f'mean chi_square: {np.mean(self.chi_squares[converged]):.2f}')
        print(f'iterations:      {np.mean(self.number_iterations[converged]):.2f}')
        print(f'time:            {self.execution_time:.2f} s')

        # get fit states
        number_converged = np.sum(self.states == 0)
        print(f'\nratio converged       {number_converged    / number_fits * 100:6.2f} %')
        print(f'ratio max it. exceeded  {np.sum(self.states == 1) / number_fits * 100:6.2f} %')
        print(f'ratio singular hessian  {np.sum(self.states == 2) / number_fits * 100:6.2f} %')
        print(f'ratio neg curvature MLE {np.sum(self.states == 3) / number_fits * 100:6.2f} %')
            
    
    
    def lindex(self,x_coord,y_coord) -> int:
        """return linear index of 2d coordinate
        """
        _,nx,_ = self.shape
        return x_coord*nx + y_coord
    
    def visualize_fit(self,coords: list = None, show_initial = False) -> None:
        """ creates a plot with some sample fits
        """
        if coords is None:
            coords = [
                        [736,892],
                        [599,1463],
                        [620,889],
                        [619,906],
                        [992,1481],
                        [1600,1200],
                        [1200,1600],
                        [1021,941],
                        [913,911],
                    ]
        _,nx,ny = self.shape
        x_smooth = np.linspace(self.phase_steps[0],self.phase_steps[-1],100)
        fig_,ax_ = plt.subplots(1,1)
        ax_.imshow(self.image_stack[0])
        fig_edge_shape = int(np.ceil(np.sqrt(len(coords))))
        fig,ax = plt.subplots(fig_edge_shape,fig_edge_shape, figsize = (8,8), sharex = True, sharey = True)
        ax = ax.flatten()
        ax_size_diff = fig_edge_shape**2-len(coords)
        if ax_size_diff > 0:
            [a.axis(False) for a in ax[-ax_size_diff:]]
        for i,(x_coord,y_coord) in enumerate(coords):
            lindex = self.lindex(x_coord,y_coord)
            params = self.parameters[lindex]
            converge_ = True if self.states[lindex] == 0 else False
            data = self.data_flat[lindex]
            ax_.plot(y_coord,x_coord, marker = 'x', color = colors(i))
            ax[i].scatter(self.phase_steps,data, color = colors(i))
            if show_initial:
                initial_params = self.initial_parameters[lindex]
                ax[i].scatter(self.phase_steps,self.eval_sine_fit(self.phase_steps,initial_params),
                        color = colors(i), s = 0.5, alpha = 0.5)
            ax[i].plot(x_smooth,self.eval_sine_fit(x_smooth,params),
                    linestyle = '--', linewidth = 0.5, color = colors(i))
            ax[i].set_title(f"converged:{converge_}\namplitude:{params[0]:.2f}\nperiod:{params[1]:.2f}\nphase:{params[2]:.2f}\noffset:{params[3]:.2f}")
        fig_.tight_layout()
        fig.tight_layout()
            
    def eval_sine_fit(self, x, parameters) -> np.array:
        """ returns the sinusoidal paramterized fit over x, note the argument
        order here is the same as it is for sinecurve.cuh that gpufit uses
        
        Args:
        -----
            - x: array to evaluate sine function
            - parameters: array-like with:
                                            - 0: amplitude
                                            - 1: period
                                            - 2: phase
                                            - 3: offset
        """
        amplitude,period,phase,offset = parameters
        return offset + amplitude * np.sin(x*2*np.pi/period + phase)
    
    def visibility(self) -> np.array:
        """ Equation 2 from Lynch et al.
        """
        _,nx,ny = self.shape
        amplitude = self.parameters[:,0].reshape(nx,ny)
        offset = self.parameters[:,3].reshape(nx,ny)
        return amplitude/offset
    
    def phase(self) -> np.array:
        """ from INFER data processing slide -> returns the phase image
        """
        _,nx,ny = self.shape
        return self.parameters[:,2].reshape(nx,ny)
    
    def offset(self) -> np.array:
        """ from INFER data processing slide
        """
        _,nx,ny = self.shape
        return self.parameters[:,3].reshape(nx,ny)
