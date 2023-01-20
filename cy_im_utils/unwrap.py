from cupyx.scipy.ndimage import median_filter
from scipy.interpolate import make_interp_spline, RegularGridInterpolator
import cupy as cp
import numpy as np
import logging
import napari
from magicgui import magicgui


class unwrapper:

    def __init__(self,
                 points_array: np.array,
                 image: np.array,
                 med_kernel: int = 3):
        """
        points array are the spline interpolation points
        """
        self.points_array = points_array
        self.image = image
        if med_kernel > 1:
            self.image = median_filter(cp.array(image, dtype=cp.float32),
                                       (med_kernel, med_kernel)).get()

    def fit_spline(self, boundary_condition='natural'):
        self.points_ndim = self.points_array.ndim
        nx, ny = self.image.shape[-2:]
        x_, y_, theta = self.points_to_angles(self.points_array, nx, ny)
        self.spl = make_interp_spline(theta,
                                      np.stack([x_, y_]).T,
                                      bc_type=boundary_condition)

    def points_to_angles(self,
                         points_arr: np.array,
                         nx: int,
                         ny: int
                         ) -> np.array:
        """
        convert points set into angular coordinate set

        Note that this assumes the spiral is increasing radially clockwise and
        I haven't exactly reasoned about how it would be different otherwise

        possible edge cases:
            - dy == 0 -> dx/dy is undefined
        """
        if nx != ny:
            logging.warning(f"image is not square -> {nx} != {ny}")
        center_x = nx // 2
        center_y = ny // 2
        if self.points_ndim == 2:
            y_ = points_arr[:, 0]
            x_ = points_arr[:, 1]
        elif self.points_ndim == 3:
            y_ = points_arr[:, 1]
            x_ = points_arr[:, 2]
        dx, dy = x_ - center_x, y_ - center_y
        if np.sum(dy == 0) > 0:
            logging.warning("Unknown behavior when dy == 0. incrementing by 1")
            dy[dy == 0] += 1
        theta = np.rad2deg(np.arctan(dx / dy, where=dy != 0))

        # These operations might need to change just a bit of a hack to make
        # monotonically increasing angles
        theta[dy > 0] += 180
        theta[theta < 0] += 360
        theta -= theta[0]
        theta[0] = 360

        self.theta = theta
        return x_[::-1], y_[::-1], theta[::-1]

    def compute_num_angular_samples(self) -> None:
        """
        this loop determines how many angular samples to take (assuming one
        sample per pixel?)

        numerically integrates the arc length of the spline and iteratively
        tries to match num_samples to the arc length, the new x and y are also
        computed in this loop
        """
        num_samples = 100
        diff = np.inf
        j = 0
        while diff > 1:
            self.angles = np.linspace(np.min(self.theta),
                                      np.max(self.theta),
                                      num_samples)
            x_new, y_new = self.spl(self.angles).T
            length = np.sum(
                np.sqrt((x_new[:-1]-x_new[1:])**2 + (y_new[:-1]-y_new[1:])**2)
                )
            diff = length - num_samples
            num_samples = int(np.round(length))
            j += 1
            if j > 10:
                logging.warning(f"not converged to sampling in {j} iterations")

        self.x_new, self.y_new = x_new, y_new
        self.num_samples = num_samples

    def compute_resampling_coordinates(self, samples: int = 25) -> None:
        """
        this computes all the coordinates to resample the image in the spline
        coordinate system
        """
        dx_new, dy_new = self.spl.derivative(1)(self.angles).T
        norm_arr = np.sqrt(dx_new**2+dy_new**2)
        dx_norm = -dy_new / norm_arr
        dy_norm = dx_new / norm_arr
        sampling_vector = np.linspace(-samples, samples, samples*2)

        dx = dx_norm[:, None] @ sampling_vector[None, :]
        dy = dy_norm[:, None] @ sampling_vector[None, :]
        x_coords = self.x_new[:, None] + dx
        y_coords = self.y_new[:, None] + dy
        self.coords_vectorized = np.dstack([y_coords, x_coords])

    def interpolate_image(self):
        """
        this instantiates the image interpolator to re-sample the image
        """
        bounds_error = False
        fill_value = None
        nx, ny = self.image.shape
        x_vector = np.linspace(0, nx-1, nx)
        y_vector = np.linspace(0, ny-1, ny)
        self.image_interpolator = RegularGridInterpolator(
                                                          (x_vector, y_vector),
                                                          self.image,
                                                          bounds_error=bounds_error,
                                                          fill_value=fill_value
                                                          )

    def compute_unwrapped_image(self):
        """
        just a wrapper to return the unwrapped image
        """
        return self.image_interpolator(self.coords_vectorized)


class napari_unwrapper(unwrapper):

    def __init__(self):
        self.viewer = napari.Viewer()
        self.viewer.window.add_dock_widget(self.unwrap_widget(),
                                           name='unwrap',
                                           add_vertical_stretch=True,
                                           area='right')

    def _initial_ops_(self):
        super().__init__(self.points, self.image)
        self.fit_spline()
        self.compute_num_angular_samples()
        self.compute_resampling_coordinates(samples=self.sampling)
        self.interpolate_image()
        self.unwrapped = self.compute_unwrapped_image()

    def unwrap_widget(self):
        @magicgui(call_button="unwrap points")
        def inner(sampling: int = 1):
            self.points = self.viewer.layers[-1].data
            self.image = self.viewer.layers[0].data
            self.sampling = sampling
            self._initial_ops_()
        return inner

if __name__ == "__main__":
    inst = napari_unwrapper()
    napari.run()
