"""

Basically a copy of Dan's IDL code for finding focus

"""
from PIL import Image
from functools import partial
try:
    from magicgui import magicgui
    from magicgui.tqdm import tqdm
    import napari
except:
    print("errors importing napari and/or magicgui")

from pathlib import Path
from scipy.optimize import least_squares
from scipy.special import erf
import matplotlib.pyplot as plt
import numpy as np

plt.style.use("dark_background")


class napari_focus_gui:
    """
    This is the big boy: it holds the projections and can facilitate the recon
    operations grapically in Napari, interactively in Jupyter or in a script
    """
    def __init__(self, images_dir: Path) -> None:
        files = sorted(list(images_dir.glob("*.tif")))
        nx, ny = self.imread(files[0]).shape
        self.images = np.zeros([len(files), nx, ny],  dtype=np.float32)
        self.x_s = np.zeros(len(files), dtype=np.float32)
        for i, f in tqdm(enumerate(files)):
            self.images[i] = self.imread(f)
            self.x_s[i] = self.f_name_to_length(f)

        self.viewer = napari.Viewer()
        self.viewer.add_image(self.images,
                              name='focus images',
                              colormap='Spectral')
        self.widgets = {
            'Find Focus': [
                                self.crop_image(),
                                self.fit_widget(),
                                self.fit_sigmas(),
                                ]
            }

        for i, (key, val) in enumerate(self.widgets.items()):
            self.viewer.window.add_dock_widget(val,
                                               name=key,
                                               add_vertical_stretch=True,
                                               area='right'
                                               )

    def f_name_to_length(self, f_name, delimiter="_", dist_position=1):
        file_name_iso = str(f_name).split("/")[-1]
        dist = file_name_iso.split(delimiter)[dist_position]
        return float(dist.replace("d", ".").replace("p", ""))

    def imread(self, x):
        with Image.open(x) as im:
            return np.array(im, dtype=np.float32)

    def mute_all(self) -> None:
        """ this suppresses all image layers """
        for elem in self.viewer.layers:
            elem.visible = False

    def _test_crops(self):
        assert hasattr(self, 'crop_image_x'), "No x-stride crop"
        assert hasattr(self, 'crop_image_y'), "No y-stride crop"

    # -------------------------------------------------------------------------
    #              NAPARI WIDGETS AND FUNCTIONS
    # -------------------------------------------------------------------------
    def crop_image(self):
        """ This returns the widget that selects the crop portion of the image
        Note: it also mutes the full image and
        """
        @magicgui(call_button="Crop Image",
                  stride={'widget_type': 'RadioButtons',
                          'choices': [('x', 0), ('y', 1)]}
                  )
        def inner(stride=0):
            verts = np.round(self.viewer.layers[-1].data[0][:, -2:]
                             ).astype(np.uint32)
            x0, x1 = np.min(verts[:, 0]), np.max(verts[:, 0])
            y0, y1 = np.min(verts[:, 1]), np.max(verts[:, 1])
            slice_y = slice(y0, y1)
            slice_x = slice(x0, x1)
            if stride == 0:
                crop_key = 'crop x-stride'
                self.crop_image_x = self.images[:, slice_x, slice_y]
                self.viewer.add_image(self.crop_image_x,
                                      colormap='twilight_shifted',
                                      name=crop_key,
                                      visible=False)
            elif stride == 1:
                crop_key = 'crop y-stride'
                self.crop_image_y = self.images[:, slice_x, slice_y].transpose(0, 2, 1)
                self.viewer.add_image(self.crop_image_y,
                                      colormap='twilight_shifted',
                                      name=crop_key,
                                      visible=False)
            if crop_key not in self.viewer.layers:
                self.viewer.layers[-2].name = f'{crop_key} shape'

        return inner

    def fit_widget(self):
        @magicgui(call_button='Fit Erfs',
                  show_erfs={'label': 'Show Error Function Fits',
                             'value': False}
                  )
        def inner(show_erfs: bool):
            self._test_crops()
            self.out_x = fit_x_focus(self.crop_image_x, show_erfs)
            self.sigmas_x = self.out_x[:, 3]
            self.out_y = fit_x_focus(self.crop_image_y, show_erfs)
            self.sigmas_y = self.out_y[:, 3]
        return inner

    def fit_sigmas(self):
        @magicgui(call_button='Fit sigmas')
        def inner(degree: int = 2):
            self._test_crops()
            nx = len(self.sigmas_x)
            x_ = np.linspace(0, nx-1, nx)
            # x_s_fine = np.linspace(self.x_s[0],self.x_s[-1],500)
            fig, ax = plt.subplots(1, 1)
            arrs_dict = {'x-stride': self.sigmas_x, 'y-stride': self.sigmas_y}
            for label, arr in arrs_dict.items():
                fit_ = np.polyfit(x_, arr, degree)
                ax.plot(x_, arr, linestyle='', marker='s', label=label)
                ax.plot(x_, np.polyval(fit_, x_), linestyle='--', label=f'{label} fit')
            fig.tight_layout()
            plt.show()
        return inner

    def plot_medians(self):
        temp = np.median(self.crop_image, axis=1)
        fig, ax = plt.subplots(1, 1)
        for j in temp:
            ax.plot(j)
        fig.tight_layout()
        plt.show()


def fit_x_focus(crop_image,
                x_plot=None,
                show_erfs: bool = False,
                plot: bool = True) -> np.array:
    crop_median = np.median(crop_image, axis=1)
    n_curves, n_points = crop_median.shape
    out = []
    for i in range(n_curves):
        data = crop_median[i].copy()
        data -= data.min()
        data /= data.max()
        data -= 0.5
        data *= 2
        x0 = [
                0,      # offset
                1.0,    # amplitude
                0.0,    # center
                0.01    # sigma
                ]

        bounds = ([-np.inf, -np.inf, -1, 0], [np.inf, np.inf, 1, np.inf])
        nx = len(data)
        x_arg = np.linspace(-1, 1, nx)

        out.append(least_squares(partial(residual, data=data, x=x_arg),
                                 x0=x0, bounds=bounds
                                 ).x)
        if show_erfs:
            fig_, ax_ = plt.subplots(1, 1)
            ax_.plot(x_arg, data, linestyle='', marker='o')
            ax_.plot(x_arg, erf_fit(out[i], x_arg), linestyle='--')
            fig_.tight_layout()
    out = np.stack(out)
    sigmas = out[:, 3]
    if plot:
        fig, ax = plt.subplots(1, 1)
        if x_plot is None:
            x_plot = range(n_curves)
        ax.plot(x_plot, sigmas, marker='x', linestyle='')
        fit_2 = np.polyfit(range(n_curves), sigmas, 10)
        x_2 = np.arange(0, n_curves, 0.1)
        x_2_plot = np.linspace(x_plot[0], x_plot[-1], len(x_2))
        ax.plot(x_2_plot, np.polyval(fit_2, x_2), linestyle='--')
        plt.show()
    return out


def residual(params, data, x) -> np.array:
    """  this minimizes the difference between the data and the model
    """
    return erf_fit(params, x) - data


def erf_fit(params, x=[]):
    """
    From the gpufit cuh "error_function.cuh" file:

    Params: array-like
    ------
    0: offset
    1: amplitude
    2: center
    3: sigma
    """
    offset, amplitude, center, sigma = params
    argx = (x - center) / (np.sqrt(2.0) * sigma)
    oneperf = 0.5 * (1 + erf(argx))
    return amplitude * oneperf + offset


if __name__ == "__main__":
    files = Path("D:/Data/MIT/focus2")
    inst = napari_focus_gui(files)
    napari.run()
